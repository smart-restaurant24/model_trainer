import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType, PeftModel
)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
import logging
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the list of possible intents
INTENT_LABELS = [
    "ENQUIRY_MENU",
    "ENQUIRY_CUISINE",
    "ENQUIRY_DISH",
    "ENQUIRY_RESTAURANT",
    "ORDER_RELATED",
    "RESERVATION_RELATED",
    "PAYMENT_RELATED",
    "GENERAL",
    "SERVICE_RELATED",
    "NON_RELATED"
]


class RestaurantIntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Created dataset with {len(data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]

        # Convert intents to multi-hot encoding
        labels = torch.zeros(len(INTENT_LABELS))
        for intent in item["intents"]:
            labels[INTENT_LABELS.index(intent)] = 1

        # Tokenize the text
        encoding = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Debug print for first few samples
        if idx < 3:
            logger.debug(f"Sample {idx}:")
            logger.debug(f"Query: {query}")
            logger.debug(f"Intents: {item['intents']}")
            logger.debug(f"Encoded length: {len(encoding['input_ids'][0])}")

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels
        }


def print_example_predictions(predictions, labels, queries, tokenizer):
    """Print detailed prediction examples for debugging"""
    predictions = torch.sigmoid(torch.Tensor(predictions))
    binary_predictions = (predictions > 0.5).float()

    for i in range(min(5, len(queries))):  # Print first 5 examples
        logger.info("\n--- Example Prediction ---")
        logger.info(f"Query: {queries[i]}")
        logger.info("True intents: " + ", ".join([INTENT_LABELS[j] for j, label in enumerate(labels[i]) if label == 1]))
        logger.info("Predicted intents: " + ", ".join(
            [INTENT_LABELS[j] for j, pred in enumerate(binary_predictions[i]) if pred == 1]))
        logger.info("Prediction scores: ")
        for j, score in enumerate(predictions[i]):
            logger.info(f"{INTENT_LABELS[j]}: {score:.4f}")
        logger.info("-------------------")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Apply sigmoid to get probabilities
    predictions = torch.sigmoid(torch.Tensor(predictions))
    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).float()

    # Calculate metrics
    f1_micro = f1_score(labels, binary_predictions, average='micro')
    f1_macro = f1_score(labels, binary_predictions, average='macro')
    accuracy = accuracy_score(labels, binary_predictions)

    # Get per-class metrics
    class_report = classification_report(
        labels,
        binary_predictions,
        target_names=INTENT_LABELS,
        output_dict=True
    )

    # Log detailed metrics
    logger.info("\n=== Evaluation Metrics ===")
    logger.info(f"Micro F1: {f1_micro:.4f}")
    logger.info(f"Macro F1: {f1_macro:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nPer-class metrics:")
    for intent in INTENT_LABELS:
        logger.info(f"{intent}:")
        logger.info(f"  Precision: {class_report[intent]['precision']:.4f}")
        logger.info(f"  Recall: {class_report[intent]['recall']:.4f}")
        logger.info(f"  F1: {class_report[intent]['f1-score']:.4f}")

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "accuracy": accuracy
    }


class DebugTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step % 10 == 0:  # Log every 10 steps
            logger.info(f"\n=== Training Step {self.state.global_step} ===")
            logger.info(f"Loss: {loss.item():.4f}")

            # Log batch statistics
            logger.info(f"Batch size: {inputs['input_ids'].shape[0]}")
            logger.info(f"Sequence length: {inputs['input_ids'].shape[1]}")

            # Log memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(f"GPU Memory used: {gpu_memory:.2f} MB")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Log predictions vs actual labels periodically
        if self.state.global_step % 50 == 0 and not prediction_loss_only and outputs is not None:
            loss, logits, labels = outputs
            predictions = torch.sigmoid(logits)  # Apply sigmoid for multi-label

            # Get the original text for logging
            input_ids = inputs["input_ids"]
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            logger.info("\n=== Prediction Examples ===")
            for i in range(min(2, len(predictions))):  # Log first 2 examples
                logger.info(f"\nExample {i + 1}:")
                logger.info(f"Text: {texts[i]}")
                logger.info("Predicted intents:")
                for j, pred in enumerate(predictions[i]):
                    if pred > 0.5:  # Threshold for positive prediction
                        logger.info(f"- {INTENT_LABELS[j]}: {pred.item():.4f}")

                logger.info("True intents:")
                for j, label in enumerate(labels[i]):
                    if label == 1:
                        logger.info(f"- {INTENT_LABELS[j]}")

                logger.info("---")

        return outputs

    def log_metrics(self, split, metrics):
        """Enhanced metrics logging"""
        super().log_metrics(split, metrics)

        if split == "eval":
            logger.info("\n=== Evaluation Metrics ===")
            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")

            # Add additional custom metric logging if needed
            if "eval_loss" in metrics:
                logger.info(f"Current eval loss: {metrics['eval_loss']:.4f}")


def main():
    # Load and prepare data
    logger.info("Loading data...")
    with open("training_data.json", "r") as f:
        data = json.load(f)

    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model...")
    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split data into train and validation sets
    full_dataset = RestaurantIntentDataset(data, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Configure quantization
    logger.info("Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENT_LABELS),
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="./model_cache"
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./intent_classification_model",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        push_to_hub=False,
    )

    # Initialize debug trainer
    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer  # Added tokenizer for debug logging
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("Performing final evaluation...")
    eval_results = trainer.evaluate()
    logger.info("Final evaluation results:")
    logger.info(eval_results)

    # Save the model
    logger.info("Saving model...")
    trainer.save_model("./final_model")
    logger.info("Training completed!")


def test_model_predictions(model_path, test_queries):
    """Test the model on some example queries"""
    logger.info("Testing model predictions...")
    model, tokenizer = load_trained_model(model_path)

    for query in test_queries:
        predicted_intents = predict_intents(query, model, tokenizer)
        logger.info(f"\nQuery: {query}")
        logger.info(f"Predicted intents: {predicted_intents}")


def load_trained_model(model_path):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "google/gemma-2-2b-it",
        num_labels=len(INTENT_LABELS)
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    return model, tokenizer


def predict_intents(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities > 0.5).float()

    predicted_intents = [
        INTENT_LABELS[i]
        for i, pred in enumerate(predictions[0])
        if pred == 1
    ]
    return predicted_intents


if __name__ == "__main__":
    # Create output directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./intent_classification_model", exist_ok=True)

    main()

    # Test the model with some example queries
    test_queries = [
        "What's in the pasta and are you specialized in Mediterranean cuisine?",
        "Can I make a reservation for tonight and do you accept credit cards?",
        "Is the chef available to discuss the menu?"
    ]
    # test_model_predictions("./final_model", test_queries)