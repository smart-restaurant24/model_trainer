import torch
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import logging
from dataclasses import dataclass

from preference_model import PreferenceModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_training_data(raw_data, tokenizer, label_sets, max_length=128):
    """Process raw training data into the format expected by the model."""
    processed_data = []
    max_tokens_needed = 0  # Track maximum tokens needed

    # Create label to index mappings
    label_to_idx = {
        category: {label: idx for idx, label in enumerate(labels)}
        for category, labels in label_sets.items()
    }

    for item in raw_data:
        # Tokenize text without truncation first to check length
        full_encodings = tokenizer(
            item['input_text'],
            truncation=False,
            add_special_tokens=True
        )
        max_tokens_needed = max(max_tokens_needed, len(full_encodings['input_ids']))

        # Now tokenize with truncation for processing
        encodings = tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            add_special_tokens=True,
            padding_side='right'
        )

        # Initialize label tensors
        label_vectors = {
            category: torch.zeros(len(labels))
            for category, labels in label_sets.items()
        }
        sentiment_vectors = {
            category: torch.zeros(len(labels))
            for category, labels in label_sets.items()
        }

        # Process labels for each category
        for category in label_sets.keys():
            for label_info in item['labels'][category]:
                label_lower = label_info['label'].lower()

                if label_lower in label_to_idx[category]:
                    label_idx = label_to_idx[category][label_lower]
                    label_vectors[category][label_idx] = 1.0
                    sentiment_vectors[category][label_idx] = float(label_info['sentiment'] > 0)

        # Create training example
        training_example = {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
        }

        # Add labels and sentiments
        for category in label_sets:
            training_example[f"{category}_labels"] = label_vectors[category]
            training_example[f"{category}_sentiment"] = sentiment_vectors[category]

        processed_data.append(training_example)

    # Log the maximum tokens needed
    logger.info(f"Maximum tokens needed in dataset: {max_tokens_needed}")
    logger.info(f"Current max_length setting: {max_length}")
    if max_tokens_needed > max_length:
        logger.warning(f"Some examples exceed max_length and will be truncated. "
                       f"Consider increasing max_length to {max_tokens_needed} if possible.")

    print(raw_data[0])
    print(processed_data[0])
    return processed_data


def verify_processed_data(processed_data, label_sets):
    """Verify that processed data matches expected format and dimensions."""
    if not processed_data:
        logger.error("No processed data found!")
        return False

    example = processed_data[0]

    # Check required keys
    required_keys = {'input_ids', 'attention_mask'}
    for category in label_sets:
        required_keys.add(f"{category}_labels")
        required_keys.add(f"{category}_sentiment")

    missing_keys = required_keys - set(example.keys())
    if missing_keys:
        logger.error(f"Missing required keys: {missing_keys}")
        return False

    # Check tensor dimensions
    for category, labels in label_sets.items():
        label_tensor = example[f"{category}_labels"]
        sentiment_tensor = example[f"{category}_sentiment"]

        expected_size = len(labels)
        if label_tensor.size(0) != expected_size:
            logger.error(f"Incorrect {category}_labels dimension. Expected {expected_size}, got {label_tensor.size(0)}")
            return False
        if sentiment_tensor.size(0) != expected_size:
            logger.error(
                f"Incorrect {category}_sentiment dimension. Expected {expected_size}, got {sentiment_tensor.size(0)}")
            return False

    logger.info("Data format verification passed!")
    return True


class PreferenceMetrics:
    """Metrics calculator for preference classification"""

    def __init__(self, label_sets: Dict[str, list]):
        self.label_sets = label_sets

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        results = {}

        # Calculate start indices for each category
        start_indices = {}
        current_idx = 0
        for category, category_labels in self.label_sets.items():
            start_indices[category] = current_idx
            current_idx += len(category_labels)

        # Process each category
        for category, category_labels in self.label_sets.items():
            idx = start_indices[category]
            size = len(category_labels)

            # Get category-specific predictions and labels
            presence_preds = torch.sigmoid(torch.tensor(logits[idx:idx + size])) > 0.5
            presence_labels = labels[idx:idx + size]

            sentiment_preds = torch.sigmoid(torch.tensor(logits[idx + size:idx + 2 * size])) > 0.5
            sentiment_labels = labels[idx + size:idx + 2 * size]

            # Calculate metrics for presence
            precision, recall, f1, _ = precision_recall_fscore_support(
                presence_labels,
                presence_preds,
                average='weighted'
            )
            accuracy = accuracy_score(presence_labels, presence_preds)

            # Store results
            results[f"{category}_precision"] = precision
            results[f"{category}_recall"] = recall
            results[f"{category}_f1"] = f1
            results[f"{category}_accuracy"] = accuracy

            # Calculate sentiment metrics where label is present
            mask = presence_labels > 0.5
            if mask.any():
                sent_precision, sent_recall, sent_f1, _ = precision_recall_fscore_support(
                    sentiment_labels[mask],
                    sentiment_preds[mask],
                    average='weighted'
                )
                sent_accuracy = accuracy_score(
                    sentiment_labels[mask],
                    sentiment_preds[mask]
                )

                results[f"{category}_sentiment_precision"] = sent_precision
                results[f"{category}_sentiment_recall"] = sent_recall
                results[f"{category}_sentiment_f1"] = sent_f1
                results[f"{category}_sentiment_accuracy"] = sent_accuracy

        return results


class PreferenceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_sets = self.model.label_sets
        self.metrics = PreferenceMetrics(self.label_sets)

        self.pred_logger = logging.getLogger("prediction_logger")
        if not self.pred_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.pred_logger.addHandler(handler)
            self.pred_logger.setLevel(logging.INFO)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step that logs input query and detailed predictions"""
        # Get the regular prediction output
        if not model.training:
            prediction_loss_only = False

        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        loss, logits, labels = outputs

        is_training = model.training
        self.pred_logger.info(f"Current mode: {'training' if is_training else 'evaluation'}")
        # Only log during eval steps (not during training)

        # print("\nIn prediction_step:")
        # print("Loss type:", type(loss))
        # print("Logits type:", type(logits))
        # print("Labels type:", type(labels))
        #
        # if isinstance(logits, tuple):
        #     print("Logits tuple length:", len(logits))
        #     print("Logits tuple contents:", [type(x) for x in logits])

        # Convert tuple to tensor if needed
        if isinstance(logits, tuple) and len(logits) > 0:
            logits = logits[0] if isinstance(logits[0], torch.Tensor) else None

        if not prediction_loss_only and logits is not None:
            batch_size = inputs['input_ids'].size(0)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Process each item in the batch
            for batch_idx in range(batch_size):
                # Decode and log the input query
                input_text = self.tokenizer.decode(
                    inputs['input_ids'][batch_idx],
                    skip_special_tokens=True
                )
                self.pred_logger.info("\n" + "=" * 50)  # Use pred_logger instead of logger
                self.pred_logger.info(f"Input Query: {input_text}")
                self.pred_logger.info("=" * 50)

                # Track current position in the concatenated logits
                current_idx = 0

                # Process each category
                for category, labels_list in self.label_sets.items():
                    num_labels = len(labels_list)

                    # Get presence probabilities for this category
                    presence_probs = probs[batch_idx, current_idx:current_idx + num_labels]
                    sentiment_probs = probs[batch_idx, current_idx + num_labels:current_idx + 2 * num_labels]

                    # Log predictions above threshold
                    threshold = 0.3
                    predictions_found = False
                    category_output = [f"\n{category.upper()}:"]

                    for label_idx, (label, prob) in enumerate(zip(labels_list, presence_probs)):
                        if prob > threshold:
                            predictions_found = True
                            sentiment = "positive" if sentiment_probs[label_idx] > 0.5 else "negative"
                            category_output.append(f"  - {label}: {prob:.3f} ({sentiment})")

                    print(predictions_found)
                    # Only log category if we found predictions above threshold
                    if predictions_found:
                        self.pred_logger.info("\n".join(category_output))  # Use pred_logger

                    # Move to next category
                    current_idx += 2 * num_labels

        return loss, logits, labels


@dataclass
class PreferenceTrainingConfig:
    model_name: str = "google/gemma-2b"
    output_dir: str = "./preference_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 27
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 30
    max_length: int = 64
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    do_eval: bool = True  # Should be added
    do_predict: bool = True  # Should be added
    max_grad_norm: float = 0.5  # Added gradient clipping
    gradient_accumulation_steps: int = 8

def train_preference_model(
        config: PreferenceTrainingConfig,
        train_data: list,
        val_data: Optional[list] = None
):
    """Main training function with direct data processing"""
    # Initialize model
    print("lora rank is")
    print(config.lora_r)
    model = PreferenceModel(
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    tokenizer = model.tokenizer

    # Process training data directly
    processed_train_data = process_training_data(
        train_data,
        tokenizer,
        model.label_sets,
        config.max_length
    )

    # Verify training data format
    if not verify_processed_data(processed_train_data, model.label_sets):
        raise ValueError("Training data format verification failed!")

    # Process validation data if provided
    processed_val_data = None
    if val_data:
        processed_val_data = process_training_data(
            val_data,
            tokenizer,
            model.label_sets,
            config.max_length
        )
        if not verify_processed_data(processed_val_data, model.label_sets):
            raise ValueError("Validation data format verification failed!")

    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        learning_rate=config.learning_rate,
        do_eval=config.do_eval,
        do_predict=config.do_predict,
        fp16=True,
        save_strategy="no",
        gradient_accumulation_steps=8,
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type="cosine",
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=0.1,
        weight_decay=config.weight_decay,  # Important for regularization
        logging_first_step=True,
        label_names=[
            f"{cat}_{label_type}"
            for cat in model.label_sets.keys()
            for label_type in ['labels', 'sentiment']
        ]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,  # Increased epsilon for stability
        betas=(0.9, 0.999)  # Default Adam betas
    )
    # Initialize trainer with processed data
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_data,
        eval_dataset=processed_val_data,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )

    # Start training
    trainer.train()

    # Save model and tokenizer
    model.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    model_state = {
        'classifiers': model.classifiers.state_dict(),
        'attention_weights': model.attention_weights,
        'label_sets': model.label_sets,
        'model_config': {  # Save configuration for reproducibility
            'lora_r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout
        }
    }
    torch.save(model_state, f"{config.output_dir}/preference_model_state.pt")

    return model, trainer


def main():
    # Load training data
    with open('./restaurant_dataset.json', 'r') as f:
        data = json.load(f)

    # Split data into train and validation
    train_size = int(0.9 * len(data['data']))
    train_data = data['data'][:train_size]
    val_data = data['data'][train_size:]

    # Training configuration
    config = PreferenceTrainingConfig(
        output_dir="./preference_model_output",
        num_train_epochs=3,
        learning_rate=2e-4
    )

    # Train model
    model, trainer = train_preference_model(
        config=config,
        train_data=train_data,
        val_data=val_data
    )

    # Example inference
    test_texts = [
        "I want a spicy vegetarian dish with no mushrooms",
        "Looking for a gluten-free appetizer that's not too salty"
    ]

    predictions = model.predict(test_texts)

    print("\nPredictions:")
    for text, preds in zip(test_texts, predictions):
        print(f"\nInput: {text}")
        for category, category_preds in preds.items():
            if category_preds:
                print(f"\n{category.upper()}:")
                for pred in category_preds:
                    sentiment = "positive" if pred['sentiment'] > 0 else "negative"
                    print(f"- {pred['label']}: {pred['confidence']:.3f} ({sentiment})")


if __name__ == "__main__":
    main()