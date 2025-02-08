import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from typing import Dict, List
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceRangeDataset:
    def __init__(self, json_file: str):
        with open(json_file, 'r') as f:
            self.raw_data = json.load(f)['data']
        self.label2id = {
            "O": 0,
            "not_price": 1,
            "min_price": 2,
            "max_price": 3
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Log dataset statistics
        logger.info(f"Loaded {len(self.raw_data)} examples")
        label_counts = {label: 0 for label in self.label2id.keys()}
        for item in self.raw_data:
            for num in item['numbers']:
                label_counts[num['type']] += 1
        logger.info(f"Label distribution in raw data: {label_counts}")

    def preprocess_data(self, tokenizer):
        """Convert raw JSON data into format suitable for training with focus on numbers"""
        processed_data = []
        skipped_items = 0
        lengths = []
        for idx, item in enumerate(self.raw_data):
            text = item['input_text']
            numbers = item['numbers']

            # Debug logging for each example
            logger.debug(f"\nProcessing example {idx}:")
            logger.info(f"Text: {text}")
            logger.debug(f"Numbers: {numbers}")

            # Tokenize the text
            tokenized = tokenizer(
                text,
                truncation=False,
                max_length=256,
                return_offsets_mapping=True
            )

            # Initialize labels for all tokens as -100 (ignored in loss computation)
            labels = [-100] * len(tokenized["input_ids"])

            # Sort numbers by their position in text to handle overlapping matches
            numbers = sorted(numbers, key=lambda x: text.find(str(x['value'])))

            # Process each number
            for num in numbers:
                value = str(num['value'])
                type_label = num['type']

                # Build patterns that handle various formats including hyphenated ranges
                base_patterns = [
                    value,  # Basic number
                    value.rstrip('.0')  # Without trailing .0
                ]

                for base in base_patterns:
                    # Patterns that handle numbers in ranges (e.g., $300-375 or 300-375)
                    if type_label == 'min_price':
                        patterns = [
                            fr'\$?{re.escape(base)}(?:\.\d*)?-',  # Matches "$300-" or "300-"
                            fr'\$?{re.escape(base)}(?:\.\d*)?(?=\s|$|[.,!?;:]|\))'  # Standalone number
                        ]
                    elif type_label == 'max_price':
                        patterns = [
                            fr'-{re.escape(base)}(?:\.\d*)?(?=\s|$|[.,])',  # Matches "-375" or "-375.0"
                            fr'\$?{re.escape(base)}(?:\.\d*)?(?=\s|$|[.,!?;:]|\))' # Standalone number
                        ]
                    else:
                        patterns = [
                            fr'\$?{re.escape(base)}(?:\.\d*)?(?=\s|$|[.,!?;:]|\))'  # Standard pattern for other types
                        ]

                    # Try each pattern
                    for pattern in patterns:
                        # logger.debug(f"Trying pattern: {pattern}")
                        matches = list(re.finditer(pattern, text))

                        for match in matches:
                            start_char, end_char = match.span()
                            matched_text = text[start_char:end_char]

                            # Debug logging
                            # logger.debug(f"Found match '{matched_text}' at positions {start_char}:{end_char}")
                            # logger.debug(f"Type label: {type_label}")

                            # Find corresponding tokens
                            token_start = None
                            token_end = None

                            for token_idx, (token_start_char, token_end_char) in enumerate(tokenized["offset_mapping"]):
                                # Check if token overlaps with the number
                                if (token_start_char <= start_char < token_end_char or
                                        token_start_char < end_char <= token_end_char or
                                        (start_char <= token_start_char and end_char >= token_end_char)):
                                    if token_start is None:
                                        token_start = token_idx
                                    token_end = token_idx + 1

                            if token_start is not None:
                                # Assign labels only to number tokens
                                for idx in range(token_start, token_end):
                                    labels[idx] = self.label2id[type_label]
                                logger.debug(f"Assigned label {type_label} to tokens {token_start}:{token_end}")

                    # Log current state of labels
                    logger.debug(f"Current labels after processing {value}: {labels}")

            # Count successful matches
            successful_matches = sum(1 for label in labels if label != -100)

            lengths.append(len(tokenized["input_ids"])*2 + len(labels) + len(tokenized["attention_mask"]))
            if successful_matches > 0:
                processed_data.append({
                    'input_ids': tokenized["input_ids"],
                    'attention_mask': tokenized["attention_mask"],
                    'labels': labels
                })
            else:
                skipped_items += 1
                logger.warning(f"Skipped item due to no successful number matches: {text}")

            # Log label distribution for this example
            label_counts = {self.id2label[i]: labels.count(i) for i in self.label2id.values()}
            logger.debug(f"Label distribution for example {idx}: {label_counts}")

        logger.info(f"Processed {len(processed_data)} examples successfully")
        logger.info(f"Skipped {skipped_items} examples due to no successful number matches")
        lengths = np.array(lengths)

        # Print statistics
        # print("\nToken Length Statistics:")
        # print(f"Number of examples: {len(lengths)}")
        # print(f"Maximum length: {np.max(lengths)}")
        # print(f"Minimum length: {np.min(lengths)}")
        # print(f"Mean length: {np.mean(lengths):.2f}")
        # print(f"Median length: {np.median(lengths):.2f}")
        # print(f"90th percentile: {np.percentile(lengths, 90):.2f}")
        # print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
        # print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
        return processed_data


def compute_metrics(pred):
    """Custom metrics function that only evaluates number tokens"""
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids

    # Only consider tokens that have actual labels (not -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    # Calculate metrics only for number tokens
    metrics = {
        'accuracy': (predictions == labels).mean(),
        'num_number_tokens': len(labels)
    }

    # Calculate per-class metrics
    for label_id in range(4):  # 0 to 3
        label_mask = labels == label_id
        if label_mask.sum() > 0:
            class_accuracy = (predictions[label_mask] == labels[label_mask]).mean()
            metrics[f'class_{label_id}_accuracy'] = class_accuracy
            metrics[f'class_{label_id}_count'] = label_mask.sum()

    return metrics


def create_model_and_tokenizer():
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model and tokenizer
    model_name = "google/gemma-2b-it"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        num_labels=4,  # O, not_price, min_price, max_price
        id2label={0: "O", 1: "not_price", 2: "min_price", 3: "max_price"},
        label2id={"O": 0, "not_price": 1, "min_price": 2, "max_price": 3}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.TOKEN_CLS
    )

    # Create PEFT model
    model = get_peft_model(model, lora_config)

    logger.info(f"Model parameters: {model.num_parameters()}")
    return model, tokenizer


class DebugTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step % 10 == 0:  # Log every 10 steps
            logger.info(f"Step {self.state.global_step}: loss = {loss.item():.4f}")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        # Log a sample of predictions vs actual labels
        if self.state.global_step % 50 == 0:  # Log every 50 steps
            if not prediction_loss_only and outputs is not None:
                predictions = outputs[1].argmax(-1)
                labels = inputs["labels"]
                for i in range(min(2, len(predictions))):  # Log first 2 examples
                    logger.info(f"\nExample {i}:")
                    logger.info(inputs["input_ids"])
                    logger.info(f"Predictions: {predictions[i]}")
                    logger.info(f"True labels: {labels[i]}")
        return outputs


def create_trainer(model, tokenizer, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir="./price_classifier",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        push_to_hub=False,
        weight_decay=0.05,  # Increased weight decay
        warmup_ratio=0.1,  # Add warmup
        max_grad_norm=1.0,  # Add gradient clipping
        lr_scheduler_type="cosine_with_restarts",
        metric_for_best_model="accuracy",
    )

    # Create data collator for token classification
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Add custom metrics
    )

    return trainer


def main():
    # Create model and tokenizer first
    model, tokenizer = create_model_and_tokenizer()

    # Load and preprocess data
    dataset = PriceRangeDataset('price_training_data.json')
    processed_data = dataset.preprocess_data(tokenizer)

    # Split into train and eval
    train_size = int(0.8 * len(processed_data))

    # Log split sizes
    logger.info(f"Train size: {train_size}")
    logger.info(f"Eval size: {len(processed_data) - train_size}")

    # Create datasets
    train_dataset = Dataset.from_dict({
        'input_ids': [item['input_ids'] for item in processed_data[:train_size]],
        'attention_mask': [item['attention_mask'] for item in processed_data[:train_size]],
        'labels': [item['labels'] for item in processed_data[:train_size]]
    })

    eval_dataset = Dataset.from_dict({
        'input_ids': [item['input_ids'] for item in processed_data[train_size:]],
        'attention_mask': [item['attention_mask'] for item in processed_data[train_size:]],
        'labels': [item['labels'] for item in processed_data[train_size:]]
    })

    # Create and start trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset)

    # Log dataset statistics
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")

    # Save the model
    trainer.save_model("./price_classifier_final")
    logger.info("Model saved")


if __name__ == "__main__":
    main()