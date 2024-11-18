import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BertConfig
)
import os
import json
from typing import List, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import re
from tqdm import tqdm


def prepare_training_data() -> List[Dict]:
    """
    Prepare multi-intent training data for restaurant queries.
    Returns a list of dictionaries with 'query' and 'intents' keys.
    """
    return [
        # ENQUIRY_MENU Examples

        # Multi-Intent Examples: RESERVATION + Other

        # Special Service Requests
        {"query": "Can I see the menu please?",
         "intents": ["ENQUIRY_MENU"]},
        {"query": "What's on your specials menu today?",
         "intents": ["ENQUIRY_MENU"]},
        {"query": "Show me the dessert menu",
         "intents": ["ENQUIRY_MENU"]},

        # ENQUIRY_CUISINE Examples
        {"query": "Do you serve Italian food?",
         "intents": ["ENQUIRY_CUISINE"]},
        {"query": "What type of cuisine do you specialize in?",
         "intents": ["ENQUIRY_CUISINE"]},
        {"query": "Is this an Indian restaurant?",
         "intents": ["ENQUIRY_CUISINE"]},
        {"query": "Do you serve authentic Mexican food?",
         "intents": ["ENQUIRY_CUISINE"]},
        {"query": "What's your signature cuisine?",
         "intents": ["ENQUIRY_CUISINE"]},
        {"query": "Are you a fusion restaurant?",
         "intents": ["ENQUIRY_CUISINE"]},

        # ENQUIRY_DISH Examples
        {"query": "Is your pasta gluten-free?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "How spicy is the curry?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "What's in the house special sandwich?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "Are there nuts in this dish?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "Is the steak grass-fed?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "What vegetables come with the fish?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "Do you have vegan options?",
         "intents": ["ENQUIRY_DISH"]},
        {"query": "Are your desserts made in-house?",
         "intents": ["ENQUIRY_DISH"]},

        # ENQUIRY_RESTAURANT Examples
        {"query": "Do you have outdoor seating?",
         "intents": ["ENQUIRY_RESTAURANT"]},
        {"query": "What are your opening hours?",
         "intents": ["ENQUIRY_RESTAURANT"]},
        {"query": "Is there parking available?",
         "intents": ["ENQUIRY_RESTAURANT"]},
        {"query": "Do you have private dining rooms?",
         "intents": ["ENQUIRY_RESTAURANT"]},


        # ORDER_RELATED Examples
        {"query": "I want to place a takeout order",
         "intents": ["ORDER_RELATED"]},
        {"query": "Can I modify my order?",
         "intents": ["ORDER_RELATED"]},
        {"query": "How long for delivery?",
         "intents": ["ORDER_RELATED"]},
        {"query": "I need to cancel my order",
         "intents": ["ORDER_RELATED"]},
        {"query": "Do you offer catering services?",
         "intents": ["ORDER_RELATED"]},
        {"query": "Can I place a large group order?",
         "intents": ["ORDER_RELATED"]},
        {"query": "What's the status of my order?",
         "intents": ["ORDER_RELATED"]},

        # RESERVATION_RELATED Examples
        {"query": "I'd like to book a table for tonight",
         "intents": ["RESERVATION_RELATED"]},
        # PAYMENT_RELATED Examples
        {"query": "Do you accept credit cards?",
         "intents": ["PAYMENT_RELATED"]},
        {"query": "Can I pay with Apple Pay?",
         "intents": ["PAYMENT_RELATED"]},


        # SERVICE_RELATED Examples
        {"query": "Can I speak to the manager?",
         "intents": ["SERVICE_RELATED"]},
        {"query": "The service is too slow",
         "intents": ["SERVICE_RELATED"]},


        # Multi-Intent Examples: ENQUIRY_MENU + Others
        {"query": "Show me the menu and book a table",
         "intents": ["ENQUIRY_MENU", "RESERVATION_RELATED"]},
        {"query": "What's on the menu and do you deliver?",
         "intents": ["ENQUIRY_MENU", "ORDER_RELATED"]},

        # Multi-Intent Examples: RESERVATION + Others
        {"query": "Book a table and specify dietary requirements",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_DISH"]},
        {"query": "Make reservation and ask about parking",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_RESTAURANT"]},

        # Multi-Intent Examples: ORDER + Others
        {"query": "Place order and pay with card",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED"]},
        {"query": "Order delivery and check ingredients",
         "intents": ["ORDER_RELATED", "ENQUIRY_DISH"]},
        {"query": "Modify order and update payment",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED"]},
        {"query": "Cancel order and get refund",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED"]},

        # Complex Multi-Intent Examples
        {"query": "Book table, check menu, and ask about parking",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU", "ENQUIRY_RESTAURANT"]},
        {"query": "Order food, pay online, and check delivery time",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED", "SERVICE_RELATED"]},

        # Specific Time-Related Queries
        {"query": "Book table for tonight and check menu",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},
        {"query": "Order lunch delivery and pay online",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED"]},
        {"query": "Weekend reservation and dietary requirements",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_DISH"]},

        # Location-Specific Queries
        {"query": "Indoor seating reservation and parking info",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_RESTAURANT"]},
        {"query": "Outdoor table booking and menu check",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},
        {"query": "Private room booking and catering menu",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},

        # Special Occasion Queries
        {"query": "Birthday party reservation and special menu",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},
        {"query": "Anniversary dinner booking and wine list",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},
        {"query": "Group booking and set menu options",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},

        # Dietary Requirement Queries
        {"query": "Check gluten-free options and make reservation",
         "intents": ["ENQUIRY_DISH", "RESERVATION_RELATED"]},


        # Service and Payment Combined
        {"query": "Speak to manager about payment issue",
         "intents": ["SERVICE_RELATED", "PAYMENT_RELATED"]},


        # Advanced Booking Scenarios
        {"query": "Change reservation date and update dietary needs",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_DISH"]},


        # Complex Order Modifications
        {"query": "Update delivery address and payment method",
         "intents": ["ORDER_RELATED", "PAYMENT_RELATED"]},


        # Special Service Requests
        {"query": "Request high chair and check kids menu",
         "intents": ["ENQUIRY_RESTAURANT", "ENQUIRY_MENU"]},
        {"query": "Need wheelchair access and special assistance",
         "intents": ["ENQUIRY_RESTAURANT", "SERVICE_RELATED"]},
        {"query": "Check dress code and parking availability",
         "intents": ["ENQUIRY_RESTAURANT"]},

        # Seasonal and Special Event Queries
        {"query": "Holiday menu options and booking availability",
         "intents": ["ENQUIRY_MENU", "RESERVATION_RELATED"]},
        {"query": "Christmas dinner booking and menu preview",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},
        {"query": "New Year's Eve reservation and special menu",
         "intents": ["RESERVATION_RELATED", "ENQUIRY_MENU"]},

        # Advanced Payment Scenarios
        {"query": "Split bill for large group and service charge",
         "intents": ["PAYMENT_RELATED", "SERVICE_RELATED"]},

    ]


class RestaurantDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


class IntentClassifier:
    def __init__(self,
                 model_name: str = "google-bert/bert-large-uncased",
                 cache_dir: str = './model_cache',
                 output_dir: str = "./intent_classifier",
                 max_length: int = 128):

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.max_length = max_length

        # Define intent labels
        self.intent_labels = [
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

        # Initialize label mappings
        self.label2id = {label: i for i, label in enumerate(self.intent_labels)}
        self.id2label = {i: label for i, label in enumerate(self.intent_labels)}

        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def get_training_args(self):
        """Define training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            # Optimal batch size for fine-tuning
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            # Increased epochs
            num_train_epochs=20,
            # Optimal learning rate found through experimentation
            learning_rate=1e-5,
            # Strong regularization
            weight_decay=0.1,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=5,
            # Very frequent evaluation
            evaluation_strategy="steps",
            eval_steps=5,
            save_strategy="steps",
            save_steps=5,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=3,
            # Gradient accumulation for larger effective batch size
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),
            # Longer warmup
            warmup_ratio=0.2,
            # Stronger gradient clipping
            max_grad_norm=0.3,
            # Advanced scheduling
            lr_scheduler_type="cosine_with_restarts"
        )

    def setup_model(self):
        """Setup model for multi-label classification"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )

        print("Loading model...")
        config = BertConfig.from_pretrained(
            self.model_name,
            num_labels=len(self.intent_labels),
            problem_type="multi_label_classification",
            label2id=self.label2id,
            id2label=self.id2label,
            hidden_dropout_prob=0.15,
            attention_probs_dropout_prob=0.15,
            classifier_dropout=0.15,
            hidden_act="gelu_new",
            layer_norm_eps=1e-7,
            position_embedding_type="absolute",
            use_cache=True,
            # Add focal loss parameters
            loss_type="focal",
            gamma=2.0,
            alpha=0.25
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            cache_dir=self.cache_dir
        )

        class AdvancedClassificationHead(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dropout = torch.nn.Dropout(config.classifier_dropout)
                self.intermediate = torch.nn.Linear(config.hidden_size, config.hidden_size * 2)
                self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps)
                self.output = torch.nn.Linear(config.hidden_size * 2, config.num_labels)
                self.activation = torch.nn.GELU()

            def forward(self, features):
                x = self.dropout(features)
                x = self.intermediate(x)
                x = self.activation(x)
                x = self.layer_norm(x)
                x = self.dropout(x)
                x = self.output(x)
                return x

        self.model.classifier = AdvancedClassificationHead(config)
        self.model.to(self.device)

    def prepare_data(self, custom_data: List[Dict] = None):
        """
        Prepare data for multi-intent classification.
        Maintains original augmentation and processing while handling multiple intents.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            data = custom_data if custom_data else prepare_training_data()

            augmented_data = []
            for item in tqdm(data, desc="Augmenting data"):
                query = item["query"]
                intents = item["intents"]

                # Create multi-hot encoded label vector
                label_vector = [0] * len(self.intent_labels)
                for intent in intents:
                    label_vector[self.label2id[intent]] = 1

                # Original data
                augmented_data.append({"query": query, "label_vector": label_vector})

                # Basic augmentations
                augmented_data.append({"query": query.lower(), "label_vector": label_vector})
                augmented_data.append({"query": re.sub(r'[^\w\s]', '', query), "label_vector": label_vector})

                # Add intent-specific markers
                if "ENQUIRY_MENU" in intents:
                    augmented_data.append({"query": f"[MENU] {query}", "label_vector": label_vector})
                elif "RESERVATION_RELATED" in intents:
                    augmented_data.append({"query": f"[RESERVATION] {query}", "label_vector": label_vector})
                elif "PAYMENT_RELATED" in intents:
                    augmented_data.append({"query": f"[PAYMENT] {query}", "label_vector": label_vector})
                elif "SERVICE_RELATED" in intents:
                    augmented_data.append({"query": f"[SERVICE] {query}", "label_vector": label_vector})

                # Add variations with different sentence structures
                if "?" not in query:
                    augmented_data.append({"query": f"{query}?", "label_vector": label_vector})
                if "please" not in query.lower():
                    augmented_data.append({"query": f"Please {query}", "label_vector": label_vector})
                if "can you" not in query.lower():
                    augmented_data.append({"query": f"Can you {query}", "label_vector": label_vector})
                if "i want to" not in query.lower():
                    augmented_data.append({"query": f"I want to {query}", "label_vector": label_vector})

                # Add negations for better context learning
                if "SERVICE_RELATED" in intents:
                    augmented_data.append({"query": f"Not happy with {query}", "label_vector": label_vector})
                    augmented_data.append({"query": f"Issue with {query}", "label_vector": label_vector})

                # Add time-related variations
                augmented_data.append({"query": f"{query} now", "label_vector": label_vector})
                augmented_data.append({"query": f"{query} today", "label_vector": label_vector})

                # Add urgency variations
                augmented_data.append({"query": f"{query} immediately", "label_vector": label_vector})
                augmented_data.append({"query": f"Urgent: {query}", "label_vector": label_vector})

            # Convert to DataFrame and clean
            df = pd.DataFrame(augmented_data)
            df['query'] = df['query'].apply(self._clean_text)
            df = df.drop_duplicates(subset=['query'])

            # Simple random split instead of stratified split
            train_df, val_df = train_test_split(
                df,
                test_size=0.1,
                random_state=42
            )

            # Create datasets
            train_dataset = self._create_dataset(train_df)
            val_dataset = self._create_dataset(val_df)

            print("\nDataset statistics:")
            print(f"Total samples after augmentation: {len(df)}")
            print(f"Training samples: {len(train_df)}")
            print(f"Validation samples: {len(val_df)}")

            # Print intent distribution
            print("\nIntent distribution in training set:")
            train_intent_dist = {intent: 0 for intent in self.intent_labels}
            for label_vector in train_df['label_vector']:
                for i, val in enumerate(label_vector):
                    if val == 1:
                        train_intent_dist[self.id2label[i]] += 1

            for intent, count in train_intent_dist.items():
                print(f"{intent}: {count}")

            return train_dataset, val_dataset

        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _create_dataset(self, df: pd.DataFrame) -> RestaurantDataset:
        """Create a dataset from a DataFrame"""
        encodings = self.tokenizer(
            df['query'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return RestaurantDataset(encodings, df['label_vector'].tolist())

    def compute_metrics(self, eval_pred):
        """Compute metrics for multi-label classification"""
        predictions, labels = eval_pred
        predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()

        # Calculate metrics
        metrics = {}

        # Per-label metrics
        for i, intent in enumerate(self.intent_labels):
            pred_i = predictions[:, i]
            label_i = labels[:, i]

            tp = np.sum((pred_i == 1) & (label_i == 1))
            fp = np.sum((pred_i == 1) & (label_i == 0))
            fn = np.sum((pred_i == 0) & (label_i == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[f"eval_{intent}_f1"] = f1

        # Overall metrics
        exact_match = np.mean(np.all(predictions == labels, axis=1))
        macro_f1 = np.mean([metrics[f"eval_{intent}_f1"] for intent in self.intent_labels])

        metrics["eval_exact_match"] = exact_match
        metrics["eval_macro_f1"] = macro_f1
        metrics["eval_accuracy"] = exact_match  # Added accuracy metric

        return metrics

    def train(self, custom_data=None):
        """Train the model"""
        try:
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_data(custom_data)

            # Setup trainer
            training_args = self.get_training_args()
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )

            print("Starting training...")
            train_result = self.trainer.train()

            print("Evaluating...")
            eval_result = self.trainer.evaluate()

            # Save model
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            return train_result, eval_result

        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        # Keep original implementation
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\?\!\.]', '', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'([.!,;:])\1+', r'\1', text)
        text = text.strip()
        return text

    def predict(self, text: str, threshold: float = 0.5):
        """Modified prediction for multi-label while keeping original structure"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Keep original preprocessing
        text = self._clean_text(text)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        # Modified prediction logic for multi-label
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities > threshold).bool()

        # Get predicted intents and probabilities
        predicted_intents = []
        intent_probabilities = {}

        for idx, (prob, pred) in enumerate(zip(probabilities[0], predictions[0])):
            intent = self.id2label[idx]
            prob_value = prob.item()
            intent_probabilities[intent] = prob_value

            if pred:
                predicted_intents.append(intent)

        return {
            "intents": predicted_intents,
            "probabilities": dict(sorted(intent_probabilities.items(), key=lambda x: x[1], reverse=True))
        }


def main():
    """Main function to demonstrate usage"""
    try:
        print("\n=== Restaurant Multi-Intent Classification System ===\n")

        # Initialize classifier
        classifier = IntentClassifier()
        classifier.setup_model()

        # Train the model
        print("\nStarting training phase...")
        train_result, eval_result = classifier.train()

        print("\n=== Training Results ===")
        print(f"Training Loss: {train_result.training_loss:.4f}")
        print(f"Evaluation Macro F1: {eval_result['eval_macro_f1']:.4f}")
        print(f"Exact Match Score: {eval_result['eval_exact_match']:.4f}")

        # Test predictions
        test_queries = [
            "Split bill for large group and service charge",
            "I want to book a table and know about parking",
            "Request high chair and check kids menu",
            "Do you have parking, take reservations, and serve vegan food?",
            "What's on the menu, do you take reservations, and accept cards?"
        ]

        print("\nTesting predictions:")
        for query in test_queries:
            prediction = classifier.predict(query)
            print(f"\nQuery: '{query}'")
            print(f"Predicted intents: {prediction['intents']}")
            print("Probabilities:")
            for intent, prob in prediction['probabilities'].items():
                if prob > 0.3:  # Show only significant probabilities
                    print(f"  {intent}: {prob:.4f}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()