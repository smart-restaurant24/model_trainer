import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from preference_model import PreferenceModel, PreferenceHead
import json
from typing import List, Dict, Union
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreferenceModelInference(PreferenceModel):
    """Subclass for inference that skips initial model loading"""

    def __init__(self, model_path: str):
        # Skip the parent class __init__ entirely
        torch.nn.Module.__init__(self)

        # Load saved state
        logger.info(f"Loading model state from {model_path}")
        model_state = torch.load(f"{model_path}/preference_model_state.pt")

        # Set up attributes without loading model
        self.label_sets = model_state['label_sets']

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='right',
            truncation_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the quantized model once
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load base model with LoRA
        logger.info("Loading base model with adapters")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            cache_dir="./cache_model"
        )

        # Get device from the model
        device = next(self.model.parameters()).device
        logger.info(f"Model is on device: {device}")

        # Initialize and load classifiers
        hidden_size = self.model.config.hidden_size
        self.classifiers = torch.nn.ModuleDict({
            category: PreferenceHead(hidden_size, len(labels))
            for category, labels in self.label_sets.items()
        })

        # Load classifier state and move to correct device
        self.classifiers.load_state_dict(model_state['classifiers'])
        self.classifiers = self.classifiers.to(device)

        # Load attention weights and move to correct device
        self.attention_weights = model_state['attention_weights'].to(device)

        # Set to eval mode
        self.eval()
        logger.info("Model loading complete")

    def predict(self, text: Union[str, List[str]], threshold: float = 0.3):
        """Override predict method to ensure device consistency"""
        self.eval()
        if isinstance(text, str):
            text = [text]

        # Get device from the model
        device = next(self.model.parameters()).device

        # Tokenize and move to correct device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            logits = outputs[1]

        batch_predictions = []
        current_idx = 0

        for batch_idx in range(len(text)):
            predictions = {}

            for category, labels in self.label_sets.items():
                num_labels = len(labels)
                presence_logits = logits[batch_idx, current_idx:current_idx + num_labels]
                sentiment_logits = logits[batch_idx, current_idx + num_labels:current_idx + 2 * num_labels]
                print(f"sentiment for category - {category} is {sentiment_logits}")
                presence_probs = torch.sigmoid(presence_logits)
                sentiment_probs = torch.sigmoid(sentiment_logits)

                predictions[category] = [
                    {
                        'label': label,
                        'confidence': float(prob),
                        'sentiment': float(sent_prob)
                    }
                    for label, prob, sent_prob in zip(labels, presence_probs, sentiment_probs)
                    if prob > threshold
                ]

                predictions[category].sort(key=lambda x: x['confidence'], reverse=True)
                current_idx += 2 * num_labels

            batch_predictions.append(predictions)

        return batch_predictions[0] if len(text) == 1 else batch_predictions


def format_predictions(predictions: Dict, confidence_threshold: float = 0.3) -> str:
    """Format predictions in a readable way"""
    output = []
    for category, preds in predictions.items():
        if preds:  # If there are predictions for this category
            category_preds = [
                f"  - {p['label']}: {p['confidence']:.3f} "
                f"({'positive' if p['sentiment'] > 0.5 else 'negative'})"
                for p in preds
                if p['confidence'] > confidence_threshold
            ]
            if category_preds:
                output.append(f"\n{category.upper()}:")
                output.extend(category_preds)

    return '\n'.join(output) if output else "No significant preferences detected."


def run_predictions(model: PreferenceModel, queries: List[str]):
    """Run predictions on a list of queries"""
    logger.info("Starting predictions...")

    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Query {i}: {query}")
        logger.info('=' * 50)

        # Get predictions
        predictions = model.predict(query)

        # Format and print results
        results = format_predictions(predictions)
        logger.info(f"\nPredictions:\n{results}\n")


def main():
    model_path = "./preference_model_output"  # Update with your model path

    # Test queries
    test_queries = [
        "I want a spicy vegetarian dish with no mushrooms",
        "Looking for a light Italian appetizer that's not too salty",
        "Can you recommend a gluten-free dessert that's not too sweet?",
        "I need a halal main course with chicken and rice",
        "What's a good keto-friendly Thai dish without dairy?",
        "I want something creamy and savory, maybe French cuisine",
        "Need a pescatarian option that's spicy and tangy",
        "Looking for a traditional Japanese soup, nothing too bitter",
        "I want a Mediterranean dish with lots of garlic and olive oil",
        "Can you suggest a dairy-free pasta dish that's not too heavy?"
    ]

    try:
        # Load model once using the inference class
        model = PreferenceModelInference(model_path)

        # Run predictions
        run_predictions(model, test_queries)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    main()