import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import logging
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define intent labels
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


def load_model(base_model_name, lora_weights_path):
    """
    Load the quantized base model and LoRA weights
    """
    logger.info(f"Loading base model: {base_model_name}")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model with quantization
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(INTENT_LABELS),
        quantization_config=bnb_config,
        device_map="auto"
    )

    logger.info("Loading LoRA weights")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer


def predict_intents(query, model, tokenizer, threshold=0.5):
    """
    Predict intents for a given query
    """
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)

    # Get raw probabilities for all intents
    raw_predictions = probabilities[0].cpu().numpy()

    # Get binary predictions
    binary_predictions = (probabilities > threshold).float()[0].cpu().numpy()

    return binary_predictions, raw_predictions


def analyze_model(model, tokenizer, test_data_path, output_dir):
    """
    Analyze model performance and save incorrect predictions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    with open(test_data_path, 'r') as f:
        data = json.load(f)

    # Extract test_data from the nested structure
    test_data = data["test_data"]

    logger.info(f"Loaded {len(test_data)} test samples")

    # Lists to store results
    all_true_labels = []
    all_predicted_labels = []
    all_raw_predictions = []
    incorrect_predictions = []

    # Process each test sample
    for idx, sample in enumerate(test_data):
        query = sample["query"]
        true_intents = sample["intents"]

        # Convert true intents to multi-hot encoding
        true_labels = np.zeros(len(INTENT_LABELS))
        for intent in true_intents:
            true_labels[INTENT_LABELS.index(intent)] = 1

        # Get model predictions
        binary_predictions, raw_predictions = predict_intents(query, model, tokenizer)

        # Store results
        all_true_labels.append(true_labels)
        all_predicted_labels.append(binary_predictions)
        all_raw_predictions.append(raw_predictions)

        # Check if prediction is incorrect
        if not np.array_equal(true_labels, binary_predictions):
            incorrect_predictions.append({
                "query": query,
                "true_intents": true_intents,
                "predicted_intents": [INTENT_LABELS[i] for i, pred in enumerate(binary_predictions) if pred == 1],
                "raw_probabilities": {INTENT_LABELS[i]: float(prob) for i, prob in enumerate(raw_predictions)}
            })

        # Log progress
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1} samples")

    # Convert lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)

    # Calculate metrics
    classification_metrics = classification_report(
        all_true_labels,
        all_predicted_labels,
        target_names=INTENT_LABELS,
        output_dict=True
    )

    # Save results
    results = {
        "classification_report": classification_metrics,
        "incorrect_predictions": incorrect_predictions,
        "analysis_summary": {
            "total_samples": len(test_data),
            "correct_predictions": len(test_data) - len(incorrect_predictions),
            "incorrect_predictions": len(incorrect_predictions),
            "accuracy": (len(test_data) - len(incorrect_predictions)) / len(test_data)
        }
    }

    # Save detailed results to files
    output_base = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Save main results
    with open(f"{output_base}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save incorrect predictions to a separate CSV for easier analysis
    if incorrect_predictions:
        df_incorrect = pd.DataFrame(incorrect_predictions)
        df_incorrect.to_csv(f"{output_base}_incorrect_predictions.csv", index=False)

    # Log summary results
    logger.info("\n=== Analysis Results ===")
    logger.info(f"Total samples: {results['analysis_summary']['total_samples']}")
    logger.info(f"Correct predictions: {results['analysis_summary']['correct_predictions']}")
    logger.info(f"Incorrect predictions: {results['analysis_summary']['incorrect_predictions']}")
    logger.info(f"Overall accuracy: {results['analysis_summary']['accuracy']:.4f}")

    # Log per-class metrics
    logger.info("\n=== Per-Class Metrics ===")
    for intent in INTENT_LABELS:
        metrics = classification_metrics[intent]
        logger.info(f"\n{intent}:")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-score: {metrics['f1-score']:.4f}")

    return results


def main():
    # Configuration
    base_model_name = "google/gemma-2-2b-it"
    lora_weights_path = "./final_model"
    test_data_path = "test_data.json"  # Path to your test data
    output_dir = "./analysis_results"

    try:
        # Load model
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model(base_model_name, lora_weights_path)

        # Run analysis
        logger.info("Starting model analysis...")
        results = analyze_model(model, tokenizer, test_data_path, output_dir)

        logger.info(f"Analysis completed. Results saved in {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()