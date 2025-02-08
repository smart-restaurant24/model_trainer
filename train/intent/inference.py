import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer


def predict_intents(query, model, tokenizer, threshold=0.5):
    """
    Predict intents for a given query
    """
    # Tokenize input
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)

    # Get predicted intents
    predictions = (probabilities > threshold).float()
    predicted_intents = []

    # Log probabilities for all intents
    logger.info(f"\nProbabilities for query: {query}")
    for idx, (intent, prob) in enumerate(zip(INTENT_LABELS, probabilities[0])):
        prob_value = prob.item()
        logger.info(f"{intent}: {prob_value:.4f}")
        if prob_value > threshold:
            predicted_intents.append((intent, prob_value))

    # Sort predicted intents by probability
    predicted_intents.sort(key=lambda x: x[1], reverse=True)

    return predicted_intents


def main():
    # Model configuration
    base_model_name = "google/gemma-2-2b-it"
    lora_weights_path = "./intent_classification_model/checkpoint-2400"  # Path to your saved LoRA weights

    try:
        # Load model
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model(base_model_name, lora_weights_path)

        # Test queries
        test_queries = [
            "What's on your menu for today?",
            "Can I make a reservation for 4 people tonight at 8 PM?",
            "Do you accept credit card payments?",
            "Is the chef available to discuss allergies?",
            "What cuisine do you specialize in?",
            "How spicy is your pad thai?",
            "Can you modify dishes for dietary restrictions?",
            "What's the waiting time right now?",
            "Do you have vegetarian options?",
            "Where is your restaurant located?"
        ]

        # Process each query
        logger.info("\nProcessing test queries...")
        for query in test_queries:
            logger.info("\n" + "=" * 50)
            logger.info(f"Query: {query}")

            # Get predictions
            predicted_intents = predict_intents(query, model, tokenizer)

            # Display results
            if predicted_intents:
                logger.info("Predicted intents:")
                for intent, probability in predicted_intents:
                    logger.info(f"- {intent}: {probability:.4f}")
            else:
                logger.info("No intents predicted above threshold")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()