import json
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import numpy as np

INTENT_LABELS = [
    "ENQUIRY_MENU", "ENQUIRY_CUISINE", "ENQUIRY_DISH", "ENQUIRY_RESTAURANT",
    "ORDER_RELATED", "RESERVATION_RELATED", "PAYMENT_RELATED", "GENERAL",
    "SERVICE_RELATED", "NON_RELATED"
]


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        model_dir = args['model_repository']
        model_version = args['model_version']

        # Load tokenizer and model
        base_model_name = "google/gemma-2-2b-it"
        model_path = f"{model_dir}/{model_version}"

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
            device_map="auto",
            cache_dir=model_path
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set threshold for intent classification
        self.threshold = 0.5

    def execute(self, requests):
        """Execute the model for inference requests"""
        responses = []

        for request in requests:
            # Get input text
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            query = input_tensor.as_numpy()[0].decode('utf-8')

            # Tokenize input
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )

            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)

            # Get predicted intents
            predictions = (probabilities > self.threshold).float()
            predicted_intents = []
            intent_probs = []

            # Collect predictions and probabilities
            for idx, (prob, pred) in enumerate(zip(probabilities[0], predictions[0])):
                if pred == 1:
                    predicted_intents.append(INTENT_LABELS[idx])
                    intent_probs.append(prob.item())

            # Create output tensors
            intents_tensor = pb_utils.Tensor(
                "INTENTS",
                np.array([intent.encode('utf-8') for intent in predicted_intents], dtype=np.object_)
            )
            probs_tensor = pb_utils.Tensor(
                "PROBABILITIES",
                np.array(intent_probs, dtype=np.float32)
            )

            # Create and append response
            response = pb_utils.InferenceResponse(
                output_tensors=[intents_tensor, probs_tensor]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up and free resources"""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()