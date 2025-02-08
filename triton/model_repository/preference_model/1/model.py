# model.py
import json
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Dict, List, Union
import numpy as np

from preference_model import PreferenceHead


class TritonPythonModel:
    def __init__(self):
        self.attention_weights = None
        self.classifiers = None
        self.model = None
        self.label_sets = None
        self.tokenizer = None
        self.model_config = None

    def initialize(self, args):
        """Initialize the model."""
        # Parse model config

        model_dir = args['model_repository']
        model_version = args['model_version']
        model_path = f"{model_dir}/{model_version}"

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='right',
            truncation_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model state
        model_state = torch.load(f"{model_path}/preference_model_state.pt")
        self.label_sets = model_state['label_sets']

        # Load base model with LoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="cuda:0",
            cache_dir=model_path
        )

        device = next(self.model.parameters()).device

        # Initialize classifiers
        hidden_size = self.model.config.hidden_size
        self.classifiers = torch.nn.ModuleDict({
            category: PreferenceHead(hidden_size, len(labels))
            for category, labels in self.label_sets.items()
        })

        # Load classifier state
        self.classifiers.load_state_dict(model_state['classifiers'])
        self.classifiers = self.classifiers.to(device)

        # Load attention weights
        self.attention_weights = model_state['attention_weights']

        # Set model to evaluation mode
        self.model.eval()
        self.classifiers.eval()

    def execute(self, requests):
        """Execute the model for inference requests."""
        responses = []

        for request in requests:
            # Get input text
            input_text = pb_utils.get_input_tensor_by_name(request, 'text')
            input_text = input_text.as_numpy()[0].decode()

            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )

                # Process hidden states
                last_hidden_states = torch.stack(outputs.hidden_states[-4:])
                weighted_hidden_states = torch.mean(last_hidden_states, dim=0)

                # Apply attention weights
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                hidden_states = weighted_hidden_states * mask
                weights = torch.softmax(self.attention_weights, dim=0)
                pooled_output = torch.sum(hidden_states * weights, dim=1) / torch.sum(mask, dim=1)

                # Get predictions from classifiers
                predictions = {}
                for category, head in self.classifiers.items():
                    presence_logits, sentiment_logits = head(pooled_output)
                    presence_probs = torch.sigmoid(presence_logits)
                    sentiment_probs = torch.sigmoid(sentiment_logits)

                    category_preds = []
                    for label, prob, sent_prob in zip(self.label_sets[category], presence_probs[0], sentiment_probs[0]):
                        if prob > 0.3:  # Threshold
                            category_preds.append({
                                'label': label,
                                'confidence': float(prob),
                                'sentiment': float(sent_prob)
                            })

                    predictions[category] = sorted(category_preds, key=lambda x: x['confidence'], reverse=True)

            # Convert predictions to JSON
            output_tensor = pb_utils.Tensor(
                'predictions',
                np.array([json.dumps(predictions).encode()], dtype=np.object_)
            )

            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up model resources."""
        self.model = None
        self.tokenizer = None
        self.classifiers = None
        torch.cuda.empty_cache()