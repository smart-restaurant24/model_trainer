import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Dict, List, Union


class PreferenceHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        # Simplified but effective architecture for each head
        self.dropout1 = nn.Dropout(0.1)
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

        # Separate branches for presence and sentiment with proper initialization
        self.presence_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )

        self.sentiment_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Initialize weights using proper scaling
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.intermediate(x)
        return self.presence_branch(x), self.sentiment_branch(x)

class PreferenceModel(nn.Module):
    def __init__(
            self,
            model_name: str = "google/gemma-2b",
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1
    ):
        super().__init__()

        # Define fixed label sets
        self.label_sets = {
            'taste': ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy', 'crispy', 'creamy', 'smoky', 'tangy'],
            'dietary': ['vegetarian', 'pescatarian', 'gluten-free', 'dairy-free', 'kosher', 'halal', 'keto', 'paleo',
                        'mediterranean', 'low-fodmap', 'raw'],
            'course': ['appetizer', 'soup', 'salad', 'main-course', 'side-dish', 'dessert', 'beverage', 'small-plate',
                       'cheese-course', 'fish-course', 'palate-cleanser', 'amuse-bouche', 'bread-service',
                       'chef-special', 'digestif'],
            'cuisine': ['chinese', 'italian', 'japanese', 'indian', 'mexican', 'thai', 'mediterranean', 'korean',
                        'french', 'vietnamese'],
            'ingredient': ['chicken', 'beef', 'fish', 'pork', 'shrimp', 'tofu', 'rice', 'noodles', 'pasta', 'tomatoes',
                           'mushrooms', 'onions', 'garlic', 'ginger', 'cheese', 'eggs']
        }

        # Calculate total number of labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right',
            truncation_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Improved quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model with improved settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        # Improved LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)

        # Create improved classification heads
        hidden_size = self.model.config.hidden_size
        self.classifiers = nn.ModuleDict({
            category: PreferenceHead(hidden_size, len(labels))
            for category, labels in self.label_sets.items()
        })

        # Add weighted pooling layer
        self.attention_weights = nn.Parameter(torch.ones(hidden_size))

    def _weighted_pooling(self, hidden_states, attention_mask):
        # Compute attention weights
        weights = F.softmax(self.attention_weights, dim=0)

        # Apply attention mask
        mask = attention_mask.unsqueeze(-1).float()
        hidden_states = hidden_states * mask

        # Weighted sum across sequence length
        weighted_sum = torch.sum(hidden_states * weights, dim=1)
        return weighted_sum / torch.sum(mask, dim=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs
    ) -> Union[tuple, Dict[str, torch.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Improved pooling strategy - use weighted combination of last 4 layers
        last_hidden_states = torch.stack(outputs.hidden_states[-4:])
        weighted_hidden_states = torch.mean(last_hidden_states, dim=0)
        pooled_output = self._weighted_pooling(weighted_hidden_states, attention_mask)

        all_logits = []
        all_labels = []
        total_loss = 0.0

        # Process each category with improved loss calculation
        for category, head in self.classifiers.items():
            presence_logits, sentiment_logits = head(pooled_output)

            all_logits.extend([presence_logits, sentiment_logits])

            if f"{category}_labels" in kwargs:
                labels = kwargs[f"{category}_labels"].float()
                sentiment = kwargs.get(f"{category}_sentiment", None)

                all_labels.append(labels)
                if sentiment is not None:
                    all_labels.append(sentiment.float())

                # Improved loss calculation with class weights
                pos_weight = torch.ones_like(labels) * 2.0  # Adjust for class imbalance
                presence_loss = F.binary_cross_entropy_with_logits(
                    presence_logits,
                    labels,
                    pos_weight=pos_weight
                )
                total_loss += presence_loss

                if sentiment is not None:
                    label_mask = labels > 0.3
                    if label_mask.any():
                        sentiment_loss = F.binary_cross_entropy_with_logits(
                            sentiment_logits[label_mask],
                            sentiment[label_mask].float(),
                            reduction='mean'
                        )
                        total_loss += sentiment_loss

        stacked_logits = torch.cat(all_logits, dim=-1) if all_logits else None
        stacked_labels = torch.cat(all_labels, dim=-1) if all_labels else None

        return (total_loss if total_loss > 0 else None,
                stacked_logits,
                stacked_labels)

    def predict(self, text: Union[str, List[str]], threshold: float = 0.3):  # Lowered threshold
        self.eval()
        if isinstance(text, str):
            text = [text]

        # Improved tokenization with longer sequence length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # Increased from 128
            padding=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            logits = outputs[1]

        batch_predictions = []
        current_idx = 0

        for batch_idx in range(len(text)):
            predictions = {}

            for category, labels in self.label_sets.items():
                num_labels = len(labels)
                presence_logits = logits[batch_idx, current_idx:current_idx + num_labels]
                sentiment_logits = logits[batch_idx, current_idx + num_labels:current_idx + 2 * num_labels]

                # Apply softmax to get better-calibrated probabilities
                presence_probs = torch.sigmoid(presence_logits)
                sentiment_probs = torch.sigmoid(sentiment_logits)

                # Improved prediction filtering
                predictions[category] = [
                    {
                        'label': label,
                        'confidence': float(prob),
                        'sentiment': float(sent_prob),
                        'sentiment_strength': abs(float(sent_prob) - 0.5) * 2  # Normalized strength
                    }
                    for label, prob, sent_prob in zip(labels, presence_probs, sentiment_probs)
                    if prob > threshold
                ]

                # Sort by confidence
                predictions[category].sort(key=lambda x: x['confidence'], reverse=True)

                current_idx += 2 * num_labels

            batch_predictions.append(predictions)

        return batch_predictions[0] if len(text) == 1 else batch_predictions