import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModel, BertConfig
from safetensors.torch import load_file
import re
import os
from torch import nn


class AdvancedClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.activation = nn.GELU()

    def forward(self, features):
        x = self.dropout(features)
        x = self.intermediate(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class CustomBertForClassification(nn.Module):
    def __init__(self, bert_model, config):
        super().__init__()
        self.bert = bert_model
        self.classifier = AdvancedClassificationHead(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits


class TritonPythonModel:
    def initialize(self, args):
        try:
            self.model_dir = args['model_repository']
            self.model_version = args['model_version']
            model_path = f"{self.model_dir}/{self.model_version}"

            # Validate paths
            if not os.path.exists(model_path):
                raise pb_utils.TritonError(f"Model path {model_path} does not exist")

            mappings_path = f"{model_path}/label_mappings.json"
            if not os.path.exists(mappings_path):
                raise pb_utils.TritonError(f"Label mappings not found at {mappings_path}")

            # Load label mappings
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.label2id = mappings['label2id']
                self.id2label = mappings['id2label']

            # Load config
            with open(f"{model_path}/config.json", 'r') as f:
                config_dict = json.load(f)
                self.config = BertConfig(**config_dict)

            self.max_length = 128
            self.threshold = 0.5
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load base BERT model
            bert_model = AutoModel.from_pretrained(model_path)

            # Create custom model
            self.model = CustomBertForClassification(
                bert_model=bert_model,
                config=self.config
            )

            # Load weights from safetensors
            state_dict = load_file(f"{model_path}/model.safetensors")

            # Rename state dict keys to match our custom model
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('bert'):
                    new_state_dict[k] = v
                elif k.startswith('classifier'):
                    new_state_dict[k] = v

            # Load the state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise Exception(f"Failed to initialize model: {str(e)}")

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\?\!\.]', '', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'([.!,;:])\1+', r'\1', text)
        return text.strip()

    def execute(self, requests):
        responses = []

        try:
            for request in requests:
                if request is None:
                    raise pb_utils.TritonError("Received empty request")

                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                if input_tensor is None:
                    raise pb_utils.TritonError("Input tensor 'text' not found")

                input_text = [t.decode('utf-8') for t in input_tensor.as_numpy().reshape(-1)]
                cleaned_text = [self._clean_text(text) for text in input_text]

                inputs = self.tokenizer(
                    cleaned_text,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs)
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > self.threshold).bool()

                probabilities = probabilities.cpu()
                predictions = predictions.cpu()
                del inputs
                torch.cuda.empty_cache()

                batch_intents = []
                batch_probs = probabilities.numpy()

                for pred_idx in range(predictions.shape[0]):
                    pred_intents = []
                    for idx, pred in enumerate(predictions[pred_idx]):
                        if pred:
                            pred_intents.append(self.id2label[str(idx)])
                    batch_intents.append(pred_intents)

                intents_tensor = pb_utils.Tensor(
                    "intents",
                    np.array(batch_intents, dtype=np.object_)
                )
                probs_tensor = pb_utils.Tensor(
                    "probabilities",
                    batch_probs.astype(np.float32)
                )

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[intents_tensor, probs_tensor]
                )
                responses.append(inference_response)

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return [pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("GPU out of memory. Try reducing batch size.")
                )]
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error processing request: {str(e)}")
                )
            )
        except Exception as e:
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error processing request: {str(e)}")
                )
            )

        return responses

    def finalize(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()