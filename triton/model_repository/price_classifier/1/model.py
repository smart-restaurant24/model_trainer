import json
import re
from typing import List, Dict

import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict
import logging
import sys
import traceback


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        # Get model directory from args
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/tmp/triton_model.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)


        self.logger.info("Initializing model...")

        model_dir = args['model_repository']
        model_version = args['model_version']
        model_path = f"{model_dir}/{model_version}"

        self.logger.info(f"Loading tokenizer from {model_path}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"** Model Path - {model_path} **")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.logger.info("Loading model...")

        base_model = AutoModelForTokenClassification.from_pretrained(
            "google/gemma-2b-it",
            quantization_config=bnb_config,
            num_labels=4,
            ignore_mismatched_sizes=True,
            cache_dir=model_path
        )

        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        self.id2label = {
            0: "O",
            1: "not_price",
            2: "min_price",
            3: "max_price"
        }
        self.logger.info("Model initialization completed successfully")

    def extract_numbers_with_positions(self, text: str) -> List[Dict]:
        """Extract numbers and their positions from text"""
        self.logger.debug(f"Extracting numbers from text: {text}")
        numbers_info = []
        for match in re.finditer(r'\b\d+\b', text):
            number = match.group()
            char_start, char_end = match.span()
            numbers_info.append({
                'value': int(number),
                'char_start': char_start,
                'char_end': char_end
            })
        self.logger.debug(f"Extracted numbers: {numbers_info}")
        return numbers_info

    def get_token_positions(self, numbers_info: List[Dict], tokenized_output: Dict) -> List[Dict]:
        """Map character positions to token positions"""
        self.logger.debug("Mapping character positions to token positions")
        self.logger.debug(f"Tokenizer output keys: {tokenized_output.keys()}")

        offset_mapping = tokenized_output.pop("offset_mapping")[0].tolist()

        self.logger.debug(f"Offset mapping length: {len(offset_mapping)}")

        for number_info in numbers_info:
            char_start = number_info['char_start']
            char_end = number_info['char_end']
            token_positions = []

            for idx, offset in enumerate(offset_mapping):
                if offset[0] is not None and offset[1] is not None:
                    start, end = offset
                    if (start <= char_start and end > char_start) or \
                            (start >= char_start and end <= char_end) or \
                            (start < char_end and end >= char_end):
                        token_positions.append(idx)

            number_info['token_positions'] = token_positions
            self.logger.debug(f"Token positions for number {number_info['value']}: {token_positions}")

        return numbers_info

    def execute(self, requests):
        """Process inference requests"""
        responses = []

        try:
            for request in requests:
                # Get input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                input_arr = input_tensor.as_numpy()

                # Process each text in the batch
                batch_results = []
                for batch_item in input_arr:
                    # Properly decode the text from bytes
                    if isinstance(batch_item[0], bytes):
                        text = batch_item[0].decode('utf-8')
                    else:
                        text = str(batch_item[0])

                    self.logger.info(f"Processing input text: {text}")

                    # Extract and classify numbers
                    numbers_info = self.extract_numbers_with_positions(text)
                    if not numbers_info:
                        batch_results.append([])  # Empty result for this item
                        continue

                    # Tokenize input
                    tokenized = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=512,
                        return_offsets_mapping=True,
                        padding=True,
                        return_tensors="pt"
                    )

                    # Get token positions
                    numbers_info = self.get_token_positions(numbers_info, dict(tokenized))

                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**{k: v for k, v in tokenized.items() if k != 'offset_mapping'})
                        predictions = outputs.logits.argmax(dim=-1)[0]

                    # Process results for this text
                    text_results = []
                    for number_info in numbers_info:
                        number_predictions = [predictions[pos].item() for pos in number_info['token_positions']]

                        if number_predictions:
                            from collections import Counter
                            most_common_pred = Counter(number_predictions).most_common(1)[0][0]
                            pred_label = self.id2label[most_common_pred]

                            if pred_label != "O":
                                text_results.append({
                                    "value": number_info['value'],
                                    "type": pred_label
                                })

                    batch_results.append(text_results)

                # Convert all results to JSON strings
                json_results = [json.dumps(result) for result in batch_results]

                # Create response tensor
                output_tensor = pb_utils.Tensor(
                    "price_classifications",
                    np.array(json_results, dtype=object)
                )

                # Add to responses
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)

        except Exception as e:
            self.logger.error(f"Error in execute: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

        return responses