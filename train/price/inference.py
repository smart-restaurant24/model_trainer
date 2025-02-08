import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
from typing import List, Dict, Tuple


class PriceRangeClassifier:
    def __init__(self, model_path: str):
        """Initialize the classifier with the trained model"""
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForTokenClassification.from_pretrained(
            "google/gemma-2b-it",
            quantization_config=bnb_config,
            num_labels=4,
            ignore_mismatched_sizes=True  # Added to handle new token classification head
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        self.id2label = {
            0: "O",
            1: "not_price",
            2: "min_price",
            3: "max_price"
        }

    def extract_numbers_with_positions(self, text: str) -> List[Dict]:
        """
        Extract numbers from text with their character positions and token positions
        """
        numbers_info = []

        # Find all numbers in the text
        for match in re.finditer(r'\b\d+\b', text):
            number = match.group()
            char_start, char_end = match.span()

            numbers_info.append({
                'value': int(number),
                'char_start': char_start,
                'char_end': char_end
            })

        return numbers_info

    def get_token_positions(self, numbers_info: List[Dict], tokenized_output: Dict) -> List[Dict]:
        """
        Map character positions to token positions
        """
        # Extract offset mapping from the first item (since we use batch size 1)
        offset_mapping = tokenized_output.pop("offset_mapping")[0].tolist()

        for number_info in numbers_info:
            char_start = number_info['char_start']
            char_end = number_info['char_end']

            token_positions = []

            # Find which tokens correspond to this number
            for idx, offset in enumerate(offset_mapping):
                if offset[0] is not None and offset[1] is not None:  # Check for special tokens
                    start, end = offset
                    # Check if token overlaps with the number
                    if (start <= char_start and end > char_start) or \
                            (start >= char_start and end <= char_end) or \
                            (start < char_end and end >= char_end):
                        token_positions.append(idx)

            number_info['token_positions'] = token_positions

        return numbers_info

    def classify_prices(self, text: str) -> List[Dict]:
        """
        Classify numbers in the text as price-related or not
        Returns: List of dictionaries containing number value and its classification
        """
        # Extract numbers and their positions
        numbers_info = self.extract_numbers_with_positions(text)
        if not numbers_info:
            return []

        # Tokenize the input text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt"
        )

        # Get token positions for each number
        numbers_info = self.get_token_positions(numbers_info, dict(tokenized))

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in tokenized.items() if k != 'offset_mapping'})
            predictions = outputs.logits.argmax(dim=-1)[0]

        # Process predictions only for number tokens
        results = []
        for number_info in numbers_info:
            # Get predictions for all tokens of this number
            number_predictions = [predictions[pos].item() for pos in number_info['token_positions']]

            # Use majority voting if number spans multiple tokens
            if number_predictions:
                from collections import Counter
                most_common_pred = Counter(number_predictions).most_common(1)[0][0]
                pred_label = self.id2label[most_common_pred]

                # Only include if it's a price-related prediction
                if pred_label != "O":
                    results.append({
                        "value": number_info['value'],
                        "type": pred_label
                    })

        return results


def main():
    # Example usage
    classifier = PriceRangeClassifier("./price_classifier_final")

    # Example queries
    test_queries = [
        "Show me dishes between USD 50 and USD 150 for 4 people",
        "I want to book a table for 3 people with budget under 75 dollars",
        "Find restaurants with average cost of 25 dollars per person"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = classifier.classify_prices(query)
            print("Classifications:")
            for result in results:
                print(f"Number {result['value']}: Classified as {result['type']}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()