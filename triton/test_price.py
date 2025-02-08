import json

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np

# Initialize client
client = httpclient.InferenceServerClient(url="localhost:8000")


def process_batch(texts):
    """
    Process a batch of texts through the Triton server
    Args:
        texts (list): List of input texts to process
    """
    # Convert texts to numpy array with proper shape
    input_data = np.array([[text.encode()] for text in texts], dtype=np.object_)

    # Create input tensor. Note the shape should be [batch_size, 1]
    inputs = [httpclient.InferInput("text", input_data.shape, np_to_triton_dtype(input_data.dtype))]
    inputs[0].set_data_from_numpy(input_data)

    # Perform inference
    result = client.infer("price_classifier", inputs)

    # Process results
    classifications = result.as_numpy("price_classifications")
    return [json.loads(r.decode()) for r in classifications]


# Example usage with single input
single_text = "Show me dishes between USD 50 and USD 150"
result = process_batch([single_text])
print(f"\nSingle input result:")
print(f"Input: {single_text}")
print(f"Classifications: {result[0]}")

# Example usage with batch
batch_texts = [
    "Show me dishes between USD 50 and USD 150",
    "I want to book a table for 3 people with budget under 75 dollars",
    "Find restaurants with average cost of 25 dollars per person"
]
results = process_batch(batch_texts)

print(f"\nBatch processing results:")
for text, classification in zip(batch_texts, results):
    print(f"\nInput: {text}")
    print(f"Classifications: {classification}")