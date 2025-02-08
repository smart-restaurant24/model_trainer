import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def test_intent_classifier():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Prepare input
    query = "Can I make a reservation for tonight? and what cards are accepted and where is the menu "
    input_data = np.array([query.encode('utf-8')], dtype=np.object_)

    # Create input tensor
    inputs = [
        httpclient.InferInput(
            "TEXT", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]
    inputs[0].set_data_from_numpy(input_data)

    # Get inference results
    results = client.infer("intent_classifier", inputs)

    # Process results
    intents = results.as_numpy("INTENTS")
    probabilities = results.as_numpy("PROBABILITIES")

    # Print results
    print("Query:", query)
    print("Predicted Intents:")
    for intent, prob in zip(intents, probabilities):
        print(f"- {intent.decode('utf-8')}: {prob:.4f}")


if __name__ == "__main__":
    test_intent_classifier()