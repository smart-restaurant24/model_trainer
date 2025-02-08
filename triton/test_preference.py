import tritonclient.http as httpclient
import numpy as np
import json


def test_preference_model(text: str, url: str = "localhost:8000"):
    client = httpclient.InferenceServerClient(url=url)

    # Prepare input data
    input_text = np.array([text.encode()], dtype=np.object_)

    # Create input tensor
    inputs = [httpclient.InferInput("text", input_text.shape, "BYTES")]
    inputs[0].set_data_from_numpy(input_text)

    # Run inference
    result = client.infer("preference_model", inputs)

    # Get and parse output
    output = result.as_numpy("predictions")[0].decode()
    predictions = json.loads(output)

    return predictions


# Example usage
text = "I don't want a spicy vegetarian dish with no mushrooms"
predictions = test_preference_model(text)
print(json.dumps(predictions, indent=2))