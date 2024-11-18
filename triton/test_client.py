import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def test_intent_classifier():
    try:
        # Create client
        client = httpclient.InferenceServerClient(url="localhost:8000")

        # Test single query first
        text = "Can I see the menu please?"
        print(f"\nTesting single query: {text}")

        # Prepare input data - encode text as bytes and reshape properly
        input_data = np.array([[text.encode('utf-8')]], dtype=np.object_)

        # Create input tensor with correct shape
        input_tensor = httpclient.InferInput(
            "text",
            [1, 1],  # shape [batch_size, 1] to match config
            "BYTES"
        )
        input_tensor.set_data_from_numpy(input_data)

        # Create output tensors
        output_tensor_1 = httpclient.InferRequestedOutput("intents")
        output_tensor_2 = httpclient.InferRequestedOutput("probabilities")

        # Send request
        response = client.infer(
            "intent_classifier",
            model_version="1",
            inputs=[input_tensor],
            outputs=[output_tensor_1, output_tensor_2]
        )

        # Get and print results
        intents = response.as_numpy("intents")
        probabilities = response.as_numpy("probabilities")

        print(f"Predicted Intents: {intents[0]}")
        print("\nProbabilities for each intent:")

        intent_mapping = {
            0: "ENQUIRY_MENU",
            1: "ENQUIRY_CUISINE",
            2: "ENQUIRY_DISH",
            3: "ENQUIRY_RESTAURANT",
            4: "ORDER_RELATED",
            5: "RESERVATION_RELATED",
            6: "PAYMENT_RELATED",
            7: "GENERAL",
            8: "SERVICE_RELATED",
            9: "NON_RELATED",
            10: "RECOMMENDATION"
        }

        # Print top 3 probabilities
        probs = probabilities[0]
        top_indices = np.argsort(probs)[-3:][::-1]

        for idx in top_indices:
            print(f"{intent_mapping[idx]}: {probs[idx]:.4f}")

        print("\nTest completed successfully!")

        # Optional: Test batch processing
        texts = [
            "Book table for tonight and check menu",
            "What cards do you accept?",
            "I'd like to make a reservation",
            "What cards you accept , Is there any discount"
        ]
        print(f"\nTesting batch processing with {len(texts)} queries...")

        # Prepare batch input
        batch_data = np.array([[text.encode('utf-8')] for text in texts], dtype=np.object_)

        # Create input tensor for batch
        batch_input_tensor = httpclient.InferInput(
            "text",
            [len(texts), 1],  # shape [batch_size, 1]
            "BYTES"
        )
        batch_input_tensor.set_data_from_numpy(batch_data)

        # Send batch request
        batch_response = client.infer(
            "intent_classifier",
            model_version="1",
            inputs=[batch_input_tensor],
            outputs=[output_tensor_1, output_tensor_2]
        )

        # Process batch results
        batch_intents = batch_response.as_numpy("intents")
        batch_probabilities = batch_response.as_numpy("probabilities")

        print("\nBatch Results:")
        for i, (text, ints, probs) in enumerate(zip(texts, batch_intents, batch_probabilities)):
            print(f"\nQuery {i + 1}: {text}")
            print(f"Predicted Intents: {ints}")

            top_indices = np.argsort(probs)[-3:][::-1]
            print("Top 3 probabilities:")
            for idx in top_indices:
                print(f"{intent_mapping[idx]}: {probs[idx]:.4f}")

        print("\nBatch test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("\nDetailed error information:")
        if hasattr(e, 'message'):
            print(f"Message: {e.message()}")
        if hasattr(e, 'status'):
            print(f"Status: {e.status()}")


if __name__ == "__main__":
    test_intent_classifier()