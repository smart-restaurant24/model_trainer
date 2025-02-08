import json
import time
from typing import Dict, List
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class TritonAccuracyAnalyzer:
    def __init__(self, server_url: str, model_name: str, test_data_path: str):
        """Initialize the analyzer with Triton server settings and test data"""
        self.client = httpclient.InferenceServerClient(url=server_url)
        self.model_name = model_name
        self.test_data = self.load_test_data(test_data_path)
        self.error_cases = []
        self.metrics = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
        self.inference_times = []
        self.latencies = defaultdict(list)
        self.server_url = server_url

    def load_test_data(self, path: str) -> List[Dict]:
        """Load test data from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
            return data['data'] if 'data' in data else data

    def infer_single(self, text: str) -> List[Dict]:
        """Make a single inference request to Triton server"""
        input_data = np.array([text.encode()], dtype=np.object_)
        inputs = [httpclient.InferInput("text", input_data.shape, np_to_triton_dtype(input_data.dtype))]
        inputs[0].set_data_from_numpy(input_data)

        # Measure complete inference time (including network latency)
        start_time = time.perf_counter()
        result = self.client.infer(self.model_name, inputs)
        end_time = time.perf_counter()

        # Store end-to-end latency
        self.latencies['complete_time'].append(end_time - start_time)

        # Parse the result
        classifications = result.as_numpy("price_classifications")[0].decode()
        return json.loads(classifications)

    def infer_batch(self, texts: List[str], batch_size: int = 8) -> List[List[Dict]]:
        """Make batched inference requests to Triton server"""
        all_results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Prepare batch input
            input_data = np.array([[text.encode()] for text in batch_texts], dtype=np.object_)
            inputs = [httpclient.InferInput("text", input_data.shape, np_to_triton_dtype(input_data.dtype))]
            inputs[0].set_data_from_numpy(input_data)

            # Measure complete inference time (including network latency)
            start_time = time.perf_counter()
            result = self.client.infer(self.model_name, inputs)
            end_time = time.perf_counter()

            # Store batch latency
            batch_time = end_time - start_time
            self.latencies['complete_time'].append(batch_time)
            self.latencies['per_request_time'].extend([batch_time / len(batch_texts)] * len(batch_texts))

            # Parse results
            classifications = result.as_numpy("price_classifications")
            batch_results = [json.loads(r.decode()) for r in classifications]
            all_results.extend(batch_results)

        return all_results

    def compare_numbers(self, true_numbers: List[Dict], pred_numbers: List[Dict]) -> Dict:
        """Compare predicted numbers with ground truth"""
        true_dict = {num["value"]: num["type"] for num in true_numbers}
        pred_dict = {num["value"]: num["type"] for num in pred_numbers}

        results = {
            "matched": [],
            "missed": [],
            "misclassified": [],
            "extra": []
        }

        # Find matches, misses, and misclassifications
        for value, true_type in true_dict.items():
            if value in pred_dict:
                if pred_dict[value] == true_type:
                    results["matched"].append({"value": value, "type": true_type})
                else:
                    results["misclassified"].append({
                        "value": value,
                        "true_type": true_type,
                        "pred_type": pred_dict[value]
                    })
            else:
                results["missed"].append({"value": value, "type": true_type})

        # Find extra predictions
        for value, pred_type in pred_dict.items():
            if value not in true_dict:
                results["extra"].append({"value": value, "type": pred_type})

        return results

    def analyze_accuracy(self, batch_size: int = 8) -> Dict:
        """Analyze accuracy across the test dataset using batched processing"""
        all_true_labels = []
        all_pred_labels = []
        all_predictions = []

        # Prepare batches
        texts = []
        true_numbers_list = []

        for example in self.test_data:
            texts.append(example["input_text"])
            true_numbers_list.append(example["numbers"])

        # Process all examples in batches
        try:
            predictions = self.infer_batch(texts, batch_size)

            # Process results
            for idx, (text, true_numbers, pred_numbers) in enumerate(zip(texts, true_numbers_list, predictions)):
                if idx % 10 == 0:
                    print(f"Processing results for example {idx}/{len(self.test_data)}")

                prediction_entry = {
                    "text": text,
                    "true_numbers": true_numbers,
                    "predicted_numbers": pred_numbers,
                    "inference_time": self.latencies['per_request_time'][idx]
                }
                all_predictions.append(prediction_entry)

                # Compare predictions with ground truth
                comparison = self.compare_numbers(true_numbers, pred_numbers)

                # Track error cases
                if comparison["misclassified"] or comparison["missed"] or comparison["extra"]:
                    self.error_cases.append({
                        "text": text,
                        "true_numbers": true_numbers,
                        "pred_numbers": pred_numbers,
                        "comparison": comparison
                    })

                # Collect labels for confusion matrix
                for num in true_numbers:
                    all_true_labels.append(num["type"])
                    pred_type = next(
                        (p["type"] for p in pred_numbers if p["value"] == num["value"]),
                        "missed"
                    )
                    all_pred_labels.append(pred_type)

                # Update type-specific metrics
                for num in true_numbers:
                    num_type = num["type"]
                    self.metrics[num_type]["total"] += 1
                    if any(p["value"] == num["value"] and p["type"] == num["type"]
                           for p in pred_numbers):
                        self.metrics[num_type]["correct"] += 1

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            raise

        # Save predictions to file
        with open('triton_predictions.json', 'w') as f:
            json.dump({
                "predictions": all_predictions,
                "metadata": {
                    "total_examples": len(self.test_data),
                    "batch_size": batch_size,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "server_url": self.server_url,
                    "model_name": self.model_name
                }
            }, f, indent=2)

        # Calculate latency metrics
        complete_times = self.latencies['complete_time']
        per_request_times = self.latencies['per_request_time']

        latency_metrics = {
            "batch_time": {
                "mean": mean(complete_times),
                "std": stdev(complete_times) if len(complete_times) > 1 else 0,
                "min": min(complete_times),
                "max": max(complete_times),
                "p50": np.percentile(complete_times, 50),
                "p95": np.percentile(complete_times, 95),
                "p99": np.percentile(complete_times, 99)
            },
            "per_request_time": {
                "mean": mean(per_request_times),
                "std": stdev(per_request_times) if len(per_request_times) > 1 else 0,
                "min": min(per_request_times),
                "max": max(per_request_times),
                "p50": np.percentile(per_request_times, 50),
                "p95": np.percentile(per_request_times, 95),
                "p99": np.percentile(per_request_times, 99)
            }
        }

        # Calculate overall metrics
        total_numbers = sum(m["total"] for m in self.metrics.values())
        total_correct = sum(m["correct"] for m in self.metrics.values())

        return {
            "overall_accuracy": total_correct / total_numbers if total_numbers > 0 else 0,
            "type_metrics": {
                t: {
                    "accuracy": m["correct"] / m["total"] if m["total"] > 0 else 0,
                    "total": m["total"],
                    "correct": m["correct"]
                }
                for t, m in self.metrics.items()
            },
            "latency_metrics": latency_metrics,
            "throughput": {
                "requests_per_second": len(self.test_data) / sum(complete_times),
                "batches_per_second": len(complete_times) / sum(complete_times)
            },
            "classification_report": classification_report(
                all_true_labels,
                all_pred_labels,
                labels=["not_price", "min_price", "max_price"],
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                all_true_labels,
                all_pred_labels,
                labels=["not_price", "min_price", "max_price"]
            ),
            "total_examples": len(self.test_data),
            "error_cases": len(self.error_cases),
            "batch_size": batch_size
        }

    def plot_latency_distribution(self):
        """Plot distribution of latency"""
        plt.figure(figsize=(12, 6))
        plt.hist(self.latencies['complete_time'], bins=50, edgecolor='black')
        plt.axvline(mean(self.latencies['complete_time']), color='red', linestyle='dashed',
                    label=f'Mean: {mean(self.latencies["complete_time"]) * 1000:.2f}ms')
        plt.title("Distribution of End-to-End Inference Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig("triton_latency_distribution.png")
        plt.close()

    def plot_confusion_matrix(self, confusion_mat):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            xticklabels=["not_price", "min_price", "max_price"],
            yticklabels=["not_price", "min_price", "max_price"]
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("triton_confusion_matrix.png")
        plt.close()

    def generate_error_analysis(self) -> pd.DataFrame:
        """Generate detailed error analysis"""
        error_records = []
        for case in self.error_cases:
            error_records.append({
                "text": case["text"],
                "true_numbers": case["true_numbers"],
                "predicted_numbers": case["pred_numbers"],
                "missed_numbers": case["comparison"]["missed"],
                "misclassified": case["comparison"]["misclassified"],
                "extra_predictions": case["comparison"]["extra"]
            })
        return pd.DataFrame(error_records)


def main():
    # Initialize analyzer
    analyzer = TritonAccuracyAnalyzer(
        server_url="localhost:8000",
        model_name="price_classifier",
        test_data_path="test_data.json"
    )

    # Run analysis
    print("\nRunning accuracy and latency analysis...")
    metrics = analyzer.analyze_accuracy(batch_size=8)

    # Print overall results
    print("\n=== Overall Results ===")
    print(f"Total examples analyzed: {metrics['total_examples']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Number of error cases: {metrics['error_cases']}")

    # Print latency metrics
    print("\n=== Latency Metrics ===")
    # Print batch-level metrics
    batch_time = metrics['latency_metrics']['batch_time']
    print("Batch Processing Time:")
    print(f"Mean: {batch_time['mean'] * 1000:.2f}ms")
    print(f"Std: {batch_time['std'] * 1000:.2f}ms")
    print(f"Min: {batch_time['min'] * 1000:.2f}ms")
    print(f"Max: {batch_time['max'] * 1000:.2f}ms")
    print(f"P50: {batch_time['p50'] * 1000:.2f}ms")
    print(f"P95: {batch_time['p95'] * 1000:.2f}ms")
    print(f"P99: {batch_time['p99'] * 1000:.2f}ms")

    # Print per-request metrics
    print("\nPer-Request Processing Time:")
    per_request = metrics['latency_metrics']['per_request_time']
    print(f"Mean: {per_request['mean'] * 1000:.2f}ms")
    print(f"Std: {per_request['std'] * 1000:.2f}ms")
    print(f"Min: {per_request['min'] * 1000:.2f}ms")
    print(f"Max: {per_request['max'] * 1000:.2f}ms")
    print(f"P50: {per_request['p50'] * 1000:.2f}ms")
    print(f"P95: {per_request['p95'] * 1000:.2f}ms")
    print(f"P99: {per_request['p99'] * 1000:.2f}ms")

    # Print throughput metrics
    print("\nThroughput Metrics:")
    print(f"Requests per second: {metrics['throughput']['requests_per_second']:.2f}")
    print(f"Batches per second: {metrics['throughput']['batches_per_second']:.2f}")
    print(f"Batch size: {metrics['batch_size']}")

    # Print type-specific metrics
    print("\n=== Metrics by Number Type ===")
    for type_name, type_metrics in metrics["type_metrics"].items():
        print(f"\n{type_name}:")
        print(f"Accuracy: {type_metrics['accuracy']:.2%}")
        print(f"Correct predictions: {type_metrics['correct']}/{type_metrics['total']}")

    # Print classification report
    print("\n=== Classification Report ===")
    print(metrics["classification_report"])

    # Plot confusion matrix
    analyzer.plot_confusion_matrix(metrics["confusion_matrix"])
    print("\nConfusion matrix has been saved as 'triton_confusion_matrix.png'")

    # Plot latency distribution
    analyzer.plot_latency_distribution()
    print("\nLatency distribution has been saved as 'triton_latency_distribution.png'")

    # Generate error analysis
    error_df = analyzer.generate_error_analysis()
    error_df.to_csv("triton_error_analysis.csv", index=False)

    # Print sample error cases
    print("\n=== Sample Error Cases ===")
    for idx, error in error_df.head().iterrows():
        print(f"\nExample {idx + 1}:")
        print(f"Text: {error['text']}")
        print(f"True numbers: {error['true_numbers']}")
        print(f"Predicted numbers: {error['predicted_numbers']}")
        if error['misclassified']:
            print(f"Misclassified: {error['misclassified']}")
        if error['missed_numbers']:
            print(f"Missed numbers: {error['missed_numbers']}")
        if error['extra_predictions']:
            print(f"Extra predictions: {error['extra_predictions']}")
        print("-" * 80)


if __name__ == "__main__":
    main()