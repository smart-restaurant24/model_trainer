import torch
from preference_model import PreferenceModel
from inference import PreferenceModelInference, format_predictions
import json
import time
from typing import Dict, List
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = PreferenceModelInference(model_path)
        self.label_sets = self.model.label_sets
        self.incorrect_predictions = []
        self.predictions = []

    def evaluate_predictions(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate model predictions against ground truth labels
        Returns metrics and timing information
        """
        total_time = 0
        predictions = []
        metrics = defaultdict(lambda: defaultdict(list))
        total_predictions = defaultdict(int)
        correct_predictions = defaultdict(int)

        logger.info("Starting evaluation...")

        for idx, item in enumerate(test_data):
            # Measure inference time
            start_time = time.time()
            pred = self.model.predict(item['input_text'])
            inference_time = time.time() - start_time
            total_time += inference_time

            # Store predictions
            predictions.append(pred)

            # Compare with ground truth for each category
            incorrect_item = {
                'input_text': item['input_text'],
                'categories': {}
            }

            prediction = {
                'input_text': item['input_text'],
                'categories': {}
            }
            has_errors = False

            for category in self.label_sets.keys():
                pred_labels = set([p['label'] for p in pred[category]])
                true_labels = set([label_info['label'].lower() for label_info in item['labels'][category]])

                # Get predicted sentiments and confidence
                predicted_with_sentiment = {
                    p['label']: {
                        'sentiment': p['sentiment'],
                        'confidence': p['confidence']
                    } for p in pred[category]
                }

                # Get actual sentiments
                actual_with_sentiment = {
                    label_info['label'].lower(): {
                        'sentiment': label_info['sentiment']
                    } for label_info in item['labels'][category]
                }

                # Store detailed predictions including sentiments
                prediction['categories'][category] = {
                    'predicted': [
                        {
                            'label': label,
                            'sentiment': predicted_with_sentiment[label]['sentiment'],
                            'confidence': predicted_with_sentiment[label]['confidence']
                        } for label in pred_labels
                    ],
                    'actual': [
                        {
                            'label': label,
                            'sentiment': actual_with_sentiment[label]['sentiment']
                        } for label in true_labels
                    ]
                }

                # Track total and correct predictions
                total_predictions[category] += len(true_labels)
                correct_predictions[category] += len(pred_labels & true_labels)

                # Calculate metrics for this prediction
                tp = len(pred_labels & true_labels)
                fp = len(pred_labels - true_labels)
                fn = len(true_labels - pred_labels)

                # Store metrics
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0

                metrics[category]['precision'].append(precision)
                metrics[category]['recall'].append(recall)

                # Track incorrect predictions
                if fp > 0 or fn > 0:
                    has_errors = True
                    incorrect_item['categories'][category] = {
                        'predicted': [
                            {
                                'label': label,
                                'sentiment': predicted_with_sentiment[label]['sentiment'],
                                'confidence': predicted_with_sentiment[label]['confidence']
                            } for label in pred_labels
                        ],
                        'actual': [
                            {
                                'label': label,
                                'sentiment': actual_with_sentiment[label]['sentiment']
                            } for label in true_labels
                        ],
                        'false_positives': [
                            {
                                'label': label,
                                'sentiment': predicted_with_sentiment[label]['sentiment'],
                                'confidence': predicted_with_sentiment[label]['confidence']
                            } for label in (pred_labels - true_labels)
                        ],
                        'false_negatives': [
                            {
                                'label': label,
                                'sentiment': actual_with_sentiment[label]['sentiment']
                            } for label in (true_labels - pred_labels)
                        ]
                    }

                # Calculate sentiment accuracy where labels match
                sentiment_correct = 0
                sentiment_total = 0

                # Check sentiment accuracy for true positives
                for label in (pred_labels & true_labels):
                    sentiment_total += 1
                    pred_sentiment = predicted_with_sentiment[label]['sentiment'] > 0.5
                    true_sentiment = actual_with_sentiment[label]['sentiment'] > 0
                    if pred_sentiment == true_sentiment:
                        sentiment_correct += 1

                if sentiment_total > 0:
                    metrics[category]['sentiment_accuracy'].append(sentiment_correct / sentiment_total)

            # Save incorrect predictions
            if has_errors:
                self.incorrect_predictions.append(incorrect_item)
            self.predictions.append(prediction)

        # Calculate average metrics
        results = {
            'inference_metrics': {
                'total_time': total_time,
                'average_time': total_time / len(test_data),
                'samples_processed': len(test_data)
            },
            'performance_metrics': {},
            'accuracy_percentages': {}
        }

        # Calculate average metrics and percentages per category
        for category in self.label_sets.keys():
            # Calculate percentage accuracy
            accuracy_percentage = (
                (correct_predictions[category] / total_predictions[category] * 100)
                if total_predictions[category] > 0 else 0
            )

            results['accuracy_percentages'][category] = accuracy_percentage

            # Calculate other metrics
            category_metrics = metrics[category]
            results['performance_metrics'][category] = {
                'precision': sum(category_metrics['precision']) / len(category_metrics['precision']),
                'recall': sum(category_metrics['recall']) / len(category_metrics['recall']),
                'f1': self._calculate_f1(
                    sum(category_metrics['precision']) / len(category_metrics['precision']),
                    sum(category_metrics['recall']) / len(category_metrics['recall'])
                )
            }

            if category_metrics['sentiment_accuracy']:
                results['performance_metrics'][category]['sentiment_accuracy'] = (
                        sum(category_metrics['sentiment_accuracy']) /
                        len(category_metrics['sentiment_accuracy'])
                )

        return results

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report from evaluation results"""
        report = []
        report.append("=" * 50)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 50)

        # Inference Metrics
        report.append("\nINFERENCE METRICS:")
        report.append("-" * 20)
        inf_metrics = results['inference_metrics']
        report.append(f"Total Processing Time: {inf_metrics['total_time']:.2f} seconds")
        report.append(f"Average Inference Time: {inf_metrics['average_time'] * 1000:.2f} ms per sample")
        report.append(f"Samples Processed: {inf_metrics['samples_processed']}")

        # Accuracy Percentages
        report.append("\nACCURACY PERCENTAGES:")
        report.append("-" * 20)
        for category, accuracy in results['accuracy_percentages'].items():
            report.append(f"{category.upper()}: {accuracy:.1f}%")

        # Detailed Performance Metrics
        report.append("\nDETAILED PERFORMANCE METRICS:")
        report.append("-" * 20)
        for category, metrics in results['performance_metrics'].items():
            report.append(f"\n{category.upper()}:")
            report.append(f"  Precision: {metrics['precision']:.3f}")
            report.append(f"  Recall: {metrics['recall']:.3f}")
            report.append(f"  F1 Score: {metrics['f1']:.3f}")
            if 'sentiment_accuracy' in metrics:
                report.append(f"  Sentiment Accuracy: {metrics['sentiment_accuracy']:.3f}")

        # Incorrect Predictions Summary
        report.append(f"\nINCORRECT PREDICTIONS:")
        report.append("-" * 20)
        report.append(f"Total incorrect predictions: {len(self.incorrect_predictions)}")

        return "\n".join(report)

    def save_incorrect_predictions(self, filename: str = "incorrect_predictions.json"):
        """Save incorrect predictions to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.incorrect_predictions, f, indent=2)
        logger.info(f"Incorrect predictions saved to {filename}")
        with open("predictions", 'w') as f:
            json.dump(self.predictions, f, indent=2)
        logger.info(f"Incorrect predictions saved to {filename}")


def main():
    # Configuration
    model_path = "./preference_model_output"
    test_data_path = "./test_data.json"

    try:
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)

        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)

        # Run evaluation
        results = evaluator.evaluate_predictions(test_data['data'])

        # Generate and print report
        report = evaluator.generate_report(results)
        print(report)

        # Save report to file
        with open('evaluation_report.txt', 'w') as f:
            f.write(report)

        # Save incorrect predictions
        evaluator.save_incorrect_predictions()

        logger.info("Evaluation completed. Report and incorrect predictions saved.")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()