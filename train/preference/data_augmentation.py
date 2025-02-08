import random
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any


class RestaurantDataAugmenter:
    def __init__(self):
        # Initialize all label categories (same as before)
        self.taste_labels = [
            "sweet", "sour", "salty", "bitter", "umami",
            "spicy", "crispy", "creamy", "smoky", "tangy"
        ]

        self.dietary_labels = [
            "vegetarian", "pescatarian", "gluten-free", "dairy-free",
            "kosher", "halal", "keto", "paleo", "mediterranean",
            "low-fodmap", "raw"
        ]

        self.course_labels = [
            "appetizer", "soup", "salad", "main-course", "side-dish",
            "dessert", "beverage", "small-plate", "cheese-course",
            "fish-course", "palate-cleanser", "amuse-bouche",
            "bread-service", "chef-special", "digestif"
        ]

        self.cuisine_labels = [
            "Chinese", "Italian", "Japanese", "Indian", "Mexican",
            "Thai", "Mediterranean", "Korean", "French", "Vietnamese"
        ]

        self.ingredient_labels = [
            "chicken", "beef", "fish", "pork", "shrimp", "tofu",
            "rice", "noodles", "pasta", "tomatoes", "mushrooms",
            "onions", "garlic", "ginger", "cheese", "eggs"
        ]

        # Basic query starters
        self.intros = [
            "I'm looking for",
            "Can you suggest",
            "I want",
            "I need",
            "Suggest",
            "Could you recommend",
            "I'd like",
            "I'm in the mood for",
            "Help me find",
            "Show me"
        ]

        # Natural language patterns for preferences by category and sentiment
        self.first_preference_templates = {
            'dietary': {
                1: [
                    "{label} food",
                    "something {label}",
                    "{label} options"
                ],
                -1: [
                    "something that's not {label}",
                    "non-{label} food"
                ]
            },
            'cuisine': {
                1: [
                    "{label} food",
                    "some {label} food",
                    "a {label} place"
                ],
                -1: [
                    "anything except {label}",
                    "something other than {label}"
                ]
            },
            'ingredient': {
                1: [
                    "something with {label}",
                    "a dish with {label}"
                ],
                -1: [
                    "something without {label}",
                    "a dish without {label}"
                ]
            },
            'taste': {
                1: [
                    "something {label}",
                    "a {label} dish"
                ],
                -1: [
                    "something that's not too {label}",
                    "nothing too {label}"
                ]
            },
            'course': {
                1: [
                    "a {label}",
                    "{label}"
                ],
                -1: [
                    "anything except {label}",
                    "something other than {label}"
                ]
            }
        }

        # Additional preference templates (for second and subsequent preferences)
        self.additional_preference_templates = {
            'dietary': {
                1: [
                    "it should be {label}",
                    "make sure it's {label}"
                ],
                -1: [
                    "but not {label}",
                    "nothing {label}"
                ]
            },
            'cuisine': {
                1: [
                    "from a {label} restaurant",
                    "{label} style"
                ],
                -1: [
                    "but not {label}",
                    "just not {label}"
                ]
            },
            'ingredient': {
                1: [
                    "with {label} in it",
                    "that has {label}"
                ],
                -1: [
                    "without any {label}",
                    "no {label}"
                ]
            },
            'taste': {
                1: [
                    "and {label}",
                    "that's {label}"
                ],
                -1: [
                    "but not too {label}",
                    "not very {label}"
                ]
            },
            'course': {
                1: [
                    "as a {label}",
                    "for {label}"
                ],
                -1: [
                    "but not a {label}",
                    "not as a {label}"
                ]
            }
        }

        # Conjunctions for combining preferences
        self.conjunctions = [
            " and ",
            ", but ",
            ". Also, ",
            ", and "
        ]

        self.stats = {
            'total_samples': 0,
            'category_distribution': {
                'taste': Counter(),
                'dietary': Counter(),
                'course': Counter(),
                'cuisine': Counter(),
                'ingredient': Counter()
            },
            'sentiment_distribution': Counter(),
            'combination_patterns': Counter()
        }

    def get_balanced_preference(self, used_categories=None) -> tuple:
        """Get a preference ensuring balanced distribution across categories and labels."""
        if used_categories is None:
            used_categories = set()

        # Get available categories
        available_categories = [cat for cat in ['taste', 'dietary', 'course', 'cuisine', 'ingredient']
                                if cat not in used_categories]

        if not available_categories:
            available_categories = ['taste', 'dietary', 'course', 'cuisine', 'ingredient']

        # Select category with lowest count
        category_counts = {cat: sum(self.stats['category_distribution'][cat].values())
                           for cat in available_categories}
        category = min(category_counts.items(), key=lambda x: x[1])[0]

        # Get label with lowest count from that category
        label_list = getattr(self, f"{category}_labels")
        label_counts = self.stats['category_distribution'][category]

        # If no counts yet, initialize with 0
        for label in label_list:
            if label not in label_counts:
                label_counts[label] = 0

        label = min(label_list, key=lambda x: label_counts[x])

        # Balance positive and negative sentiments
        pos_count = self.stats['sentiment_distribution'][1]
        neg_count = self.stats['sentiment_distribution'][-1]
        sentiment = -1 if pos_count > neg_count else 1

        return category, label, sentiment

    def generate_balanced_dataset(self, num_samples: int, output_file: str = None) -> Dict:
        """Generate a balanced dataset and save to JSON file."""
        dataset = []

        # Calculate desired samples per category
        samples_per_category = num_samples // 5  # 5 categories

        for _ in range(num_samples):
            # Randomly choose number of filters (1-3)
            num_filters = random.randint(1, 3)

            # Generate sample with balanced preferences
            used_categories = set()
            sample = self.generate_sentence(num_filters, used_categories)

            dataset.append(sample)

            # Update statistics
            self.update_statistics(sample)

        # Prepare final output
        output = {
            'dataset': dataset,
            'statistics': self.get_statistics(),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_samples': num_samples,
                'version': '1.0'
            }
        }

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        return output

    def update_statistics(self, sample: Dict):
        """Update statistics based on generated sample."""
        if not sample or 'labels' not in sample:
            return

        self.stats['total_samples'] += 1

        # Update category and label distributions
        for category, labels in sample['labels'].items():
            if labels:  # Only update if there are labels
                for label_info in labels:
                    self.stats['category_distribution'][category][label_info['label']] += 1
                    self.stats['sentiment_distribution'][label_info['sentiment']] += 1

        # Update combination patterns
        used_categories = [cat for cat, labels in sample['labels'].items() if labels]
        if used_categories:
            combination = '+'.join(sorted(used_categories))
            self.stats['combination_patterns'][combination] += 1

    def get_statistics(self) -> Dict:
        """Get current statistics in a structured format."""
        return {
            'total_samples': self.stats['total_samples'],
            'category_distribution': {
                category: dict(counts)
                for category, counts in self.stats['category_distribution'].items()
            },
            'sentiment_distribution': dict(self.stats['sentiment_distribution']),
            'combination_patterns': dict(self.stats['combination_patterns']),
            'average_preferences_per_query': sum(
                count for count in self.stats['combination_patterns'].values()
            ) / max(1, self.stats['total_samples'])
        }

    def get_random_preference(self) -> tuple:
        """Get a random preference category and label."""
        category = random.choice(['taste', 'dietary', 'course', 'cuisine', 'ingredient'])
        label_list = getattr(self, f"{category}_labels")
        label = random.choice(label_list)
        sentiment = random.choice([-1, 1])
        return category, label, sentiment

    def get_phrase_for_category(self, category: str, label: str, sentiment: int, is_first: bool = True) -> str:
        """Get appropriate phrasing for a category and sentiment."""
        templates = self.first_preference_templates if is_first else self.additional_preference_templates
        template = random.choice(templates[category][sentiment])
        return template.format(label=label.lower())

    def generate_sentence(self, num_filters: int = None, used_categories: set = None) -> dict:
        """Generate a sentence using balanced preferences."""
        if num_filters is None:
            num_filters = random.randint(1, 3)

        if used_categories is None:
            used_categories = set()

        intro = random.choice(self.intros)
        filters = []
        labels_dict = {}  # Initialize empty dict here

        for i in range(num_filters):
            category, label, sentiment = self.get_balanced_preference(used_categories)
            used_categories.add(category)

            # Get appropriate phrasing based on position
            filter_text = self.get_phrase_for_category(category, label, sentiment, is_first=(i == 0))

            # Add conjunction if not first filter
            if i > 0:
                filter_text = f"{random.choice(self.conjunctions)}{filter_text}"

            filters.append(filter_text)

            # Add to labels dictionary
            if category not in labels_dict:
                labels_dict[category] = []
            labels_dict[category].append({
                "label": label,
                "sentiment": sentiment
            })

        # Combine all parts into final text
        text = f"{intro} {' '.join(filters)}"
        text = ' '.join(text.split())
        text = text.replace(" .", ".").replace(" ,", ",")

        # Ensure all categories are present in output
        all_categories = {
            'taste': [],
            'dietary': [],
            'course': [],
            'cuisine': [],
            'ingredient': []
        }

        # Update with any found labels
        for category, labels in labels_dict.items():
            all_categories[category] = labels

        return {
            "input_text": text,
            "labels": all_categories
        }

    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate multiple samples for the dataset."""
        dataset = []
        for _ in range(num_samples):
            sample = self.generate_sentence()
            dataset.append(sample)
        return dataset


# Usage example
if __name__ == "__main__":
    augmenter = RestaurantDataAugmenter()

    # Generate a balanced dataset with 100 samples
    output_file = "test_data.json"
    result = augmenter.generate_balanced_dataset(200, output_file)

    # Print some statistics
    stats = result['statistics']
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print("\nCategory Distribution:")
    for category, counts in stats['category_distribution'].items():
        print(f"\n{category.capitalize()}:")
        for label, count in counts.items():
            print(f"  {label}: {count}")

    print("\nSentiment Distribution:")
    print(f"Positive: {stats['sentiment_distribution'].get(1, 0)}")
    print(f"Negative: {stats['sentiment_distribution'].get(-1, 0)}")

    print("\nMost Common Combinations:")
    for pattern, count in sorted(stats['combination_patterns'].items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
        print(f"{pattern}: {count}")

    print(f"\nAverage preferences per query: {stats['average_preferences_per_query']:.2f}")
    print(f"\nDataset saved to: {output_file}")