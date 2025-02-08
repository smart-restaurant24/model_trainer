import random
import json
import re
from typing import List, Dict
import itertools


class RestaurantDataAugmenter:
    def __init__(self):
        self.price_points = [
            8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 150, 180, 200, 250, 300
        ]

        self.table_numbers = list(range(1, 31))  # Common table numbers
        self.party_sizes = list(range(1, 16))  # Common party sizes
        self.portion_sizes = [1, 2, 3, 4, 5]  # Common portion quantities
        self.menu_item_numbers = list(range(1, 51))  # Menu item numbers
        self.time_slots = [
            "11:30", "12:00", "12:30", "1:00", "1:30", "2:00",
            "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00"
        ]

        self.cuisines = {
            "Italian": ["pasta", "pizza", "risotto", "bruschetta", "lasagna", "gnocchi", "carpaccio",
                        "osso buco", "tiramisu", "cannoli"],
            "Chinese": ["dim sum", "dumplings", "fried rice", "noodles", "spring rolls", "kung pao chicken",
                        "mapo tofu", "peking duck", "hot pot", "chow mein"],
            "Japanese": ["sushi", "ramen", "tempura", "udon", "sashimi", "miso soup", "katsu",
                         "yakitori", "donburi", "gyoza"],
            "Thai": ["pad thai", "green curry", "tom yum", "satay", "papaya salad",
                     "massaman curry", "pad see ew", "thai basil chicken", "tom kha gai"],
            "Indian": ["butter chicken", "biryani", "tikka masala", "naan", "curry", "samosa",
                       "tandoori chicken", "dal makhani", "palak paneer", "korma"],
            "Mexican": ["tacos", "enchiladas", "guacamole", "quesadillas", "burritos",
                        "fajitas", "mole", "churros", "tostadas", "tamales"]
        }

        self.meal_types = ["appetizer", "starter", "main course", "entrée", "dessert", "side dish"]
        self.dietary_preferences = ["vegetarian", "vegan", "gluten-free", "dairy-free", "halal"]
        self.spice_levels = ["mild", "medium", "spicy", "extra spicy"]

        self.currency_formats = ["${price}", "{price} dollars", "USD {price}", "{price}"]

        # Enhanced templates with explicit non-price number scenarios
        self.templates = {
            "beginner": {
        "max_limit": [
            "Do you have anything under {price}?",
            "What dishes are less than {price}?",
            "I'd like something under {price}",
            "Can I get a meal within {price}?",
            "What's available below {price}?"
        ],
        "min_limit": [
            "What specialties do you have over {price}?",
            "Can you show me dishes above {price}?",
            "I want something more than {price}",
            "What premium dishes are over {price}?",
            "Any special items above {price}?"
        ],
        "price_range": [
            "What can you recommend between {price1} and {price2}?",
            "I'm looking in the {price1} to {price2} range",
            "Show me dishes from {price1} to {price2}",
            "What options do you have between {price1}-{price2}?",
            "Can I see items in the {price1}-{price2} range?"
        ],
        "no_price_limit": [
            # Table/Seating Related
            "Can we get a table for {num_people}?",
            "Is table {table_num} available?",
            "Could you seat {num_people} of us?",
            "We'd like table {table_num} by the window",
            "Can you combine {table_num} and {table_num2} for our group?",

            # Order Quantity Related
            "I'll have {num_portions} of the pasta",
            "Can we get {num_portions} orders of spring rolls?",
            "We need {num_portions} more plates",
            "{num_portions} servings of the soup please",
            "Make that {num_portions} orders",

            # Time Relate
        ]
    },
    "intermediate": {
        "max_limit": [
            "For {num_people} people, what can we get under {price}?",
            "Table {table_num} wants options under {price}",
            "Can we get {num_portions} dishes under {price} each?",
            "What combos for {num_people} stay below {price}?",
            "Is item #{menu_num} under {price}?"
        ],
        "min_limit": [
            "What premium dishes over {price} would you recommend for {num_people}?",
            "Table {table_num} wants to see specials above {price}",
            "Can you suggest {num_portions} items over {price}?",
            "Show us your signature dishes above {price}",
            "Any chef's specials over {price} for table {table_num}?"
        ],
        "price_range": [
            "What can {num_people} people get between {price1}-{price2}?",
            "Show table {table_num} options from {price1} to {price2}",
            "We need {num_portions} plates in the {price1}-{price2} range",
            "What dishes between {price1}-{price2} work for sharing?",
            "Can you suggest items {price1}-{price2} for our group?"
        ],
        "no_price_limit": [
            # Complex Table Scenarios
            "Can you combine tables for {num_people} near table {table_num}?",
            "We need high chairs for {num_portions} kids at table {table_num}",
            "Table {table_num} needs {num_portions} extra settings",
            "Could you split table {table_num} for {num_people} people?",
            "Can we move our {num_people} guests to table {table_num}?",

            # Detailed Order Quantities
            "For table {table_num}, {num_portions} appetizers and {num_portions2} mains",
            "We'll start with {num_portions} shares for {num_people}",
            "Item #{menu_num}, but split for {num_portions} people",
            "Can we get {num_portions} regular and {num_portions2} spicy?",
            "{num_portions} of #{menu_num} for table {table_num}",

            # Specific Timing Requests

        ]
    },
    "advanced": {
        "max_limit": [
            "For our party of {num_people} at table {table_num}, what sharing platters are under {price}?",
            "Can you arrange {num_portions} course tasting menu under {price} for {num_people} people?",
            "Table {table_num} needs dietary options for {num_people} under {price} each",
            "What wine pairings under {price} go with {num_portions} courses?",
            "Private dining for {num_people} with {num_portions} courses under {price}"
        ],
        "min_limit": [
            "Premium tasting menu above {price} for table {table_num} of {num_people}",
            "Chef's special course above {price} with {num_portions} wines",
            "Could you do {num_portions} premium dishes over {price} for {num_people}?",
            "Table {table_num} wants your premium pairing above {price}",
            "What's your best experience over {price} for {num_people}?"
        ],
        "price_range": [
            "Tasting menu between {price1}-{price2} for {num_people} at table {table_num}",
            "What {num_portions} course options are available {price1}-{price2}?",
            "Wine pairing menu {price1}-{price2} for party of {num_people}",
            "Special event menu {price1}-{price2} with {num_portions} courses",
            "Custom menu for {num_people} between {price1}-{price2}"
        ],
        "no_price_limit": [
            # Special Event Scenarios
            "Can you arrange {num_portions} course tasting for {num_people} at table {table_num}?",
            "Special wine pairing for {num_portions} courses at table {table_num}",
            "Chef's table for {num_people} with {num_portions} custom courses",
            "Private room near table {table_num} for {num_people} dinner",
            "Custom menu for table {table_num}, {num_portions} dietary restrictions",

            # Complex Timing and Service

        ]
    }
        }

        self.additional_context = {
            "beginner": [
                "for tonight",
                "for lunch",
                "walk-in available",
                "near the window",
                "with a view"
            ],
            "intermediate": [
                "with {dietary} options",
                "birthday celebration",
                "business casual",
                "date night",
                "family dinner"
            ],
            "advanced": [
                "wine pairing",
                "private dining room",
                "chef's table experience",
                "customized menu",
                "special occasion"
            ]
        }

    def _format_price(self, price: float) -> str:
        """Format price with random currency style"""
        price_format = random.choice(self.currency_formats)
        return price_format.format(price=price)

    def _generate_price_range(self) -> tuple:
        """Generate a realistic price range"""
        price1 = random.choice(self.price_points)
        higher_prices = [p for p in self.price_points if p > price1]
        price2 = random.choice(higher_prices) if higher_prices else price1 * 1.25
        return price1, price2

    def _fill_template_variables(self, template: str) -> tuple:
        """Fill template variables and track numbers"""
        numbers = []
        query = template

        # Handle price-related replacements
        if "{price1}" in template and "{price2}" in template:
            price1, price2 = self._generate_price_range()
            query = query.replace("{price1}", self._format_price(price1))
            query = query.replace("{price2}", self._format_price(price2))
            numbers.extend([
                {"value": price1, "type": "min_price"},
                {"value": price2, "type": "max_price"}
            ])
        elif "{price}" in template:
            price = random.choice(self.price_points)
            query = query.replace("{price}", self._format_price(price))
            price_type = "max_price" if "under" in template or "less than" in template else "min_price"
            numbers.append({"value": price, "type": price_type})

        # Handle non-price replacements
        if "{num_people}" in query:
            num = random.choice(self.party_sizes)
            query = query.replace("{num_people}", str(num))
            numbers.append({"value": num, "type": "not_price"})

        if "{table_num}" in query:
            num = random.choice(self.table_numbers)
            query = query.replace("{table_num}", str(num))
            numbers.append({"value": num, "type": "not_price"})

        if "{num_portions}" in query:
            num = random.choice(self.portion_sizes)
            query = query.replace("{num_portions}", str(num))
            numbers.append({"value": num, "type": "not_price"})

        if "{menu_num}" in query:
            num = random.choice(self.menu_item_numbers)
            query = query.replace("{menu_num}", str(num))
            numbers.append({"value": num, "type": "not_price"})

        if "{time}" in query:
            query = query.replace("{time}", random.choice(self.time_slots))

        # Replace other variables
        if "{cuisine}" in query:
            query = query.replace("{cuisine}", random.choice(list(self.cuisines.keys())))
        if "{dish}" in query:
            cuisine = random.choice(list(self.cuisines.keys()))
            query = query.replace("{dish}", random.choice(self.cuisines[cuisine]))
        if "{dietary}" in query:
            query = query.replace("{dietary}", random.choice(self.dietary_preferences))

        return query, numbers

    def _add_context(self, query: str, difficulty: str) -> str:
        """Add contextual information based on difficulty level"""
        if random.random() < 0.7:  # 70% chance to add context
            context = random.choice(self.additional_context[difficulty])
            if "{dietary}" in context:
                context = context.replace("{dietary}", random.choice(self.dietary_preferences))
            return f"{query}, {context}"
        return query

    def generate_example(self, difficulty: str) -> Dict:
        """Generate a single training example"""
        context_type = random.choice(list(self.templates[difficulty].keys()))
        template = random.choice(self.templates[difficulty][context_type])

        # Fill template and get numbers
        query, numbers = self._fill_template_variables(template)

        # Add context
        query = self._add_context(query, difficulty)

        return {
            "input_text": query,
            "context": context_type,
            "numbers": numbers,
            "difficulty": difficulty
        }

    def generate_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Generate a balanced dataset across difficulties"""
        dataset = []
        difficulty_percent = {
            "beginner": 45,
            "intermediate": 35,
            "advanced": 20
        }

        # Generate examples for each difficulty
        for difficulty, percent in difficulty_percent.items():
            num_diff_examples = round(num_examples * percent / 100)
            for _ in range(num_diff_examples):
                example = self.generate_example(difficulty)
                dataset.append(example)

        # Shuffle dataset
        random.shuffle(dataset)
        return dataset


def main():
    """Generate and save training dataset with detailed statistics"""
    augmenter = RestaurantDataAugmenter()
    dataset = augmenter.generate_dataset(500)  # Generate 1000 examples

    # Collect statistics
    stats = {
        "total": len(dataset),
        "by_difficulty": {},
        "by_context": {},
        "by_number_type": {},
        "avg_numbers_per_example": 0,
        "template_distribution": {
            "with_table_numbers": 0,
            "with_party_size": 0,
            "with_portions": 0,
            "with_menu_items": 0,
            "with_time": 0
        }
    }

    total_numbers = 0
    for example in dataset:
        # Count by difficulty
        diff = example["difficulty"]
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        # Count by context
        context = example["context"]
        stats["by_context"][context] = stats["by_context"].get(context, 0) + 1

        # Count by number type
        for num in example["numbers"]:
            num_type = num["type"]
            stats["by_number_type"][num_type] = stats["by_number_type"].get(num_type, 0) + 1
            total_numbers += 1

        # Count template distributions
        text = example["input_text"].lower()
        if any(f"table {i}" in text for i in range(1, 31)):
            stats["template_distribution"]["with_table_numbers"] += 1
        if any(f"party of {i}" in text for i in range(1, 16)):
            stats["template_distribution"]["with_party_size"] += 1
        if any(f"{i} portion" in text for i in range(1, 6)):
            stats["template_distribution"]["with_portions"] += 1
        if "#" in text or "item" in text:
            stats["template_distribution"]["with_menu_items"] += 1
        if any(time in text for time in augmenter.time_slots):
            stats["template_distribution"]["with_time"] += 1

    stats["avg_numbers_per_example"] = total_numbers / len(dataset)

    # Save dataset and statistics
    output = {
        "data": dataset,
        "statistics": stats
    }

    with open("test_data.json", "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print detailed statistics
    print("\nDataset Statistics:")
    print(f"\nTotal examples: {stats['total']}")

    print("\nBy Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"{diff}: {count} examples ({count/stats['total']*100:.1f}%)")

    print("\nBy Context Type:")
    for context, count in stats["by_context"].items():
        print(f"{context}: {count} examples ({count/stats['total']*100:.1f}%)")

    print("\nBy Number Type:")
    for num_type, count in stats["by_number_type"].items():
        print(f"{num_type}: {count} numbers ({count/total_numbers*100:.1f}%)")

    print("\nTemplate Distribution:")
    for template_type, count in stats["template_distribution"].items():
        print(f"{template_type}: {count} examples ({count/stats['total']*100:.1f}%)")

    print(f"\nAverage numbers per example: {stats['avg_numbers_per_example']:.2f}")

    # Print example queries for verification
    print("\nExample queries from each difficulty level:")
    for difficulty in ["beginner", "intermediate", "advanced"]:
        examples = [ex for ex in dataset if ex["difficulty"] == difficulty]
        if examples:
            sample = random.choice(examples)
            print(f"\n{difficulty.title()}:")
            print(f"Query: {sample['input_text']}")
            print(f"Context: {sample['context']}")
            print(f"Numbers: {sample['numbers']}")


if __name__ == "__main__":
    main()