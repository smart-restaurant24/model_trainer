import random
import json
from typing import List, Dict, Union
from datetime import datetime


class DynamicTemplateEngine:
    def __init__(self):
        self.templates = {}
        self.variables = {}

    def add_template(self, intent: str, templates: List[str]):
        self.templates[intent] = templates

    def add_variables(self, var_name: str, values: List[str]):
        self.variables[var_name] = values

    def get_template(self, intent: str) -> str:
        return random.choice(self.templates.get(intent, []))

    def fill_template(self, template: str) -> str:
        filled = template
        for var_name, values in self.variables.items():
            pattern = f"{{{var_name}}}"
            while pattern in filled:
                filled = filled.replace(pattern, random.choice(values), 1)
        return filled


class RestaurantQueryAugmenter:
    def __init__(self):
        self.template_engine = DynamicTemplateEngine()
        self._initialize_templates()
        self._initialize_variables()
        self._initialize_difficulty_params()

    def _initialize_templates(self):
        templates = {
            "ENQUIRY_MENU": [
                "what's on the {menu_type}",
                "can i see your {menu_type}",
                "what are the {meal_time} specials",
                "what's good on the {menu_type}",
                "any recommendations from {menu_type}",
                "do you have {cuisine} dishes on the menu",
                "is {dish} on the menu",
                "show me some {dietary_restriction} options",
                "what comes in the {combo_name}",
                "price for {dish}"
            ],
            "ENQUIRY_CUISINE": [
                "how's your {cuisine} food",
                "is your {cuisine} spicy",
                "which {cuisine} dish should i try",
                "are your {cuisine} dishes authentic",
                "what's your best {cuisine} dish",
                "can you make {cuisine} food {spice_level}",
                "do you have {regional} style {cuisine}",
                "is chef {cuisine}",
                "how authentic is your {cuisine}"
            ],
            "ENQUIRY_DISH": [
                "what's in the {dish}",
                "is {dish} {attribute}",
                "how spicy is your {dish}",
                "can i get {dish} {dietary_restriction}",
                "what comes with {dish}",
                "portion size for {dish}",
                "can you customize the {dish}",
                "any nuts in {dish}",
                "is {dish} for one person",
                "{dish} good for sharing"
            ],
            "ENQUIRY_RESTAURANT": [
                "do you have {restaurant_feature}",
                "how late are you open",
                "when do you close",
                "can you do {special_requirement}",
                "is there {amenity}",
                "how busy is it right now",
                "do i need a reservation",
                "can i bring {special_item}",
                "where are you located",
                "is there parking"
            ],
            "ORDER_RELATED": [
                "can i order {order_type}",
                "how long for {order_type}",
                "do you deliver to {location}",
                "minimum order for delivery",
                "can i order for later",
                "how to cancel my order",
                "where's my order",
                "can i add to my order",
                "is my order ready",
                "delivery time to {location}"
            ],
            "RESERVATION_RELATED": [
                "table for {party_size} tonight",
                "can i book for {party_size}",
                "reservation for {day_time}",
                "got space for {party_size}",
                "need a table for {special_occasion}",
                "how long is the wait",
                "can i get a booth",
                "do you have outdoor seating",
                "private room available",
                "can i modify my booking"
            ],
            "PAYMENT_RELATED": [
                "do you take {payment_method}",
                "can we split the bill",
                "is there a minimum charge",
                "do you accept {payment_method}",
                "service charge included",
                "any discounts today",
                "do i need to pay now",
                "how much is the deposit",
                "can i pay online",
                "card machine working"
            ],
            "SERVICE_RELATED": [
                "need help with my order",
                "order is taking long",
                "food is cold",
                "missing items in order",
                "wrong order received",
                "can i speak to manager",
                "issue with my booking",
                "need special assistance",
                "allergies to check",
                "dietary restrictions"
            ],
            "GENERAL": [
                "what's your specialty",
                "when are you busy",
                "how long have you been open",
                "is the owner here",
                "best time to come",
                "any events today",
                "do you cater",
                "what's your rating",
                "are you on uber eats",
                "any entertainment tonight"
            ],
            "NON_RELATED": [
                "where can i park",
                "nearest metro station",
                "is area safe at night",
                "other restaurants nearby",
                "where's the atm",
                "any shops nearby",
                "good hotels around here",
                "call taxi please",
                "weather forecast today",
                "tourist spots nearby"
            ]
        }

        for intent, intent_templates in templates.items():
            self.template_engine.add_template(intent, intent_templates)

    def _initialize_variables(self):
        variables = {
            "menu_type": [
                "dinner menu", "lunch menu", "breakfast menu",
                "kids menu", "dessert menu", "drinks menu",
                "wine list", "cocktail menu", "special menu"
            ],
            "meal_time": [
                "lunch", "dinner", "breakfast", "today's", "weekend"
            ],
            "cuisine": [
                "Italian", "Chinese", "Indian", "Mexican", "Japanese",
                "Thai", "Korean", "Mediterranean", "French", "Vietnamese"
            ],
            "dish": [
                "chicken curry", "pad thai", "pizza", "burger",
                "sushi roll", "pasta", "steak", "fish and chips",
                "fried rice", "noodle soup"
            ],
            "combo_name": [
                "family meal", "lunch special", "dinner combo",
                "party platter", "set menu", "value meal"
            ],
            "attribute": [
                "spicy", "vegetarian", "gluten-free", "fresh",
                "homemade", "popular"
            ],
            "dietary_restriction": [
                "vegetarian", "vegan", "gluten-free", "dairy-free",
                "nut-free", "halal"
            ],
            "spice_level": [
                "extra spicy", "mild", "medium spicy", "not spicy"
            ],
            "regional": [
                "North", "South", "East", "West", "Central"
            ],
            "restaurant_feature": [
                "outdoor seating", "private rooms", "parking",
                "wifi", "tv screens", "bar"
            ],
            "special_requirement": [
                "birthday party", "large group", "business meeting",
                "private event", "wedding party"
            ],
            "amenity": [
                "wifi", "parking", "high chairs", "wheelchair access",
                "baby changing", "smoking area"
            ],
            "special_item": [
                "cake", "wine", "decorations", "gifts", "food"
            ],
            "order_type": [
                "takeout", "delivery", "pickup", "catering",
                "party order", "large order"
            ],
            "location": [
                "downtown", "uptown", "west side", "east side",
                "business district", "campus area"
            ],
            "party_size": [
                "2", "4", "6", "8", "10", "large group"
            ],
            "day_time": [
                "tonight", "tomorrow", "weekend", "friday night",
                "saturday evening", "sunday brunch"
            ],
            "special_occasion": [
                "birthday", "anniversary", "date", "business lunch",
                "family dinner", "celebration"
            ],
            "payment_method": [
                "credit card", "debit card", "Apple Pay", "Google Pay",
                "PayPal", "gift card"
            ]
        }

        for var_name, values in variables.items():
            self.template_engine.add_variables(var_name, values)

    def _initialize_difficulty_params(self):
        self.difficulty_params = {
            "easy": {
                "multi_intent_prob": 0.1,
                "max_intents": 1
            },
            "medium": {
                "multi_intent_prob": 0.7,
                "max_intents": 2
            },
            "hard": {
                "multi_intent_prob": 0.9,
                "max_intents": 3
            }
        }

    def generate_query(self, intents: List[str], difficulty: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a natural chatbot-style query"""
        queries = []
        for intent in intents:
            template = self.template_engine.get_template(intent)
            query = self.template_engine.fill_template(template)
            queries.append(query)

        # Combine queries in a more natural way
        if len(queries) > 1:
            # Use different combinations based on number of queries
            if len(queries) == 2:
                combined_query = f"{queries[0]} and {queries[1]}"
            else:
                # For three queries, use comma and 'and'
                combined_query = f"{queries[0]}, {queries[1]} and {queries[2]}"
        else:
            combined_query = queries[0]

        # Add question mark if not present
        if not combined_query.endswith("?"):
            combined_query += "?"

        return {
            "query": combined_query,
            "intents": intents,
            "difficulty": difficulty
        }

    def augment_data(self, num_samples: int = 100, difficulty: str = "medium") -> List[
        Dict[str, Union[str, List[str]]]]:
        """Generate augmented dataset with specified difficulty"""
        params = self.difficulty_params[difficulty]
        all_intents = list(self.template_engine.templates.keys())
        augmented_data = []

        for _ in range(num_samples):
            if random.random() < params["multi_intent_prob"]:
                num_intents = random.randint(1, params["max_intents"])
                selected_intents = random.sample(all_intents, num_intents)
            else:
                selected_intents = [random.choice(all_intents)]

            sample = self.generate_query(selected_intents, difficulty)
            augmented_data.append(sample)

        return augmented_data


def main():
    augmenter = RestaurantQueryAugmenter()

    # Generate samples for each difficulty
    difficulties = ["easy", "medium", "hard"]
    all_data = []

    for difficulty in difficulties:
        data = augmenter.augment_data(num_samples=200, difficulty=difficulty)
        all_data.extend(data)

        # Print examples
        print(f"\n{difficulty.upper()} examples:")
        for i, sample in enumerate(random.sample(data, 3)):
            print(f"{i + 1}. {sample['query']}")
            print(f"   Intents: {', '.join(sample['intents'])}")

    # Save to JSON
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(all_data)} samples")


if __name__ == "__main__":
    main()