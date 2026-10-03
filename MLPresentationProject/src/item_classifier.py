import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import MODEL_DIR
from src.features import CATEGORIES

ITEMS = {
    "Groceries": [
        "milk", "bread", "butter", "eggs", "rice", "dal", "atta", "flour",
        "vegetables", "fruits", "apple", "banana", "onion", "potato", "tomato",
        "sugar", "salt", "cooking oil", "tea leaves", "coffee powder",
        "biscuits", "juice", "chips", "snacks", "curd", "paneer", "chicken",
        "fish", "meat", "soap", "shampoo", "toothpaste", "detergent",
    ],
    "Food & Dining": [
        "pizza", "burger", "biryani", "samosa", "dosa", "idli", "coffee",
        "chai", "tea", "lunch", "dinner", "breakfast", "restaurant", "swiggy",
        "zomato", "cafe", "noodles", "pasta", "sandwich", "roll", "thali",
        "paratha", "momos", "ice cream", "coke", "cold drink",
    ],
    "Transport": [
        "uber", "ola", "auto", "bus", "metro", "taxi", "petrol", "diesel",
        "train", "rapido", "fuel", "parking", "toll", "cab", "rickshaw",
    ],
    "Utilities": [
        "electricity", "water", "wifi", "internet", "recharge", "gas",
        "bill", "broadband", "mobile recharge", "dth", "cable", "municipal",
    ],
    "Shopping": [
        "clothes", "shirt", "tshirt", "jeans", "shoes", "watch", "phone",
        "laptop", "headphones", "bag", "wallet", "belt", "socks", "dress",
        "jacket", "perfume", "gift", "earbuds", "charger", "cable", "kurti",
        "saree", "mobile", "screen guard", "case",
    ],
    "Entertainment": [
        "movie", "netflix", "spotify", "game", "concert", "ticket",
        "theatre", "pvr", "bookmyshow", "steam", "music", "gaming",
        "playstation", "xbox", "match", "show",
    ],
    "Health": [
        "medicine", "doctor", "gym", "pharmacy", "tablet", "vitamins",
        "hospital", "dentist", "consultation", "bandage", "syrup", "protein",
        "yoga", "clinic", "blood test",
    ],
    "Subscriptions": [
        "netflix", "spotify", "prime", "icloud", "youtube", "subscription",
        "membership", "amazon prime", "hotstar", "disney", "audible",
    ],
    "Travel": [
        "flight", "hotel", "airbnb", "trip", "bus ticket", "train ticket",
        "cab", "stay", "visa", "passport", "resort", "sightseeing",
    ],
    "Rent": [
        "rent", "hostel", "pg", "deposit", "maintenance", "flat",
        "house rent", "room rent",
    ],
    "Stationery": [
        "pen", "pencil", "eraser", "notebook", "paper", "marker", "stapler",
        "sharpener", "ruler", "folder", "file", "glue", "scissors",
        "highlighter", "book", "textbook", "diary", "calculator", "scale",
        "register", "chart paper", "sketch pen",
    ],
}

CATEGORY_ALIASES = {
    "groceries": "Groceries",
    "grocery": "Groceries",
    "food": "Food & Dining",
    "dining": "Food & Dining",
    "transport": "Transport",
    "travel": "Travel",
    "utilities": "Utilities",
    "utility": "Utilities",
    "bills": "Utilities",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "fun": "Entertainment",
    "health": "Health",
    "medical": "Health",
    "subscription": "Subscriptions",
    "subscriptions": "Subscriptions",
    "rent": "Rent",
    "stationery": "Stationery",
    "stationary": "Stationery",
    "misc": "Shopping",
    "other": "Shopping",
}


def build_training_df():
    rows = []
    for cat, items in ITEMS.items():
        for it in items:
            rows.append({"text": it.lower(), "category": cat})
    return pd.DataFrame(rows)


def build_model():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def train():
    df = build_training_df()
    model = build_model()
    model.fit(df["text"], df["category"])
    path = MODEL_DIR / "item_classifier.joblib"
    joblib.dump(model, path)
    print(f"Trained item classifier on {len(df)} examples -> {path}")
    return model


def load():
    return joblib.load(MODEL_DIR / "item_classifier.joblib")


def resolve_category(text):
    """Return an explicit category if the text names one, else None."""
    for tok in text.lower().split():
        if tok in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[tok]
    return None


if __name__ == "__main__":
    train()
