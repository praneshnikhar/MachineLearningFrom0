import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import BASE_DIR, DATA_DIR
from src.features import CATEGORIES

MERCHANTS = {
    "Groceries": ["BigBasket", "DMart", "Whole Foods", "Reliance Fresh", "More Supermarket"],
    "Food & Dining": ["Swiggy", "Zomato", "McDonalds", "Dominos", "Local Restaurant", "Cafe Coffee Day"],
    "Transport": ["Uber", "Ola", "Metro Card", "IndianOil Petrol", "Rapido"],
    "Utilities": ["Electricity Board", "Water Utility", "BSNL Broadband", "Jio Recharge", "Gas Cylinder"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "H&M", "Nykaa"],
    "Entertainment": ["PVR Cinemas", "BookMyShow", "Steam", "Playstation Store"],
    "Health": ["Apollo Pharmacy", "Practo", "MedPlus", "Gym Membership", "Cult.fit"],
    "Subscriptions": ["Netflix", "Spotify", "Amazon Prime", "YouTube Premium", "iCloud"],
    "Travel": ["Indigo Airlines", "IRCTC", "MakeMyTrip", "Airbnb", "OYO"],
    "Rent": ["Monthly Rent", "Hostel Rent", "PG Rent"],
    "Stationery": ["Amazon Stationery", "WH Smith", "Classmate", "Local Bookstore", "Flipkart Stationery"],
}

AMOUNT_RANGES = {
    "Groceries": (150, 2500),
    "Food & Dining": (100, 1200),
    "Transport": (50, 600),
    "Utilities": (200, 2500),
    "Shopping": (200, 5000),
    "Entertainment": (150, 1500),
    "Health": (100, 2000),
    "Subscriptions": (99, 1000),
    "Travel": (500, 15000),
    "Rent": (8000, 25000),
    "Stationery": (20, 1500),
}

CATEGORY_WEIGHTS = {
    "Groceries": 15,
    "Food & Dining": 25,
    "Transport": 18,
    "Utilities": 8,
    "Shopping": 10,
    "Entertainment": 8,
    "Health": 5,
    "Subscriptions": 5,
    "Travel": 3,
    "Rent": 1,
    "Stationery": 4,
}

SUFFIXES = [
    "ORDER-{n}",
    "TXN-{n}",
    "REF#{n}",
    "PAYMENT {n}",
    "*{n}",
    "INV-{n}",
]


def _random_date(rng, start, end):
    span = (end - start).days
    return start + timedelta(days=rng.randint(0, span))


def _description(rng, merchant):
    suffix = rng.choice(SUFFIXES).format(n=rng.randint(100, 99999))
    if rng.random() < 0.4:
        return merchant
    return f"{merchant} {suffix}"


def _sample_category(rng, category_weights=None):
    if category_weights is None:
        return rng.choice(CATEGORIES)
    cats = list(category_weights.keys())
    weights = [category_weights[c] for c in cats]
    return random.choices(cats, weights=weights, k=1)[0]


def generate_transactions(n, start=None, end=None, seed=42, category_weights=None):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if end is None:
        end = datetime.now()
    if start is None:
        start = end - timedelta(days=365)

    rows = []
    for _ in range(n):
        category = _sample_category(rng, category_weights)
        merchant = rng.choice(MERCHANTS[category])
        lo, hi = AMOUNT_RANGES[category]
        amount = round(float(np.random.lognormal(mean=np.log((lo + hi) / 2), sigma=0.6)), 2)
        amount = min(max(amount, lo), hi * 1.5)
        date = _random_date(random, start, end)
        rows.append(
            {
                "merchant": merchant,
                "description": _description(random, merchant),
                "amount": amount,
                "date": date.strftime("%Y-%m-%d"),
                "category": category,
            }
        )

    return pd.DataFrame(rows)


def main():
    train = generate_transactions(4000, seed=42, category_weights=CATEGORY_WEIGHTS)
    new = generate_transactions(
        120, seed=7, start=datetime.now() - timedelta(days=30), category_weights=CATEGORY_WEIGHTS
    )
    new = new.drop(columns=["category"])

    today = datetime.now()
    anomalies = pd.DataFrame(
        [
            {"merchant": "BigBasket", "description": "BigBasket ORDER-9999", "amount": 9500.0,
             "date": (today - timedelta(days=3)).strftime("%Y-%m-%d")},
            {"merchant": "Amazon", "description": "Amazon ORDER-8888", "amount": 48000.0,
             "date": (today - timedelta(days=1)).strftime("%Y-%m-%d")},
            {"merchant": "Swiggy", "description": "Swiggy ORDER-7777", "amount": 6200.0,
             "date": today.strftime("%Y-%m-%d")},
        ]
    )
    new = pd.concat([new, anomalies], ignore_index=True)

    train_path = DATA_DIR / "transactions_train.csv"
    new_path = DATA_DIR / "transactions_new.csv"
    train.to_csv(train_path, index=False)
    new.to_csv(new_path, index=False)
    print(f"Wrote {len(train)} training rows -> {train_path}")
    print(f"Wrote {len(new)} unlabeled rows -> {new_path}")
    print(f"\nCategory distribution:\n{train['category'].value_counts()}")


if __name__ == "__main__":
    main()
