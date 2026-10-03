import pandas as pd

CATEGORIES = [
    "Groceries",
    "Food & Dining",
    "Transport",
    "Utilities",
    "Shopping",
    "Entertainment",
    "Health",
    "Subscriptions",
    "Travel",
    "Rent",
    "Stationery",
]


def build_features(df):
    """Create a consistent feature frame used by both training and inference."""
    df = df.copy()
    df["merchant"] = df.get("merchant", pd.Series(index=df.index, dtype=str)).fillna("")
    df["description"] = df.get("description", pd.Series(index=df.index, dtype=str)).fillna("")
    df["text"] = (df["merchant"].astype(str) + " " + df["description"].astype(str)).str.lower()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day_of_week"] = df["date"].dt.dayofweek.fillna(0).astype(int)
    else:
        df["day_of_week"] = 0
    return df
