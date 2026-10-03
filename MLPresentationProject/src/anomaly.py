import json

import pandas as pd

from config import MODEL_DIR


def load_category_stats():
    path = MODEL_DIR / "category_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_budgets():
    path = MODEL_DIR / "budgets.json"
    return json.loads(path.read_text()) if path.exists() else {}


def flag_outliers(df, stats, k=3.0):
    """Flag transactions whose amount is unusually high for their predicted category."""
    flags = []
    for _, row in df.iterrows():
        cat = row["category"]
        amount = float(row["amount"])
        s = stats.get(cat, {})
        mean = s.get("mean", amount)
        std = s.get("std", 0.0)
        p99 = s.get("p99", amount)
        threshold = mean + k * std if std > 0 else p99
        is_outlier = bool(amount > threshold) and bool(amount > p99 * 1.2)
        flags.append(is_outlier)
    return flags


def budget_overruns(df, budgets):
    """Compare month-to-date spend per category against monthly budgets.

    Only the most recent month present in the data is evaluated — that is the
    "current" month for a weekly watchdog run.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    if df["month"].nunique() == 0:
        return []
    current = df["month"].max()
    df = df[df["month"] == current]
    overruns = []
    for cat, spend in df.groupby("category")["amount"].sum().items():
        budget = budgets.get(cat)
        if not budget:
            continue
        ratio = float(spend) / float(budget)
        if ratio >= 0.9:
            overruns.append(
                {
                    "month": str(current),
                    "category": cat,
                    "spend": round(float(spend), 2),
                    "budget": float(budget),
                    "ratio": round(ratio, 2),
                    "status": "OVER" if ratio >= 1.0 else "WARNING",
                }
            )
    return overruns
