import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def get_budgets():
    raw = os.getenv("MONTHLY_BUDGETS", "")
    budgets = {}
    if raw:
        for chunk in raw.split(";"):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                try:
                    budgets[k.strip()] = float(v.strip())
                except ValueError:
                    continue
    return budgets


DEFAULT_BUDGETS = {
    "Groceries": 8000,
    "Food & Dining": 5000,
    "Transport": 3000,
    "Utilities": 4000,
    "Shopping": 5000,
    "Entertainment": 2000,
    "Health": 2000,
    "Subscriptions": 1500,
    "Travel": 6000,
    "Rent": 20000,
    "Stationery": 1000,
}


def budgets():
    merged = dict(DEFAULT_BUDGETS)
    merged.update(get_budgets())
    return merged
