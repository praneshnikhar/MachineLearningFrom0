from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config import MODEL_DIR
from src.anomaly import flag_outliers, load_category_stats
from src.features import build_features
from src.pipeline import load_model

app = FastAPI(title="Expense Classifier", version="1.0")


class Transaction(BaseModel):
    merchant: str
    description: str = ""
    amount: float
    date: str


class Batch(BaseModel):
    transactions: list[Transaction]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    model = load_model()
    stats = load_category_stats()
    df = build_features(
        pd.DataFrame(
            [
                {
                    "merchant": tx.merchant,
                    "description": tx.description,
                    "amount": tx.amount,
                    "date": tx.date,
                }
            ]
        )
    )
    category = model.predict(df["text"])[0]
    df["category"] = category
    outlier = bool(flag_outliers(df, stats)[0])
    return {"category": category, "is_anomaly": outlier}


@app.post("/predict_batch")
def predict_batch(batch: Batch):
    rows = [t.model_dump() for t in batch.transactions]
    df = pd.DataFrame(rows)
    model = load_model()
    stats = load_category_stats()
    df = build_features(df)
    df["category"] = model.predict(df["text"])
    df["is_anomaly"] = flag_outliers(df, stats)
    return df.to_dict(orient="records")


@app.get("/digest")
def digest():
    from src.pipeline import run

    summary, _, _ = run(notify=False)
    return summary
