import json
import sys
from datetime import datetime

import joblib
import pandas as pd

from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from src.anomaly import budget_overruns, flag_outliers, load_budgets, load_category_stats
from src.features import build_features
from src.slack_notifier import post_alert, post_summary

MODEL_PATH = MODEL_DIR / "classifier.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def ingest(path=None):
    path = path or (DATA_DIR / "transactions_new.csv")
    return pd.read_csv(path)


def classify(model, df):
    df = build_features(df)
    df["category"] = model.predict(df["text"])
    return df


def summarize(df, budgets):
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    total = round(float(df["amount"].sum()), 2)
    by_cat = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )
    top_merchants = (
        df.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(5).round(2).to_dict()
    )
    current_month = str(df["month"].max())
    return {
        "total": total,
        "transactions": int(len(df)),
        "current_month": current_month,
        "by_category": by_cat,
        "top_merchants": top_merchants,
    }


def run(path=None, notify=True):
    model = load_model()
    stats = load_category_stats()
    budgets = load_budgets()

    raw = ingest(path)
    df = classify(model, raw)
    df["is_outlier"] = flag_outliers(df, stats)
    overruns = budget_overruns(df, budgets)

    summary = summarize(df, budgets)
    summary["budgets"] = budgets

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(OUTPUT_DIR / f"classified_{ts}.csv", index=False)
    df.to_csv(OUTPUT_DIR / "latest.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "overruns.json").write_text(json.dumps(overruns, indent=2))

    print(json.dumps(summary, indent=2))

    if notify:
        _notify(summary, df, overruns)

    return summary, df, overruns


def _notify(summary, df, overruns):
    cat_lines = "\n".join(
        f"{k}: ₹{v:,.0f}" for k, v in summary["by_category"].items()
    )
    post_summary(
        "Weekly Spend Digest",
        [
            ("Total", f"₹{summary['total']:,.0f} across {summary['transactions']} transactions"),
            ("By category", cat_lines),
        ],
    )

    outliers = df[df["is_outlier"]]
    if len(outliers):
        lines = [
            f"{r['merchant']} — ₹{r['amount']:,.0f} ({r['category']}, {r['date']})"
            for _, r in outliers.iterrows()
        ]
        post_alert("⚠️ Unusual spending detected", lines[:20])

    if overruns:
        lines = [
            f"{o['category']} — ₹{o['spend']:,.0f} / ₹{o['budget']:,.0f} ({o['ratio']*100:.0f}%) [{o['status']}]"
            for o in overruns
        ]
        post_alert("💰 Budget watch", lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run(path=path)
