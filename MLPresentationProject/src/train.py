import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_DIR, MODEL_DIR, budgets
from src.features import build_features


def build_model():
    return Pipeline(
        [
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]
    )


def category_stats(df):
    stats = {}
    for cat, grp in df.groupby("category"):
        stats[cat] = {
            "mean": float(grp["amount"].mean()),
            "std": float(grp["amount"].std()),
            "p99": float(grp["amount"].quantile(0.99)),
        }
    return stats


def main():
    train_path = DATA_DIR / "transactions_train.csv"
    df = build_features(pd.read_csv(train_path))
    df = df[df["category"].notna()]

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = MODEL_DIR / "classifier.joblib"
    joblib.dump(model, model_path)

    stats = category_stats(df)
    stats_path = MODEL_DIR / "category_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    budgets_path = MODEL_DIR / "budgets.json"
    budgets_path.write_text(json.dumps(budgets(), indent=2))

    metrics = {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "classes": list(report.keys()),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model -> {model_path}")
    print(f"Saved category stats -> {stats_path}")
    print(f"Saved budgets -> {budgets_path}")


if __name__ == "__main__":
    main()
