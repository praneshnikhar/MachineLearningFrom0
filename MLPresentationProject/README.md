# Expense Classifier + Budget Watchdog

An ML project that answers a real daily-life problem: **you don't know where your money
goes, and you miss unusual or over-budget spending until it's too late.**

It automates the whole loop — ingest transactions → classify them → flag anomalies →
send a weekly digest and alerts to Slack. Built with `scikit-learn`, `pandas`, `FastAPI`,
and wired up with cron / n8n / Slack webhooks.

## Why ML is the right tool here

- **Classification** turns a raw transaction string (`"SWIGGY ORDER-1234"`) into a category
  (`Food & Dining`). Hand-writing rules for every merchant doesn't scale — the model learns
  the pattern from examples.
- **Anomaly detection** flags transactions that are unusually large for their category
  (a ₹12,000 "Groceries" charge is suspicious).
- **Budget watchdog** compares month-to-date spend against a monthly budget per category
  and raises a warning before you blow the budget.

This is the exact same "classification + anomaly detection" machinery behind Gmail spam
filtering and bank fraud detection — applied to personal finance.

## How it works

```
transactions_new.csv
        │
        ▼
  build_features()        text = merchant + description (lowercased)
        │                 amount, day_of_week
        ▼
  classifier.joblib       TF-IDF (text) → LogisticRegression
        │
        ▼
  predicted category ──► flag_outliers()   (amount > mean + k·std AND > 99th pct)
        │                 budget_overruns() (month-to-date vs monthly budget)
        ▼
  summary.json + Slack digest + alerts
```

### Model details
- **Features**: TF-IDF n-grams (1–2) over the merchant + description text. Amount is
  deliberately *not* a classification feature — it's used only for anomaly detection, so
  an unusually large amount can't mislead the category.
- **Model**: multinomial `LogisticRegression` (balanced class weights).
- **Anomaly**: per-category mean/σ/99th-percentile from training data.
- **Budget**: configurable monthly budget per category (see `.env.example`).

## Project layout

```
├── config.py               # paths, Slack webhook, Telegram token, budgets
├── app.py                  # FastAPI: /predict, /predict_batch, /digest
├── bot.py                  # Telegram bot: chat → classify → log expense
├── demo.py                 # one-command end-to-end run
├── src/
│   ├── generate_data.py    # synthetic transactions (train + new)
│   ├── features.py         # shared feature engineering
│   ├── train.py            # train + evaluate + save model/stats
│   ├── item_classifier.py  # free-text item → category model (for the bot)
│   ├── anomaly.py          # outlier + budget logic
│   ├── pipeline.py         # ingest → classify → alert → digest
│   └── slack_notifier.py   # Slack webhook posting
├── automation/
│   ├── run_pipeline.sh     # shell wrapper for cron
│   └── n8n_workflow.json   # example n8n schedule → run → Slack
└── data/ models/ output/   # generated artifacts
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate data, 2. train, 3. run pipeline (all in one)
python demo.py
```

Or step by step:

```bash
python -m src.generate_data     # data/transactions_train.csv + transactions_new.csv
python -m src.train             # models/classifier.joblib + category_stats.json
python -m src.pipeline          # output/latest.csv + summary.json + Slack
```

## Telegram bot (log expenses by chat)

Chat to the bot and it classifies + logs each expense automatically:

```
you: pen 20            ->  ✅ Added ₹20 — Stationery (pen)
you: pizza 250         ->  ✅ Added ₹250 — Food & Dining (pizza)
you: netflix 199       ->  ✅ Added ₹199 — Subscriptions (netflix)
you: pen 20 stationery ->  ✅ Added ₹20 — Stationery (pen stationery)   # optional category override
you: /today            ->  Total: ₹X (n items) ...
you: /month            ->  month-to-date breakdown by category
```

Setup:
1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, copy the token.
2. `cp .env.example .env` and set `TELEGRAM_BOT_TOKEN=...`.
3. Train the item classifier once: `python -m src.item_classifier`.
4. Run the bot: `python bot.py`, then message it.

The bot stores every entry in `data/expenses_log.csv` (date, item, amount, category),
which you can feed back into the main pipeline for budgeting.

## Enable Slack alerts

1. Create an incoming webhook at <https://api.slack.com/messaging/webhooks>.
2. `cp .env.example .env` and set `SLACK_WEBHOOK_URL=...`.
3. Re-run `python -m src.pipeline`. Without a webhook it prints the digest to the console.

## Automate it

**cron** (weekly, Sunday 9am):
```
0 9 * * 0  cd /path/to/MLPresentationProject && bash automation/run_pipeline.sh
```

**n8n**: import `automation/n8n_workflow.json` (Schedule → Execute Command → Slack).

**FastAPI** (serve the model):
```bash
uvicorn app:app --reload
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"merchant":"Swiggy","description":"ORDER-88","amount":450,"date":"2026-08-16"}'
```

## Results (example)

```
Accuracy: 0.97
Total: ₹84,320 across 250 transactions
Food & Dining: ₹18,450   Groceries: ₹16,100   Shopping: ₹13,800 ...
⚠️ Unusual spending: BigBasket — ₹7,500 (Groceries)
💰 Budget watch: Entertainment — ₹4,200 / ₹2,000 (210%) [OVER]
```
