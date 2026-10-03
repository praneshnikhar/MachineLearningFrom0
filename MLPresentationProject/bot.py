import re
from datetime import date, datetime, timedelta

import pandas as pd

from config import DATA_DIR, TELEGRAM_BOT_TOKEN
from src.item_classifier import load, resolve_category

LOG_PATH = DATA_DIR / "expenses_log.csv"
BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

AMOUNT_RE = re.compile(r"(\d+(?:[.,]\d{1,2})?)")


def parse_expense(text):
    """Extract (item, amount) from free text like 'pen 20', 'pen ₹20', '20 pen'."""
    m = AMOUNT_RE.search(text)
    if not m:
        return None, None
    amount = float(m.group(1).replace(",", "."))
    item = AMOUNT_RE.sub(" ", text)
    item = re.sub(r"[₹$]|rs\.?|rupees?|inr", " ", item, flags=re.I)
    item = re.sub(r"[^\w\s]", " ", item)
    item = " ".join(item.split())
    return item, amount


def classify_item(item):
    override = resolve_category(item)
    if override:
        return override
    model = load()
    return model.predict([item.lower()])[0]


def log_expense(item, amount, category, when=None):
    row = {
        "date": (when or date.today()).isoformat(),
        "item": item,
        "amount": amount,
        "category": category,
    }
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)
    return row


def _summarize(since):
    if not LOG_PATH.exists():
        return "No expenses logged yet."
    df = pd.read_csv(LOG_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= since]
    if df.empty:
        return "No expenses in that period."
    total = df["amount"].sum()
    lines = [f"Total: ₹{total:,.0f} ({len(df)} items)"]
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    for cat, val in by_cat.items():
        lines.append(f"  {cat}: ₹{val:,.0f}")
    return "\n".join(lines)


def handle_command(text):
    cmd = text.split()[0].lower().strip("/")
    today = date.today()
    if cmd == "today":
        return _summarize(pd.Timestamp(today))
    if cmd == "week":
        return _summarize(pd.Timestamp(today - timedelta(days=7)))
    if cmd == "month":
        return _summarize(pd.Timestamp(today.replace(day=1)))
    if cmd in ("total", "summary"):
        return _summarize(pd.Timestamp("1970-01-01"))
    if cmd == "categories":
        from src.features import CATEGORIES

        return "Categories:\n" + "\n".join(f"  • {c}" for c in CATEGORIES)
    return (
        "Commands:\n"
        "/today — spend today\n"
        "/week — last 7 days\n"
        "/month — this month\n"
        "/total — all time\n"
        "/categories — list categories\n\n"
        "Or just type: pen 20  (or: pizza 250, uber 120, pen 20 stationery)"
    )


def handle(text):
    if text.startswith("/"):
        return handle_command(text)
    item, amount = parse_expense(text)
    if amount is None:
        return "How much? Try: pen 20"
    if not item:
        return "What did you buy? Try: pen 20"
    category = classify_item(item)
    log_expense(item, amount, category)
    return f"✅ Added ₹{amount:g} — {category} ({item})"


def send_message(chat_id, text):
    import requests

    requests.post(f"{BASE}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=10)


def poll():
    import requests

    offset = 0
    print(f"Telegram bot polling (token {'set' if TELEGRAM_BOT_TOKEN else 'MISSING'})...")
    while True:
        try:
            r = requests.get(f"{BASE}/getUpdates", params={"timeout": 30, "offset": offset}, timeout=40)
            for u in r.json().get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message")
                if not msg or not msg.get("text"):
                    continue
                reply = handle(msg["text"])
                send_message(msg["chat"]["id"], reply)
        except requests.RequestException as e:
            print(f"[telegram] {e}")


if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env first (get one from @BotFather).")
        raise SystemExit(1)
    poll()
