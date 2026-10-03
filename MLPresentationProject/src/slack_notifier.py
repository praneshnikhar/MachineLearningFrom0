import json

import requests

from config import SLACK_WEBHOOK_URL


def send_slack(payload):
    if not SLACK_WEBHOOK_URL:
        print("[slack] no webhook configured — printing message instead:\n")
        print(_render(payload))
        return False
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[slack] send failed: {e}")
        return False


def _render(payload):
    if "blocks" in payload:
        lines = []
        for b in payload["blocks"]:
            t = b.get("text") or b.get("fields") or []
            if isinstance(t, dict):
                lines.append(t.get("text", ""))
            elif isinstance(t, list):
                lines.extend(f.get("text", "") for f in t)
        return "\n".join(l for l in lines if l)
    return payload.get("text", "")


def post_summary(title, sections):
    blocks = [{"type": "header", "text": {"type": "plain_text", "text": title}}]
    for heading, body in sections:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{heading}*\n{body}"}})
    return send_slack({"blocks": blocks})


def post_alert(title, items):
    if not items:
        return False
    blocks = [{"type": "header", "text": {"type": "plain_text", "text": title}}]
    for it in items:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"{it}"}})
    return send_slack({"blocks": blocks})
