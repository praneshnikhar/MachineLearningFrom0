#!/usr/bin/env bash
# Run the expense pipeline: ingest -> classify -> anomaly -> digest -> Slack.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
python -m src.pipeline "$@"
