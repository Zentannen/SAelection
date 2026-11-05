#!/usr/bin/env bash
# One refresh cycle during the count: pull the IEC feed, re-project, export for the web app.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m saelection.iec_live
python scripts/run_live.py
python -m saelection.frontend
