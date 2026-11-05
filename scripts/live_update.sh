#!/usr/bin/env bash
# Election-night loop: every INTERVAL seconds fetch the IEC per-VD file; when there is a new
# snapshot, re-run the projection (method from config/live.json) and re-export the ward-map
# data. Optionally copy the export into a web app's public/data (APP_DATA=...).
#   bash scripts/live_update.sh            # 600 s interval
#   INTERVAL=300 APP_DATA=../SAelection/public/data bash scripts/live_update.sh
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${INTERVAL:-600}"
[ -f .venv/bin/activate ] && . .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
while true; do
  out=$(python scripts/fetch_iec_2026.py 2>&1); echo "$out"
  if echo "$out" | grep -q "NEW snapshot" || [ ! -f output/webapp/predictions.json ]; then
    if python scripts/run_live.py >/tmp/live_run.log 2>&1 && python scripts/export_webapp.py; then
      grep -E "% of VDs counted|^(DA|ANC|ACTIONSA) " /tmp/live_run.log
      if [ -n "${APP_DATA:-}" ]; then cp output/webapp/*.json output/webapp/*.geojson "$APP_DATA"/ && echo "copied to $APP_DATA"; fi
    else
      echo "$(date +%H:%M:%S) projection FAILED, see /tmp/live_run.log"; tail -5 /tmp/live_run.log
    fi
  fi
  sleep "$INTERVAL"
done
