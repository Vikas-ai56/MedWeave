#!/bin/zsh
# Serve the MedWeave dashboard at http://127.0.0.1:8080 (override with PORT=...).
# It reads the coordinator's database, so start it before or during
# coordinator/flower_app/run_local.sh. Builds the frontend on first use.
set -e
HERE=${0:A:h}
REPO=${HERE:h}
export PYTHONPATH="$REPO"
export MEDWEAVE_STATE_DIR="${MEDWEAVE_STATE_DIR:-$REPO/.medweave_local}"   # same default as run_local.sh

if [[ ! -d "$HERE/frontend/dist" || "$1" == "--rebuild" ]]; then
  echo "Building the dashboard frontend..."
  (cd "$HERE/frontend" && npm install --no-audit --no-fund && npm run build)
fi

echo "MedWeave dashboard: http://127.0.0.1:${PORT:-8080}"
exec "$REPO/venv/bin/uvicorn" dashboard.backend.app:app --host 127.0.0.1 --port "${PORT:-8080}" --log-level warning
