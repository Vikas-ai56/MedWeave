#!/bin/zsh
# Run a full MedWeave session on one machine: 1 SuperLink (coordinator) and
# 4 SuperNodes (institutions). Extra arguments go to `flwr run`, e.g.
#   ./run_local.sh --run-config "num-rounds=2"
# State (checkpoints) goes to $MEDWEAVE_STATE_DIR, logs to $MEDWEAVE_STATE_DIR/logs.
set -e
HERE=${0:A:h}
REPO=${HERE:h:h}
BIN="$REPO/venv/bin"
export PATH="$BIN:$PATH" PYTHONPATH="$REPO"
export MEDWEAVE_STATE_DIR="${MEDWEAVE_STATE_DIR:-$REPO/.medweave_local}"
export FLWR_HOME="$MEDWEAVE_STATE_DIR/flwr_home"   # keeps Flower's config out of ~/.flwr
LOGS="$MEDWEAVE_STATE_DIR/logs"
mkdir -p "$LOGS" "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" <<'TOML'
[superlink]
default = "local-deployment"

[superlink.local-deployment]
address = "127.0.0.1:8000"
insecure = true
TOML

trap 'pkill -f "$BIN/flower-" || true' EXIT
"$BIN/flower-superlink" --insecure > "$LOGS/superlink.log" 2>&1 &
sleep 4
for i in 1 2 3 4; do
  "$BIN/flower-supernode" --insecure --superlink 127.0.0.1:8000 --port $((9093 + i)) \
    --node-config "institution-id=\"inst-$i\"" > "$LOGS/inst-$i.log" 2>&1 &
done
sleep 4
cd "$HERE" && "$BIN/flwr" run . local-deployment --stream "$@"
