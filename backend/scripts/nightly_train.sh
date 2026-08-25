#!/usr/bin/env bash
# Nightly DAXRO 1.0 training wrapper — meant to be run by cron, not by hand.
#
# Reads the current step count from the shipped checkpoint's train_log.json,
# adds STEP_INCREMENT more steps as tonight's target, and calls auto_train.sh
# toward that new target. Every run is additive and resumable — if a run gets
# killed halfway, tomorrow night's run just picks up from wherever training
# actually got to, not from the target that was set (checkpoint state on
# disk is always the source of truth, never this script's own logic).
#
# Usage (see scripts/setup_cron.sh to install this on a schedule):
#   bash scripts/nightly_train.sh                    # +4000 steps tonight
#   bash scripts/nightly_train.sh 8000                # +8000 steps tonight
#   NIGHTLY_MAX_STEPS=200000 bash scripts/nightly_train.sh   # stop auto-growing past this total

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$BACKEND_DIR/models/native/checkpoint"
LOG_FILE="$CHECKPOINT_DIR/nightly_train.log"
STEP_INCREMENT="${1:-4000}"
MAX_STEPS="${NIGHTLY_MAX_STEPS:-100000}"

mkdir -p "$CHECKPOINT_DIR"

if [ -f "$CHECKPOINT_DIR/PAUSED" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] PAUSED file present - skipping tonight's training (run scripts/resume_training.sh to re-enable)" >> "$LOG_FILE"
    exit 0
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] nightly_train.sh starting, +$STEP_INCREMENT steps requested" >> "$LOG_FILE"

CURRENT_STEP=$(python3 - "$CHECKPOINT_DIR/train_log.json" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        log = json.load(f)
    print(log[-1]["step"] if log else 0)
except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError):
    print(0)
PYEOF
)

TARGET_STEP=$((CURRENT_STEP + STEP_INCREMENT))

if [ "$TARGET_STEP" -gt "$MAX_STEPS" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] target $TARGET_STEP exceeds NIGHTLY_MAX_STEPS=$MAX_STEPS, capping there and stopping future growth" >> "$LOG_FILE"
    TARGET_STEP="$MAX_STEPS"
    if [ "$CURRENT_STEP" -ge "$MAX_STEPS" ]; then
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] already at max, nothing to do tonight" >> "$LOG_FILE"
        exit 0
    fi
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] current step=$CURRENT_STEP, tonight's target=$TARGET_STEP" >> "$LOG_FILE"

cd "$BACKEND_DIR" || exit 1
bash "$SCRIPT_DIR/auto_train.sh" "$TARGET_STEP" >> "$LOG_FILE" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] nightly run finished OK" >> "$LOG_FILE"
else
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] nightly run exited with status $STATUS - will resume next scheduled run" >> "$LOG_FILE"
fi
