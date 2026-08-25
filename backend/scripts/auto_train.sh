#!/usr/bin/env bash
# Automated, unattended training for DATORYX's native model.
#
# Runs training toward a target total step count, in chunks, and keeps
# going even if a chunk gets interrupted (laptop sleep, closed terminal,
# a crash) - just re-run this same script and it picks up exactly where
# it left off, because every chunk uses --resume against the real
# checkpoint on disk.
#
# Usage:
#   bash scripts/auto_train.sh                      # trains to 20,000 total steps
#   bash scripts/auto_train.sh 50000                 # trains to 50,000 total steps
#   bash scripts/auto_train.sh 50000 /path/to/notes   # + your own text folder
#
# Runs in the foreground by default (so you see progress). To let it run
# in the background and survive closing the terminal:
#   nohup bash scripts/auto_train.sh 50000 > /dev/null 2>&1 &
#   disown
# Check on it later with: tail -f models/native/checkpoint/auto_train.log

set -uo pipefail
cd "$(dirname "$0")/.."   # run from backend/, regardless of where this was invoked from

TARGET_STEPS="${1:-20000}"
EXTRA_TEXT_DIR="${2:-}"
CHECKPOINT_DIR="models/native/checkpoint"
LOG_FILE="$CHECKPOINT_DIR/auto_train.log"
CHUNK_STEPS=2000          # steps per chunk - small enough that a crash never loses much progress
MAX_RETRIES=5

mkdir -p "$CHECKPOINT_DIR"

get_current_step() {
  if [ -f "$CHECKPOINT_DIR/train_log.json" ]; then
    python3 -c "
import json
try:
    with open('$CHECKPOINT_DIR/train_log.json') as f:
        log = json.load(f)
    print(log[-1]['step'] if log else 0)
except Exception:
    print(0)
"
  else
    echo 0
  fi
}

echo "=== DATORYX native model - automated training ===" | tee -a "$LOG_FILE"
echo "Target: $TARGET_STEPS total steps | started: $(date)" | tee -a "$LOG_FILE"

EXTRA_ARGS=()
if [ -n "$EXTRA_TEXT_DIR" ]; then
  EXTRA_ARGS+=(--extra-text-dir "$EXTRA_TEXT_DIR")
  echo "Extra text source: $EXTRA_TEXT_DIR" | tee -a "$LOG_FILE"
fi

current=$(get_current_step)
echo "Currently at step $current" | tee -a "$LOG_FILE"

retries=0
while [ "$current" -lt "$TARGET_STEPS" ]; do
  remaining=$((TARGET_STEPS - current))
  this_chunk=$(( remaining < CHUNK_STEPS ? remaining : CHUNK_STEPS ))

  echo "--- $(date): training chunk of $this_chunk steps (currently at $current/$TARGET_STEPS) ---" | tee -a "$LOG_FILE"

  python3 -m models.native.train \
    --steps "$this_chunk" \
    --out-dir "$CHECKPOINT_DIR" \
    --resume \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_FILE" 2>&1
  exit_code=$?

  new_current=$(get_current_step)

  if [ "$exit_code" -ne 0 ] && [ "$new_current" -eq "$current" ]; then
    # Chunk failed AND made no progress at all (not even a partial
    # checkpoint save) - genuine failure, not just an interruption.
    retries=$((retries + 1))
    echo "Chunk failed with no progress (attempt $retries/$MAX_RETRIES)" | tee -a "$LOG_FILE"
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
      echo "Giving up after $MAX_RETRIES failed attempts with no progress. Check $LOG_FILE for the actual error." | tee -a "$LOG_FILE"
      exit 1
    fi
    sleep 5
  else
    retries=0   # made progress (even partial) - reset the failure counter
  fi

  current=$(get_current_step)
  echo "Now at step $current/$TARGET_STEPS" | tee -a "$LOG_FILE"
done

echo "=== Done: reached $current/$TARGET_STEPS steps at $(date) ===" | tee -a "$LOG_FILE"
