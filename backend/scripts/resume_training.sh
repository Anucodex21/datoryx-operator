#!/usr/bin/env bash
# Undoes scripts/pause_training.sh - removes the PAUSED marker so the next
# scheduled cron run trains normally again.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/../models/native/checkpoint"

if [ -f "$CHECKPOINT_DIR/PAUSED" ]; then
    rm "$CHECKPOINT_DIR/PAUSED"
    echo "Nightly training resumed - it'll run normally starting with the next scheduled time."
else
    echo "Nightly training wasn't paused - nothing to do."
fi
