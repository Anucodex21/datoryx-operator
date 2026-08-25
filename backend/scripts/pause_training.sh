#!/usr/bin/env bash
# Pauses nightly training without touching the cron job itself - the cron
# entry stays installed and fires every night as scheduled, but
# nightly_train.sh sees this PAUSED marker and skips the run (just logs
# that it skipped). Whatever step count DAXRO 1.0 was already trained to
# is untouched and safe.
#
# Undo with: bash scripts/resume_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/../models/native/checkpoint"
mkdir -p "$CHECKPOINT_DIR"
touch "$CHECKPOINT_DIR/PAUSED"
echo "Nightly training paused. The cron schedule is still installed and will fire,"
echo "but nightly_train.sh will skip actual training and just log that it's paused."
echo "Resume any time with: bash scripts/resume_training.sh"
