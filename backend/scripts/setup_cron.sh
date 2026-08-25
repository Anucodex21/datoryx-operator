#!/usr/bin/env bash
# One-time setup: installs a cron entry that runs nightly_train.sh every
# night, fully hands-off from then on. Safe to re-run (won't duplicate the
# entry).
#
# Usage:
#   bash scripts/setup_cron.sh                 # runs nightly at 2:30 AM local time
#   bash scripts/setup_cron.sh "0 4 * * *"      # custom cron schedule (here: 4:00 AM)
#
# Check it's installed:    crontab -l
# See last run's activity: tail -f backend/models/native/checkpoint/nightly_train.log
# Remove it later:         crontab -e   (delete the datoryx-nightly-train line)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIGHTLY_SCRIPT="$SCRIPT_DIR/nightly_train.sh"
SCHEDULE="${1:-30 2 * * *}"
MARKER="# datoryx-nightly-train"

chmod +x "$NIGHTLY_SCRIPT"

if ! command -v crontab >/dev/null 2>&1; then
    echo "crontab isn't available on this system."
    echo "macOS: cron works out of the box but needs Full Disk Access granted to"
    echo "  /usr/sbin/cron in System Settings > Privacy & Security."
    echo "Alternative: use launchd (macOS) or Task Scheduler (Windows, see setup_cron.ps1"
    echo "  equivalent - ask if you want that generated) instead."
    exit 1
fi

EXISTING=$(crontab -l 2>/dev/null || true)
if echo "$EXISTING" | grep -qF "$MARKER"; then
    echo "Nightly training cron entry already installed. Current crontab:"
    crontab -l | grep -A1 -F "$MARKER"
    exit 0
fi

NEW_LINE="$SCHEDULE bash \"$NIGHTLY_SCRIPT\" >> \"$SCRIPT_DIR/../models/native/checkpoint/nightly_train.log\" 2>&1 $MARKER"

{ echo "$EXISTING"; echo "$MARKER"; echo "$NEW_LINE"; } | crontab -

echo "Installed. DAXRO 1.0 will now train automatically on this schedule: $SCHEDULE"
echo "(cron format: minute hour day month weekday — default is 2:30 AM every night)"
echo
echo "This machine must be ON and awake at that time for cron to fire — if it's a"
echo "laptop that sleeps overnight, either change the schedule to a time it's usually"
echo "on, or run the equivalent on a server/Render cron job instead (see DEPLOY.md)."
echo
echo "Verify:  crontab -l"
echo "Watch:   tail -f backend/models/native/checkpoint/nightly_train.log"
