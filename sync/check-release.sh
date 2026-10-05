#!/bin/bash
# Release watcher: polls GitHub for the latest earendil-works/pi release and, on a new
# tag, launches an omp headless session that applies the release to this repo.
#
# State machine (sync/.release-state):
#   no state file      -> record baseline tag, exit (never syncs on first run)
#   tag == state       -> nothing to do
#   tag != state       -> lock; run omp sync; advance state only on success
#
# Env:
#   PI_RS_SYNC_DRY_RUN=1  log the omp command instead of running it
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/sync/.release-state"
LOCK="$ROOT/sync/sync.lock"
LOG="$ROOT/sync/sync.log"
API="https://api.github.com/repos/earendil-works/pi/releases/latest"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

TAG=$(curl -fsSL --max-time 30 -H "Accept: application/vnd.github+json" "$API" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])') || { log "ERROR: failed to query releases API"; exit 1; }

if [ ! -f "$STATE" ]; then
  printf '%s\n' "$TAG" > "$STATE"
  log "first run: recorded baseline $TAG"
  exit 0
fi

PREV=$(cat "$STATE")
[ "$TAG" = "$PREV" ] && exit 0

# Atomic mutex via mkdir (portable on macOS; no flock dependency).
if ! mkdir "$LOCK" 2>/dev/null; then
  log "sync already in progress; skipping $TAG (will retry next interval)"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

log "new upstream release detected: $PREV -> $TAG"

OMP_ARGS=(omp -p --cwd "$ROOT" --approval-mode yolo --max-time 2h
  "@$ROOT/sync/SYNC_PROMPT.md" "Apply upstream release $TAG (previous synced: $PREV).")

if [ "${PI_RS_SYNC_DRY_RUN:-0}" = "1" ]; then
  log "DRY RUN would execute: ${OMP_ARGS[*]}"
  exit 0
fi

if "${OMP_ARGS[@]}" >> "$LOG" 2>&1; then
  printf '%s\n' "$TAG" > "$STATE"
  log "sync of $TAG complete; state advanced"
else
  # State stays at $PREV so the next interval retries this release.
  log "ERROR: omp sync of $TAG failed; state NOT advanced (will retry)"
  exit 1
fi
