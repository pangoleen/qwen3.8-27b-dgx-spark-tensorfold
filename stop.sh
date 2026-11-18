#!/bin/bash
# Stop the server started by start.sh and free its memory.
cd "$(dirname "$0")"
. scripts/config.sh
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  for i in $(seq 1 30); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null && kill -9 "$(cat "$PIDFILE")"
  echo "[stop.sh] stopped"
else
  echo "[stop.sh] not running"
fi
rm -f "$PIDFILE"
