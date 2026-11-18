#!/bin/bash
# Serve Qwen3.8-27B on one DGX Spark with TensorFold native 1.2.9.
#   ./start.sh            install what is missing, then serve (background)
#   ./start.sh restart    stop, then serve
#   FOREGROUND=1 ./start.sh   serve in this terminal
set -euo pipefail
cd "$(dirname "$0")"
. scripts/config.sh
log() { printf '[start.sh] %s\n' "$*"; }
die() { printf '[start.sh] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == restart ]]; then ./stop.sh; shift; fi   # extra arguments go to tensorfold serve

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "already running (pid $(cat "$PIDFILE")) on port $PORT; ./start.sh restart to apply new settings"; exit 0
fi

# 1. engine: the release archive, checked against its published sha256
if [[ ! -x "$TF_BIN" ]]; then
  log "installing TensorFold $TF_VERSION into $TF_DIR"
  mkdir -p "$TF_DIR"; cd "$TF_DIR"
  A=tensorfold-$TF_VERSION-linux-aarch64.tar.gz
  U=https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION
  curl -fsSLO "$U/$A"; curl -fsSLO "$U/$A.sha256"
  sha256sum -c "$A.sha256" || die "checksum mismatch for $A"
  tar -xzf "$A"; cd - >/dev/null
fi
"$TF_BIN" --version | grep -q "$TF_VERSION" || die "$TF_BIN is not version $TF_VERSION"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk -F. '$1<580{exit 1}' \
  || die "NVIDIA driver R580 or newer is required (CUDA 13)"

# 2. weights, at their pinned revisions (a complete cached snapshot is reused)
snap() { echo "$HF_HUB_CACHE/models--${1//\//--}/snapshots/$2"; }
for pair in "$MODEL_REPO@$MODEL_REV" "$DRAFT_REPO@$DRAFT_REV"; do
  repo=${pair%@*}; rev=${pair#*@}
  [[ -f "$(snap "$repo" "$rev")/config.json" ]] || { log "downloading $pair"; "$TF_BIN" pull "$pair"; }
done
MODEL_DIR=$(snap "$MODEL_REPO" "$MODEL_REV"); DRAFT_DIR=$(snap "$DRAFT_REPO" "$DRAFT_REV")
[[ -f "$MODEL_DIR/config.json" && -f "$DRAFT_DIR/config.json" ]] || die "weights missing under $HF_HUB_CACHE"

# 3. serve
ARGS=(serve "$MODEL_DIR" --drafter "$DRAFT_DIR" --name "$NAME" --host "$HOST" --port "$PORT"
      --context "$CONTEXT" --kv-cache "$KV_CACHE" --no-update-check)
[[ "$THINKING" == 1 ]] && ARGS+=(--thinking) || ARGS+=(--no-thinking)
[[ -n "$PARALLEL" ]] && ARGS+=(--parallel "$PARALLEL")
[[ -n "$API_KEY_FILE" ]] && ARGS+=(--api-key-file "$API_KEY_FILE")
ARGS+=("$@")
log "tensorfold-native ${ARGS[*]}"
log "TF_QWEN_REQUEST_STARTS=$TF_QWEN_REQUEST_STARTS"
if [[ "${FOREGROUND:-0}" == 1 ]]; then exec "$TF_BIN" "${ARGS[@]}"; fi
nohup "$TF_BIN" "${ARGS[@]}" > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

# 4. wait for the model, then one request
for i in $(seq 1 200); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$NAME" && break
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -30 "$LOGFILE"; die "the server exited; log: $LOGFILE"; }
  sleep 3
done
curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q "$NAME" || die "not ready after 600 s; log: $LOGFILE"
grep -E "streams at once|cut at replies|serving" "$LOGFILE" | sed 's/^/  /' | cut -c1-220
H=(); [[ -n "$API_KEY_FILE" ]] && H=(-H "Authorization: Bearer $(cat "$API_KEY_FILE")")
r=$(curl -sf -m 60 "${H[@]}" "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word ready.\"}],\"max_tokens\":8}") \
  || die "smoke test failed; log: $LOGFILE"
log "ready: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1 (model $NAME); log $LOGFILE"
