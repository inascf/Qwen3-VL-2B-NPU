#!/bin/bash
# VLM_server startup script — runs in background, logs to server.log
# Usage:
#   ./start_server.sh          start
#   ./start_server.sh stop     stop
#   ./start_server.sh status   check if running
#   ./start_server.sh log      tail the log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/VLM_server"
VLM_MODEL="${SCRIPT_DIR}/models/qwen3-vl-2b-vision_rk3588.rknn"
LLM_MODEL="${SCRIPT_DIR}/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"
PID_FILE="${SCRIPT_DIR}/server.pid"
LOG_FILE="${SCRIPT_DIR}/server.log"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-2b}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CONTEXT="${CONTEXT:-4096}"

# ── Helper: check if server is running ───────────────────────────────────────
is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# ── stop ─────────────────────────────────────────────────────────────────────
cmd_stop() {
    if ! is_running; then
        echo "[info] VLM_server is not running."
        return
    fi
    PID=$(cat "$PID_FILE")
    kill "$PID"
    rm -f "$PID_FILE"
    echo "[info] VLM_server stopped (pid $PID)."
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    if is_running; then
        echo "[info] VLM_server is running (pid $(cat "$PID_FILE"))."
        echo "       http://${HOST}:${PORT}"
    else
        echo "[info] VLM_server is not running."
    fi
}

# ── log ──────────────────────────────────────────────────────────────────────
cmd_log() {
    [ -f "$LOG_FILE" ] && tail -f "$LOG_FILE" || echo "[info] No log file yet."
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
    if is_running; then
        echo "[info] VLM_server is already running (pid $(cat "$PID_FILE"))."
        exit 0
    fi

    if [ ! -f "$BINARY" ]; then
        echo "[error] VLM_server binary not found: $BINARY"
        echo "        Build it first: mkdir build && cd build && cmake .. && make -j4"
        exit 1
    fi
    if [ ! -f "$VLM_MODEL" ]; then
        echo "[error] VLM model not found: $VLM_MODEL"
        echo "        See models/README.md for download instructions."
        exit 1
    fi
    if [ ! -f "$LLM_MODEL" ]; then
        echo "[error] LLM model not found: $LLM_MODEL"
        echo "        See models/README.md for download instructions."
        exit 1
    fi

    echo "[info] Starting VLM_server in background..."
    echo "[info]   Address   : http://${HOST}:${PORT}"
    echo "[info]   Model name: ${MODEL_NAME}"
    echo "[info]   Log       : ${LOG_FILE}"

    nohup "$BINARY" \
        "$VLM_MODEL" \
        "$LLM_MODEL" \
        --host       "$HOST" \
        --port       "$PORT" \
        --model-name "$MODEL_NAME" \
        --max-tokens "$MAX_TOKENS" \
        --context    "$CONTEXT" \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "[info] VLM_server started (pid $(cat "$PID_FILE"))."
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${1:-start}" in
    start)  cmd_start  ;;
    stop)   cmd_stop   ;;
    status) cmd_status ;;
    log)    cmd_log    ;;
    restart)
        cmd_stop
        sleep 1
        cmd_start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
