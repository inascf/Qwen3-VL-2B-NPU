#!/bin/bash
# VLM_server startup script
# Place this file next to the VLM_server binary (project root after cmake build).

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/VLM_server"
VLM_MODEL="${SCRIPT_DIR}/models/qwen3-vl-2b-vision_rk3588.rknn"
LLM_MODEL="${SCRIPT_DIR}/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"

# ── Server settings (override via environment variables) ─────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-2b}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CONTEXT="${CONTEXT:-4096}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "[error] VLM_server binary not found: $BINARY"
    echo "        Build it first:"
    echo "          mkdir build && cd build && cmake .. && make -j4"
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

# ── Start ─────────────────────────────────────────────────────────────────────
echo "Starting VLM_server..."
echo "  VLM model : $VLM_MODEL"
echo "  LLM model : $LLM_MODEL"
echo "  Address   : http://${HOST}:${PORT}"
echo "  Model name: $MODEL_NAME"
echo "  Max tokens: $MAX_TOKENS  Context: $CONTEXT"
echo ""

exec "$BINARY" \
    "$VLM_MODEL" \
    "$LLM_MODEL" \
    --host       "$HOST" \
    --port       "$PORT" \
    --model-name "$MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --context    "$CONTEXT"
