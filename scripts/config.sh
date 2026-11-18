#!/bin/bash
# Every setting, with the value measured on one DGX Spark (GB10) on 2026-11-18.
# Override any of them from the environment or a .env file next to start.sh.
[[ -f "$(dirname "${BASH_SOURCE[0]}")/../.env" ]] && set -a && . "$(dirname "${BASH_SOURCE[0]}")/../.env" && set +a

# Engine: TensorFold's native (Zig) release. CUDA kernels for sm_121 are built in.
TF_VERSION="${TF_VERSION:-1.2.9}"
TF_DIR="${TF_DIR:-$HOME/tf-native}"                  # where the release archive is unpacked
TF_BIN="$TF_DIR/tensorfold-$TF_VERSION-linux-aarch64/bin/tensorfold-native"

# Weights, pinned. RadixArk's NVFP4 export (NVFP4 MLP and head, FP8 attention and DeltaNet)
# and incoai's DFlash2 drafter. Both measured; see README "What was tested".
MODEL_REPO="${MODEL_REPO:-RadixArk/Qwen3.8-27B-NVFP4}"
MODEL_REV="${MODEL_REV:-554ebba9b5f1b79dc11246341960360e6ef05ef4}"
DRAFT_REPO="${DRAFT_REPO:-incoai/Qwen3.8-27B-DFlash2}"
DRAFT_REV="${DRAFT_REV:-dedf8df68adfb1afeaf7b7480c0a0243108177b4}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# Serving
NAME="${NAME:-qwen3.8-27b}"                          # the model id clients send
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"
CONTEXT="${CONTEXT:-262144}"                         # the model's native window, per request
KV_CACHE="${KV_CACHE:-fp8}"                          # fp8: 8 streams at 262k; bf16: 4
THINKING="${THINKING:-0}"                            # 0: --no-thinking (as measured); 1: --thinking
PARALLEL="${PARALLEL:-}"                             # empty: auto (8 with fp8 KV at 262k)
API_KEY_FILE="${API_KEY_FILE:-}"                     # set to require a bearer key

# Prefix reuse at message starts instead of a 4,096-row grid: each agent turn resumes at its own
# history. The single largest gain for agent traffic (fan-out 88 s -> 68 s median).
export TF_QWEN_REQUEST_STARTS="${TF_QWEN_REQUEST_STARTS:-1}"
export TENSORFOLD_NO_UPDATE_CHECK=1

PIDFILE="${PIDFILE:-$TF_DIR/qwen38-27b.pid}"
LOGFILE="${LOGFILE:-$TF_DIR/qwen38-27b.log}"
