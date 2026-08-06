#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Serve the local generator-LLM (sglang OR vllm) from an ISOLATED venv.
#
# WHY isolated: sglang/vllm pin OLDER torch (<2.12) AND OLDER transformers (<5.9) than this
# project (torch>=2.12, transformers>=5.9), so they CANNOT co-install in the project's .venv.
# This script owns a separate `.venv-serve` — the project venv stays untouched and you never
# manage the serving env by hand.
#
#   bash scripts/llm_server.sh setup sglang     # one-time: create .venv-serve + install sglang
#   bash scripts/llm_server.sh sglang           # serve (foreground; leave it running)
#     ... or vllm:
#   bash scripts/llm_server.sh setup vllm  &&  bash scripts/llm_server.sh vllm
#
# Both serve the SAME OpenAI endpoint (http://localhost:8000/v1) the pipeline calls
# (Verifier/llm_machine.py), so the backend is interchangeable with no code change.
# The automation (run_all.sh) never starts a server — it only checks this endpoint is up.
#
# Knobs (env): LLM_MODEL  LLM_PORT  LLM_CONTEXT  LLM_MEM_FRAC  LLM_VENV
# ─────────────────────────────────────────────────────────────────────────────
set -eu
ROOT=/home/third/Desktop/simulationv2          # <-- repo path; edit if the machine differs
cd "$ROOT"
SERVE_VENV="${LLM_VENV:-$ROOT/.venv-serve}"
PY="$SERVE_VENV/bin/python"
MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${LLM_PORT:-8000}"
CONTEXT="${LLM_CONTEXT:-4096}"                  # ctx 4096: richer evidence overflowed 2048
MEM_FRAC="${LLM_MEM_FRAC:-0.4}"                 # VRAM fraction; 0.4 of 25 GB leaves room for NLI on GPU

setup(){
  local backend="${1:-}"
  [ -x "$PY" ] || { echo "creating isolated serve venv: $SERVE_VENV"; uv venv "$SERVE_VENV" --python 3.11; }
  case "$backend" in
    sglang) echo "installing sglang[all] into $SERVE_VENV (heavy; its own torch)"; uv pip install --python "$PY" "sglang[all]" ;;
    vllm)   echo "installing vllm into $SERVE_VENV (heavy; its own torch)";       uv pip install --python "$PY" "vllm" ;;
    *) echo "usage: llm_server.sh setup {sglang|vllm}"; exit 1 ;;
  esac
  echo "ready. serve with:  bash scripts/llm_server.sh $backend"
}

case "${1:-}" in
  setup) shift; setup "${1:-}" ;;
  sglang)
    "$PY" -c "import sglang" 2>/dev/null || { echo "sglang not set up -> bash scripts/llm_server.sh setup sglang"; exit 1; }
    echo "serving $MODEL via sglang on :$PORT (ctx $CONTEXT, mem-frac $MEM_FRAC)"
    exec env PATH="$SERVE_VENV/bin:$PATH" "$PY" -m sglang.launch_server \
      --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
      --context-length "$CONTEXT" --mem-fraction-static "$MEM_FRAC" ;;
  vllm)
    "$PY" -c "import vllm" 2>/dev/null || { echo "vllm not set up -> bash scripts/llm_server.sh setup vllm"; exit 1; }
    echo "serving $MODEL via vllm on :$PORT (ctx $CONTEXT, gpu-util $MEM_FRAC)"
    exec "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
      --max-model-len "$CONTEXT" --gpu-memory-utilization "$MEM_FRAC" ;;
  *) echo "usage: llm_server.sh {setup {sglang|vllm} | sglang | vllm}"; exit 1 ;;
esac
