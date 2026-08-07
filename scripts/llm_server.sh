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
# Model is configurable two ways -- they must AGREE with the pipeline client, which also reads
# LLM_MODEL (vllm rejects a mismatched name):
#   export LLM_MODEL=<hf/model>                  # server AND pipeline (run_all workers inherit it) [canonical]
#   bash scripts/llm_server.sh vllm <hf/model>   # server only (quick test)
# Knobs (env): LLM_MODEL  LLM_PORT  LLM_CONTEXT  LLM_MEM_FRAC  LLM_VENV  LLM_QUANT  LLM_EXTRA_ARGS
# ─────────────────────────────────────────────────────────────────────────────
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root = parent of scripts/, auto-detected
cd "$ROOT"
SERVE_VENV="${LLM_VENV:-$ROOT/.venv-serve}"
PY="$SERVE_VENV/bin/python"
MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${LLM_PORT:-8000}"
CONTEXT="${LLM_CONTEXT:-4096}"                  # ctx 4096: richer evidence overflowed 2048
MEM_FRAC="${LLM_MEM_FRAC:-0.5}"                 # sglang SHARES the card with the QA workers' NLI+embeddings
                                               # (NLI_DEVICE=cuda) -- ~0.9 GB each, so ~7 GB for 8 workers.
                                               # 0.5 of 25 GB = ~12.5 GB sglang leaves ~12 GB headroom.
                                               # Raise toward 0.65 ONLY if nvidia-smi shows spare VRAM;
                                               # >0.7 with 8 GPU workers will OOM.
QUANT="${LLM_QUANT:-}"                          # force/override quantization: fp8, gptq, awq, bitsandbytes, ...
                                               # pre-quantized repos auto-detect (leave empty); set this to ONLINE-quantize
                                               # a base fp16 model, e.g. LLM_QUANT=fp8 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"                # any extra backend flags, passed through verbatim (e.g. "--max-num-seqs 64")

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
    MODEL="${2:-$MODEL}"                          # optional cmd arg: llm_server.sh sglang <model>
    "$PY" -c "import sglang" 2>/dev/null || { echo "sglang not set up -> bash scripts/llm_server.sh setup sglang"; exit 1; }
    echo "serving $MODEL via sglang on :$PORT (ctx $CONTEXT, mem-frac $MEM_FRAC)"
    exec env PATH="$SERVE_VENV/bin:$PATH" "$PY" -m sglang.launch_server \
      --model-path "$MODEL" --host 0.0.0.0 --port "$PORT" \
      --context-length "$CONTEXT" --mem-fraction-static "$MEM_FRAC" \
      ${QUANT:+--quantization $QUANT} $EXTRA_ARGS ;;
  vllm)
    MODEL="${2:-$MODEL}"                          # optional cmd arg: llm_server.sh vllm <model>
    "$PY" -c "import vllm" 2>/dev/null || { echo "vllm not set up -> bash scripts/llm_server.sh setup vllm"; exit 1; }
    echo "serving $MODEL via vllm on :$PORT (ctx $CONTEXT, gpu-util $MEM_FRAC)"
    exec "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
      --max-model-len "$CONTEXT" --gpu-memory-utilization "$MEM_FRAC" \
      ${QUANT:+--quantization $QUANT} $EXTRA_ARGS ;;
  *) echo "usage: llm_server.sh {setup {sglang|vllm} | sglang | vllm}"; exit 1 ;;
esac
