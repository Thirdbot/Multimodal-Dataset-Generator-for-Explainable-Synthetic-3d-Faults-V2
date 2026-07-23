#!/usr/bin/env bash
# Launch the local Qwen server for QA generation.
# Must ACTIVATE the venv (not just call .venv/bin/python3): sglang shells out to `ninja` to
# build kernels, and ninja only lives in .venv/bin -- without activation it is not on PATH and
# the server dies with FileNotFoundError: 'ninja'.
set -eu
cd /home/third/Desktop/sglang
# shellcheck disable=SC1091
source .venv/bin/activate
exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --context-length 2048 \
  --mem-fraction-static 0.7 \
  --port 8000
