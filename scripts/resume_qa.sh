#!/usr/bin/env bash
# Resume QA generation from where it stopped. Appends to verified_qa.jsonl (does NOT truncate):
# _seed_processed_graphs() skips the ~383 graphs already turned into rows and runs only the rest.
set -eu
cd /home/third/Desktop/simulationv2
# 1) local Qwen server (needs the venv activated for ninja) -- see scripts/run_sglang.sh
systemctl --user reset-failed sglang-qwen 2>/dev/null || true
systemd-run --user --unit=sglang-qwen --property=Restart=on-failure --property=RestartSec=20 \
  /bin/bash "$PWD/scripts/run_sglang.sh"
sleep 40
# 2) QA worker -- NO DATASET_REGEN so it appends and resumes
systemctl --user reset-failed seismic-qa 2>/dev/null || true
systemd-run --user --unit=seismic-qa --property=Restart=on-failure --property=RestartSec=20 \
  --setenv=NLI_DEVICE=cpu --setenv=MPLBACKEND=Agg --setenv=SKIP_QA= \
  --setenv=BUILD_CONCURRENCY=1 --setenv=IMAGE_GEN_CONCURRENCY=1 \
  --working-directory="$PWD" \
  "$PWD/.venv/bin/python" -m scripts.watcher.process
echo "resumed: sglang + seismic-qa started; QA appends from ~3635 rows, ~49 samples remain"
