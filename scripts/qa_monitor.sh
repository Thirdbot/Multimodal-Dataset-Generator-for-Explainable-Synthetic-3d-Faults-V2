#!/usr/bin/env bash
# Watches the clean-regen QA run (seismic-qa-new + sglang-qwen). Logs progress every 120s,
# exits (re-invoking the agent) when the QA unit finishes/fails or on a ~30min heartbeat.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=qa_monitor.log
MAX_ITERS=15
for i in $(seq 1 $MAX_ITERS); do
  qa=$(systemctl --user is-active seismic-qa-new 2>/dev/null)
  sg=$(systemctl --user is-active sglang-qwen 2>/dev/null)
  rows=$(wc -l < Dataset/verified_qa.jsonl 2>/dev/null)
  ramA=$(free -g | awk 'NR==2{print $7}')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
  trej=$(journalctl --user -u seismic-qa-new --no-pager 2>/dev/null | grep -ac 'topic mismatch')
  crej=$(journalctl --user -u seismic-qa-new --no-pager 2>/dev/null | grep -ac 'count question not')
  echo "$(date +%H:%M:%S) qa=$qa sglang=$sg rows=$rows ramA=${ramA}G gpu=$gpu topic_rej=$trej count_rej=$crej" >>"$LOG"
  # QA finished (clean exit -> inactive) or died -> wake agent to build CSV / investigate
  if [ "$qa" != "active" ]; then
    res=$(systemctl --user show seismic-qa-new -p Result --value 2>/dev/null)
    echo "QA_ENDED state=$qa result=$res rows=$rows"; exit 0
  fi
  # sglang died -> QA will stall on LLM calls; wake agent
  if [ "$sg" = "failed" ]; then echo "SGLANG_DOWN rows=$rows"; exit 0; fi
  sleep 120
done
echo "HEARTBEAT rows=$(wc -l < Dataset/verified_qa.jsonl 2>/dev/null) qa=$(systemctl --user is-active seismic-qa-new)"
