#!/usr/bin/env bash
# Watches the N sharded QA workers (seismic-qa-0..N-1) + sglang. Logs combined rows every 120s,
# exits (re-invoking the agent) if a worker dies, RAM goes critical, all workers finish, or ~30min.
# ASSUMES qa_shards.sh start|resume created the seismic-qa-0..N-1 systemd --user units (run_all's
# own qa_pass uses nohup, NOT these units) -- set N_SHARDS to match the run (qa_shards default 5,
# run_all default 8).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=qa_shard_monitor.log
N="${N_SHARDS:-5}"; MAX_ITERS=15
rows_total(){ local t=0 s; for s in $(seq 0 $((N-1))); do t=$((t + $(wc -l < Dataset/verified_qa_shard_$s.jsonl 2>/dev/null || echo 0))); done; echo $t; }
for i in $(seq 1 $MAX_ITERS); do
  states=""; active=0; dead=0
  for s in $(seq 0 $((N-1))); do
    st=$(systemctl --user is-active seismic-qa-$s 2>/dev/null); states+="${st:0:1}"
    [ "$st" = "active" ] && active=$((active+1))
    [ "$st" = "failed" ] && dead=$((dead+1))
  done
  rows=$(rows_total); ramA=$(free -g|awk 'NR==2{print $7}'); swap=$(free -g|awk 'NR==3{print $3}')
  sg=$(systemctl --user is-active sglang-qwen 2>/dev/null)
  echo "$(date +%H:%M:%S) rows=$rows workers=$states active=$active sglang=$sg ramA=${ramA}G swap=${swap}G" >>"$LOG"
  # a worker failed (OOM/crash) -> wake to restart that shard
  if [ "$dead" -gt 0 ]; then echo "WORKER_DEAD states=$states rows=$rows"; exit 0; fi
  # all workers finished (inactive, exit 0) -> time to merge + build CSV
  if [ "$active" -eq 0 ]; then echo "ALL_DONE rows=$rows"; exit 0; fi
  # RAM critical
  if [ "${ramA:-9}" -lt 1 ]; then echo "RAM_CRIT ramA=${ramA}G rows=$rows"; exit 0; fi
  sleep 120
done
echo "HEARTBEAT rows=$(rows_total) workers=$(for s in $(seq 0 $((N-1))); do systemctl --user is-active seismic-qa-$s|head -c1; done)"
