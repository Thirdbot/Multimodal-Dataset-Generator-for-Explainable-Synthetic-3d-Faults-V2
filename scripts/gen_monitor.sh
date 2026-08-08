#!/usr/bin/env bash
# Overnight watchdog: logs GPU/RAM/CPU/disk + generation health every 120s, guards resources,
# and exits (re-invoking the agent) on a meaningful EVENT or a ~30min heartbeat.
# ASSUMES generation runs as the seismic-gen systemd --user unit (created out-of-band, NOT by
# run_all.sh, which launches the build via `nohup bash scripts/run_generation.sh`) -- an absent
# unit reads as inactive here; the STALL_HEAL / disk / OOM guards still work off the filesystem.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=gen_monitor.log
MAX_ITERS=15   # ~30 min heartbeat
train_alive(){ pgrep -f "hybrid.train.train" >/dev/null && echo 1 || echo 0; }
for i in $(seq 1 $MAX_ITERS); do
  unit=$(systemctl --user is-active seismic-gen 2>/dev/null)
  scenes=$(ls build_objects/images 2>/dev/null | wc -l)
  raw=$(ls -d builds/seismic__* 2>/dev/null | wc -l)
  oom=$(journalctl --user -u seismic-gen --since '-3min' --no-pager 2>/dev/null | grep -ac 'oom-kill')
  ramA=$(free -g | awk 'NR==2{print $7}')
  swap=$(free -g | awk 'NR==3{print $3}')
  load=$(cut -d' ' -f1 /proc/loadavg)
  disk=$(df -h --output=avail . | tail -1 | tr -d ' ')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')
  tr=$(train_alive)
  pending=$(tail -20 gen_driver.log 2>/dev/null | grep -oE 'pending=[0-9]+' | tail -1 | tr -dc '0-9')
  echo "$(date +%H:%M:%S) unit=$unit scenes=$scenes raw=$raw pend=${pending:-?} oom3m=$oom ramA=${ramA}G swap=${swap}G load=$load disk=$disk gpu=$gpu train=$tr" >>"$LOG"
  # SELF-HEAL: orphan-stall. The builds_config watcher only builds on Change.added and its
  # startup loop does NOT rebuild pre-existing unbuilt configs; when the unit recycles it orphans
  # in-flight configs, and once pending>=QUEUE_AHEAD(3) the driver stops populating -> deadlock.
  # Signature: an active queue (pending>=3) but NO build running (raw==0) for 2 consecutive checks.
  if [ "$raw" -eq 0 ] && [ "${pending:-0}" -ge 3 ]; then idle=$((idle+1)); else idle=0; fi
  if [ "${idle:-0}" -ge 2 ]; then
    moved=0
    for cfg in build_configs/*.json; do
      [ -e "$cfg" ] || continue
      stem=$(basename "$cfg" .json)
      # only unbuilt (no scene/graph) AND settled (>3min old, not mid-pickup)
      if ls build_objects/images/ 2>/dev/null | grep -qF -- "$stem"; then continue; fi
      if [ -n "$(find "$cfg" -mmin +3 2>/dev/null)" ]; then
        mkdir -p build_configs_orphaned; mv "$cfg" build_configs_orphaned/ && moved=$((moved+1))
      fi
    done
    echo "$(date +%H:%M:%S) STALL_HEAL moved=$moved orphaned configs (pending=$pending raw=0)" >>"$LOG"
    idle=0
  fi
  # EVENT: repeated OOM kills -> generation is thrashing, needs attention
  if [ "${oom:-0}" -ge 2 ]; then echo "EVENT OOM_THRASH oom=$oom at $(date +%H:%M)" >>"$LOG"; echo "OOM_THRASH oom=$oom scenes=$scenes"; exit 0; fi
  # EVENT: unit dead and not recovering
  if [ "$unit" = "failed" ]; then
    systemctl --user reset-failed seismic-gen 2>/dev/null
    sleep 20
    if [ "$(systemctl --user is-active seismic-gen)" = "failed" ]; then echo "EVENT UNIT_DOWN at $(date +%H:%M)" >>"$LOG"; echo "UNIT_DOWN scenes=$scenes oom=$oom"; exit 0; fi
  fi
  # EVENT: disk critically low
  dnum=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  if [ "${dnum:-99}" -lt 20 ]; then echo "EVENT DISK_LOW ${dnum}G at $(date +%H:%M)" >>"$LOG"; echo "DISK_LOW ${dnum}G"; exit 0; fi
  sleep 120
done
echo "HEARTBEAT scenes=$(ls build_objects/images 2>/dev/null|wc -l) oom=$(journalctl --user -u seismic-gen --since '-30min' --no-pager 2>/dev/null|grep -ac oom-kill)"
