#!/usr/bin/env bash
# Detached generation driver: keeps the watcher alive across shell/session exits and tops up
# recipes (2 fault_complex each) until TARGET_SCENES scene folders exist.
#   TARGET_SCENES : how many distinct scenes (each renders 2 view-images) to reach
#   QUEUE_AHEAD   : how many un-built configs to keep queued so the builder never idles
#   MIN_FREE_GB   : hard stop if the disk drops below this
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET_SCENES="${TARGET_SCENES:-250}"
QUEUE_AHEAD="${QUEUE_AHEAD:-9}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
MAX_WATCHER_GB="${MAX_WATCHER_GB:-5}"   # recycle the leaking watcher before it OOMs the unit
MAX_RAW_BACKLOG="${MAX_RAW_BACKLOG:-4}"   # pause building while unextracted raw builds (~1.2GB each) pile up
LOG=gen_run.log
DRIVER_LOG=gen_driver.log

scenes()  { ls build_objects/images 2>/dev/null | wc -l; }
freegb()  { df --output=avail -BG . | tail -1 | tr -dc '0-9'; }

# Count only UNBUILT configs. Build configs are NOT removed after a successful build (the
# watcher just skips them via _config_already_built), so counting *.json would count finished
# work as queued -> the queue never looks empty -> we stop populating -> deadlock.
# A config is built when a scene folder carries its stem (seismic__DATE_<stem>).
pending() {
  local n=0 f stem
  for f in build_configs/*.json; do
    [ -e "$f" ] || continue
    stem=$(basename "$f" .json)
    if ! ls build_objects/images 2>/dev/null | grep -q -- "$stem"; then n=$((n+1)); fi
  done
  echo "$n"
}

# builds_config_watcher only builds a config on Change.added *while it is watching*; its startup
# loop merely skips already-built ones. So any UNBUILT config that predates the current watcher
# would sit there forever, and (since pending() counts it) the queue would look full and we'd
# stop populating -> deadlock. Drop them right after a watcher (re)start so the populate loop
# recreates equivalent configs with the watcher attached. Safe there: no build is in flight at
# start, and deleting an unbuilt config's file has no build folder for on_build_delete to remove.
drop_stale_configs() {
  local f stem
  for f in build_configs/*.json; do
    [ -e "$f" ] || continue
    stem=$(basename "$f" .json)
    if ! ls build_objects/images 2>/dev/null | grep -q -- "$stem"; then
      echo "$(date +%H:%M:%S) dropping pre-watcher config: $stem" >>"$DRIVER_LOG"
      rm -f "$f"
    fi
  done
}

# Kill our watcher AND its build subprocesses when this driver exits. Without this, stopping /
# OOM-killing the driver left orphaned builds running: a second driver then started its own
# watcher, each watcher enforces its OWN BUILD_CONCURRENCY semaphore, so 2 watchers x 2 = 4
# concurrent builds -> ~1.8GB free -> load 147 -> OOM. Orphans are the real hazard here.
cleanup() {
  pkill -9 -P $$ -f "scripts.watcher.process" 2>/dev/null || true
  pkill -9 -f "scripts.build.sample_generator" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) watcher -- NO setsid: under systemd, setsid makes the watcher escape the unit's cgroup, so
# systemd kills the driver on OOM/stop while the watcher survives as an orphan and keeps
# building. Keeping it a child means systemd owns the whole tree (driver+watcher+builds).
if ! pgrep -f "[p]ython -m scripts.watcher.process" >/dev/null; then
  env NLI_DEVICE=cpu SKIP_QA=${SKIP_QA:-1} BUILD_CONCURRENCY=${BUILD_CONCURRENCY:-2} IMAGE_GEN_CONCURRENCY=${IMAGE_GEN_CONCURRENCY:-2} \
    .venv/bin/python -m scripts.watcher.process >>"$LOG" 2>&1 </dev/null &
  sleep 20   # let awatch attach before any populate, else configs are not seen as "added"
fi
drop_stale_configs   # anything unbuilt predates this watcher and would never build

# 2) top-up loop: populate 2-at-a-time while the queue is short and target not reached
while true; do
  s=$(scenes); p=$(pending); f=$(freegb)
  echo "$(date +%H:%M:%S) scenes=$s/$TARGET_SCENES pending=$p freeGB=$f" >>"$DRIVER_LOG"
  if [ "$s" -ge "$TARGET_SCENES" ]; then
    echo "$(date +%H:%M:%S) TARGET REACHED ($s scenes)" >>"$DRIVER_LOG"; break
  fi
  if [ "${f:-99}" -lt "$MIN_FREE_GB" ]; then
    echo "$(date +%H:%M:%S) STOP: disk below ${MIN_FREE_GB}G" >>"$DRIVER_LOG"; break
  fi
  # The watcher LEAKS: measured ~2.0GB at start -> ~8.9GB after ~70min / ~330 graphs, which
  # OOM-killed the whole unit (builds are innocent: ~0.6-1.3GB each). The per-graph RAG vector
  # stores / NLI tracing accumulate. Recycle it BEFORE the kernel does: a controlled kill loses
  # nothing (graphs are on disk, QA is regenerated in the final clean pass) whereas an OOM takes
  # the driver and in-flight builds with it. The "watcher died" branch below then restarts it.
  w_rss_kb=$(ps -eo rss,args | grep -a "[p]ython -m scripts.watcher.process" | awk '{print $1; exit}')
  if [ -n "${w_rss_kb:-}" ] && [ "$w_rss_kb" -gt $(( MAX_WATCHER_GB * 1048576 )) ]; then
    echo "$(date +%H:%M:%S) watcher RSS $(( w_rss_kb / 1048576 ))GB > ${MAX_WATCHER_GB}GB - recycling" >>"$DRIVER_LOG"
    pkill -9 -f "[p]ython -m scripts.watcher.process" 2>/dev/null || true
    sleep 5
  fi
  if ! pgrep -f "[p]ython -m scripts.watcher.process" >/dev/null; then
    echo "$(date +%H:%M:%S) watcher died - restarting" >>"$DRIVER_LOG"
    env NLI_DEVICE=cpu SKIP_QA=${SKIP_QA:-1} BUILD_CONCURRENCY=${BUILD_CONCURRENCY:-2} IMAGE_GEN_CONCURRENCY=${IMAGE_GEN_CONCURRENCY:-2} \
      .venv/bin/python -m scripts.watcher.process >>"$LOG" 2>&1 </dev/null &
    sleep 20
    drop_stale_configs   # same reason as at startup: they predate the new watcher
  fi
  # Backpressure on the RAW-BUILD backlog, not just the config queue.
  #
  # Builds emit a ~1.2GB raw folder each and it is only deleted after image-gen extracts it.
  # Building (2 concurrent, ~1.5min each) outruns single-threaded image-gen, so raw builds pile
  # up: the disk fell 17G -> 9G while gaining only 2 scenes, and the run hit the disk floor.
  # Stop feeding new work while the extractor is behind; it drains, then we resume.
  raw_backlog=$(ls -d builds/seismic__* 2>/dev/null | wc -l)
  if [ "$raw_backlog" -ge "$MAX_RAW_BACKLOG" ]; then
    echo "$(date +%H:%M:%S) raw backlog $raw_backlog >= $MAX_RAW_BACKLOG - pausing populate (extractor catching up)" >>"$DRIVER_LOG"
  else
    # keep the queue shallow: one recipe = sample_population samples
    while [ "$(pending)" -lt "$QUEUE_AHEAD" ] && [ $(( $(scenes) + $(pending) )) -lt "$TARGET_SCENES" ]; do
      .venv/bin/python -m scripts.config.control_parameter >>"$DRIVER_LOG" 2>&1
    done
  fi
  sleep 60
done
