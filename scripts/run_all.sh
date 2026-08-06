#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end 2-class dataset run (FAULT + CLOSURE + empty-mask negatives), tuned for a
#   12-core / 64 GB RAM / 25 GB VRAM / 500 GB-disk box.
#
# Phases:  preflight -> build (CPU) -> qa (GPU) -> finalize
#   build    scenes -> property graphs -> images        (run_generation.sh, CPU only)
#   qa       sglang + N NLI-on-GPU workers over graph shards, RESUME-safe
#   finalize class-balance -> final CSV                 (qa_shards.sh)
#
# WHY this is fast here: the 25 GB card fits Qwen-1.5B AND the NLI model, so NLI runs on GPU
# (NLI_DEVICE=cuda) instead of CPU -- that removes the CPU-NLI throughput wall of the small box.
#
# WHY it won't crash: aborts up-front if VRAM<8 GB (training using the GPU) or disk<MIN_FREE_GB;
# waits for sglang health before starting workers; workers are RESUME-safe (re-run = no redo);
# build keeps its disk floor + watcher-recycle backpressure.
#
# Usage:  bash scripts/run_all.sh [all|preflight|build|qa|finalize]   (default: all)
#   TARGET_SCENES=1000 bash scripts/run_all.sh all      # override any tunable via env
# ─────────────────────────────────────────────────────────────────────────────
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root = parent of scripts/, auto-detected
cd "$ROOT" || { echo "cannot cd to repo root $ROOT"; exit 1; }

# ── tunables (sized for 12c / 64 GB / 25 GB) ────────────────────────────────
TARGET_SCENES="${TARGET_SCENES:-2000}"                 # ~40 GB disk, overnight QA on this box
N_SHARDS="${N_SHARDS:-8}"                               # more workers -> more concurrent sglang requests -> bigger GPU batch
                                                       #   (NLI on GPU = light CPU, so >cores is fine). qa_shards honors N_SHARDS.
NLI_DEVICE="${NLI_DEVICE:-cuda}"                        # THE unlock: NLI on the 25 GB GPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"; export MKL_NUM_THREADS="$OMP_NUM_THREADS"
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-6}"            # 64 GB holds ~6 parallel Synthoseis builds
IMAGE_GEN_CONCURRENCY="${IMAGE_GEN_CONCURRENCY:-3}"
# `overlap` mode runs build + QA at the SAME time (GPU isn't idle during the build); they share
# the 12 cores, so each runs leaner than a full-machine sequential phase:
OVERLAP_BUILD_CONCURRENCY="${OVERLAP_BUILD_CONCURRENCY:-3}"
OVERLAP_N_SHARDS="${OVERLAP_N_SHARDS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://localhost:8000/v1}"   # started by YOU (scripts/llm_server.sh); we only check it
GRAPH_DIR="graphs/properties_2d_graph"

log(){ echo "$(date +%H:%M:%S) [run_all] $*"; }
graphs(){ ls "$GRAPH_DIR"/*_properties_graph_*.json 2>/dev/null | wc -l; }

preflight(){
  command -v nvidia-smi >/dev/null || { log "ABORT: no nvidia-smi"; exit 1; }
  local vram; vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "GPU free VRAM: ${vram:-?} MiB"
  # the LLM server (user-run) has already taken its VRAM; NLI just needs a few GB of headroom
  if [ "${NLI_DEVICE}" = "cuda" ] && [ "${vram:-0}" -lt 3000 ]; then
    log "ABORT: <3 GB free VRAM for NLI. Lower the server's LLM_MEM_FRAC, or run with NLI_DEVICE=cpu."; exit 1
  fi
  local d; d=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  log "disk free: ${d:-?} GB"; [ "${d:-0}" -lt "$MIN_FREE_GB" ] && { log "ABORT: disk < ${MIN_FREE_GB} GB"; exit 1; }
  log "preflight OK (target=$TARGET_SCENES shards=$N_SHARDS nli=$NLI_DEVICE build=$BUILD_CONCURRENCY/$IMAGE_GEN_CONCURRENCY)"
}

build(){
  log "BUILD -> $TARGET_SCENES scenes (regimes fault_only/fault_complex/boring)"
  TARGET_SCENES="$TARGET_SCENES" MIN_FREE_GB="$MIN_FREE_GB" SKIP_QA=1 \
    BUILD_CONCURRENCY="$BUILD_CONCURRENCY" IMAGE_GEN_CONCURRENCY="$IMAGE_GEN_CONCURRENCY" \
    bash scripts/run_generation.sh                     # blocks until target or disk floor, then cleans up
  log "BUILD done: $(ls build_objects/images | wc -l) scenes, $(graphs) 2D graphs"
}

require_llm(){
  # We do NOT start the server. YOU run it (scripts/llm_server.sh sglang|vllm); here we only verify.
  curl -s "$LLM_ENDPOINT/models" >/dev/null 2>&1 && { log "LLM endpoint up: $LLM_ENDPOINT"; return; }
  log "ABORT: LLM endpoint not reachable ($LLM_ENDPOINT). Start it first, e.g.:"
  log "    uv sync --extra sglang && bash scripts/llm_server.sh sglang    (or --extra vllm / ... vllm)"
  exit 1
}

shard(){
  local n; n=$(graphs); [ "$n" -eq 0 ] && { log "ABORT: no graphs in $GRAPH_DIR (run build first)"; exit 1; }
  for s in $(seq 0 $((N_SHARDS-1))); do rm -rf "graphs/_shard_$s"; mkdir -p "graphs/_shard_$s"; done
  for f in "$GRAPH_DIR"/*_properties_graph_*.json; do          # key by SCENE stem so both views co-locate
    local stem=${f##*/}; stem=${stem%%_properties_graph_*}
    local b=$(( $(cksum <<<"$stem" | cut -d' ' -f1) % N_SHARDS ))
    ln -sf "$(realpath "$f")" "graphs/_shard_$b/${f##*/}"
  done
  log "sharded $n graphs: $(for s in $(seq 0 $((N_SHARDS-1))); do echo -n "$s:$(ls graphs/_shard_$s|wc -l) "; done)"
}

# One QA pass over whatever graphs exist now: (re)shard, launch N workers RESUME-mode, wait.
# Deterministic sharding + resume => passes never double-process a graph, so it is safe to call
# repeatedly (overlap mode does, as the build produces more graphs).
qa_pass(){
  shard
  log "QA pass: $N_SHARDS workers, NLI on $NLI_DEVICE"
  local pids=() s
  for s in $(seq 0 $((N_SHARDS-1))); do
    NLI_DEVICE="$NLI_DEVICE" MPLBACKEND=Agg PYTHONPATH="$ROOT" \
      QA_GRAPH_ROOT="graphs/_shard_$s" QA_OUTPUT="Dataset/verified_qa_shard_$s.jsonl" QA_RESUME=1 \
      nohup "$ROOT/.venv/bin/python" scripts/qa_new_only.py >>"qa_shard_$s.log" 2>&1 &
    pids+=($!); sleep 3
  done
  wait "${pids[@]}" || true                                    # a dead worker: resume-safe, just re-run
  local t=0; for s in $(seq 0 $((N_SHARDS-1))); do t=$((t + $(wc -l < Dataset/verified_qa_shard_$s.jsonl 2>/dev/null || echo 0))); done
  log "QA pass done: $t rows across $N_SHARDS shards"
}

qa(){ require_llm; qa_pass; }                                   # sequential: build already finished

# Run build and QA CONCURRENTLY. Build streams graphs in the background; we keep launching QA
# passes that consume new graphs (resume skips done ones) until the build finishes and drains.
overlap(){
  N_SHARDS="$OVERLAP_N_SHARDS"; BUILD_CONCURRENCY="$OVERLAP_BUILD_CONCURRENCY"
  preflight; require_llm
  log "OVERLAP: build($BUILD_CONCURRENCY) + QA($N_SHARDS) concurrent, sharing 12 cores -> GPU not idle during build"
  TARGET_SCENES="$TARGET_SCENES" MIN_FREE_GB="$MIN_FREE_GB" SKIP_QA=1 \
    BUILD_CONCURRENCY="$BUILD_CONCURRENCY" IMAGE_GEN_CONCURRENCY="$IMAGE_GEN_CONCURRENCY" \
    nohup bash scripts/run_generation.sh >>gen_run.log 2>&1 &
  local bpid=$!; log "overlap: build pid $bpid (background)"
  local seen=0 drain=0 n
  while :; do
    n=$(graphs)
    if [ "$n" -gt "$seen" ]; then qa_pass; seen="$n"; fi        # new graphs appeared -> process them
    if kill -0 "$bpid" 2>/dev/null; then sleep 30
    else drain=$((drain+1)); [ "$drain" -ge 2 ] && break; fi    # build done: 2 drain passes catch the tail
  done
  wait "$bpid" 2>/dev/null || true
  finalize
}

finalize(){
  log "balance + CSV over $N_SHARDS shards"
  N_SHARDS="$N_SHARDS" bash scripts/qa_shards.sh balance        # merges shard_0..N-1 -> Dataset/verified_qa.jsonl (balanced)
  bash scripts/qa_shards.sh csv Dataset/verified_qa.jsonl       # -> Dataset/multimodal_multi_image_dataset.csv
  log "final CSV built. stats:"
  .venv/bin/python scripts/dataset_stats.py 2>/dev/null | head -24
}

case "${1:-all}" in
  preflight) preflight ;;
  build)     build ;;
  qa)        preflight; qa ;;
  finalize)  finalize ;;
  all)       preflight; build; qa; finalize ;;   # sequential: build (full machine) then QA (full machine)
  overlap)   overlap ;;                          # build + QA at once (faster wall-clock; shares cores)
  *) echo "usage: run_all.sh {all|overlap|preflight|build|qa|finalize}"; exit 1 ;;
esac
