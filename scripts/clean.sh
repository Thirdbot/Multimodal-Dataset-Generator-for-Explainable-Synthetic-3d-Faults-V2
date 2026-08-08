#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Clean pipeline artifacts for a re-run or post-run tidy.
#
# DRY-RUN BY DEFAULT: prints what it WOULD remove (with sizes) and deletes nothing.
# Add --yes to actually delete. It NEVER removes Dataset/*.csv (your finished dataset).
#
#   scripts/clean.sh after      # post-run tidy: drop regenerable scaffolding
#                               #   removes: build_configs recipes builds graphs/_shard_*
#                               #   keeps:   build_objects/images  graphs/  Dataset/ (csv+masks+jsonl)
#   scripts/clean.sh qa         # re-run QA only (scenes/graphs kept)
#                               #   removes: Dataset/verified_qa*.jsonl  graphs/_shard_*
#   scripts/clean.sh rebuild    # re-BUILD from scratch (run_generation starts at 0 scenes)
#                               #   removes: build_objects/{images,objects}  build_configs recipes builds
#                               #            graphs/{properties_graph,properties_2d_graph,_shard_*,*_db_extract.json}
#                               #   keeps:   Dataset/ (⚠ the OLD csv then references deleted images)
#
#   ...add --yes to execute:    scripts/clean.sh rebuild --yes
# ─────────────────────────────────────────────────────────────────────────────
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
# accept --yes in ANY position; MODE = first non-flag arg (safe default: dry-run unless --yes present)
MODE=""; YES=""
for a in "$@"; do
  if [ "$a" = "--yes" ]; then YES=1; elif [ -z "$MODE" ]; then MODE="$a"; fi
done

case "$MODE" in
  after)   TARGETS="build_configs recipes builds graphs/_shard_*" ;;
  qa)      TARGETS="Dataset/verified_qa.jsonl Dataset/verified_qa_shard_*.jsonl graphs/_shard_*" ;;
  rebuild) TARGETS="build_objects/images build_objects/objects graphs/properties_graph graphs/properties_2d_graph graphs/_shard_* graphs/*_db_extract.json build_configs recipes builds"
           echo "⚠ rebuild removes the scene images -> the current Dataset/*.csv will reference deleted files." ;;
  *) echo "usage: clean.sh {after|qa|rebuild} [--yes]"; exit 1 ;;
esac

# hard safety: a *.csv must never be in the target set
case "$TARGETS" in *.csv*) echo "ABORT: target set includes a .csv"; exit 1 ;; esac

echo "mode: $MODE   (Dataset/*.csv is always kept)"
paths=()
for glob in $TARGETS; do
  for p in $glob; do [ -e "$p" ] && paths+=("$p"); done
done
[ "${#paths[@]}" -eq 0 ] && { echo "  (nothing to remove)"; exit 0; }
for p in "${paths[@]}"; do printf "  %-42s %s\n" "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"; done
printf "total: %s\n" "$(du -sch "${paths[@]}" 2>/dev/null | tail -1 | cut -f1)"

if [ -z "$YES" ]; then
  echo "DRY-RUN — nothing deleted. Re-run with --yes to remove the above."
  exit 0
fi
rm -rf "${paths[@]}"
echo "cleaned ($MODE): removed ${#paths[@]} paths."
