#!/usr/bin/env bash
# Manage the 5 sharded QA workers (seismic-qa-0..4).
#   qa_shards.sh start    -> FRESH run: each worker truncates its shard output and starts over
#   qa_shards.sh resume   -> APPEND: skip graphs already done, continue accumulating (safe to re-run)
#   qa_shards.sh stop     -> stop all workers (progress in verified_qa_shard_*.jsonl is preserved)
#   qa_shards.sh status   -> worker states + combined row count
#   qa_shards.sh merge          -> combine shard files -> Dataset/verified_qa.jsonl (dedup by row_id)
#   qa_shards.sh balance [N]     -> auto object-class balance -> Dataset/verified_qa.jsonl
#   qa_shards.sh csv [jsonl]     -> build the CSV (default verified_qa.jsonl; pass balanced one to use it)
#   qa_shards.sh finalize        -> stop + merge + csv (full-set end-of-run wrap-up)
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${N_SHARDS:-5}"
CMD="${1:-status}"

rows_total(){ local t=0 s; for s in $(seq 0 $((N-1))); do t=$((t + $(wc -l < Dataset/verified_qa_shard_$s.jsonl 2>/dev/null || echo 0))); done; echo $t; }

case "$CMD" in
  start|resume)
    RESUME=""; [ "$CMD" = "resume" ] && RESUME=1
    for s in $(seq 0 $((N-1))); do
      systemctl --user reset-failed seismic-qa-$s 2>/dev/null || true
      systemd-run --user --unit=seismic-qa-$s \
        --setenv=NLI_DEVICE=cpu --setenv=MPLBACKEND=Agg --setenv=CUDA_VISIBLE_DEVICES= \
        --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
        --setenv=QA_GRAPH_ROOT="graphs/_shard_$s" --setenv=QA_OUTPUT="Dataset/verified_qa_shard_$s.jsonl" \
        --setenv=QA_RESUME="$RESUME" --setenv=PYTHONPATH="$PWD" \
        --property=MemoryHigh=2200M --property=MemoryMax=2900M \
        --working-directory="$PWD" \
        "$PWD/.venv/bin/python" scripts/qa_new_only.py
      sleep 2
    done
    echo "[$CMD] workers: $(for s in $(seq 0 $((N-1))); do echo -n "$s:$(systemctl --user is-active seismic-qa-$s) "; done)"
    ;;
  stop)
    for s in $(seq 0 $((N-1))); do systemctl --user stop seismic-qa-$s 2>/dev/null || true; systemctl --user reset-failed seismic-qa-$s 2>/dev/null || true; done
    echo "stopped. combined rows preserved: $(rows_total)"
    ;;
  status)
    echo "workers: $(for s in $(seq 0 $((N-1))); do echo -n "$s:$(systemctl --user is-active seismic-qa-$s) "; done)"
    echo "combined rows: $(rows_total)"
    ;;
  merge)
    # Combine the shard files -> Dataset/verified_qa.jsonl, dedup by row_id. Shards are disjoint
    # (distinct graphs), so this is safe; run 'stop' first for a COMPLETE snapshot, not a live one.
    active=0; for s in $(seq 0 $((N-1))); do [ "$(systemctl --user is-active seismic-qa-$s 2>/dev/null)" = active ] && active=$((active+1)); done
    [ "$active" -gt 0 ] && echo "note: $active worker(s) still running -- merging a LIVE snapshot; run 'stop' (or 'finalize') for the final one"
    "$PWD/.venv/bin/python" - <<'PY'
import json, glob
seen, out = set(), []
shards = sorted(glob.glob('Dataset/verified_qa_shard_*.jsonl'))
for f in shards:
    for line in open(f):
        if not line.strip(): continue
        try: rid = json.loads(line).get('row_id')
        except Exception: continue
        if rid in seen: continue
        seen.add(rid); out.append(line if line.endswith('\n') else line + '\n')
open('Dataset/verified_qa.jsonl', 'w').writelines(out)
print(f'merged {len(out)} rows from {len(shards)} shards -> Dataset/verified_qa.jsonl')
PY
    ;;
  balance)
    # Automatic object-class balancer. Down-samples the over-represented class(es) by dropping
    # their LOWEST-COLLATERAL rows first (e.g. onlap-ONLY rows before onlap+fault rows), so the
    # other classes stay intact. Default target = cap the top class to the 2nd-highest count;
    # pass a number as $2 for an explicit per-class target. Writes a SEPARATE balanced file --
    # the full verified_qa.jsonl is untouched. Build its CSV with:  $0 csv Dataset/verified_qa.jsonl
    [ -s Dataset/verified_qa.jsonl ] || { echo "Dataset/verified_qa.jsonl empty -- run '$0 merge' first"; exit 1; }
    BALANCE_TARGET="${2:-}" "$PWD/.venv/bin/python" - <<'PY'
import json, os
from collections import Counter
CLASSES = ('fault', 'closure', 'salt', 'onlap')
def otype(oid):
    oid = str(oid)
    for t in CLASSES:
        if oid.startswith(t):
            return t
    return None
rows = []                                    # [line, Counter(class->regions), keep]
for l in open('Dataset/verified_qa.jsonl'):
    if not l.strip():
        continue
    try: r = json.loads(l)
    except Exception: continue
    ev = r.get('evidence')
    if isinstance(ev, str):
        try: ev = json.loads(ev)
        except Exception: ev = []
    cl, seen = Counter(), set()
    for e in (ev or []):
        if isinstance(e, dict):
            oid = e.get('object_id') or e.get('source'); t = otype(oid)
            if t and oid not in seen:
                seen.add(oid); cl[t] += 1
    rows.append([l if l.endswith('\n') else l + '\n', cl, True])
before = Counter()
for _, cl, _ in rows: before.update(cl)
arg = os.environ.get('BALANCE_TARGET', '').strip()
if arg.isdigit():
    target = int(arg)
else:
    v = sorted(before.values(), reverse=True); target = v[1] if len(v) > 1 else v[0]
cur = Counter(before)
for cls in sorted(CLASSES, key=lambda c: cur[c] - target, reverse=True):
    if cur[cls] <= target:
        continue
    cands = [i for i, (_, cl, keep) in enumerate(rows) if keep and cl.get(cls, 0) > 0]
    cands.sort(key=lambda i: sum(v for k, v in rows[i][1].items() if k != cls))   # onlap-only first
    for i in cands:
        if cur[cls] <= target:
            break
        rows[i][2] = False; cur.subtract(rows[i][1])
out = [r[0] for r in rows if r[2]]
open('Dataset/verified_qa.jsonl', 'w').writelines(out)
after = Counter()
for _, cl, keep in rows:
    if keep: after.update(cl)
print(f'target per class: {target}')
print(f'before: {dict((c, before[c]) for c in CLASSES)}  ({len(rows)} rows)')
print(f'after : {dict((c, after[c]) for c in CLASSES)}  ({len(out)} rows, dropped {len(rows)-len(out)})')
print('-> Dataset/verified_qa.jsonl')
PY
    ;;
  csv)
    # Build the CSV (regions/masks + <SEG> in evidence). Defaults to verified_qa.jsonl; pass an
    # alternate jsonl as $2 (e.g. the balanced one) -- honored via QA_INPUT in DatasetMaker.
    IN="${2:-Dataset/verified_qa.jsonl}"
    [ -s "$IN" ] || { echo "$IN is empty -- run '$0 merge' (or 'balance') first"; exit 1; }
    echo "building CSV from $IN ($(wc -l < "$IN") rows) ..."
    QA_INPUT="$IN" "$PWD/.venv/bin/python" Dataset/DatasetMaker.py
    ;;
  finalize)
    # End-of-run wrap-up: stop workers, merge to a complete snapshot, build the CSV (full set).
    "$0" stop; "$0" merge; "$0" csv
    ;;
  *) echo "usage: $0 {start|resume|stop|status|merge|balance [target]|csv [jsonl]|finalize}"; exit 1;;
esac
