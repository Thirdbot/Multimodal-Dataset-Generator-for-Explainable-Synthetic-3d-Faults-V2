"""Corpus statistics extractor for dataset_research.md (§11, and inputs to §5/§6/§12).

Reads the final training CSV (Dataset/multimodal_multi_image_dataset.csv) and prints exact,
provenance-friendly distribution tables. Deterministic; no sampling. Run:

    .venv/bin/python scripts/dataset_stats.py

Every number the paper cites from the corpus should come from here (cite "scripts/dataset_stats.py").
"""
import csv
import json
import re
import statistics
from collections import Counter

csv.field_size_limit(10 ** 7)
CSV = "Dataset/multimodal_multi_image_dataset.csv"
CLASSES = ("fault", "closure", "salt", "onlap")


def load():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def regime_of(sample_id):
    # sample_id = seismic__<date>_<time>_<regime>_<hash>; regime is the middle token(s)
    m = re.match(r"seismic__\d+_\d+_([a-z_]+)_[0-9a-f]+$", sample_id)
    return m.group(1) if m else "UNPARSED"


def regions_of(row):
    try:
        return json.loads(row["regions"]) or []
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[BAD REGIONS] {row.get('sample_id', '?')}: {exc}")
        return []


def q_type(q):
    ql = q.lower()
    if re.search(r"\bhow many\b|\bnumber of\b", ql): return "count"
    if re.search(r"\bwhere\b|\blocat|\bsits?\b|\bposition\b", ql): return "location"
    if re.search(r"\bdip|\bsteep|\borientation|\bangle", ql): return "orientation/dip"
    if re.search(r"\bthrow|displac", ql): return "throw"
    if re.search(r"\barea|\bcover|percent", ql): return "area/coverage"
    if re.search(r"\bcompare|\bwhich\b|steeper|larger|greater|deeper|relation|relative|\bboth\b", ql): return "comparison/relation"
    if re.search(r"\bfluid|hydrocarbon|gas|oil|brine|water-bearing", ql): return "fluid"
    if re.search(r"\bpattern|horst|graben|relay|branch|staircase", ql): return "structural pattern"
    if re.search(r"\bpresent|\bshow|\bvisible|any ", ql): return "presence/absence"
    return "other"


def a_style(a, n_obj):
    a = re.sub(r"</?answer>", "", a)
    has_num = bool(re.search(r"\d", a))
    comparative = bool(re.search(r"steeper|gentler|larger|smaller|greater|less|more than|versus|compared|deeper|shallower", a, re.I)) and n_obj >= 2
    if comparative: return "comparative"
    if not has_num: return "qualitative"
    return "extractive"


def n_objects(row):
    # distinct object regions that are NOT the section-level region ("the section")
    objs = [r for r in regions_of(row) if str(r.get("object_name", "")).lower() != "the section"]
    return len(objs)


def pct(n, d): return f"{100.0*n/d:.1f}%" if d else "0%"


def main():
    rows = load()
    n = len(rows)
    print(f"CORPUS: {CSV}")
    print(f"[TOTAL] rows = {n}")

    scenes = {r["sample_id"] for r in rows}
    imgs = set()
    for r in rows:
        try:
            imgs.update(json.loads(r["images"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"[BAD IMAGES] {r.get('sample_id', '?')}: {exc}")
    views = {(r["sample_id"], reg.get("view")) for r in rows for reg in regions_of(r)}
    print(f"[TOTAL] distinct scenes (sample_id) = {len(scenes)}")
    print(f"[TOTAL] distinct images (unique image paths) = {len(imgs)}")
    print(f"[TOTAL] distinct (scene,view) pairs = {len(views)}")

    print("\n[SCENES per regime]  (regime parsed from sample_id)")
    scene_regime = {r["sample_id"]: regime_of(r["sample_id"]) for r in rows}
    for reg, c in Counter(scene_regime.values()).most_common():
        print(f"   {reg:20} {c:4}  ({pct(c, len(scenes))})")

    print("\n[ROWS per regime]")
    for reg, c in Counter(regime_of(r['sample_id']) for r in rows).most_common():
        print(f"   {reg:20} {c:5}  ({pct(c, n)})")

    print("\n[REGIONS per object class]  (segmentation targets; excludes section-level regions)")
    region_class = Counter()
    section_regions = 0
    class_id_map = {}
    for r in rows:
        for reg in regions_of(r):
            ot = reg.get("object_type")
            if str(reg.get("object_name", "")).lower() == "the section":
                section_regions += 1
                continue                              # section region is not a segmentation target
            region_class[ot] += 1
            class_id_map.setdefault(ot, reg.get("class_id"))
    tot_reg = sum(region_class.values())
    for ot, c in region_class.most_common():
        print(f"   {str(ot):10} (class_id={class_id_map.get(ot)}) {c:5}  ({pct(c, tot_reg)})")
    print(f"   [section-level regions (excluded above): {section_regions}]")
    print(f"   [total regions/masks: {tot_reg + section_regions}]")

    print("\n[ROWS per question type]  (regex heuristic; see q_type())")
    for t, c in Counter(q_type(r["question"]) for r in rows).most_common():
        print(f"   {t:22} {c:5}  ({pct(c, n)})")

    print("\n[ROWS per answer style]")
    for s, c in Counter(a_style(r["answer"], n_objects(r)) for r in rows).most_common():
        print(f"   {s:14} {c:5}  ({pct(c, n)})")

    print("\n[OBJECTS per row]  (distinct non-section object regions)")
    for k, c in sorted(Counter(n_objects(r) for r in rows).items()):
        print(f"   {k} object(s): {c:5}  ({pct(c, n)})")

    print("\n[ROWS per scene]")
    rps = Counter(r["sample_id"] for r in rows)
    vals = list(rps.values())
    if not vals:
        print("   (no rows)")
    else:
        print(f"   min={min(vals)} median={statistics.median(vals):.1f} mean={statistics.mean(vals):.2f} max={max(vals)}")

    print("\n[ANSWER length]  (words, tags stripped)")
    lens = [len(re.sub(r"</?answer>", "", r["answer"]).split()) for r in rows]
    lens.sort()
    if n == 0:
        print("   (no rows)")
    else:
        print(f"   min={lens[0]} p25={lens[n//4]} median={statistics.median(lens):.0f} p75={lens[3*n//4]} max={lens[-1]} mean={statistics.mean(lens):.1f}")

    print("\n[MEASURE/DERIVE value coverage]  (rows whose regions carry each value type)")
    measure_keys, derive_keys = Counter(), Counter()
    for r in rows:
        for reg in regions_of(r):
            v = reg.get("values", {}) or {}
            for k in (v.get("measure") or {}): measure_keys[k] += 1
            for k in (v.get("derive") or {}): derive_keys[k] += 1
    print("   measure:", dict(measure_keys))
    print("   derive :", dict(derive_keys))

    print("\n[SEG / region alignment check]")
    seg_counts = Counter()
    for r in rows:
        segs = r["evidence"].count("<SEG>")
        regs = len(regions_of(r))
        seg_counts["aligned" if segs == regs else "MISALIGNED"] += 1
    print("  ", dict(seg_counts))


if __name__ == "__main__":
    main()
