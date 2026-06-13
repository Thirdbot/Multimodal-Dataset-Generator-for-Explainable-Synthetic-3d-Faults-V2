"""Build a multi-image dataset manifest from verified graph QA rows.

The input rows already contain graph evidence with object ids. This script maps
those object ids to image/mask/overlay triples under build_objects/images and
writes a JSONL file for training plus a CSV file for inspection.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Verifier.llm_machine import multimodal_dataset_instruction


DATASET_DIR = ROOT / "Dataset"
DEFAULT_INPUT = DATASET_DIR / "hybrid_verified_qa.jsonl"
DEFAULT_JSONL_OUTPUT = DATASET_DIR / "multimodal_multi_image_dataset.jsonl"
DEFAULT_CSV_OUTPUT = DATASET_DIR / "multimodal_multi_image_dataset.csv"
DEFAULT_IMAGE_ROOT = ROOT / "build_objects" / "images"


CATEGORY_TO_GLOBAL_IMAGE_TYPES = {
    "category:boring": ["closure"],
    "category:fault_only": ["fault"],
    "category:fault_complex": ["fault", "closure"],
    "category:salt_only": ["salt", "closure"],
    "category:salt_fault_mixed": ["fault", "salt", "closure"],
    "category:onlap": ["onlap", "closure"],
    "category:depositional": ["closure", "lithology"],
    "category:full_mixed": ["fault", "salt", "onlap", "closure"],
}

CATEGORY_EDGE_TO_IMAGE_TYPES = {
    "number_faults": ["fault"],
    "fault_mode": ["fault"],
    "number_fault_intersections": ["fault"],
    "salt_inserted": ["salt"],
    "number_hc_closures": ["closure"],
    "fluid": ["closure"],
    "number_onlap_episodes": ["onlap"],
    "onlaps_horizon_list": ["onlap"],
    "number_fan_episodes": ["lithology"],
    "fan_horizon_list": ["lithology"],
    "sand_voxel_pct": ["lithology"],
    "sand_layer_percent_a_posteriori": ["lithology"],
}

OBJECT_TYPES = {"fault", "closure", "salt", "onlap", "lithology", "age_depth"}


def main():
    args = parse_args()
    rows = [
        make_row(row, Path(args.image_root), min_image_score=args.min_image_score)
        for row in read_jsonl(args.input)
    ]
    rows = [row for row in rows if row["images"]]
    write_jsonl(rows, args.jsonl_output)
    write_csv(rows, args.csv_output)
    print(json.dumps({
        "rows": len(rows),
        "jsonl_output": str(args.jsonl_output),
        "csv_output": str(args.csv_output),
    }, indent=2))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def make_row(row, image_root, min_image_score=0.4):
    sample_id = row["sample_id"]
    view = row.get("view") or "inline"
    evidence = row.get("evidence", [])
    sample_image_dir = image_root / sample_id

    images = collect_images(sample_image_dir, view, row.get("category", ""), evidence, min_image_score)

    return {
        "row_id": row.get("row_id") or stable_id(row),
        "sample_id": sample_id,
        "category": row.get("category", ""),
        "view": view,
        "instruction": multimodal_dataset_instruction(),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "reason": row.get("trace", {}).get("reason", ""),
        "images": images,
        "image_paths": [item["image_path"] for item in images],
        "mask_paths": [item["mask_path"] for item in images],
        "overlay_paths": [item["overlay_path"] for item in images],
        "object_ids": [item["object_id"] for item in images],
        "evidence": compact_evidence(evidence),
        "verification": row.get("verification", {}),
        "verification_verdict": row.get("verification", {}).get("verdict", ""),
        "verification_score": row.get("verification", {}).get("score", ""),
    }


def collect_images(sample_image_dir, view, category, evidence, min_image_score=0.4):
    images = []
    seen = set()
    requested_global_types = []

    for item in evidence:
        if evidence_score(item) < min_image_score:
            continue
        object_id = item.get("object_id") or item.get("source") or ""
        edge = item.get("edge") or item.get("fact_name") or ""

        for object_type in image_types_from_evidence(object_id, edge, category):
            requested_global_types.append(object_type)

        if is_object_instance(object_id):
            object_type = object_id.split("_", 1)[0]
            add_image(images, seen, sample_image_dir, view, object_type, object_id, "evidence_object")
            add_image(images, seen, sample_image_dir, view, object_type, object_type, "global")
        elif object_id in OBJECT_TYPES:
            add_image(images, seen, sample_image_dir, view, object_id, object_id, "global")

    for object_type in dedupe(requested_global_types):
        add_image(images, seen, sample_image_dir, view, object_type, object_type, "global")

    if not any(item["role"] == "global" for item in images):
        for object_type in fallback_global_types(category):
            if add_image(images, seen, sample_image_dir, view, object_type, object_type, "global"):
                break

    if not images:
        add_first_available_global(images, seen, sample_image_dir, view)

    return images


def image_types_from_evidence(object_id, edge, category):
    if str(object_id).startswith("category:"):
        edge_types = CATEGORY_EDGE_TO_IMAGE_TYPES.get(edge)
        if edge_types:
            return edge_types
        return CATEGORY_TO_GLOBAL_IMAGE_TYPES.get(object_id, fallback_global_types(category))

    if object_id in OBJECT_TYPES:
        return [object_id]

    if is_object_instance(object_id):
        return [object_id.split("_", 1)[0]]

    return []


def fallback_global_types(category):
    category_node = category if str(category).startswith("category:") else f"category:{category}"
    return CATEGORY_TO_GLOBAL_IMAGE_TYPES.get(category_node, ["closure", "fault", "salt", "onlap"])


def add_first_available_global(images, seen, sample_image_dir, view):
    for image_path in sorted(sample_image_dir.glob(f"*/*/{view}.png")):
        object_type = image_path.parent.parent.name
        object_id = image_path.parent.name
        if object_type != object_id:
            continue
        if add_image(images, seen, sample_image_dir, view, object_type, object_id, "global"):
            return True
    return False


def add_image(images, seen, sample_image_dir, view, object_type, object_id, role):
    image_path = sample_image_dir / object_type / object_id / f"{view}.png"
    mask_path = sample_image_dir / object_type / object_id / f"{view}_mask.png"
    overlay_path = sample_image_dir / object_type / object_id / f"{view}_overlay.png"

    if not image_path.exists():
        return False

    key = (object_type, object_id, view, role)
    if key in seen:
        return True
    seen.add(key)

    images.append({
        "role": role,
        "object_type": object_type,
        "object_id": object_id,
        "view": view,
        "image_path": image_path.as_posix(),
        "mask_path": mask_path.as_posix() if mask_path.exists() else "",
        "overlay_path": overlay_path.as_posix() if overlay_path.exists() else "",
    })
    return True


def compact_evidence(evidence):
    compact = []
    for item in evidence:
        compact.append({
            "text": item.get("text", ""),
            "score": item.get("score", ""),
            "trace_type": item.get("trace_type", ""),
            "source": item.get("source", ""),
            "object_id": item.get("object_id", ""),
            "edge": item.get("edge", ""),
            "target": item.get("target", ""),
            "relation": item.get("relation", ""),
        })
    return compact


def evidence_score(item):
    try:
        return float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, default=str) + "\n")


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_id",
        "sample_id",
        "category",
        "view",
        "instruction",
        "question",
        "answer",
        "reason",
        "image_paths",
        "mask_paths",
        "overlay_paths",
        "object_ids",
        "evidence",
        "verification_verdict",
        "verification_score",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], default=str) if isinstance(row.get(key), (list, dict)) else row.get(key, "")
                for key in columns
            })


def is_object_instance(object_id):
    object_id = str(object_id)
    return any(object_id.startswith(f"{object_type}_") for object_type in OBJECT_TYPES)


def dedupe(items):
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def stable_id(row):
    payload = "|".join([
        row.get("sample_id", ""),
        row.get("view", ""),
        row.get("question", ""),
        row.get("answer", ""),
    ])
    return hashlib.sha1(payload.encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Create a multi-image multimodal dataset manifest.")
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, type=Path)
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT, type=Path)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, type=Path)
    parser.add_argument("--min-image-score", default=0.4, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main()
