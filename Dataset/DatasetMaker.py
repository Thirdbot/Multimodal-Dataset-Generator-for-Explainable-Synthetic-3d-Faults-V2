"""Create a small HuggingFace-friendly multimodal dataset table.

Input: Dataset/verified_qa.jsonl
Output: Dataset/multimodal_multi_image_dataset.csv and .jsonl
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "Dataset" / "verified_qa.jsonl"
IMAGE_ROOT = ROOT / "build_objects" / "images"
CSV_OUTPUT = ROOT / "Dataset" / "multimodal_multi_image_dataset.csv"

INSTRUCTION = (
    "Inspect all provided seismic images and their paired masks. Answer the question "
    "from visible geological evidence only. For each supporting region, write one "
    "<region> block containing the observed object, its class_id, the evidence used, "
    "its <bbox>[x_min, y_min, x_max, y_max]</bbox>, and one <SEG> token that refers "
    "to the paired mask. Then give the final response inside <answer>...</answer>. "
    "Use concise geological wording and do not add facts that are not supported by "
    "the shown regions."
)

# Visual-object policy notes:
# - fault and closure are the strongest object-grounded dataset targets.
# - salt can be useful as aggregate visual context.
# - onlap should usually stay aggregate/count-based, not many local components.
# - lithology is broad and can over-expand rows; remove it from these maps if it
#   starts dominating evidence/image selection.
OBJECT_TYPES = {
    "fault",
    "closure",
    "salt",
    # "onlap",  # broad visual context; keep out of object-level dataset rows for now
    # "lithology",  # broad volume; too noisy for current region-grounded rows
}

CLASS_IDS = {
    "fault": 1,
    "closure": 2,
    "salt": 3,
    # "onlap": 4,
    # "lithology": 5,
}

CLASS_COLORS = {
    1: "red",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "green",
    6: "orange",
}


CATEGORY_TYPES = {
    "boring": ["closure"],
    "fault_only": ["fault"],
    "fault_complex": ["fault", "closure"],
    "salt_only": ["salt", "closure"],
    "salt_fault_mixed": ["fault", "salt", "closure"],
    "onlap": ["closure"],  # "onlap" commented out: aggregate/count evidence only for now
    "depositional": ["closure"],  # "lithology" commented out: broad/noisy visual evidence
    "full_mixed": ["fault", "salt", "closure"],  # "onlap" commented out
}
EDGE_TYPES = {
    "number_faults": ["fault"],
    "fault_mode": ["fault"],
    "number_fault_intersections": ["fault"],
    "salt_inserted": ["salt"],
    "number_hc_closures": ["closure"],
    "fluid": ["closure"],
    # "number_onlap_episodes": ["onlap"],
    # "number_fan_episodes": ["lithology"],
}


def main():
    rows = [build_row(item) for item in read_jsonl(INPUT)]
    rows = [row for row in rows if row and row["images"] and row["masks"]]
    write_csv(rows, CSV_OUTPUT)


def build_row(item):
    sample_id = item.get("sample_id", "")
    view = item.get("view") or "inline"
    sample_dir = IMAGE_ROOT / sample_id
    image_items = collect_image_items(sample_dir, view, item.get("category", ""), item.get("evidence", []))
    regions = collect_regions(sample_dir, image_items)
    evidences = compact_evidences(item.get("evidence", []))
    regions_box = ""
    used_indices = []

    evidence_text = ""
    for region in regions:
        matching_evidences = [
            evidence for evidence in evidences
            if evidence_matches_region(evidence, region)
        ]
        if not matching_evidences:
            continue

        image_idx = region.get("image_idx")
        if image_idx is None:
            continue
        used_indices.append(image_idx)

        evidence_texts = evidence_text.join(
            f"{evidence.get('text', '')}.\n"
            for evidence in matching_evidences
        )
        regions_box += (
            "<region>\n"
            f"{evidence_texts}"
            "<SEG>\n"
            "</region>\n"
        )

    used_indices = sorted(set(used_indices))
    used_image_items = [image_items[index] for index in used_indices]
    used_regions = [
        {**region, "image_idx": new_index, "mask_idx": new_index, "region_idx": new_index}
        for new_index, region in enumerate(regions)
        if region.get("image_idx") in used_indices
    ]

    return {
        "sample_id":sample_id,
        "images": [image["image"] for image in used_image_items],
        "masks": [image["mask"] for image in used_image_items],
        "instruction": INSTRUCTION,
        "question": f"{item.get('question', '')}",
        "reason": f'<think>{item.get("trace", {}).get("reason", "")}</think>',
        "answer": f'<answer>{item.get("answer", "")}</answer>',
        "evidence": regions_box,
        "regions": used_regions,
    }


def collect_image_items(sample_dir, view, category, evidence):
    items, seen = [], set()
    for evidence_item in evidence:
        object_id = evidence_item.get("object_id") or evidence_item.get("source") or ""
        edge = evidence_item.get("edge") or evidence_item.get("fact_name") or ""
        target = evidence_item.get("target") or ""
        for object_type, object_name, role in requested_objects(object_id, edge, target, category):
            add_image_item(items, seen, sample_dir, view, object_type, object_name, role)

    return items


def requested_objects(object_id, edge, target, category):
    if is_object_id(object_id):
        object_type = object_id.split("_", 1)[0]
        return [(object_type, object_id, "evidence")]
    if object_id in OBJECT_TYPES:
        return [(object_id, object_id, "context")]
    if edge == "HAS_VISUAL_OBJECT" and target in OBJECT_TYPES:
        return [(target, target, "evidence")]
    if str(object_id).startswith("category:"):
        return [(object_type, object_type, "context") for object_type in EDGE_TYPES.get(edge, CATEGORY_TYPES.get(category, []))]
    return []


def add_image_item(items, seen, sample_dir, view, object_type, object_id, role):
    image = sample_dir / object_type / object_id / f"{view}.png"
    mask = sample_dir / object_type / object_id / f"{view}_mask.png"
    key = (object_type, object_id, view)
    if key in seen:
        return True
    if not image.exists() or not mask.exists():
        return False
    seen.add(key)
    class_id = CLASS_IDS.get(object_type, 0)
    items.append({
        "object_type": object_type,
        "object_id": object_id,
        "view": view,
        "role": role,
        "class_id": class_id,
        "image": image.as_posix(),
        "mask": mask.as_posix(),
    })
    return True


def collect_regions(sample_dir, image_items):
    positions = load_positions(sample_dir)
    regions = []
    for index, image_item in enumerate(image_items):
        position = positions.get((image_item["object_type"], image_item["object_id"], image_item["view"]))
        if not position:
            continue
        bbox = position.get("bbox") or {}
        center = position.get("center") or {}
        regions.append({
            "image_idx":index,
            "mask_idx":index,
            "region_idx":index,
            "object_type": image_item["object_type"],
            "view": image_item["view"],
            "object_id":image_item["object_id"],
            "class_id": position.get("class_id", image_item["class_id"]),
            "bbox": [bbox.get("x_min"), bbox.get("y_min"), bbox.get("x_max"), bbox.get("y_max")],
            "center": [center.get("x"), center.get("y")],
        })
    return regions


def load_positions(sample_dir):
    positions = {}
    for path in sample_dir.glob("*_object_position.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for item in data.get("objects", []):
            key = (item.get("object_type"), item.get("object_id"), item.get("view"))
            positions[key] = item
    return positions


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compact_evidences(evidence):
    output = []
    for item in evidence:
        output.append({
            "object_id": item.get("object_id") or item.get("source", ""),
            "text": item.get("text") or item.get("page_content") or "",
            "edge": item.get("edge", ""),
            "target": item.get("target", ""),
        })
    return output


def evidence_matches_region(evidence, region):
    evidence_object_id = str(evidence.get("object_id") or "")
    region_object_id = str(region.get("object_id") or "")
    region_object_type = str(region.get("object_type") or "")

    if evidence_object_id == region_object_id:
        return True
    if evidence_object_id == region_object_type:
        return True
    if evidence.get("edge") == "HAS_VISUAL_OBJECT" and str(evidence.get("target")) == region_object_type:
        return True
    if is_object_id(evidence_object_id):
        return False
    if evidence_object_id.startswith("category:"):
        return region_object_type in EDGE_TYPES.get(evidence.get("edge"), [])
    return False


def write_csv(rows, path):
    columns = ["sample_id","images", "masks", "instruction", "question", "answer", "evidence","reason","regions"] # "reason" when there is actually reason
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list) else row[key]
                for key in columns
            })


def is_object_id(value):
    return any(str(value).startswith(f"{object_type}_") for object_type in OBJECT_TYPES)


def evidence_score(item):
    try:
        return float(item.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
