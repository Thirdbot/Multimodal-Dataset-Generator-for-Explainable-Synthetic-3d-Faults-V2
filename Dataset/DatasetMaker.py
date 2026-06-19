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
    "Inspect the seismic images, use the marked regions as visual evidence, "
    "and answer the question with concise geological reasoning."
)

MIN_SCORE = 0.4
OBJECT_TYPES = {"fault", "closure", "salt", "onlap", "lithology", "age_depth"}

CLASS_IDS = {"fault": 1, "closure": 2, "salt": 3, "onlap": 4, "lithology": 5, "age_depth": 6}

CLASS_COLORS = {
    1: [255, 0, 0],
    2: [0, 128, 255],
    3: [180, 0, 255],
    4: [255, 220, 0],
    5: [0, 200, 100],
    6: [255, 140, 0],
}

CLASS_ID_COLORS = {
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
    "onlap": ["onlap", "closure"],
    "depositional": ["closure", "lithology"],
    "full_mixed": ["fault", "salt", "onlap", "closure"],
}
EDGE_TYPES = {
    "number_faults": ["fault"],
    "fault_mode": ["fault"],
    "number_fault_intersections": ["fault"],
    "salt_inserted": ["salt"],
    "number_hc_closures": ["closure"],
    "fluid": ["closure"],
    "number_onlap_episodes": ["onlap"],
    "number_fan_episodes": ["lithology"],
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

    for region in regions:
        matching_evidences = [
            evidence for evidence in evidences
            if evidence_matches_region(evidence, region)
        ]
        if not matching_evidences:
            continue

        object_name = region.get("object_type") or region.get("class_name") or ""
        class_id = region.get("class_id", "")
        class_color_name = region.get("class_color_name", "")
        bbox = region.get("bbox") or []
        center = region.get("center") or []
        evidence_text = "\n".join(
            f"<evidence>{evidence.get('text', '')}</evidence>"
            for evidence in matching_evidences
        )
        regions_box += (
            "<region>\n"
            f"<object>{object_name}</object>\n"
            f"<class_id>{class_id}</class_id>\n"
            f"<color>{class_color_name}</color>\n"
            f"{evidence_text}\n"
            f"<bbox>{json.dumps(bbox)}</bbox>\n"
            "<SEG>"
            "</region>\n"
        )

    return {
        "images": [image["image"] for image in image_items],
        "masks": [image["mask"] for image in image_items],
        "instruction": INSTRUCTION,
        "question": f"{'<image>'*len(image_items)}{item.get('question', '')}",
        "reason": f'<think>{item.get("trace", {}).get("reason", "")}</think>',
        "answer": f'<answer>{item.get("answer", "")}</answer>',
        "evidence": regions_box,
    }


def collect_image_items(sample_dir, view, category, evidence):
    items, seen = [], set()
    for evidence_item in evidence:
        if evidence_score(evidence_item) < MIN_SCORE:
            continue
        object_id = evidence_item.get("object_id") or evidence_item.get("source") or ""
        edge = evidence_item.get("edge") or evidence_item.get("fact_name") or ""
        for object_type, object_name, role in requested_objects(object_id, edge, category):
            add_image_item(items, seen, sample_dir, view, object_type, object_name, role)

    if not items:
        for object_type in CATEGORY_TYPES.get(category, []):
            if add_image_item(items, seen, sample_dir, view, object_type, object_type, "global"):
                break
    return items


def requested_objects(object_id, edge, category):
    if is_object_id(object_id):
        object_type = object_id.split("_", 1)[0]
        return [(object_type, object_id, "evidence"), (object_type, object_type, "context")]
    if object_id in OBJECT_TYPES:
        return [(object_id, object_id, "context")]
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
        "class_name": object_type,
        "class_color": CLASS_COLORS.get(class_id, [255, 255, 255]),
        "class_color_name": CLASS_ID_COLORS.get(class_id, "white"),
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
            "object_type": image_item["object_type"],
            "view": image_item["view"],
            "object_id":image_item["object_id"],
            "class_id": position.get("class_id", image_item["class_id"]),
            "class_name": position.get("class_name", image_item["class_name"]),
            "class_color": position.get("class_color", image_item["class_color"]),
            "class_color_name": CLASS_ID_COLORS.get(
                int(position.get("class_id", image_item["class_id"]) or 0),
                image_item.get("class_color_name", "white"),
            ),
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
    if is_object_id(evidence_object_id):
        return False
    if evidence_object_id.startswith("category:"):
        return region_object_type in EDGE_TYPES.get(evidence.get("edge"), [])
    return False


def write_csv(rows, path):
    columns = ["images", "masks", "instruction", "question", "reason", "answer", "evidence"]
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
