"""Create a small HuggingFace-friendly multimodal dataset table.

Input: Dataset/verified_qa.jsonl
Output: Dataset/multimodal_multi_image_dataset.csv and .jsonl
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "Dataset" / "verified_qa.jsonl"
IMAGE_ROOT = ROOT / "build_objects" / "images"
CSV_OUTPUT = ROOT / "Dataset" / "multimodal_multi_image_dataset.csv"
MASK_OUTPUT_DIR = ROOT / "Dataset" / "masks"

INSTRUCTION = (
    "Answer the question with concise geological evidence. "
        "Reference specific objects using object tags. "
        "Insert one segmentation marker at the end of each region-specific evidence line. "
        "Use slot placeholders where numeric values, bounding boxes, or centers belong. "
        "Do not add facts unsupported by the image."
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
    # "onlap": ["onlap"],  # "onlap" commented out: aggregate/count evidence only for now
    "depositional": ["closure"],  # "lithology" commented out: broad/noisy visual evidence
    "full_mixed": ["fault", "salt", "closure"],  # "onlap" commented out
}
# Model-level (category-node) count/presence evidence -> the object type it describes.
# Only reached for NON-object-specific evidence, so a per-object property can never
# bleed across siblings of the same type.
EDGE_TYPES = {
    "number_faults": ["fault"],
    "fault_mode": ["fault"],
    "number_fault_intersections": ["fault"],
    "salt_inserted": ["salt"],
    "number_hc_closures": ["closure"],
    # "fluid" removed: per-closure property (object_id=closure_N), already routed by
    #   object_id; listing it here would bleed one closure's fluid onto every closure.
    # "intersects_fault" moved to CROSS_REFERENCE_EDGES below (it is object-specific,
    #   so it never actually reached this map).
    # "number_onlap_episodes": ["onlap"],
    # "number_fan_episodes": ["lithology"],
}

# A property on ONE object that implicates ANOTHER type: a closure's intersects_fault /
# intersects_salt. We can't know WHICH fault it meets, so the whole referenced class
# stands in. Consulted even for object-specific evidence (unlike EDGE_TYPES) -- safe
# because the referenced type differs from the object's own, so it can't bleed onto
# same-type siblings. The object keeps its own mask via the object_id match above.
CROSS_REFERENCE_EDGES = {
    "intersects_fault": ["fault"],
    "intersects_salt": ["salt"],
}


def main():
    rows = [build_row(item) for item in read_jsonl(INPUT)]
    # mask is optional now: object rows carry a mask, negative/featureless rows don't.
    rows = [row for row in rows if row and row["images"]]
    write_csv(rows, CSV_OUTPUT)


def build_row(item):
    sample_id = item.get("sample_id", "")
    view = item.get("view") or "inline"
    sample_dir = IMAGE_ROOT / sample_id
    # one shared scene image (all objects in the same section); the mask is built
    # below from only the objects this row retrieves, highlighted together.
    image_path, scene_objects = load_scene(sample_dir, view)
    if not image_path:
        return None

    evidences = compact_evidences(item.get("evidence", []))
    regions = []
    retrieved = []
    regions_box = ""

    # Individual scene objects (fault_0, closure_1, ...); used so object-specific
    # evidence only falls back to the type-global mask when its own mask is absent.
    individual_ids = frozenset(oid for oid in scene_objects if is_object_id(oid))

    for object_id, scene_object in scene_objects.items():
        object_type = scene_object.get("object_type", "")
        if object_type not in CLASS_IDS:            # keep only dataset object classes
            continue
        region = {"object_type": object_type, "object_id": object_id, "view": view}
        matching_evidences = [
            evidence for evidence in evidences
            if evidence_matches_region(evidence, region, individual_ids)
        ]
        if not matching_evidences:
            continue

        bbox = scene_object.get("bbox") or {}
        center = scene_object.get("center") or {}
        evidence_texts = "".join(
            f"{evidence.get('text', '')}.\n" for evidence in matching_evidences
        )
        regions_box += (
            "<region>\n"
            f"{evidence_texts}"
            "<SEG>\n"
            "</region>\n"
        )
        regions.append({
            "image_idx": 0,                          # single shared scene image
            "mask_idx": 0,                           # single composited row mask
            "region_idx": len(regions),
            "object_type": object_type,
            "view": view,
            "object_id": object_id,
            "class_id": scene_object.get("class_id", CLASS_IDS.get(object_type, 0)),
            "bbox": [bbox.get("x_min"), bbox.get("y_min"), bbox.get("x_max"), bbox.get("y_max")],
            "center": [center.get("x"), center.get("y")],
        })
        retrieved.append(scene_object)

    if not regions:
        # Featureless / negative example (e.g. "the section shows no faulting"): there
        # is no object to outline, but the scene image plus the section-level evidence
        # is still a valid VQA row. Emit it with no mask and no regions instead of
        # dropping it, so negatives reach the dataset.
        evidence_text = "".join(f"{evidence.get('text', '')}.\n" for evidence in evidences)
        return {
            "sample_id": sample_id,
            "images": [image_path],
            "masks": [],
            "instruction": INSTRUCTION,
            "question": f"{item.get('question', '')}",
            "reason": f'<think>{item.get("trace", {}).get("reason", "")}</think>',
            "answer": f'<answer>{item.get("answer", "")}</answer>',
            "evidence": evidence_text,
            "regions": [],
        }

    mask_path = build_row_mask(sample_dir, item, view, retrieved)
    if not mask_path:
        # regions matched but the composite is empty -- this is NOT a normal negative
        # (those have no regions). It means the matched objects' scene mask PNGs are
        # missing/stale on disk. Surface it instead of silently discarding a valid row.
        print(f"[MASK EMPTY] {sample_id} {view}: {len(regions)} region(s) matched but "
              f"no mask pixels on disk (stale scene_position.json?); row dropped")
        return None

    return {
        "sample_id":sample_id,
        "images": [image_path],
        "masks": [mask_path],
        "instruction": INSTRUCTION,
        "question": f"{item.get('question', '')}",
        "reason": f'<think>{item.get("trace", {}).get("reason", "")}</think>',
        "answer": f'<answer>{item.get("answer", "")}</answer>',
        "evidence": regions_box,
        "regions": regions,
    }


def build_row_mask(sample_dir, item, view, retrieved_objects):
    # Composite only the retrieved objects' scene-registered masks into one binary
    # white mask, so the row's single mask highlights exactly what the question pulled.
    combined = None
    for scene_object in sorted(retrieved_objects, key=_object_mask_area, reverse=True):
        mask = _read_object_mask(sample_dir, scene_object.get("mask_path"))
        if mask is None:
            continue
        if combined is None:
            combined = np.zeros(mask.shape, dtype=np.uint8)
        height = min(combined.shape[0], mask.shape[0])
        width = min(combined.shape[1], mask.shape[1])
        region = mask[:height, :width]
        combined[:height, :width][region] = 255

    if combined is None or not combined.any():
        return ""

    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row_id = item.get("row_id") or hashlib.sha1(
        f"{item.get('sample_id','')}|{item.get('question','')}|{item.get('answer','')}".encode()
    ).hexdigest()
    output_path = MASK_OUTPUT_DIR / f"{row_id}_{view}_mask.png"
    Image.fromarray(combined, mode="L").save(output_path)
    return output_path.as_posix()


def _read_object_mask(sample_dir, rel_path):
    if not rel_path:
        return None
    path = sample_dir / rel_path
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 0


def _object_mask_area(scene_object):
    bbox = scene_object.get("bbox") or {}
    if not bbox:
        return 0
    return (bbox.get("x_max", 0) - bbox.get("x_min", 0)) * (bbox.get("y_max", 0) - bbox.get("y_min", 0))


def load_scene(sample_dir, view):
    # Read the shared per-view scene: (image_path, {object_id: object}). Each object
    # carries its own scene-registered mask path, combined per row downstream.
    scene_path = sample_dir / "scene_position.json"
    if not scene_path.exists():
        return "", {}
    try:
        data = json.loads(scene_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "", {}

    view_block = (data.get("views") or {}).get(view)
    if not view_block:
        return "", {}

    image_path = _scene_file(sample_dir, view_block.get("image_path"))
    objects = {
        obj.get("object_id"): obj
        for obj in view_block.get("objects", [])
        if obj.get("object_id")
    }
    return image_path, objects


def _scene_file(sample_dir, rel_path):
    if not rel_path:
        return ""
    path = sample_dir / rel_path
    return path.as_posix() if path.exists() else ""


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


def evidence_matches_region(evidence, region, individual_ids=frozenset()):
    evidence_object_id = str(evidence.get("object_id") or "")
    region_object_id = str(region.get("object_id") or "")
    region_object_type = str(region.get("object_type") or "")

    if evidence_object_id == region_object_id:
        return True
    if evidence_object_id == region_object_type:
        return True
    if evidence.get("edge") == "HAS_VISUAL_OBJECT" and str(evidence.get("target")) in (region_object_id, region_object_type):
        # HAS_VISUAL_OBJECT's target is the object node id (e.g. "salt_0"), not the
        # bare type -- so it must match the region's object_id. Comparing only against
        # object_type left presence-only visual objects (salt, visual-only nodes)
        # unrouted, which dropped them from the mask and mislabeled "present" rows as
        # maskless negatives. Keeping object_type too preserves any type-level target.
        return True
    # Cross-reference: a closure's intersects_fault also highlights the fault class
    # (which specific fault is unknown). Checked BEFORE the object-specific branch, so
    # the closure keeps its own mask (via object_id above) AND the faults light up.
    if region_object_type in CROSS_REFERENCE_EDGES.get(evidence.get("edge"), []):
        return True
    if is_object_id(evidence_object_id):
        # Object-specific evidence (e.g. fault_0). Fall back to the type-global
        # region (object_id == object_type, e.g. "fault") ONLY when this object has
        # no individual mask in the scene -- so "tilt of fault 1" still lights up the
        # fault mask instead of nothing. Never bleed onto a different individual
        # (fault_0 evidence must not mask fault_1), and prefer the individual when it
        # exists (don't also drag in the all-faults global mask).
        if (not is_object_id(region_object_id)
                and evidence_object_id.startswith(f"{region_object_type}_")
                and evidence_object_id not in individual_ids):
            return True
        return False
    return True if region_object_type in EDGE_TYPES.get(evidence.get("edge"), []) else False


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


# An object id is "<type>_<index>" (fault_0, salt_3), NOT any string that merely starts
# with a type name. The category node is "<category> structure" (e.g. "fault_complex
# structure"), which startswith("fault_") and was being misread as a fault object -- so
# all model-level count/presence evidence for fault_/salt_/closure_ categories bypassed
# the EDGE_TYPES routing (branch 5) and landed on the wrong mask (or none).
_OBJECT_ID_RE = re.compile(r"^(?:%s)_\d+$" % "|".join(sorted(OBJECT_TYPES)))


def is_object_id(value):
    return bool(_OBJECT_ID_RE.match(str(value)))


def evidence_score(item):
    try:
        return float(item.get("score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
