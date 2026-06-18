import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Dataset" / "fcot_seismic_dataset.csv"
IMAGE_LIST_COLUMNS = {"image_paths", "mask_paths", "overlay_paths", "images", "mask_images", "overlay_images"}
JSON_COLUMNS = {"regions", "region_context", "region_bboxes", "object_ids", "evidence", "verification"}
PREVIEW_IMAGE_COLUMNS = {"primary_image", "primary_mask", "primary_overlay", "image", "mask_image", "overlay_image"}


def load_rows(csv_path, limit=None):
    rows = []
    with open(csv_path, newline="") as file:
        for row in csv.DictReader(file):
            rows.append(clean_row(row))
            if limit and len(rows) >= limit:
                break
    return rows


def clean_row(row):
    cleaned = {}
    image_paths = parse_json_list(row.get("image_paths"))
    mask_paths = parse_json_list(row.get("mask_paths"))
    overlay_paths = parse_json_list(row.get("overlay_paths"))

    for key, value in row.items():
        if key in {"image", "mask_image", "overlay_image"}:
            cleaned[key] = resolve_image(value)
        elif key in IMAGE_LIST_COLUMNS:
            cleaned[key] = resolve_image_list(value)
        elif key in JSON_COLUMNS or key.endswith("_json"):
            cleaned[key] = normalize_json_string(value)
        else:
            cleaned[key] = "" if value is None else str(value)

    # Wide multimodal CSVs use list columns; FCoT rows already have direct image columns.
    if image_paths and "primary_image" not in cleaned:
        cleaned["primary_image"] = resolve_image(first_item(image_paths))
        cleaned["primary_mask"] = resolve_image(first_item(mask_paths))
        cleaned["primary_overlay"] = resolve_image(first_item(overlay_paths))
    return cleaned


def resolve_image(value):
    value = str(value or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    return path.as_posix()


def resolve_image_list(value):
    return [
        resolved
        for resolved in (resolve_image(item) for item in parse_json_list(value))
        if resolved
    ]


def parse_json_list(value):
    value = str(value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def first_item(items):
    return items[0] if items else ""


def normalize_json_string(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        return json.dumps(json.loads(value), ensure_ascii=False)
    except json.JSONDecodeError:
        return value


def build_dataset(rows):
    from datasets import Dataset, Features, Image, Sequence, Value

    if not rows:
        raise ValueError("no rows to upload")

    features = {}
    for key in rows[0]:
        if key in PREVIEW_IMAGE_COLUMNS:
            features[key] = Image()
        elif key in IMAGE_LIST_COLUMNS:
            features[key] = Sequence(Image())
        else:
            features[key] = Value("string")

    return Dataset.from_list(rows, features=Features(features))


def upload_dataset(csv_path, repo_id, private=False, token=None, limit=None, dry_run=False):
    from huggingface_hub import HfApi

    rows = load_rows(csv_path, limit=limit)
    dataset = build_dataset(rows)

    if dry_run:
        return {
            "repo_id": repo_id,
            "rows": len(rows),
            "private": private,
            "columns": dataset.column_names,
            "dry_run": True,
        }

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(repo_id, private=private, token=token)
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=dataset_card(repo_id, rows).encode("utf-8"),
        path_in_repo="README.md",
        token=token,
    )
    return {
        "repo_id": repo_id,
        "rows": len(rows),
        "private": private,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


def dataset_card(repo_id, rows):
    return f"""---
license: mit
task_categories:
- visual-question-answering
- image-segmentation
- image-to-text
language:
- en
size_categories:
- n<1K
---

# Synthetic Seismic VLM

This dataset contains synthetic seismic multimodal QA rows with raw image paths,
segmentation masks, region-grounded reasoning, answer text, and compact metadata.

Rows: {len(rows)}

Repository: https://huggingface.co/datasets/{repo_id}

Common preview columns are:

- `image`
- `mask_image`
- `overlay_image`

When uploading the wider audit CSV, the full multi-image columns are:

- `image_paths`
- `mask_paths`
- `overlay_paths`

"""


def parse_args():
    parser = argparse.ArgumentParser(description="Upload multimodal CSV rows to Hugging Face with previewable image columns.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--repo-id", required=True, help="Example: username/synthetic-seismic-vlm")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset construction without uploading.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(
        upload_dataset(
            csv_path=args.csv,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
            limit=args.limit,
            dry_run=args.dry_run,
        ),
        indent=2,
    ))
