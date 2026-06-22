import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Dataset" / "multimodal_multi_image_dataset.csv"
IMAGE_LIST_COLUMNS = {"images", "masks"}
JSON_COLUMNS = {"regions", "evidence"}


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

    for key, value in row.items():
        if key in IMAGE_LIST_COLUMNS:
            cleaned[key] = resolve_image_list(value)
        elif key in JSON_COLUMNS or key.endswith("_json"):
            cleaned[key] = normalize_json_string(value)
        else:
            cleaned[key] = "" if value is None else str(value)
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


def resolve_image_list(value):
    return [
        resolved
        for resolved in (resolve_image(item) for item in parse_json_list(value))
        if resolved
    ]


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

    columns = [
        "images", "masks",
        "instruction", "question", "answer",
        "evidence","reason","regions"
    ]
    rows = [
        {key: row.get(key, [] if key in IMAGE_LIST_COLUMNS else "") for key in columns}
        for row in rows
    ]

    features = {}
    for key in columns:
        if key in IMAGE_LIST_COLUMNS:
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

This dataset contains synthetic seismic multimodal QA rows with raw seismic
images, segmentation masks, evidence-grounded questions, answers, and compact
region metadata.

Rows: {len(rows)}

Repository: https://huggingface.co/datasets/{repo_id}

Columns:

- `images`: sequence of all raw images
- `masks`: sequence of all mask images
- `instruction`: task instruction
- `question`: question text
- `reason`: optional reasoning/description text
- `answer`: answer text
- `evidence`: JSON string of supporting text evidence
- `regions`: JSON string of bbox/class/color region metadata

"""

if __name__ == "__main__":
    upload_dataset(DEFAULT_CSV, "thirdExec/synthetic-seismic-vlm", private=False)
