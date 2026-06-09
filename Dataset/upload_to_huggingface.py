import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Dataset" / "multimodal_verified_dataset.csv"


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
        if key in {"image", "overlay_image", "mask_image"}:
            cleaned[key] = resolve_image(value)
        elif key.endswith("_json"):
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


def normalize_json_string(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        return json.dumps(json.loads(value), ensure_ascii=False)
    except json.JSONDecodeError:
        return value


def build_dataset(rows):
    from datasets import Dataset, Features, Image, Value

    if not rows:
        raise ValueError("no rows to upload")

    features = {}
    for key in rows[0]:
        if key in {"image", "overlay_image", "mask_image"}:
            features[key] = Image()
        else:
            features[key] = Value("string")

    return Dataset.from_list(rows, features=Features(features))


def upload_dataset(csv_path, repo_id, private=False, token=None, limit=None):
    from huggingface_hub import HfApi

    rows = load_rows(csv_path, limit=limit)
    dataset = build_dataset(rows)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(repo_id, private=private, token=token)
    return {
        "repo_id": repo_id,
        "rows": len(rows),
        "private": private,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Upload multimodal CSV rows to Hugging Face with previewable image columns.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--repo-id", required=True, help="Example: username/synthetic-seismic-vlm")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--limit", type=int, default=None)
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
        ),
        indent=2,
    ))
