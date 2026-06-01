import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.visualize_sample import list_arrays, load_array


DEFAULT_VERIFIED = ROOT / "Dataset" / "verified_hypotheses.jsonl"
DEFAULT_OUTPUT = ROOT / "Dataset" / "multimodal_verified_dataset.csv"
DEFAULT_IMAGE_DIR = ROOT / "Dataset" / "multimodal_images"
OUTPUTS_ROOT = ROOT / "outputs"


def load_verified_rows(path):
    rows = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"verified dataset not found: {path}")

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def resolve_sample_folder(sample_id):
    sample_path = OUTPUTS_ROOT / sample_id
    if not sample_path.exists():
        raise FileNotFoundError(f"sample folder not found: {sample_path}")
    return sample_path


def select_seismic_item(sample_path):
    preferred_tokens = [
        "seismicCubes_RFC_fullstack",
        "seismic/seismicCubes_RFC__15_degrees_normalized",
        "seismic/seismicCubes_RFC__7_degrees_normalized",
        "seismic/seismicCubes_RFC__24_degrees_normalized",
        "seismicCubes_cumsum_fullstack",
    ]
    arrays = list_arrays(sample_path)

    for token in preferred_tokens:
        for item in arrays:
            rel = item["group_path"].relative_to(sample_path).as_posix()
            if token in rel:
                return item

    for item in arrays:
        rel = item["group_path"].relative_to(sample_path).as_posix().lower()
        if "seismic" in rel:
            return item

    raise FileNotFoundError(f"no seismic array found in {sample_path}")


def clip_normalize(slice_2d):
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
        if high <= low:
            return np.zeros_like(slice_2d, dtype=np.float32)

    clipped = np.clip(slice_2d, low, high)
    scaled = (clipped - low) / (high - low)
    return scaled.astype(np.float32)


def load_graph(path):
    path = Path(path)
    if not path.exists():
        return {"nodes": [], "edges": []}
    return json.loads(path.read_text())


def choose_target_closure(graph):
    closures = [node for node in graph.get("nodes", []) if node.get("label") == "Closure"]
    if not closures:
        return None

    def closure_rank(node):
        fluid = str(node.get("fluid", "")).lower()
        fluid_rank = 0 if fluid in {"oil", "gas"} else 1
        return (fluid_rank, -int(node.get("n_voxels", 0) or 0))

    closures = sorted(closures, key=closure_rank)
    return closures[0]


def closure_indices(closure, shape):
    if not closure:
        return None

    def center(min_key, max_key, axis_len):
        min_val = closure.get(min_key)
        max_val = closure.get(max_key)
        if min_val is None or max_val is None:
            return None
        value = int(round((float(min_val) + float(max_val)) / 2.0))
        return int(np.clip(value, 0, axis_len - 1))

    return {
        "inline": center("x_min", "x_max", shape[0]),
        "crossline": center("y_min", "y_max", shape[1]),
        "timeslice": center("z_min", "z_max", shape[2]),
        "source": "closure_extent",
        "source_id": closure.get("id"),
        "fluid": closure.get("fluid"),
    }


def choose_fault_heuristic(graph, shape):
    faults = [node for node in graph.get("nodes", []) if node.get("label") == "Fault"]
    if not faults:
        return {}

    faults = sorted(faults, key=lambda node: float(node.get("throw", 0) or 0), reverse=True)
    fault = faults[0]

    def bounded_index(raw, axis_len):
        if raw is None:
            return None
        raw = float(raw)
        if 0 <= raw < axis_len:
            return int(round(raw))
        shifted = raw + (axis_len / 2.0)
        if 0 <= shifted < axis_len:
            return int(round(shifted))
        return None

    return {
        "inline": bounded_index(fault.get("x0"), shape[0]),
        "crossline": bounded_index(fault.get("y0"), shape[1]),
        "source": "fault_heuristic",
        "source_id": fault.get("id"),
        "fault_index": fault.get("fault_index"),
    }


def select_view_indices(volume, graph=None):
    amplitude = np.abs(np.nan_to_num(volume, nan=0.0))
    inline_scores = amplitude.mean(axis=(1, 2))
    crossline_scores = amplitude.mean(axis=(0, 2))
    timeslice_scores = amplitude.mean(axis=(0, 1))

    graph = graph or {"nodes": [], "edges": []}
    selected = {}
    methods = {}

    closure_choice = closure_indices(choose_target_closure(graph), volume.shape)
    if closure_choice:
        if closure_choice.get("inline") is not None:
            selected["inline"] = closure_choice["inline"]
            methods["inline"] = closure_choice["source"]
        if closure_choice.get("crossline") is not None:
            selected["crossline"] = closure_choice["crossline"]
            methods["crossline"] = closure_choice["source"]
        if closure_choice.get("timeslice") is not None:
            selected["timeslice"] = closure_choice["timeslice"]
            methods["timeslice"] = closure_choice["source"]

    fault_choice = choose_fault_heuristic(graph, volume.shape)
    if "inline" not in selected and fault_choice.get("inline") is not None:
        selected["inline"] = fault_choice["inline"]
        methods["inline"] = fault_choice["source"]
    if "crossline" not in selected and fault_choice.get("crossline") is not None:
        selected["crossline"] = fault_choice["crossline"]
        methods["crossline"] = fault_choice["source"]

    if "inline" not in selected:
        selected["inline"] = int(np.argmax(inline_scores))
        methods["inline"] = "amplitude_max"
    if "crossline" not in selected:
        selected["crossline"] = int(np.argmax(crossline_scores))
        methods["crossline"] = "amplitude_max"
    if "timeslice" not in selected:
        selected["timeslice"] = int(np.argmax(timeslice_scores))
        methods["timeslice"] = "amplitude_max"

    return {
        "inline": selected["inline"],
        "crossline": selected["crossline"],
        "timeslice": selected["timeslice"],
        "methods": methods,
    }


def build_views(volume, graph=None):
    indices = select_view_indices(volume, graph=graph)
    inline = clip_normalize(volume[indices["inline"], :, :].T)
    crossline = clip_normalize(volume[:, indices["crossline"], :].T)
    timeslice = clip_normalize(volume[:, :, indices["timeslice"]].T)
    return {
        "inline": inline,
        "crossline": crossline,
        "timeslice": timeslice,
        "indices": indices,
    }


def render_panel(sample_id, seismic_name, views, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=180)
    fig.patch.set_facecolor("black")

    panel_defs = [
        (
            "Inline",
            views["inline"],
            f"i={views['indices']['inline']} [{views['indices']['methods'].get('inline', 'unknown')}]",
        ),
        (
            "Crossline",
            views["crossline"],
            f"x={views['indices']['crossline']} [{views['indices']['methods'].get('crossline', 'unknown')}]",
        ),
        (
            "Timeslice",
            views["timeslice"],
            f"t={views['indices']['timeslice']} [{views['indices']['methods'].get('timeslice', 'unknown')}]",
        ),
    ]

    for ax, (title, image, subtitle) in zip(axes, panel_defs):
        ax.imshow(image, cmap="gray", aspect="auto", origin="lower")
        ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    fig.suptitle(f"{sample_id}\n{seismic_name}", color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def multimodal_row(source_row, image_path, seismic_relpath, slice_indices):
    instruction = source_row["instruction"]
    answer = source_row["answer"]
    sample_id = source_row["sample_id"]

    return {
        "id": sample_id,
        "sample_id": sample_id,
        "image": image_path.as_posix(),
        "instruction": instruction,
        "input": "",
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_path.as_posix()},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ],
        "metadata": {
            **source_row.get("metadata", {}),
            "source_verified_sample_id": sample_id,
            "seismic_array": seismic_relpath,
            "view_indices": slice_indices,
            "render_type": "three_view_seismic_panel",
            "view_selection": slice_indices.get("methods", {}),
        },
        "evidence": source_row.get("evidence", []),
        "verification": source_row.get("verification", {}),
    }


def export_dataset(verified_path, output_path, image_dir, limit=None):
    rows = load_verified_rows(verified_path)
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    skipped = []

    source_rows = rows[:limit] if limit else rows
    for row in source_rows:
        sample_id = row["sample_id"]
        try:
            sample_path = resolve_sample_folder(sample_id)
            seismic_item = select_seismic_item(sample_path)
            volume = load_array(seismic_item)
            graph_path = row.get("metadata", {}).get("graph_path")
            graph = load_graph(graph_path) if graph_path else {"nodes": [], "edges": []}
            views = build_views(volume, graph=graph)

            image_path = image_dir / f"{sample_id}.png"
            seismic_relpath = seismic_item["group_path"].relative_to(sample_path).as_posix()
            render_panel(
                sample_id=sample_id,
                seismic_name=seismic_relpath,
                views=views,
                output_path=image_path,
            )

            exported.append(
                multimodal_row(
                    source_row=row,
                    image_path=image_path,
                    seismic_relpath=seismic_relpath,
                    slice_indices=views["indices"],
                )
            )
        except Exception as exc:
            skipped.append({
                "sample_id": sample_id,
                "error": f"{exc.__class__.__name__}: {exc}",
            })

    if output_path.suffix.lower() == ".jsonl":
        with open(output_path, "w") as file:
            for row in exported:
                file.write(json.dumps(row) + "\n")
    else:
        write_csv(output_path, exported)

    return {
        "verified_rows": len(rows),
        "exported_rows": len(exported),
        "skipped_rows": len(skipped),
        "output_path": output_path.as_posix(),
        "image_dir": image_dir.as_posix(),
        "skipped": skipped,
    }


def write_csv(output_path, rows):
    fieldnames = [
        "id",
        "sample_id",
        "image",
        "instruction",
        "input",
        "answer",
        "metadata_json",
        "evidence_json",
        "verification_json",
        "messages_json",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "sample_id": row["sample_id"],
                "image": row["image"],
                "instruction": row["instruction"],
                "input": row["input"],
                "answer": row["answer"],
                "metadata_json": json.dumps(row.get("metadata", {})),
                "evidence_json": json.dumps(row.get("evidence", [])),
                "verification_json": json.dumps(row.get("verification", {})),
                "messages_json": json.dumps(row.get("messages", [])),
            })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert verified_hypotheses.jsonl into a 2D multimodal instruction-tuning dataset."
    )
    parser.add_argument("--verified", default=str(DEFAULT_VERIFIED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = export_dataset(
        verified_path=args.verified,
        output_path=args.output,
        image_dir=args.image_dir,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
