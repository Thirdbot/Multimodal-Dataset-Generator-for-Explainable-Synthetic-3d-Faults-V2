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


def select_fault_segments_item(sample_path):
    arrays = list_arrays(sample_path)
    for item in arrays:
        rel = item["group_path"].relative_to(sample_path).as_posix()
        if rel.startswith("fault_segments_") and rel.endswith(".zarr"):
            return item
    return None


def select_first_existing(sample_path, candidates):
    arrays = list_arrays(sample_path)
    for candidate in candidates:
        for item in arrays:
            rel = item["group_path"].relative_to(sample_path).as_posix()
            if rel == candidate or rel.endswith(candidate):
                return item
    return None


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
        "closure": closure,
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
        "timeslice": None,
        "source": "fault_heuristic",
        "source_id": fault.get("id"),
        "fault_index": fault.get("fault_index"),
        "fault": fault,
    }


def target_volume_indices(target_volume, source_name):
    occupancy = np.nan_to_num(target_volume, nan=0.0)
    inline_scores = occupancy.sum(axis=(1, 2))
    crossline_scores = occupancy.sum(axis=(0, 2))
    timeslice_scores = occupancy.sum(axis=(0, 1))

    if inline_scores.max() <= 0 and crossline_scores.max() <= 0 and timeslice_scores.max() <= 0:
        return None

    return {
        "inline": int(np.argmax(inline_scores)),
        "crossline": int(np.argmax(crossline_scores)),
        "timeslice": int(np.argmax(timeslice_scores)),
        "source": source_name,
    }


def clamp_index(index, axis_len):
    return int(np.clip(int(index), 0, axis_len - 1))


def select_view_indices(volume, graph=None, target_volume=None, target_source="target_mask"):
    amplitude = np.abs(np.nan_to_num(volume, nan=0.0))
    inline_scores = amplitude.mean(axis=(1, 2))
    crossline_scores = amplitude.mean(axis=(0, 2))
    timeslice_scores = amplitude.mean(axis=(0, 1))

    graph = graph or {"nodes": [], "edges": []}
    selected = {}
    methods = {}

    fault_choice = choose_fault_heuristic(graph, volume.shape)
    target_choice = target_volume_indices(target_volume, target_source) if target_volume is not None else None

    if target_choice:
        selected["inline"] = clamp_index(target_choice["inline"], volume.shape[0])
        methods["inline"] = target_choice["source"]
        selected["crossline"] = clamp_index(target_choice["crossline"], volume.shape[1])
        methods["crossline"] = target_choice["source"]
        selected["timeslice"] = clamp_index(target_choice["timeslice"], volume.shape[2])
        methods["timeslice"] = target_choice["source"]

    if "inline" not in selected and fault_choice.get("inline") is not None:
        selected["inline"] = fault_choice["inline"]
        methods["inline"] = fault_choice["source"]
    if "crossline" not in selected and fault_choice.get("crossline") is not None:
        selected["crossline"] = fault_choice["crossline"]
        methods["crossline"] = fault_choice["source"]

    closure_choice = closure_indices(choose_target_closure(graph), volume.shape)
    if closure_choice:
        if "inline" not in selected and closure_choice.get("inline") is not None:
            selected["inline"] = closure_choice["inline"]
            methods["inline"] = closure_choice["source"]
        if "crossline" not in selected and closure_choice.get("crossline") is not None:
            selected["crossline"] = closure_choice["crossline"]
            methods["crossline"] = closure_choice["source"]
        if "timeslice" not in selected and closure_choice.get("timeslice") is not None:
            selected["timeslice"] = closure_choice["timeslice"]
            methods["timeslice"] = closure_choice["source"]

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
        "target_volume_used": bool(target_choice),
        "target_closure": closure_choice.get("closure") if closure_choice else None,
        "target_fault": fault_choice.get("fault") if fault_choice else None,
    }


def build_views(volume, graph=None, target_volume=None, target_source="target_mask"):
    indices = select_view_indices(
        volume,
        graph=graph,
        target_volume=target_volume,
        target_source=target_source,
    )
    inline = clip_normalize(volume[indices["inline"], :, :].T)
    crossline = clip_normalize(volume[:, indices["crossline"], :].T)
    timeslice = clip_normalize(volume[:, :, indices["timeslice"]].T)
    return {
        "inline": inline,
        "crossline": crossline,
        "timeslice": timeslice,
        "indices": indices,
    }


def build_overlay_views(mask_volume, indices):
    binary = np.nan_to_num(mask_volume, nan=0.0)
    inline_idx = clamp_index(indices["inline"], binary.shape[0])
    crossline_idx = clamp_index(indices["crossline"], binary.shape[1])
    timeslice_idx = clamp_index(indices["timeslice"], binary.shape[2])
    inline = (binary[inline_idx, :, :].T > 0).astype(np.float32)
    crossline = (binary[:, crossline_idx, :].T > 0).astype(np.float32)
    timeslice = (binary[:, :, timeslice_idx].T > 0).astype(np.float32)
    return {
        "inline": inline,
        "crossline": crossline,
        "timeslice": timeslice,
    }


def align_overlay_mask(overlay_mask, target_shape):
    target_h, target_w = target_shape
    src_h, src_w = overlay_mask.shape

    aligned = np.zeros((target_h, target_w), dtype=np.float32)
    copy_h = min(target_h, src_h)
    copy_w = min(target_w, src_w)
    aligned[:copy_h, :copy_w] = overlay_mask[:copy_h, :copy_w]
    return aligned


def choose_overlay_source(sample_path, source_row, graph, fault_segments_item, task="structural_interpretation"):
    evidence = source_row.get("evidence", [])
    answer = str(source_row.get("answer", "")).lower()
    fact_names = [str(item.get("fact_name", "")) for item in evidence]

    if task == "fault_detection":
        if fault_segments_item is None:
            return None
        return {
            "kind": "fault",
            "item": fault_segments_item,
            "color": "#ef4444",
        }

    target_closure = None
    for node in graph.get("nodes", []):
        if node.get("label") == "Closure":
            target_closure = node
            break

    # Answer topic should dominate overlay choice.
    if "fault" in answer and fault_segments_item is not None:
        return {
            "kind": "fault",
            "item": fault_segments_item,
            "color": "#ef4444",
        }

    if any(token in answer for token in ("closure", "oil", "gas", "brine")) and target_closure:
        fluid = str(target_closure.get("fluid", "")).lower()
        fluid_item = select_first_existing(sample_path, [f"closures/{fluid}.zarr"]) if fluid else None
        if fluid_item:
            return {
                "kind": f"closure_{fluid}",
                "item": fluid_item,
                "color": overlay_color_for_kind(fluid),
            }

    fault_priority = (
        fault_segments_item is not None and
        any(name in {"Fault", "HAS_FAULT", "FaultSystem"} for name in fact_names)
    )
    if fault_priority:
        return {
            "kind": "fault",
            "item": fault_segments_item,
            "color": "#ef4444",
        }

    if target_closure:
        fluid = str(target_closure.get("fluid", "")).lower()
        fluid_item = select_first_existing(sample_path, [f"closures/{fluid}.zarr"]) if fluid else None
        if fluid_item:
            return {
                "kind": f"closure_{fluid}",
                "item": fluid_item,
                "color": overlay_color_for_kind(fluid),
            }

    hc_item = select_first_existing(sample_path, ["closures/hc_labels.zarr", "all_closure_segments"])
    if hc_item:
        return {
            "kind": "closure",
            "item": hc_item,
            "color": "#f59e0b",
        }

    if fault_segments_item is not None:
        return {
            "kind": "fault",
            "item": fault_segments_item,
            "color": "#ef4444",
        }

    return None


def overlay_color_for_kind(kind):
    return {
        "oil": "#f59e0b",
        "gas": "#22c55e",
        "brine": "#38bdf8",
    }.get(str(kind).lower(), "#ef4444")


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


def render_overlay_panel(sample_id, seismic_name, views, overlay_views, overlay_kind, overlay_color, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=180)
    fig.patch.set_facecolor("black")

    panel_defs = [
        (
            "Inline",
            views["inline"],
            overlay_views["inline"],
            f"i={views['indices']['inline']} [{views['indices']['methods'].get('inline', 'unknown')}]",
        ),
        (
            "Crossline",
            views["crossline"],
            overlay_views["crossline"],
            f"x={views['indices']['crossline']} [{views['indices']['methods'].get('crossline', 'unknown')}]",
        ),
        (
            "Timeslice",
            views["timeslice"],
            overlay_views["timeslice"],
            f"t={views['indices']['timeslice']} [{views['indices']['methods'].get('timeslice', 'unknown')}]",
        ),
    ]

    rgb = np.array(plt.matplotlib.colors.to_rgb(overlay_color), dtype=np.float32)
    for ax, (title, image, overlay_mask, subtitle) in zip(axes, panel_defs):
        overlay_mask = align_overlay_mask(overlay_mask, image.shape)
        base_rgb = np.dstack([image, image, image])
        alpha = overlay_mask[..., None] * 0.55
        overlay_rgb = np.ones_like(base_rgb) * rgb
        composite = base_rgb * (1.0 - alpha) + overlay_rgb * alpha
        ax.imshow(composite, aspect="auto", origin="lower")
        ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    fig.suptitle(f"{sample_id}\n{seismic_name} + {overlay_kind}", color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_mask_panel(sample_id, overlay_views, overlay_kind, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=180)
    fig.patch.set_facecolor("black")
    panel_defs = [
        ("Inline", overlay_views["inline"]),
        ("Crossline", overlay_views["crossline"]),
        ("Timeslice", overlay_views["timeslice"]),
    ]

    for ax, (title, image) in zip(axes, panel_defs):
        ax.imshow(image, cmap="gray", aspect="auto", origin="lower")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    fig.suptitle(f"{sample_id}\n{overlay_kind} mask", color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def multimodal_row(source_row, image_path, overlay_image_path, mask_image_path, seismic_relpath, overlay_relpath, overlay_kind, slice_indices):
    instruction = source_row["instruction"]
    answer = source_row["answer"]
    sample_id = source_row["sample_id"]
    trace = source_row.get("trace", {})
    verification = source_row.get("verification", {})
    deciding_evidence = trace.get("deciding_evidence") or {}
    if not deciding_evidence and source_row.get("evidence"):
        deciding_evidence = source_row["evidence"][0]

    return {
        "id": sample_id,
        "sample_id": sample_id,
        "instruction": instruction,
        "input": "",
        "answer": answer,
        "image": image_path.as_posix(),
        "overlay_image": overlay_image_path.as_posix() if overlay_image_path else "",
        "mask_image": mask_image_path.as_posix() if mask_image_path else "",
        "overlay_kind": overlay_kind or "",
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
            "graph_path": source_row.get("metadata", {}).get("graph_path", ""),
            "category": source_row.get("metadata", {}).get("category", ""),
            "seismic_array": seismic_relpath,
            "overlay_array": overlay_relpath or "",
            "view_indices": slice_indices,
            "render_type": "three_view_seismic_panel",
        },
        "trace": {
            "graph_trace": trace.get("graph_trace", source_row.get("evidence", [])),
            "llm_prompt": trace.get("llm_prompt", ""),
            "llm_raw_output": trace.get("llm_raw_output", ""),
            "llm_answer": trace.get("llm_answer", answer),
            "nli_status": verification.get("status", ""),
            "nli_score": verification.get("score", ""),
            "nli_model": trace.get("nli_model", ""),
            "nli_deciding_evidence": deciding_evidence,
            "nli_retrieved_evidence": trace.get("retrieved_evidence", source_row.get("evidence", [])),
        },
    }


def is_exportable_answer(text):
    text = str(text or "").strip()
    if len(text.split()) < 6:
        return False
    if text[-1:] not in {".", "!", "?"}:
        return False
    lowered = text.lower()
    bad_endings = (
        "because",
        "due to",
        "indicating",
        "showing",
        "suggesting",
        "with",
        "including",
    )
    if any(lowered.endswith(f" {ending}.") or lowered == f"{ending}." for ending in bad_endings):
        return False
    return True


def is_no_fault_answer(text):
    text = str(text or "").lower()
    no_fault_phrases = (
        "no fault",
        "no faults",
        "without fault",
        "without faults",
        "does not contain fault",
        "does not contain faults",
        "no fault evidence",
        "zero faults",
    )
    return any(phrase in text for phrase in no_fault_phrases)


def export_dataset(verified_path, output_path, image_dir, limit=None, task="structural_interpretation"):
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
            if not is_exportable_answer(row.get("answer", "")):
                raise ValueError("answer is incomplete or not training-ready")

            sample_path = resolve_sample_folder(sample_id)
            seismic_item = select_seismic_item(sample_path)
            fault_segments_item = select_fault_segments_item(sample_path)
            if task == "fault_detection" and "fault" not in str(row.get("answer", "")).lower():
                raise ValueError("fault_detection row does not contain a fault statement")
            if task == "fault_detection" and fault_segments_item is None and not is_no_fault_answer(row.get("answer", "")):
                raise ValueError("positive fault_detection row has no fault mask")

            volume = load_array(seismic_item)
            graph_path = row.get("metadata", {}).get("graph_path")
            graph = load_graph(graph_path) if graph_path else {"nodes": [], "edges": []}
            overlay_source = choose_overlay_source(sample_path, row, graph, fault_segments_item, task=task)
            overlay_volume = load_array(overlay_source["item"]) if overlay_source else None
            view_graph = graph if task != "fault_detection" or fault_segments_item is not None else {"nodes": [], "edges": []}
            views = build_views(
                volume,
                graph=view_graph,
                target_volume=overlay_volume,
                target_source=overlay_source["kind"] if overlay_source else "amplitude_max",
            )

            image_path = image_dir / f"{sample_id}.png"
            overlay_image_path = image_dir / f"{sample_id}_overlay.png"
            mask_image_path = image_dir / f"{sample_id}_mask.png"
            seismic_relpath = seismic_item["group_path"].relative_to(sample_path).as_posix()
            render_panel(
                sample_id=sample_id,
                seismic_name=seismic_relpath,
                views=views,
                output_path=image_path,
            )

            overlay_relpath = ""
            overlay_kind = ""
            if overlay_source:
                overlay_views = build_overlay_views(overlay_volume, views["indices"])
                overlay_kind = overlay_source["kind"]
                overlay_relpath = overlay_source["item"]["group_path"].relative_to(sample_path).as_posix()
                render_overlay_panel(
                    sample_id=sample_id,
                    seismic_name=seismic_relpath,
                    views=views,
                    overlay_views=overlay_views,
                    overlay_kind=overlay_kind,
                    overlay_color=overlay_source["color"],
                    output_path=overlay_image_path,
                )
                render_mask_panel(
                    sample_id=sample_id,
                    overlay_views=overlay_views,
                    overlay_kind=overlay_kind,
                    output_path=mask_image_path,
                )
            else:
                overlay_image_path = None
                mask_image_path = None

            exported_row = multimodal_row(
                source_row=row,
                image_path=image_path,
                overlay_image_path=overlay_image_path,
                mask_image_path=mask_image_path,
                seismic_relpath=seismic_relpath,
                overlay_relpath=overlay_relpath,
                overlay_kind=overlay_kind,
                slice_indices=views["indices"],
            )
            exported_row["metadata"]["task"] = task
            if overlay_image_path is not None:
                exported_row["messages"][0]["content"].append(
                    {"type": "image", "image": overlay_image_path.as_posix()}
                )
            if mask_image_path is not None:
                exported_row["messages"][0]["content"].append(
                    {"type": "image", "image": mask_image_path.as_posix()}
                )
            exported.append(exported_row)
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
        "instruction",
        "input",
        "answer",
        "image",
        "overlay_image",
        "mask_image",
        "overlay_kind",
        "view_inline",
        "view_crossline",
        "view_timeslice",
        "view_mode_inline",
        "view_mode_crossline",
        "view_mode_timeslice",
        "graph_path",
        "seismic_array",
        "overlay_array",
        "llm_answer",
        "llm_raw_output",
        "nli_status",
        "nli_score",
        "nli_model",
        "nli_deciding_sentence",
        "nli_deciding_evidence_json",
        "llm_prompt",
        "graph_trace_json",
        "nli_retrieved_evidence_json",
        "messages_json",
        "metadata_json",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row.get("metadata", {})
            trace = row.get("trace", {})
            deciding = trace.get("nli_deciding_evidence", {}) or {}
            view_indices = metadata.get("view_indices", {})
            view_selection = view_indices.get("methods", {})
            writer.writerow({
                "id": row["id"],
                "sample_id": row["sample_id"],
                "instruction": row["instruction"],
                "input": row["input"],
                "answer": row["answer"],
                "image": row["image"],
                "overlay_image": row.get("overlay_image", ""),
                "mask_image": row.get("mask_image", ""),
                "overlay_kind": row.get("overlay_kind", ""),
                "view_inline": view_indices.get("inline"),
                "view_crossline": view_indices.get("crossline"),
                "view_timeslice": view_indices.get("timeslice"),
                "view_mode_inline": view_selection.get("inline", ""),
                "view_mode_crossline": view_selection.get("crossline", ""),
                "view_mode_timeslice": view_selection.get("timeslice", ""),
                "graph_path": metadata.get("graph_path", ""),
                "seismic_array": metadata.get("seismic_array", ""),
                "overlay_array": metadata.get("overlay_array", ""),
                "llm_answer": trace.get("llm_answer", row["answer"]),
                "llm_raw_output": trace.get("llm_raw_output", ""),
                "nli_status": trace.get("nli_status", ""),
                "nli_score": trace.get("nli_score", ""),
                "nli_model": trace.get("nli_model", ""),
                "nli_deciding_sentence": deciding.get("sentence", ""),
                "nli_deciding_evidence_json": json.dumps(deciding),
                "llm_prompt": trace.get("llm_prompt", ""),
                "graph_trace_json": json.dumps(trace.get("graph_trace", [])),
                "nli_retrieved_evidence_json": json.dumps(trace.get("nli_retrieved_evidence", [])),
                "messages_json": json.dumps(row.get("messages", [])),
                "metadata_json": json.dumps(metadata),
            })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert verified_hypotheses.jsonl into a 2D multimodal instruction-tuning dataset."
    )
    parser.add_argument("--verified", default=str(DEFAULT_VERIFIED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", default="structural_interpretation", choices=["structural_interpretation", "fault_detection"])
    return parser.parse_args()


def main():
    args = parse_args()
    result = export_dataset(
        verified_path=args.verified,
        output_path=args.output,
        image_dir=args.image_dir,
        limit=args.limit,
        task=args.task,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
