import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize_sample import list_arrays, load_array


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs"


def create_images_for_row(row, image_dir):
    sample_id = row["sample_id"]
    cached = load_sample_assets(sample_id, image_dir)
    if cached:
        return normalize_cached_assets(cached)

    # Dataset export should use the sample-level assets created by graph_generation.py
    # or the hybrid pipeline precheck. This keeps overlays tied to sample intent.
    if row.get("image_assets"):
        return normalize_cached_assets(row["image_assets"])

    sample_path = resolve_sample_folder(sample_id)

    seismic_item = select_seismic_item(sample_path)
    fault_segments_item = select_fault_segments_item(sample_path)
    graph_path = row.get("metadata", {}).get("graph_path")
    graph = load_graph(graph_path) if graph_path else {"nodes": [], "edges": []}
    overlay_source = choose_overlay_source(sample_path, row, graph, fault_segments_item)

    volume = load_array(seismic_item)
    overlay_volume = load_array(overlay_source["item"]) if overlay_source else None
    views = build_views(
        volume,
        graph=graph,
        target_volume=overlay_volume,
        target_source=overlay_source["kind"] if overlay_source else "amplitude_max",
    )

    image_dir = Path(image_dir)
    sample_image_dir = image_dir / sample_id
    sample_image_dir.mkdir(parents=True, exist_ok=True)
    seismic_relpath = seismic_item["group_path"].relative_to(sample_path).as_posix()

    view_info = {}
    overlay_views = None
    overlay_relpath = ""
    overlay_kind = ""
    if overlay_source:
        overlay_views = build_overlay_views(overlay_volume, views["indices"])
        overlay_kind = overlay_source["kind"]
        overlay_relpath = overlay_source["item"]["group_path"].relative_to(sample_path).as_posix()

    for view in ("inline", "crossline"):
        image_path = sample_image_dir / f"{view}.png"
        save_single_image(sample_id, seismic_relpath, view, views, image_path)

        overlay_image_path = None
        mask_image_path = None
        if overlay_source and overlay_views:
            overlay_image_path = sample_image_dir / f"{view}_overlay.png"
            mask_image_path = sample_image_dir / f"{view}_mask.png"
            save_single_overlay(
                sample_id,
                seismic_relpath,
                view,
                views,
                overlay_views,
                overlay_kind,
                overlay_source["color"],
                overlay_image_path,
            )
            save_single_mask(sample_id, view, overlay_views, overlay_kind, mask_image_path, overlay_source["color"])

        view_info[view] = {
            "image_path": image_path,
            "overlay_image_path": overlay_image_path,
            "mask_image_path": mask_image_path,
            "slice_index": views["indices"][view],
            "selection_method": views["indices"].get("methods", {}).get(view, ""),
            "fixed_axis": "inline" if view == "inline" else "crossline",
        }

    return {
        "views": view_info,
        "seismic_relpath": seismic_relpath,
        "overlay_relpath": overlay_relpath,
        "overlay_kind": overlay_kind,
        "slice_indices": views["indices"],
    }


def load_sample_assets(sample_id, image_dir):
    assets_path = Path(image_dir) / sample_id / "assets.json"
    if not assets_path.exists():
        return None
    return json.loads(assets_path.read_text())


def normalize_cached_assets(assets):
    normalized = {
        "views": {},
        "seismic_relpath": assets.get("seismic_array", ""),
        "overlay_relpath": assets.get("overlay_array", ""),
        "overlay_kind": assets.get("overlay_kind", ""),
        "overlay_arrays": assets.get("overlay_arrays", []),
        "slice_indices": assets.get("slice_indices", {}),
    }
    for view, info in assets.get("views", {}).items():
        normalized["views"][view] = {
            "image_path": Path(info["image_path"]),
            "overlay_image_path": Path(info["overlay_image_path"]) if info.get("overlay_image_path") else None,
            "mask_image_path": Path(info["mask_image_path"]) if info.get("mask_image_path") else None,
            "slice_index": info.get("slice_index"),
            "selection_method": info.get("selection_method", ""),
            "fixed_axis": info.get("fixed_axis", view),
        }
    return normalized


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
    for item in list_arrays(sample_path):
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


def load_graph(path):
    path = Path(path)
    if not path.exists():
        return {"nodes": [], "edges": []}
    return json.loads(path.read_text())


def choose_overlay_source(sample_path, source_row, graph, fault_segments_item):
    answer = str(source_row.get("answer", "")).lower()
    target_closure = next((node for node in graph.get("nodes", []) if node.get("label") == "Closure"), None)

    if "fault" in answer and fault_segments_item is not None:
        return {"kind": "fault", "item": fault_segments_item, "color": "#ef4444"}

    if any(token in answer for token in ("closure", "oil", "gas", "brine")) and target_closure:
        fluid = str(target_closure.get("fluid", "")).lower()
        fluid_item = select_first_existing(sample_path, [f"closures/{fluid}.zarr"]) if fluid else None
        if fluid_item:
            return {"kind": f"closure_{fluid}", "item": fluid_item, "color": overlay_color_for_kind(fluid)}

    if target_closure:
        fluid = str(target_closure.get("fluid", "")).lower()
        fluid_item = select_first_existing(sample_path, [f"closures/{fluid}.zarr"]) if fluid else None
        if fluid_item:
            return {"kind": f"closure_{fluid}", "item": fluid_item, "color": overlay_color_for_kind(fluid)}

    hc_item = select_first_existing(sample_path, ["closures/hc_labels.zarr", "all_closure_segments"])
    if hc_item:
        return {"kind": "closure", "item": hc_item, "color": "#f59e0b"}

    if fault_segments_item is not None:
        return {"kind": "fault", "item": fault_segments_item, "color": "#ef4444"}

    return None


def overlay_color_for_kind(kind):
    return {
        "oil": "#f59e0b",
        "gas": "#22c55e",
        "brine": "#38bdf8",
    }.get(str(kind).lower(), "#ef4444")


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
    return ((np.clip(slice_2d, low, high) - low) / (high - low)).astype(np.float32)


def build_views(volume, graph=None, target_volume=None, target_source="target_mask"):
    indices = select_view_indices(volume, graph=graph, target_volume=target_volume, target_source=target_source)
    return {
        "inline": clip_normalize(volume[indices["inline"], :, :].T),
        "crossline": clip_normalize(volume[:, indices["crossline"], :].T),
        "timeslice": clip_normalize(volume[:, :, indices["timeslice"]].T),
        "indices": indices,
    }


def select_view_indices(volume, graph=None, target_volume=None, target_source="target_mask"):
    amplitude = np.abs(np.nan_to_num(volume, nan=0.0))
    selected = {}
    methods = {}
    target_choice = target_volume_indices(target_volume, target_source) if target_volume is not None else None

    if target_choice:
        selected["inline"] = clamp_index(target_choice["inline"], volume.shape[0])
        selected["crossline"] = clamp_index(target_choice["crossline"], volume.shape[1])
        selected["timeslice"] = clamp_index(target_choice["timeslice"], volume.shape[2])
        methods.update({
            "inline": target_choice["source"],
            "crossline": target_choice["source"],
            "timeslice": target_choice["source"],
        })

    if "inline" not in selected:
        selected["inline"] = int(np.argmax(amplitude.mean(axis=(1, 2))))
        methods["inline"] = "amplitude_max"
    if "crossline" not in selected:
        selected["crossline"] = int(np.argmax(amplitude.mean(axis=(0, 2))))
        methods["crossline"] = "amplitude_max"
    if "timeslice" not in selected:
        selected["timeslice"] = int(np.argmax(amplitude.mean(axis=(0, 1))))
        methods["timeslice"] = "amplitude_max"

    return {
        "inline": selected["inline"],
        "crossline": selected["crossline"],
        "timeslice": selected["timeslice"],
        "methods": methods,
        "target_volume_used": bool(target_choice),
    }


def target_volume_indices(target_volume, source_name):
    occupancy = np.nan_to_num(target_volume, nan=0.0)
    scores = {
        "inline": occupancy.sum(axis=(1, 2)),
        "crossline": occupancy.sum(axis=(0, 2)),
        "timeslice": occupancy.sum(axis=(0, 1)),
    }
    if all(score.max() <= 0 for score in scores.values()):
        return None
    return {
        "inline": int(np.argmax(scores["inline"])),
        "crossline": int(np.argmax(scores["crossline"])),
        "timeslice": int(np.argmax(scores["timeslice"])),
        "source": source_name,
    }


def build_overlay_views(mask_volume, indices):
    binary = np.nan_to_num(mask_volume, nan=0.0)
    inline = (binary[clamp_index(indices["inline"], binary.shape[0]), :, :].T > 0).astype(np.float32)
    crossline = (binary[:, clamp_index(indices["crossline"], binary.shape[1]), :].T > 0).astype(np.float32)
    timeslice = (binary[:, :, clamp_index(indices["timeslice"], binary.shape[2])].T > 0).astype(np.float32)
    return {"inline": inline, "crossline": crossline, "timeslice": timeslice}


def clamp_index(index, axis_len):
    return int(np.clip(int(index), 0, axis_len - 1))


def align_overlay_mask(overlay_mask, target_shape):
    aligned = np.zeros(target_shape, dtype=np.float32)
    copy_h = min(target_shape[0], overlay_mask.shape[0])
    copy_w = min(target_shape[1], overlay_mask.shape[1])
    aligned[:copy_h, :copy_w] = overlay_mask[:copy_h, :copy_w]
    return aligned


def save_panel(sample_id, seismic_name, views, output_path):
    image = np.hstack([views[view] for view in ("inline", "crossline", "timeslice")])
    save_gray_image(image, output_path)


def save_overlay_panel(sample_id, seismic_name, views, overlay_views, overlay_kind, overlay_color, output_path):
    image = np.hstack([
        overlay_image(views[view], overlay_views[view], overlay_color)
        for view in ("inline", "crossline", "timeslice")
    ])
    save_rgb_image(image, output_path)


def save_mask_panel(sample_id, overlay_views, overlay_kind, output_path):
    image = np.hstack([overlay_views[view] for view in ("inline", "crossline", "timeslice")])
    save_gray_image(image, output_path)


def save_single_image(sample_id, seismic_name, view, views, output_path):
    save_gray_image(views[view], output_path)


def save_single_overlay(sample_id, seismic_name, view, views, overlay_views, overlay_kind, overlay_color, output_path):
    save_rgb_image(overlay_image(views[view], overlay_views[view], overlay_color), output_path)


def save_single_mask(sample_id, view, overlay_views, overlay_kind, output_path, overlay_color="#ef4444"):
    mask = overlay_views[view] > 0
    rgb = np.array(plt.matplotlib.colors.to_rgb(overlay_color), dtype=np.float32)
    image = np.zeros((*mask.shape, 3), dtype=np.float32)
    image[mask] = rgb
    save_rgb_image(image, output_path)


def save_composite_overlay(sample_id, seismic_name, view, views, overlay_components, output_path):
    image = views[view]
    composite = np.dstack([image, image, image])

    for component in overlay_components:
        overlay_mask = align_overlay_mask(component["views"][view], image.shape)
        rgb = np.array(plt.matplotlib.colors.to_rgb(component["color"]), dtype=np.float32)
        alpha = overlay_mask[..., None] * 0.55
        composite = composite * (1.0 - alpha) + (np.ones_like(composite) * rgb) * alpha

    save_rgb_image(composite, output_path)


def save_composite_mask(sample_id, view, overlay_components, output_path):
    first_mask = overlay_components[0]["views"][view]
    image = np.zeros((*first_mask.shape, 3), dtype=np.float32)

    for component in overlay_components:
        mask = align_overlay_mask(component["views"][view], first_mask.shape) > 0
        rgb = np.array(plt.matplotlib.colors.to_rgb(component["color"]), dtype=np.float32)
        image[mask] = rgb

    save_rgb_image(image, output_path)


def overlay_image(image, overlay_mask, overlay_color):
    overlay_mask = align_overlay_mask(overlay_mask, image.shape)
    rgb = np.array(plt.matplotlib.colors.to_rgb(overlay_color), dtype=np.float32)
    base_rgb = np.dstack([image, image, image])
    alpha = overlay_mask[..., None] * 0.55
    return base_rgb * (1.0 - alpha) + (np.ones_like(base_rgb) * rgb) * alpha


def save_gray_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.flipud(image), cmap="gray", vmin=0.0, vmax=1.0)


def save_rgb_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.flipud(np.clip(image, 0.0, 1.0)))


def view_label(view, indices):
    key = {"inline": "inline", "crossline": "crossline", "timeslice": "timeslice"}[view]
    prefix = {"inline": "i", "crossline": "x", "timeslice": "t"}[view]
    method = indices.get("methods", {}).get(key, "unknown")
    return f"{view.title()}\n{prefix}={indices[key]} [{method}]"


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
