import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Dataset.slice2d import (
    build_overlay_views,
    build_views,
    load_array,
    save_composite_mask,
    save_composite_overlay,
    save_single_image,
    save_single_mask,
    save_single_overlay,
    select_first_existing,
    select_seismic_item,
)
from scripts.visualize_sample import list_arrays


DEFAULT_IMAGE_DIR = ROOT / "Dataset" / "multimodal_images"


IMAGE_REQUIREMENTS = {
    "boring": set(),
    "fault_only": {"fault"},
    "fault_complex": {"fault"},
    "salt_only": {"salt"},
    "onlap": {"onlap"},
    "depositional": {"depositional"},
    "salt_fault_mixed": {"fault", "salt"},
    "full_mixed": {"fault", "salt", "closure", "onlap"},
}


def create_sample_images(sample_path, graph_path=None, image_dir=DEFAULT_IMAGE_DIR):
    sample_path = Path(sample_path)
    sample_id = sample_path.name
    sample_image_dir = Path(image_dir) / sample_id
    sample_image_dir.mkdir(parents=True, exist_ok=True)

    seismic_item = select_seismic_item(sample_path)
    overlay_sources = choose_sample_overlays(sample_path, sample_id)

    volume = load_array(seismic_item)
    overlay_volume = combine_overlay_volumes(overlay_sources)
    views = build_views(
        volume,
        target_volume=overlay_volume,
        target_source=overlay_kind(overlay_sources) if overlay_sources else "amplitude_max",
    )

    seismic_relpath = seismic_item["group_path"].relative_to(sample_path).as_posix()
    overlay_components = build_overlay_components(sample_path, overlay_sources, views["indices"])
    overlay_relpath = overlay_components[0]["array"] if overlay_components else ""
    overlay_kind_name = overlay_kind(overlay_components)

    assets = {
        "sample_id": sample_id,
        "sample_path": sample_path.as_posix(),
        "graph_path": Path(graph_path).as_posix() if graph_path else "",
        "seismic_array": seismic_relpath,
        "overlay_array": overlay_relpath,
        "overlay_arrays": [
            {"kind": component["kind"], "array": component["array"], "color": component["color"]}
            for component in overlay_components
        ],
        "overlay_kind": overlay_kind_name,
        "slice_indices": views["indices"],
        "views": {},
    }

    for view in ("inline", "crossline"):
        image_path = sample_image_dir / f"{view}.png"
        save_single_image(sample_id, seismic_relpath, view, views, image_path)

        overlay_image_path = None
        mask_image_path = None
        if overlay_components:
            overlay_image_path = sample_image_dir / f"{view}_overlay.png"
            mask_image_path = sample_image_dir / f"{view}_mask.png"
            if len(overlay_components) == 1:
                component = overlay_components[0]
                save_single_overlay(
                    sample_id,
                    seismic_relpath,
                    view,
                    views,
                    component["views"],
                    component["kind"],
                    component["color"],
                    overlay_image_path,
                )
                save_single_mask(sample_id, view, component["views"], component["kind"], mask_image_path, component["color"])
            else:
                save_composite_overlay(sample_id, seismic_relpath, view, views, overlay_components, overlay_image_path)
                save_composite_mask(sample_id, view, overlay_components, mask_image_path)

        assets["views"][view] = {
            "image_path": image_path.as_posix(),
            "overlay_image_path": overlay_image_path.as_posix() if overlay_image_path else "",
            "mask_image_path": mask_image_path.as_posix() if mask_image_path else "",
            "overlay_components": [
                {"kind": component["kind"], "array": component["array"], "color": component["color"]}
                for component in overlay_components
            ],
            "slice_index": views["indices"][view],
            "selection_method": views["indices"].get("methods", {}).get(view, ""),
            "fixed_axis": view,
        }

    assets_path = sample_image_dir / "assets.json"
    assets_path.write_text(json.dumps(assets, indent=2, default=str))
    return assets


def ensure_sample_images(sample_id, graph_path=None, image_dir=DEFAULT_IMAGE_DIR):
    sample_path = ROOT / "outputs" / sample_id
    assets_path = Path(image_dir) / sample_id / "assets.json"
    if assets_path.exists():
        assets = json.loads(assets_path.read_text())
    else:
        assets = create_sample_images(sample_path, graph_path=graph_path, image_dir=image_dir)

    valid, reason = validate_sample_images(assets)
    if valid:
        return assets

    # Cached assets can be stale after graph/image logic changes.
    assets = create_sample_images(sample_path, graph_path=graph_path, image_dir=image_dir)
    valid, reason = validate_sample_images(assets)
    if not valid:
        raise ValueError(reason)
    return assets


def validate_sample_images(assets):
    sample_id = assets.get("sample_id", "")
    category = category_from_sample_name(sample_id)
    kinds = {
        item.get("kind")
        for item in assets.get("overlay_arrays", [])
        if item.get("kind")
    }

    if category == "boring":
        if kinds:
            return False, "boring sample should not have overlay masks"
        return validate_base_views(assets)

    required = IMAGE_REQUIREMENTS.get(category, set())
    if category in {"salt_fault_mixed", "full_mixed"}:
        if not kinds.intersection(required):
            return False, f"{category} sample should have at least one intended overlay"
    elif required and not required.issubset(kinds):
        return False, f"{category} sample missing required overlay: {sorted(required - kinds)}"

    valid, reason = validate_base_views(assets)
    if not valid:
        return valid, reason
    return validate_overlay_views(assets, expected_overlay=bool(required))


def validate_base_views(assets):
    for view in ("inline", "crossline"):
        info = assets.get("views", {}).get(view, {})
        image_path = info.get("image_path")
        if not image_path or not Path(image_path).exists():
            return False, f"missing {view} image"
    return True, "ok"


def validate_overlay_views(assets, expected_overlay):
    for view in ("inline", "crossline"):
        info = assets.get("views", {}).get(view, {})
        overlay_path = info.get("overlay_image_path")
        mask_path = info.get("mask_image_path")
        if expected_overlay and (not overlay_path or not mask_path):
            return False, f"missing {view} overlay or mask"
        if overlay_path and not Path(overlay_path).exists():
            return False, f"missing {view} overlay image"
        if mask_path and not Path(mask_path).exists():
            return False, f"missing {view} mask image"
    return True, "ok"


def choose_sample_overlay(sample_path, sample_id):
    overlays = choose_sample_overlays(sample_path, sample_id)
    return overlays[0] if overlays else None


def choose_sample_overlays(sample_path, sample_id):
    category = category_from_sample_name(sample_id)
    if category == "boring":
        return []
    if category in {"fault_only", "fault_complex"}:
        return collect_non_empty(sample_path, [(fault_candidates(sample_path), "fault", "#ef4444")])
    if category == "salt_only":
        return collect_non_empty(sample_path, [(salt_candidates(sample_path), "salt", "#a855f7")])
    if category == "onlap":
        return collect_non_empty(sample_path, [(["onlap_segments"], "onlap", "#22c55e")])
    if category == "depositional":
        return collect_non_empty(sample_path, [(["depth_maps_fans", "depth_maps_onlaps"], "depositional", "#f59e0b")])
    if category == "salt_fault_mixed":
        return collect_non_empty(sample_path, [
            (salt_candidates(sample_path), "salt", "#a855f7"),
            (fault_candidates(sample_path), "fault", "#ef4444"),
        ])
    if category == "full_mixed":
        return collect_non_empty(sample_path, [
            (salt_candidates(sample_path), "salt", "#a855f7"),
            (fault_candidates(sample_path), "fault", "#ef4444"),
            (closure_candidates(), "closure", "#f59e0b"),
            (["onlap_segments"], "onlap", "#22c55e"),
        ])
    return collect_non_empty(sample_path, [(closure_candidates(), "closure", "#f59e0b")])


def collect_non_empty(sample_path, groups):
    overlays = []
    used_arrays = set()
    for candidates, kind, color in groups:
        selected = first_non_empty(sample_path, candidates, kind, color)
        if not selected:
            continue
        rel = selected["item"]["group_path"].relative_to(sample_path).as_posix()
        if rel in used_arrays:
            continue
        used_arrays.add(rel)
        overlays.append(selected)
    return overlays


def combine_overlay_volumes(overlay_sources):
    if not overlay_sources:
        return None
    combined = None
    for source in overlay_sources:
        volume = (np.nan_to_num(load_array(source["item"]), nan=0.0) > 0).astype(np.float32)
        if combined is None:
            combined = volume
        else:
            combined, volume = align_volumes(combined, volume)
            combined = np.maximum(combined, volume)
    return combined


def align_volumes(left, right):
    shape = tuple(min(a, b) for a, b in zip(left.shape, right.shape))
    slices = tuple(slice(0, axis_len) for axis_len in shape)
    return left[slices], right[slices]


def build_overlay_components(sample_path, overlay_sources, indices):
    components = []
    for source in overlay_sources:
        volume = load_array(source["item"])
        relpath = source["item"]["group_path"].relative_to(sample_path).as_posix()
        components.append({
            "kind": source["kind"],
            "array": relpath,
            "color": source["color"],
            "views": build_overlay_views(volume, indices),
        })
    return components


def overlay_kind(overlay_sources):
    return "+".join(source["kind"] for source in overlay_sources)


def first_non_empty(sample_path, candidates, kind, color):
    for candidate in candidates:
        item = select_first_existing(sample_path, [candidate])
        if not item:
            continue
        if array_has_content(item):
            return {"kind": kind, "item": item, "color": color}
    return None


def array_has_content(item):
    arr = load_array(item)
    values = np.nan_to_num(arr, nan=0.0)
    if not bool((np.abs(values) > 0).any()):
        return False
    unique = np.unique(values)
    if unique.size == 1:
        return False
    return True


def fault_candidates(sample_path):
    rels = []
    for item in list_arrays(sample_path):
        rel = item["group_path"].relative_to(sample_path).as_posix()
        if (
            rel.startswith("fault_segments_")
            and rel.endswith(".zarr")
            and "azimuth" not in rel
            and "throw" not in rel
        ):
            rels.append(rel)
    rels.extend([
        "fault_segments_throw",
        "fault_intersection_segments",
    ])
    return rels


def salt_candidates(sample_path):
    rels = []
    for item in list_arrays(sample_path):
        rel = item["group_path"].relative_to(sample_path).as_posix()
        if rel.startswith("salt_") and rel.endswith(".zarr"):
            rels.append(rel)
    return rels


def closure_candidates():
    return [
        "closures/oil.zarr",
        "closures/gas.zarr",
        "closures/brine.zarr",
        "closures/hc_labels.zarr",
        "all_closure_segments",
    ]


def category_from_sample_name(sample_name):
    for category in (
        "salt_fault_mixed",
        "fault_complex",
        "fault_only",
        "full_mixed",
        "salt_only",
        "depositional",
        "boring",
        "onlap",
    ):
        if f"_{category}_" in sample_name:
            return category
    return "unknown"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/sample_image_generation.py <sample_output_folder>")
    print(json.dumps(create_sample_images(Path(sys.argv[1])), indent=2))
