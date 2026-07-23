"""Create 2D-position graph copies from DB-grounded properties graphs.

The source properties graph stays unchanged. This script only copies each graph
and updates matching object nodes with 2D x/y positions from the image metadata
written by images_generator.py.
"""

import copy
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PROPERTIES_GRAPH_DIR = ROOT / "graphs" / "properties_graph"
IMAGE_OBJECT_DIR = ROOT / "build_objects" / "images"
OUTPUT_DIR = ROOT / "graphs" / "properties_2d_graph"
VIEWS = ("inline", "crossline")
# Visual attributes read straight off each object's scene mask (not the DB), so they
# reflect what is actually visible in the section; attached to nodes here at 2d time.
MASK_FEATURE_KEYS = ("dip_deg", "area_pct")
# Dip estimation (RANSAC + inlier gate). PCA alone is a moment fit, so a contaminated fault
# mask (a crossing structure, a stray blob) drags the major axis flat and produces a
# geologically absurd near-horizontal dip. RANSAC finds the dominant collinear pixel set and
# the dip is fit on THOSE inliers; masks whose dominant line covers less than
# _DIP_MIN_INLIER_FRAC of the pixels are too contaminated to trust and get no dip at all.
_DIP_MIN_PIXELS = 8
_DIP_MIN_INLIER_FRAC = 0.5      # dominant line must contain the majority of the mask's pixels
_DIP_RANSAC_THRESHOLD = 1.5     # px perpendicular distance for a pixel to count as on-line
_DIP_RANSAC_ITERS = 300
_DIP_RANSAC_SEED = 0            # fixed -> graph generation stays reproducible
# Visual-only objects can be much noisier than DB-backed objects.
# Recommended policy for a cleaner fault/closure dataset:
# - fault: keep global and local objects
# - closure: keep global and local objects
# - salt: keep aggregate/global visual context
# - onlap: keep aggregate/count evidence, but avoid numbered components
# - lithology: usually exclude from local visual QA unless explicitly needed
# - age_depth: exclude because it is broad background context, not an object
EXCLUDED_VISUAL_OBJECTS = {
    "age_depth",
    "onlap",  # broad aggregate visual evidence; comment this line back in if needed
    "lithology",  # broad facies volume; too noisy for current object-level QA
}


def main(sample_ids=None):
    # sample_ids: optional iterable to build only those samples' 2d graphs.
    # None means rebuild every graph on disk (startup catch-up).
    sample_filter = set(sample_ids) if sample_ids is not None else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    samples = []
    for graph_path in sorted(PROPERTIES_GRAPH_DIR.glob("*.json")):
        sample_id = _sample_id_from_graph_path(graph_path)
        if sample_filter is not None and sample_id not in sample_filter:
            continue
        graph = _read_json(graph_path)

        for view in VIEWS:
            # positions/bboxes now come from the shared per-view scene, so every
            # node points at a region inside the one scene image for that view.
            positions, scene_meta = _load_scene_positions(sample_id, view)
            copied_graph = _copy_graph_with_2d_positions(graph, positions, view)
            if scene_meta:
                copied_graph["scene"] = scene_meta
            output_path = OUTPUT_DIR / f"{graph_path.stem}_{view}_properties_2d_graph.json"
            output_path.write_text(json.dumps(copied_graph, indent=2, default=str))
            written.append(output_path)
            samples.append(sample_id)

    print(f"wrote {len(written)} 2d graph files to {OUTPUT_DIR}")
    return samples


def _sample_id_from_graph_path(graph_path):
    stem = Path(graph_path).stem
    return stem.removesuffix("_db_extract_properties_graph")


def _read_json(path):
    return json.loads(Path(path).read_text())


def _load_scene_positions(sample_id, view):
    # Read the shared per-view scene: object bboxes/centers are already relative to
    # the one scene image for this view, plus the scene image/mask paths themselves.
    scene_path = IMAGE_OBJECT_DIR / sample_id / "scene_position.json"
    if not scene_path.exists():
        return {}, None

    payload = _read_json(scene_path)
    view_block = (payload.get("views") or {}).get(view)
    if not view_block:
        return {}, None

    positions = {}
    for item in view_block.get("objects", []):
        object_id = item.get("object_id")
        object_type = item.get("object_type", "")
        center = item.get("center") or {}
        bbox = item.get("bbox") or {}
        color = item.get("class_color", "white")
        if object_id in EXCLUDED_VISUAL_OBJECTS or object_type in EXCLUDED_VISUAL_OBJECTS:
            continue
        if _skip_visual_component(object_id, object_type):
            continue
        if not object_id or "x" not in center or "y" not in center:
            continue
        features = _mask_features(
            IMAGE_OBJECT_DIR / sample_id / (item.get("mask_path") or ""),
            object_id,
            object_type,
        )
        positions[(object_id, view)] = {
            "object_type": object_type,
            "x": center["x"],
            "y": center["y"],
            "bbox": bbox,
            "color": color,
            **features,
        }

    def _root_rel(rel_path):
        if not rel_path:
            return ""
        return (IMAGE_OBJECT_DIR / sample_id / rel_path).resolve().relative_to(ROOT).as_posix()

    # the row mask is composited downstream from only the retrieved objects, so the
    # scene block carries just the shared image (+ preview overlay), not one mask.
    scene_meta = {
        "view": view,
        "index": view_block.get("index"),
        "image_path": _root_rel(view_block.get("image_path")),
        "overlay_path": _root_rel(view_block.get("overlay_path")),
        "class_legend": payload.get("class_legend", {}),
    }
    return positions, scene_meta


def _skip_visual_component(object_id, object_type):
    # Onlap connected components can explode into hundreds of unstable slices.
    # Keep the aggregate "onlap" object and drop numbered visual-only parts.
    # If lithology becomes too broad during generation, apply the same aggregate
    # policy here or add it to EXCLUDED_VISUAL_OBJECTS above.
    return object_type == "onlap" and re.match(r"^onlap_\d+$", str(object_id))


_OBJECT_INSTANCE_RE = re.compile(r"_\d+$")


def _is_object_instance(object_id):
    # Numbered object node (fault_0, closure_1, salt_2). The category node
    # ("fault_complex structure") and the type-hub nodes ("fault", "closure") never end
    # in _<digits>, so they are never treated as prunable instances.
    return bool(_OBJECT_INSTANCE_RE.search(str(object_id)))


def _copy_graph_with_2d_positions(graph, positions, view):
    copied_graph = copy.deepcopy(graph)

    # View-filter: keep only object instances that actually appear in THIS view's scene.
    # `positions` already holds exactly the objects rendered in this view, so an instance
    # with no position here is off-view (a 3D fault that doesn't intersect this 2D section).
    # Pruning it means the RAG built from this graph never serves facts for an object with
    # no mask in this view -- which is what produced blank-mask QA rows (asking about a
    # fault that isn't in the picture). Section/hub nodes have no per-view position and are
    # always kept; edges touching a pruned instance are dropped so nothing dangles.
    dropped = set()
    kept_nodes = []
    for node in copied_graph.get("nodes", []):
        object_id = node.get("id")
        if _is_object_instance(object_id) and (object_id, view) not in positions:
            dropped.add(object_id)
            continue
        kept_nodes.append(node)
    copied_graph["nodes"] = kept_nodes
    if dropped:
        copied_graph["edges"] = [
            edge for edge in copied_graph.get("edges", [])
            if edge.get("source") not in dropped and edge.get("target") not in dropped
        ]

    node_ids = {node.get("id") for node in copied_graph.get("nodes", [])}
    category_id = _category_id(copied_graph)

    for node in copied_graph.get("nodes", []):
        object_id = node.get("id")
        position = positions.get((object_id, view))
        if position is None:
            continue


        node["view"] = view
        node["x"] = position["x"]
        node["y"] = position["y"]
        bbox = position.get("bbox") or {}
        node['color'] = position.get("color")
        for key in ("x_min", "x_max", "y_min", "y_max"):
            if key in bbox:
                node[key] = bbox[key]
        for key in MASK_FEATURE_KEYS:
            if key in position:
                node[key] = position[key]

    for (object_id, position_view), position in sorted(positions.items()):
        if position_view != view or object_id in node_ids:
            continue
        node = _visual_node(object_id, position, view)
        copied_graph.setdefault("nodes", []).append(node)
        node_ids.add(object_id)
        if category_id:
            copied_graph.setdefault("edges", []).append({
                "source": category_id,
                "target": object_id,
                "type": "HAS_VISUAL_OBJECT",
            })

    # Recount number_faults to THIS view's surviving fault instances. After pruning off-view
    # faults, the DB total would over-count (e.g. "7 faults" on a section showing 6), so the
    # category node's count must match what is visible. number_fault_intersections and
    # fault_mode are 3D-structural (not per-slice counts) and stay as-is; number_hc_closures
    # is an HC subset, not a plain instance count, so it is left untouched too.
    fault_instances = sum(
        1 for node in copied_graph.get("nodes", [])
        if _is_object_instance(node.get("id"))
        and (node.get("object_type") or str(node.get("id")).split("_")[0]) == "fault"
    )
    for node in copied_graph.get("nodes", []):
        if "number_faults" in node:
            node["number_faults"] = fault_instances

    return copied_graph


def _category_id(graph):
    for node in graph.get("nodes", []):
        node_id = node.get("id", "")
        node_id = str(node_id).split("_")[0]
        return node_id
    return ""


def _visual_node(object_id, position, view):
    node = {
        "id": object_id,
        "object_type": position.get("object_type", ""),
        "source": "visual",
        "view": view,
        "x": position["x"],
        "y": position["y"],
        "color": position.get("color", "white"),
    }
    bbox = position.get("bbox") or {}
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key in bbox:
            node[key] = bbox[key]
    for key in MASK_FEATURE_KEYS:
        if key in position:
            node[key] = position[key]
    return node


def _mask_features(mask_path, object_id, object_type):
    # Read visual geometry off the object's scene mask: apparent dip for faults,
    # coverage for closures/salt. Skip the merged per-type mask (object_id == type),
    # whose combined geometry is meaningless.
    if str(object_id) == str(object_type):
        return {}
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return {}
    try:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except (OSError, ValueError):
        return {}
    if not mask.any():
        return {}

    features = {}
    if object_type in {"closure", "salt"}:
        features["area_pct"] = round(100.0 * float(mask.sum()) / mask.size, 1)
    if object_type == "fault":
        dip = _dip_degrees(mask)
        if dip is not None:
            features["dip_deg"] = dip
    return features


def _ransac_inliers(pts):
    # Largest set of pixels collinear within _DIP_RANSAC_THRESHOLD px of a line through two
    # sampled points. Deterministic (fixed seed) so graph generation stays reproducible.
    n = len(pts)
    rng = np.random.default_rng(_DIP_RANSAC_SEED)
    best_inliers, best_count = None, -1
    for _ in range(_DIP_RANSAC_ITERS):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        norm = float(np.hypot(d[0], d[1]))
        if norm < 1e-6:
            continue
        d = d / norm
        rel = pts - pts[i]
        perp = np.abs(rel[:, 0] * (-d[1]) + rel[:, 1] * d[0])   # perpendicular distance to line
        inliers = perp < _DIP_RANSAC_THRESHOLD
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    return best_inliers


def _dip_degrees(mask):
    # Apparent dip of the fault trace = angle of its dominant line from horizontal
    # (0 = flat, 90 = vertical). Image y runs downward, so magnitude only.
    #
    # RANSAC-then-PCA: RANSAC isolates the dominant collinear pixels (rejecting a crossing
    # structure or stray blob that would flatten a plain PCA fit), then the angle is fit by
    # PCA on those inliers. A mask whose dominant line covers less than _DIP_MIN_INLIER_FRAC
    # of the pixels is too contaminated to assign a dip and returns None. On a clean single-
    # fault mask every pixel is an inlier, so this reduces exactly to the old PCA angle.
    ys, xs = np.nonzero(mask)
    if xs.size < _DIP_MIN_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    inliers = _ransac_inliers(pts)
    if inliers is None:
        return None
    n_in = int(inliers.sum())
    if n_in < _DIP_MIN_PIXELS or n_in / len(pts) < _DIP_MIN_INLIER_FRAC:
        return None  # too contaminated to trust any single line
    p = pts[inliers]
    cov = np.cov(np.vstack([p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[-1] <= 0 or eigvals[0] / eigvals[-1] > 0.6:
        return None  # inliers too round to have a meaningful dip direction
    major = eigvecs[:, -1]
    return round(float(np.degrees(np.arctan2(abs(major[1]), abs(major[0])))), 1)


if __name__ == "__main__":
    main()
