"""Create 2D-position graph copies from DB-grounded properties graphs.

The source properties graph stays unchanged. This script only copies each graph
and updates matching object nodes with 2D x/y positions from the image metadata
written by images_generator.py.
"""

import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROPERTIES_GRAPH_DIR = ROOT / "graphs" / "properties_graph"
IMAGE_OBJECT_DIR = ROOT / "build_objects" / "images"
OUTPUT_DIR = ROOT / "graphs" / "properties_2d_graph"
VIEWS = ("inline", "crossline")
# Visual attributes read straight off each object's scene mask (not the DB), so they
# reflect what is actually visible in the section; attached to nodes here at 2d time.
# Mask -> attribute computation (apparent dip, coverage, bbox/center) lives in compute_attribute.py
# so future attributes have one organised home. DB-sourced params stay in the graph-extract layer.
from scripts.graph.compute_attribute import mask_features, MASK_FEATURE_KEYS
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
    # "onlap" removed: it is now an AGGREGATE object (one union mask). The aggregate "onlap"
    # is kept; any stray numbered onlap_N is still dropped by _skip_visual_component below.
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
        # One sample id per graph, not per view: the old per-view append double-listed every
        # id (once per VIEWS entry). No caller uses the returned length, so dedup is safe.
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
        features = mask_features(
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

    # Recount per view: every count must match what is actually VISIBLE in this section
    # ("accept reality"), the same rule for all types, not just faults. number_fault_intersections,
    # fault_mode and number_onlap_episodes are 3D-structural counts (no per-slice instances) and
    # stay as-is.
    def _instances(otype, pred=None):
        return [
            node for node in copied_graph.get("nodes", [])
            if _is_object_instance(node.get("id"))
            and (node.get("object_type") or str(node.get("id")).split("_")[0]) == otype
            and (pred is None or pred(node))
        ]

    fault_instances = len(_instances("fault"))
    # number_hc_closures is the HYDROCARBON subset -> count visible closures whose fluid is oil/gas.
    hc_closures = len(_instances("closure", lambda n: str(n.get("fluid", "")).lower() in {"oil", "gas"}))
    salt_present = len(_instances("salt")) > 0
    for node in copied_graph.get("nodes", []):
        if "number_faults" in node:
            node["number_faults"] = fault_instances
        if "number_hc_closures" in node:
            node["number_hc_closures"] = hc_closures
        if "salt_inserted" in node:                      # DB says inserted but none visible here -> false
            node["salt_inserted"] = bool(node.get("salt_inserted")) and salt_present

    return copied_graph


def _category_id(graph):
    # The category/hub node id ends in " structure" (e.g. "fault_complex structure"), the
    # convention create_rag.py uses to find it (re.search(r" structure$", ...)). The old code
    # returned on the very first node and only ever yielded a bare prefix, never the real
    # category node -- so the HAS_VISUAL_OBJECT edges below dangled from a non-existent id.
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", ""))
        if re.search(r" structure$", node_id):
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


# _mask_features / _dip_degrees / _ransac_inliers moved to scripts/graph/compute_attribute.py
# (imported at the top as mask_features). This file keeps only graph orchestration + the
# view-scoped DB recount; all mask geometry now lives in compute_attribute.py.


if __name__ == "__main__":
    main()
