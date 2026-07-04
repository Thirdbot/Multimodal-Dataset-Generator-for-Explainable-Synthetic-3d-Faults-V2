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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for graph_path in sorted(PROPERTIES_GRAPH_DIR.glob("*.json")):
        sample_id = _sample_id_from_graph_path(graph_path)
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

    print(f"wrote {len(written)} 2d graph files to {OUTPUT_DIR}")


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
        positions[(object_id, view)] = {
            "object_type": object_type,
            "x": center["x"],
            "y": center["y"],
            "bbox": bbox,
            "color": color,
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


def _copy_graph_with_2d_positions(graph, positions, view):
    copied_graph = copy.deepcopy(graph)
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

    return copied_graph


def _category_id(graph):
    for node in graph.get("nodes", []):
        node_id = node.get("id", "")
        if str(node_id).startswith("category:"):
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
    return node


if __name__ == "__main__":
    main()
