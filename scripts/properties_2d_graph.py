"""Create 2D-position graph copies from DB-grounded properties graphs.

The source properties graph stays unchanged. This script only copies each graph
and updates matching object nodes with 2D x/y positions from the image metadata
written by images_generator.py.
"""

import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROPERTIES_GRAPH_DIR = ROOT / "graphs" / "properties_graph"
IMAGE_OBJECT_DIR = ROOT / "build_objects" / "images"
OUTPUT_DIR = ROOT / "graphs" / "properties_2d_graph"
VIEWS = ("inline", "crossline", "timeslice")
EXCLUDED_VISUAL_OBJECTS = {"age_depth"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for graph_path in sorted(PROPERTIES_GRAPH_DIR.glob("*.json")):
        sample_id = _sample_id_from_graph_path(graph_path)
        graph = _read_json(graph_path)
        positions = _load_positions(sample_id)

        for view in VIEWS:
            copied_graph = _copy_graph_with_2d_positions(graph, positions, view)
            output_path = OUTPUT_DIR / f"{graph_path.stem}_{view}_properties_2d_graph.json"
            output_path.write_text(json.dumps(copied_graph, indent=2, default=str))
            written.append(output_path)

    print(f"wrote {len(written)} 2d graph files to {OUTPUT_DIR}")


def _sample_id_from_graph_path(graph_path):
    stem = Path(graph_path).stem
    return stem.removesuffix("_db_extract_properties_graph")


def _read_json(path):
    return json.loads(Path(path).read_text())


def _load_positions(sample_id):
    sample_image_dir = IMAGE_OBJECT_DIR / sample_id
    positions = {}

    for position_path in sorted(sample_image_dir.glob("*_object_position.json")):
        payload = _read_json(position_path)
        for item in payload.get("objects", []):
            object_id = item.get("object_id")
            object_type = item.get("object_type", "")
            view = item.get("view")
            center = item.get("center") or {}
            bbox = item.get("bbox") or {}
            color = item.get("class_color",'white')
            if object_id in EXCLUDED_VISUAL_OBJECTS or object_type in EXCLUDED_VISUAL_OBJECTS:
                continue
            if _skip_visual_component(object_id, object_type):
                continue
            if not object_id or not view or "x" not in center or "y" not in center:
                continue
            positions[(object_id, view)] = {
                "object_type": object_type,
                "x": center["x"],
                "y": center["y"],
                "bbox": bbox,
                "color": color
            }

    return positions


def _skip_visual_component(object_id, object_type):
    # Onlap connected components can explode into hundreds of unstable slices.
    # Keep the aggregate "onlap" object and drop numbered visual-only parts.
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
