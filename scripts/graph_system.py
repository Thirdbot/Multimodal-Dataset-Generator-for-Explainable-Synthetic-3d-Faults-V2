"""Small graph builder for DB-extracted Synthoseis metadata.

GraphSystem turns low-level JSON table dumps into NetworkX graph objects and
serializes them as simple node/edge JSON. It can also project 3D graph fields
into inline/crossline view graphs when image assets are available.
"""

import json
import re
import copy
from pathlib import Path

import networkx as nx

from yaml_helper import YAMLHelper


class GraphSystem:
    """Build, project, summarize, and save a filtered metadata graph."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.db_extract_path = None

    def build(self,db_extract_path,category_filter):
        """Load a DB extraction JSON and build a graph using a category filter."""
        self._add_db_extract(db_extract_path,category_filter)
        self.db_extract_path = db_extract_path
        return self.graph

    def change_build(self, view, image_assets=None):
        """Return a projected copy of the 3D graph for one 2D view."""
        copy_system = GraphSystem()
        copy_system.graph = copy.deepcopy(self.graph)
        copy_system.db_extract_path = self.db_extract_path
        copy_system._project_positions(view, image_assets=image_assets)
        return copy_system

    def save_to_json(self,graph_subfolder='properties_graph', suffix='properties_graph'):
        """Serialize the current NetworkX graph under the graphs directory."""
        if self.db_extract_path is None:
            raise Exception("No DB extraction path provided")
        self.db_extract_path = Path(self.db_extract_path)


        graph_subfolder = self.db_extract_path.parent / graph_subfolder
        graph_subfolder.mkdir(parents=True, exist_ok=True)
        graph_path = graph_subfolder / f"{self.db_extract_path.stem}_{suffix}.json"

        payload = {
            "nodes": [
                {"id": node_id, **attrs}
                for node_id, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in self.graph.edges(data=True)
            ],
        }
        graph_path.write_text(json.dumps(payload, indent=2, default=str))
        return graph_path

    def _project_positions(self, view, image_assets=None):
        """Project stored 3D position fields into 2D image-coordinate fields."""
        for node_id, attrs in self.graph.nodes(data=True):
            attrs["view"] = view
            if str(node_id).startswith("category:"):
                self._add_view_asset_attrs(attrs, view, image_assets)
            if view == "inline":
                self._project_point(attrs, "y0", "z0")
                self._project_extent(attrs, "y_min", "y_max", "z_min", "z_max")
                self._drop_3d_position_keys(attrs)
            elif view == "crossline":
                self._project_point(attrs, "x0", "z0")
                self._project_extent(attrs, "x_min", "x_max", "z_min", "z_max")
                self._drop_3d_position_keys(attrs)

    @staticmethod
    def _project_point(attrs, source_x, source_y):
        if source_x in attrs and source_y in attrs:
            attrs["x"] = attrs.get(source_x)
            attrs["y"] = attrs.get(source_y)

    @staticmethod
    def _project_extent(attrs, source_x_min, source_x_max, source_y_min, source_y_max):
        required = {source_x_min, source_x_max, source_y_min, source_y_max}
        if required.issubset(attrs):
            attrs["x_min"] = attrs.get(source_x_min)
            attrs["x_max"] = attrs.get(source_x_max)
            attrs["y_min"] = attrs.get(source_y_min)
            attrs["y_max"] = attrs.get(source_y_max)

    @staticmethod
    def _drop_3d_position_keys(attrs):
        for key in {"x0", "y0", "z0", "z_min", "z_max"}:
            attrs.pop(key, None)

    @staticmethod
    def _add_view_asset_attrs(attrs, view, image_assets):
        if not image_assets:
            return
        view_asset = image_assets.get("views", {}).get(view, {})
        attrs["fixed_axis"] = view_asset.get("fixed_axis", view)
        attrs["fixed_index"] = view_asset.get("slice_index")
        attrs["selection_method"] = view_asset.get("selection_method", "")
        attrs["image_path"] = view_asset.get("image_path", "")
        attrs["overlay_image_path"] = view_asset.get("overlay_image_path", "")
        attrs["mask_image_path"] = view_asset.get("mask_image_path", "")
        attrs["overlay_kind"] = image_assets.get("overlay_kind", "")
        attrs["overlay_array"] = image_assets.get("overlay_array", "")
        attrs["overlay_arrays"] = image_assets.get("overlay_arrays", [])
        attrs["overlay_components"] = view_asset.get("overlay_components", [])

    def summary(self):
        labels = {}
        for _, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", "Unknown")
            labels[label] = labels.get(label, 0) + 1

        edge_types = {}
        for _, _, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("type", "UNKNOWN")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_labels": labels,
            "edge_types": edge_types,
        }

    def _add_db_extract(self, db_extract_path,category_filter):
        data = json.loads(db_extract_path.read_text())
        model_rows = data.get("model_parameters", [])
        if not model_rows:
            return

        model = model_rows[0]
        model_id = model["model_id"]
        category = self._parse_model_id(model_id)

        category_node = f"category:{category}"

        self._add_by_filter(category_node,data,category_filter)

    def _add_by_filter(self,category_node,data,category_filter):
        """Add category, object-system, and realized-object nodes from tables."""
        tables = category_filter.get("tables")
        model_properties = self._pick(data.get("model_parameters",[{}])[0], category_filter.get("model_keys"))
        visible_fault_indexes = self._visible_fault_indexes(model_properties)
        fault_index_map = self._fault_index_map(visible_fault_indexes)
        if visible_fault_indexes is not None and "number_faults" in model_properties:
            model_properties["number_faults"] = len(visible_fault_indexes)
        if category_node == "category:boring":
            model_properties.pop("number_faults", None)
            model_properties.pop("fault_mode", None)
        self.graph.add_node(category_node, **model_properties)  # category get its properties
        for table in tables:
            if table == "model_parameters":
                continue
            table_name = table.split('_')[0] # fault and closure
            if data.get(table) is not None:
                self.graph.add_edge(category_node,f"{table_name}",type=f"HAS_{table_name.upper()}")
            for idx in range(len(data.get(table,[]))): # get all rows
                if table_name == "fault" and visible_fault_indexes is not None and idx not in visible_fault_indexes:
                    continue
                node_index = fault_index_map.get(idx, idx) if table_name == "fault" else idx
                node_attrs = self._pick(data.get(table,[{}])[idx],category_filter.get(f"{table_name}_keys"))
                if table_name == "fault" and visible_fault_indexes is not None:
                    node_attrs["original_fault_index"] = idx
                self.graph.add_node(f"{table_name}_{node_index}",**node_attrs) # get keys by table name closure_list and fault_list
                self.graph.add_edge(f"{table_name}",f"{table_name}_{node_index}",type="REALIZED")

    @staticmethod
    def _visible_fault_indexes(model_properties):
        value = model_properties.get("fault_voxel_count_list")
        if value is None:
            return None
        if isinstance(value, list):
            counts = value
        else:
            counts = re.findall(r"-?\d+(?:\.\d+)?", str(value))
        return {
            index
            for index, count in enumerate(counts)
            if float(count) > 0
        }

    @staticmethod
    def _fault_index_map(visible_fault_indexes):
        if visible_fault_indexes is None:
            return {}
        return {
            original_index: visible_index
            for visible_index, original_index in enumerate(sorted(visible_fault_indexes))
        }

    def _parse_model_id(self, model_id):
        match = re.match(r"seismic__\d{4}_\d{4}_(recipe_\d+)_(.+)", model_id)
        if not match:
            return "unknown"

        sample_id = match.group(2)
        category = sample_id.split("_", 1)[0]
        if sample_id.startswith("fault_only_"):
            category = "fault_only"
        elif sample_id.startswith("fault_complex_"):
            category = "fault_complex"
        elif sample_id.startswith("salt_fault_mixed_"):
            category = "salt_fault_mixed"
        elif sample_id.startswith("salt_only_"):
            category = "salt_only"
        elif sample_id.startswith("full_mixed_"):
            category = "full_mixed"

        return category

    @staticmethod
    def _pick(source, keys):
        return {key: source.get(key) for key in keys if key in source}


# if __name__ == "__main__":
#     setting_path = Path(__file__).parent.parent / "settings.yaml"
#     yaml_helper = YAMLHelper(setting_path)
#     root = Path(__file__).parent.parent
#     graphs_path = yaml_helper.get_data("graphs_path")
#     graphs_path = Path(graphs_path)
#     db_extract_paths = sorted(graphs_path.glob("*_db_extract.json"))
#     if db_extract_paths:
#         selected_samples = db_extract_paths[0]
#         graph_system = GraphSystem()
#         graph_system.build(selected_samples,{})
#         graph_path = graph_system.save_to_json()
#         print(graph_system.summary())
#         print(f"saved graph to {graph_path}")
