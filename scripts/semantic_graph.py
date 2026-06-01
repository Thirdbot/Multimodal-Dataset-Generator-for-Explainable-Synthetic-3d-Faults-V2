# combine properties graph and array_relation graph to make 1 big graph that about sample properties-relation, not event yet
# format relation to some semantic when doing tracing only
import json
import re
from pathlib import Path

import networkx as nx

from yaml_helper import YAMLHelper


class SemanticGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph_paths = []

    def build(self, properties_graph_path, array_relation_graph_path):
        properties_graph_path = Path(properties_graph_path)
        array_relation_graph_path = Path(array_relation_graph_path)

        self._add_graph_json(properties_graph_path, source_graph="properties_graph")
        self._add_graph_json(array_relation_graph_path, source_graph="array_relation_graph")
        self._link_sample_context()

        self.graph_paths = [properties_graph_path, array_relation_graph_path]
        return self.graph

    def save_to_json(self, output_path=None):
        if not self.graph_paths:
            raise Exception("No graph paths provided")

        if output_path is None:
            traces_path = self.graph_paths[0].parent.parent
            sub_folder = traces_path / "combine_graph"
            sub_folder.mkdir(parents=True, exist_ok=True)
            sample_name = self._sample_name_from_paths()
            output_path = sub_folder / f"{sample_name}_semantic_graph.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

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
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        return output_path

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

    def _add_graph_json(self, graph_path, source_graph):
        data = json.loads(graph_path.read_text())

        for node in data.get("nodes", []):
            node_id = node["id"]
            attrs = {key: value for key, value in node.items() if key != "id"}
            self._add_or_update_node(node_id, attrs, source_graph)

        for index, edge in enumerate(data.get("edges", []), start=1):
            source = edge["source"]
            target = edge["target"]
            attrs = {
                key: value
                for key, value in edge.items()
                if key not in {"source", "target"}
            }
            attrs["source_graph"] = source_graph
            attrs["source_edge_index"] = index
            self.graph.add_edge(source, target, **attrs)

    def _add_or_update_node(self, node_id, attrs, source_graph):
        attrs = dict(attrs)
        if self.graph.has_node(node_id):
            current = self.graph.nodes[node_id]
            sources = set(current.get("source_graphs", []))
            sources.add(source_graph)
            current.update(attrs)
            current["source_graphs"] = sorted(sources)
            return

        attrs["source_graphs"] = [source_graph]
        self.graph.add_node(node_id, **attrs)

    def _link_sample_context(self):
        for node_id, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("label") != "Sample":
                continue

            model_id = attrs.get("sample_id")
            if not isinstance(model_id, str) or not model_id.startswith("seismic__"):
                continue

            build_run_node = f"build_run:{model_id}"
            if self.graph.has_node(build_run_node):
                self.graph.add_edge(
                    build_run_node,
                    node_id,
                    type="HAS_ARRAY_RELATION_EVIDENCE",
                    source_graph="semantic_graph",
                )

            parsed = self._parse_model_id(model_id)
            sample_node = f"sample:{parsed['sample_id']}"
            if self.graph.has_node(sample_node) and sample_node != node_id:
                self.graph.add_edge(
                    sample_node,
                    node_id,
                    type="HAS_ARRAY_RELATION_EVIDENCE",
                    source_graph="semantic_graph",
                )

            recipe_node = f"recipe:{parsed['recipe_id']}"
            if self.graph.has_node(recipe_node):
                self.graph.add_edge(
                    recipe_node,
                    node_id,
                    type="HAS_ARRAY_RELATION_EVIDENCE",
                    source_graph="semantic_graph",
                )

    def _sample_name_from_paths(self):
        for path in self.graph_paths:
            match = re.search(r"(seismic__.+?)_(?:db_extract|array_relations)", path.stem)
            if match:
                return match.group(1)
        return self.graph_paths[0].stem

    @staticmethod
    def _parse_model_id(model_id):
        match = re.match(r"seismic__\d{4}_\d{4}_(recipe_\d+)_(.+)", model_id)
        if not match:
            return {
                "recipe_id": "unknown",
                "sample_id": model_id,
            }

        return {
            "recipe_id": match.group(1),
            "sample_id": match.group(2),
        }


def pair_graph_paths(traces_path):
    traces_path = Path(traces_path)
    properties_paths = sorted((traces_path / "properties_graph").glob("*_properties_graph.json"))
    array_paths = sorted((traces_path / "array_relation_graph").glob("*.json"))

    pairs = []
    for properties_path in properties_paths:
        sample_name = _sample_name_from_graph_path(properties_path)
        if sample_name is None:
            continue

        matched_array = None
        for array_path in array_paths:
            if sample_name in array_path.stem:
                matched_array = array_path
                break

        if matched_array is not None:
            pairs.append((properties_path, matched_array))

    return pairs


def _sample_name_from_graph_path(path):
    match = re.search(r"(seismic__.+?)_(?:db_extract|array_relations)", Path(path).stem)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":
    setting_path = Path(__file__).parent.parent / "settings.yaml"
    yaml_helper = YAMLHelper(setting_path)
    traces_path = Path(yaml_helper.get_data("traces_path"))

    graph_pairs = pair_graph_paths(traces_path)
    for properties_graph_path, array_relation_graph_path in graph_pairs:
        semantic_graph = SemanticGraph()
        semantic_graph.build(properties_graph_path, array_relation_graph_path)
        output_path = semantic_graph.save_to_json()
        print(semantic_graph.summary())
        print(f"saved graph to {output_path}")
