import json
import re
from pathlib import Path

import networkx as nx

from yaml_helper import YAMLHelper


class GraphSystem:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.traces_path = None
    def build(self,trace_path,the_great_filter):
        self._add_trace(trace_path,the_great_filter)
        self.traces_path = trace_path
        return self.graph

    def save_to_json(self, output_path=None):
        if self.traces_path is None:
            raise Exception("No trace path provided")
        self.traces_path = Path(self.traces_path)

        sub_folder = 'properties_graph'
        sub_folder = self.traces_path.parent / sub_folder
        sub_folder.mkdir(parents=True, exist_ok=True)
        output_path = sub_folder / f"{self.traces_path.stem}_properties_graph.json"

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

    def _add_trace(self, trace_path,the_great_filter):
        data = json.loads(trace_path.read_text())
        model_rows = data.get("model_parameters", [])
        if not model_rows:
            return

        model = model_rows[0]
        model_id = model["model_id"]
        category = self._parse_model_id(model_id)

        category_node = f"category:{category}"

        self._add_by_filter(category_node,data,the_great_filter)

    def _add_by_filter(self,category_node,data,the_great_filter):
        tables = the_great_filter.get("tables")
        model_properties = self._pick(data.get("model_parameters",[{}])[0], the_great_filter.get("model_keys"))
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
                self.graph.add_node(f"{table_name}_{idx}",**self._pick(data.get(table,[{}])[idx],the_great_filter.get(f"{table_name}_keys"))) # get keys by table name closure_list and fault_list
                self.graph.add_edge(f"{table_name}",f"{table_name}_{idx}",type="REALIZED")


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
#     traces_path = yaml_helper.get_data("traces_path")
#     traces_path = Path(traces_path)
#     trace_sample_path = sorted(traces_path.glob("*_db_extract.json"))
#     if trace_sample_path:
#         selected_samples = trace_sample_path[0]
#         graph_system = GraphSystem()
#         graph_system.build(selected_samples,{})
#         output_path = graph_system.save_to_json()
#         print(graph_system.summary())
#         print(f"saved graph to {output_path}")
