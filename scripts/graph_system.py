import json
import re
from pathlib import Path

import networkx as nx

from yaml_helper import YAMLHelper


class GraphSystem:
    def __init__(self, traces_path=None):
        self.graph = nx.MultiDiGraph()
        self.traces_path = None
    def build(self,trace_path):
        self._add_trace(trace_path)
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

    def _add_trace(self, trace_path):
        data = json.loads(trace_path.read_text())
        model_rows = data.get("model_parameters", [])
        if not model_rows:
            return

        model = model_rows[0]
        model_id = model["model_id"]
        ids = self._parse_model_id(model_id)

        recipe_id = ids["recipe_id"]
        sample_id = ids["sample_id"]
        category = ids["category"]

        recipe_node = f"recipe:{recipe_id}"
        sample_node = f"sample:{sample_id}"
        category_node = f"category:{category}"
        run_node = f"build_run:{model_id}"
        model_node = f"model_parameters:{model_id}"
        artifact_node = f"artifact:{model_id}:output_folder"

        self.graph.add_node(recipe_node, label="Recipe", recipe_id=recipe_id)
        self.graph.add_node(sample_node, label="Sample", sample_id=sample_id)
        self.graph.add_node(category_node, label="Category", name=category)
        self.graph.add_node(
            run_node,
            label="BuildRun",
            model_id=model_id,
            output_folder=model.get("work_subfolder"),
            start_time=model.get("start_time"),
            elapsed_time=model.get("elapsed_time"),
        )
        self.graph.add_node(
            model_node,
            label="ModelParameters",
            **self._pick_model_properties(model),
        )
        self.graph.add_node(
            artifact_node,
            label="Artifact",
            artifact_type="output_folder",
            path=model.get("work_subfolder"),
        )

        self.graph.add_edge(recipe_node, sample_node, type="DEFINES")
        self.graph.add_edge(sample_node, category_node, type="HAS_CATEGORY")
        self.graph.add_edge(sample_node, run_node, type="BUILT_AS")
        self.graph.add_edge(run_node, model_node, type="HAS_MODEL_PARAMETERS")
        self.graph.add_edge(run_node, artifact_node, type="PRODUCED")

        self._add_faults(run_node, model_id, data.get("fault_parameters", []))
        self._add_closures(run_node, model_id, data.get("closure_parameters", []))

    def _add_faults(self, run_node, model_id, faults):
        fault_system_node = f"fault_system:{model_id}"
        self.graph.add_node(
            fault_system_node,
            label="FaultSystem",
            realized_faults=len(faults),
        )
        self.graph.add_edge(run_node, fault_system_node, type="REALIZED")

        for index, fault in enumerate(faults, start=1):
            fault_node = f"fault:{model_id}:{index}"
            self.graph.add_node(
                fault_node,
                label="Fault",
                fault_index=index,
                **self._pick(fault, [
                    "a",
                    "b",
                    "c",
                    "x0",
                    "y0",
                    "z0",
                    "throw",
                    "tilt_pct",
                    "shear_zone_width",
                    "gouge_pctile",
                ]),
            )
            self.graph.add_edge(fault_system_node, fault_node, type="HAS_FAULT")

    def _add_closures(self, run_node, model_id, closures):
        closure_system_node = f"closure_system:{model_id}"
        self.graph.add_node(
            closure_system_node,
            label="ClosureSystem",
            realized_closures=len(closures),
        )
        self.graph.add_edge(run_node, closure_system_node, type="REALIZED")

        for index, closure in enumerate(closures, start=1):
            closure_node = f"closure:{model_id}:{index}"
            fluid = closure.get("fluid")
            self.graph.add_node(
                closure_node,
                label="Closure",
                closure_index=index,
                **self._pick(closure, [
                    "fluid",
                    "n_voxels",
                    "x_min",
                    "x_max",
                    "y_min",
                    "y_max",
                    "z_min",
                    "z_max",
                    "intersects_fault",
                    "intersects_onlap",
                    "intersects_salt",
                    "intercept_avg",
                    "gradient_avg",
                ]),
            )
            self.graph.add_edge(closure_system_node, closure_node, type="HAS_CLOSURE")

            if fluid:
                fluid_node = f"fluid:{fluid}"
                self.graph.add_node(fluid_node, label="Fluid", name=fluid)
                self.graph.add_edge(closure_node, fluid_node, type="HAS_FLUID")

    def _parse_model_id(self, model_id):
        match = re.match(r"seismic__\d{4}_\d{4}_(recipe_\d+)_(.+)", model_id)
        if not match:
            return {
                "recipe_id": "unknown",
                "sample_id": model_id,
                "category": "unknown",
            }

        recipe_id = match.group(1)
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

        return {
            "recipe_id": recipe_id,
            "sample_id": sample_id,
            "category": category,
        }

    def _pick_model_properties(self, model):
        return self._pick(model, [
            "cube_shape",
            "incident_angles",
            "number_faults",
            "fault_mode",
            "salt_inserted",
            "number_onlap_episodes",
            "onlaps_horizon_list",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
            "sand_voxel_pct",
            "sn_db",
            "bandpass_bandlimit_low",
            "bandpass_bandlimit_high",
        ])

    @staticmethod
    def _pick(source, keys):
        return {key: source.get(key) for key in keys if key in source}


if __name__ == "__main__":
    setting_path = Path(__file__).parent.parent / "settings.yaml"
    yaml_helper = YAMLHelper(setting_path)
    root = Path(__file__).parent.parent
    traces_path = yaml_helper.get_data("traces_path")
    traces_path = Path(traces_path)
    trace_sample_path = sorted(traces_path.glob("*_db_extract.json"))
    if trace_sample_path:
        selected_samples = trace_sample_path[0]
        graph_system = GraphSystem()
        graph_system.build(selected_samples)
        output_path = graph_system.save_to_json()
        print(graph_system.summary())
        print(f"saved graph to {output_path}")
