import json
import re
from pathlib import Path


root = Path(__file__).parent.parent
graph_root = root / "graphs" / "properties_graph"


class EvidenceTracer(object):
    def __init__(self, graph_path):
        self.graph_path = Path(graph_path)
        self.graph = json.loads(self.graph_path.read_text())
        self.nodes = {node["id"]: node for node in self.graph.get("nodes", [])}
        self.edges = self.graph.get("edges", [])
        self.relations = self.trace()

    def trace(self):
        relations = []
        for node in self.nodes.values():
            relations.extend(self.trace_properties(node))
        for edge in self.edges:
            relation = self.trace_edge(edge)
            if relation:
                relations.append(relation)
        return relations

    def trace_properties(self, node):
        node_id = node["id"]
        relations = []
        for key, value in self.properties(node).items():
            relations.append({
                "trace_type": "property",
                "source": node_id,
                "edge": key,
                "target": value,
                "relation": [node_id, key, value],
                "text": f"{self.clean_node_id(node_id)} {key} {value}",
            })
        return relations

    def trace_edge(self, edge):
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")

        if source not in self.nodes or target not in self.nodes:
            return None

        return {
            "trace_type": "edge",
            "source": source,
            "edge": edge_type,
            "target": target,
            "source_properties": self.properties(self.nodes[source]),
            "target_properties": self.properties(self.nodes[target]),
            "relation": [source, edge_type, target],
            "text": " ".join([
                self.clean_node_id(source),
                edge_type,
                self.clean_node_id(target),
            ]),
        }

    def retrieve(self, claim, top_k=8):
        claim_tokens = self.tokenize(claim)
        scored = [
            (self.score(claim_tokens, relation), relation)
            for relation in self.relations
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [relation for score, relation in scored if score > 0]
        return selected[:top_k] if selected else [relation for _, relation in scored[:top_k]]

    def structural_evidence(self):
        return self.relations

    @staticmethod
    def properties(node):
        return {
            key: value
            for key, value in node.items()
            if key not in {"id", "model_id"}
        }

    @staticmethod
    def score(claim_tokens, relation):
        relation_tokens = EvidenceTracer.tokenize(relation.get("text", ""))
        return len(claim_tokens & relation_tokens) / max(len(claim_tokens), 1)

    @staticmethod
    def clean_node_id(node_id):
        node_id = str(node_id)
        if node_id.startswith("category:"):
            return "category"
        if node_id.startswith("fault_") or node_id == "fault":
            return "fault"
        if node_id.startswith("closure_") or node_id == "closure":
            return "closure"
        return node_id

    @staticmethod
    def tokenize(text):
        return set(re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " ")))


def first_graph():
    graph_paths = sorted(graph_root.glob("*_properties_graph.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no properties graph found in {graph_root}")
    return graph_paths[0]


if __name__ == "__main__":
    tracer = EvidenceTracer(first_graph())
    print(json.dumps(tracer.structural_evidence(), indent=2, default=str))
