# graph tracing multihop for NLI and llm
import argparse
import json
import re
from pathlib import Path

try:
    from .evidence_template import (
        property_template,
        relation_template,
        sample_template,
    )
except ImportError:
    from evidence_template import (
        property_template,
        relation_template,
        sample_template,
    )


root = Path(__file__).parent.parent
graph_root = root / "traces" / "properties_graph"


class EvidenceTracer(object):
    def __init__(self, graph_path):
        self.graph_path = Path(graph_path)
        self.graph = json.loads(self.graph_path.read_text())
        self.nodes = {node["id"]: node for node in self.graph.get("nodes", [])}
        self.evidence = self.compose_evidence()

    def compose_evidence(self):
        evidence = []
        for node in self.graph.get("nodes", []):
            item = self.node_to_evidence(node)
            if item:
                evidence.append(item)

        for edge in self.graph.get("edges", []):
            item = self.edge_to_evidence(edge)
            if item:
                evidence.append(item)

        return evidence

    def structural_evidence(self):
        prioritized = []
        for item in self.evidence:
            if item.get("source") == "property":
                prioritized.append(item)
                continue
            if item.get("source") != "relation":
                continue
            if item.get("fact_name") in {"HAS_CATEGORY", "REALIZED", "HAS_FAULT", "HAS_CLOSURE", "HAS_FLUID"}:
                prioritized.append(item)
        return prioritized

    def retrieve(self, claim, top_k=8):
        claim_tokens = self.tokenize(claim)
        scored = []
        evidence = self.structural_evidence()
        for item in evidence:
            text = f"{item['fact_name']} {item['sentence']}"
            score = len(claim_tokens & self.tokenize(text)) / max(len(claim_tokens), 1)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [item for score, item in scored if score > 0]
        return selected[:top_k] if selected else [item for _, item in scored[:top_k]]

    def node_to_evidence(self, node):
        label = node.get("label")
        if label == "Sample":
            return self.make_evidence("sample", sample_template(node), "sample", node.get("sample_id"), vlm_visible=False, node=node)
        if label in {"Category", "ModelParameters", "FaultSystem", "Fault", "ClosureSystem", "Closure", "Fluid"}:
            return self.make_evidence("property", property_template(node), label, node.get("name", ""), vlm_visible=False, node=node)
        return None

    def edge_to_evidence(self, edge):
        edge_type = edge.get("type")
        source_node = self.nodes.get(edge.get("source"), {})
        target_node = self.nodes.get(edge.get("target"), {})

        if edge_type in {"HAS_CATEGORY", "REALIZED", "HAS_FAULT", "HAS_CLOSURE", "HAS_FLUID"}:
            sentence = relation_template(edge, source_node, target_node)
            return self.make_evidence("relation", sentence, edge_type, edge.get("metric_family", ""), vlm_visible=False, edge=edge)
        return None

    @staticmethod
    def make_evidence(source, sentence, fact_name, value, **kwargs):
        return {
            "source": source,
            "sentence": sentence,
            "fact_name": fact_name,
            "value": value if value is not None else "",
            **kwargs,
        }

    @staticmethod
    def tokenize(text):
        return set(re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " ")))


def first_graph():
    graph_paths = sorted(graph_root.glob("*_properties_graph.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no properties graph found in {graph_root}")
    return graph_paths[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", nargs="?", default=None)
    parser.add_argument("--claim", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    graph_path = Path(args.graph) if args.graph else first_graph()
    tracer = EvidenceTracer(graph_path)
    if args.claim:
        evidence = tracer.retrieve(args.claim, args.top_k)
    else:
        evidence = tracer.structural_evidence()
    print(json.dumps(evidence, indent=2, default=str))
