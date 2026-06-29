"""Visualize one properties graph JSON with NetworkX.

This script reads the local graph JSON format written by graph_system.py and
saves a simple node-link PNG. It does not require Neo4j.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import networkx as nx


ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "graphs" / "properties_2d_graph"
OUTPUT_DIR = ROOT / "graphs" / "visualizations"


def main():
    graph_path = first_graph(GRAPH_DIR)
    graph = load_graph(graph_path)
    output_path = OUTPUT_DIR / f"{graph_path.stem}_networkx.png"
    draw_graph(graph, output_path)
    print(f"saved graph visualization to {output_path}")


def first_graph(graph_dir):
    graph_paths = sorted(Path(graph_dir).glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no graph JSON found in {graph_dir}")
    return graph_paths[0]


def load_graph(graph_path):
    payload = json.loads(Path(graph_path).read_text())
    graph = nx.MultiDiGraph()

    for node in payload.get("nodes", []):
        node_id = node["id"]
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)

    for edge in payload.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
        graph.add_edge(source, target, **attrs)

    return graph


def draw_graph(graph, output_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 9))

    pos = nx.spring_layout(graph, seed=42, k=1.2)
    node_colors = [node_color(node_id) for node_id in graph.nodes]
    node_sizes = [node_size(node_id) for node_id in graph.nodes]

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#7a8794",
        width=1.2,
        alpha=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#263238",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node_id: short_label(node_id) for node_id in graph.nodes},
        font_size=8,
        font_color="#111827",
    )

    edge_labels = {}
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_labels[(source, target, key)] = attrs.get("type", "")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Properties Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def node_color(node_id):
    node_id = str(node_id)
    if node_id.startswith("category:"):
        return "#8ecae6"
    if node_id == "fault" or node_id.startswith("fault_"):
        return "#ffb4a2"
    if node_id == "closure" or node_id.startswith("closure_"):
        return "#90dbf4"
    if node_id == "salt" or node_id.startswith("salt_"):
        return "#cdb4db"
    if node_id == "onlap" or node_id.startswith("onlap_"):
        return "#ffd166"
    return "#d9e2ec"


def node_size(node_id):
    node_id = str(node_id)
    if node_id.startswith("category:"):
        return 2600
    if "_" not in node_id:
        return 2100
    return 1500


def short_label(node_id):
    return str(node_id).replace("category:", "category:\n")


if __name__ == "__main__":
    main()
