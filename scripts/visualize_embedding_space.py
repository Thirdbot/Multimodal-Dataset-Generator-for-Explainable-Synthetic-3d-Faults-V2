"""Visualize GraphRAG evidence sentence embeddings.

The script embeds evidence Documents created from a properties graph, reduces
the vectors to 2D, and saves a scatter plot colored by graph object type.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from NaturalTransform import TextTransform
from Tracer import EvidenceTracer
from Verifier.create_rag import Rag


GRAPH_DIR = ROOT / "graphs" / "properties_2d_graph"
OUTPUT_DIR = ROOT / "graphs" / "visualizations"
MAX_LABELS = 80


def main():
    graph_path = first_graph(GRAPH_DIR)
    docs = evidence_documents(graph_path)
    if not docs:
        raise ValueError(f"no evidence documents created from {graph_path}")

    vectors, embedding_name = embed_documents([doc.page_content for doc in docs])
    points = reduce_to_2d(vectors)

    output_path = OUTPUT_DIR / f"{graph_path.stem}_embedding_space.png"
    draw_embeddings(points, docs, output_path, embedding_name)
    print(f"saved embedding visualization to {output_path}")


def evidence_documents(graph_path):
    tracer = EvidenceTracer(graph_path)
    evidence = TextTransform().relations_to_evidence(tracer.structural_evidence())
    hierarchy = Rag.graph_hierarchy(graph_path)
    return Rag.prepare_content(evidence, hierarchy=hierarchy)


def embed_documents(texts):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"trust_remote_code": True},
        )
        return np.asarray(embedding.embed_documents(texts)), "all-MiniLM-L6-v2"
    except Exception as error:
        print(f"embedding model unavailable, using local TF-IDF fallback: {error}")
        return tfidf_vectors(texts), "TF-IDF fallback"


def tfidf_vectors(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()


def first_graph(graph_dir):
    graph_paths = sorted(Path(graph_dir).glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError(f"no graph JSON found in {graph_dir}")
    return graph_paths[0]


def reduce_to_2d(vectors):
    if vectors.shape[0] == 1:
        return np.array([[0.0, 0.0]])

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    points = centered @ components
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(points.shape[0])])
    return points


def draw_embeddings(points, docs, output_path, embedding_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 8))

    colors = [object_color(doc.metadata.get("object_id", "")) for doc in docs]
    plt.scatter(points[:, 0], points[:, 1], c=colors, s=90, alpha=0.82, edgecolors="#263238")

    for index, (point, doc) in enumerate(zip(points, docs)):
        if index >= MAX_LABELS:
            break
        label = evidence_label(doc)
        plt.text(point[0], point[1], label, fontsize=7, alpha=0.9)

    add_legend()
    plt.title(f"GraphRAG Evidence Embedding Space ({embedding_name})", fontsize=16)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def object_color(object_id):
    object_id = str(object_id)
    if object_id.startswith("fault"):
        return "#e63946"
    if object_id.startswith("closure"):
        return "#219ebc"
    if object_id.startswith("salt"):
        return "#8338ec"
    if object_id.startswith("onlap"):
        return "#ffb703"
    if object_id.startswith("category:"):
        return "#8ecae6"
    return "#adb5bd"


def evidence_label(doc):
    object_id = str(doc.metadata.get("object_id", ""))
    edge = str(doc.metadata.get("edge", ""))
    if object_id and edge:
        return f"{object_id}:{edge}"
    return edge or object_id or "evidence"


def add_legend():
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="fault", markerfacecolor="#e63946", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="closure", markerfacecolor="#219ebc", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="salt", markerfacecolor="#8338ec", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="onlap", markerfacecolor="#ffb703", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="other", markerfacecolor="#adb5bd", markersize=9),
    ]
    plt.legend(handles=handles, loc="best", frameon=True)


if __name__ == "__main__":
    main()
