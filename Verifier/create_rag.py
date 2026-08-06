"""
From the graph, we make a rag
"""
import inspect
import os
import re
import sys
from pathlib import Path

from graph_retriever.strategies import Eager
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_graph_retriever import GraphRetriever
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathlib import Path
from NaturalTransform import TextTransform
from Tracer import EvidenceTracer


_TAG_RE = re.compile(r"</?(?:object|nums|center|bbox|SEG)>")


class _TagStrippingEmbeddings:
    """Embed tag-free text on BOTH sides (docs + queries) while the stored
    page_content keeps its tags for display/answer-copy fidelity.

    The evidence tags (<object>, <nums>, <center>, <bbox>) mark spans that must be
    copied verbatim in answers -- they are noise to the embedder. Embedding them
    tanked same-fact similarity from 1.0 to ~0.62-0.75, so a tagged doc vs the LLM's
    untagged RETRIEVAL_QUERY scored *below* MIN_RETRIEVAL_SCORE and got rejected.
    Stripping tags before encoding makes retrieval tag-agnostic; nothing downstream
    changes because page_content itself is untouched.
    """

    def __init__(self, base):
        self._base = base

    @staticmethod
    def _clean(text):
        return _TAG_RE.sub("", str(text))

    def embed_documents(self, texts):
        return self._base.embed_documents([self._clean(t) for t in texts])

    def embed_query(self, text):
        return self._base.embed_query(self._clean(text))


class Rag:
    def __init__(self,embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embedding = _TagStrippingEmbeddings(HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # GeoGPT-Research-Project/GeoEmbedding
            # EMBED_DEVICE=cuda moves retrieval/RAG embedding off the CPU (it is rebuilt per graph);
            # default cpu keeps the small/shared-GPU behaviour unchanged.
            model_kwargs={"trust_remote_code": True, "device": os.environ.get("EMBED_DEVICE", "cpu")},
        ))
        self.strategy = Eager(
            k=20,
            start_k=6,
            select_k=20,
            max_depth=3,
        )


    def generator_rag(self,graph_path="graphs/views_graph"):
        for graph in self.get_graph(graph_directory=graph_path):
            yield self.mapping_graph_rag(graph) # generate 1 rag at a time

    def graph_retrieval(self,vector_store,edges):
        graph_rag =  GraphRetriever(store=vector_store,
                              edges=list(edges),
                              strategy=self.strategy,
                              )
        return graph_rag

    def mapping_graph_rag(self,graph_path):
        content = self.evidence_documents(graph_path)
        vector_store = self.vector_store_from_rag(content)
        edges = {
            ("object_id", "object_id"),
            ("object_id", "parent_id"),
            ("parent_id", "object_id"),
            ("parent_id", "category_id"),
            ("category_id", "parent_id"),
            ("source", "parent_id"),
            ("parent_id", "source"),
            # ("edge", "edge") dropped: it linked every doc sharing a relation
            # (all closures with intersects_fault) into one clique, so a narrow
            # seed still walked out to every object. Breadth now comes from the
            # query (generic seed hits many docs), not from cross-object fan-out.
        }
        return vector_store, edges

    def vector_store_from_rag(self,content):
        # get retrival
        vector_store = InMemoryVectorStore.from_documents(content,
                                                          embedding=self.embedding)
        return vector_store

    def evidence_documents(self,graph_path):
        graph_path = Path(graph_path)
        tracer = EvidenceTracer(graph_path)
        source_evidence = TextTransform().relations_to_evidence(tracer.structural_evidence())
        hierarchy = self.graph_hierarchy(graph_path)
        return self.prepare_content(source_evidence, hierarchy=hierarchy)

    @staticmethod
    def graph_hierarchy(graph_path):
        import json

        graph = json.loads(Path(graph_path).read_text())
        parents = {}
        category_id = ""

        for node in graph.get("nodes", []):
            node_id = node.get("id", "")
            category_id = node_id

        for edge in graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source and target:
                parents[target] = source

        return {
            "parents": parents,
            "category_id": category_id,
        }

    def format_docs(self,docs):
        return "\n".join(doc.page_content for doc in docs)

    def get_all(self,graph_path):
        content = self.evidence_documents(graph_path)
        return self.format_docs(content)

    @staticmethod
    def get_graph(graph_directory="graphs/properties_graph"):
        graph_paths = Path(graph_directory).resolve()
        return list(graph_paths.iterdir())

    @staticmethod
    def prepare_content(list_contents, hierarchy=None):
        # OBJECT-LEVEL documents: one Document per object (all its facts together),
        # not one per fact. Retrieval and NLI then operate on a coherent object rather
        # than scattered fact fragments -- so a query finds the object, verification
        # checks each answer fact against the object's whole fact set (coverage), and
        # masking stays one-object-one-region. Model-level facts group under the
        # category node into a "section" pseudo-object. The per-fact structure is kept
        # in metadata["facts"] so downstream evidence/masking is still per-fact.
        from collections import OrderedDict

        hierarchy = hierarchy or {"parents": {}, "category_id": ""}
        category_id = hierarchy.get("category_id", "")

        groups = OrderedDict()
        for content in list_contents:
            object_id = content.get("object_id") or content.get("source") or category_id
            group = groups.setdefault(object_id, {"sentences": [], "facts": []})
            sentence = content.get("sentence") or ""
            if sentence:
                group["sentences"].append(sentence)
            group["facts"].append({
                "trace_type": content.get("trace_type"),
                "edge": content.get("edge"),
                "target": content.get("target"),
                "relation": content.get("relation"),
                "text": sentence,
            })

        prepared_content = []
        for object_id, group in groups.items():
            prepared_content.append(
                Document(
                    page_content="\n".join(group["sentences"]),
                    metadata={
                        "object_id": object_id,
                        "source": object_id,
                        "parent_id": hierarchy["parents"].get(object_id, ""),
                        "category_id": category_id,
                        "facts": group["facts"],   # per-fact structure kept for evidence + mask routing
                    },
                )
            )
        return prepared_content

if __name__ == "__main__":
    rag = Rag(embedding_model="all-MiniLM-L6-v2")
    selected_graph = rag.get_graph('graphs/properties_2d_graph')[0]
    print(rag.get_all(selected_graph))
