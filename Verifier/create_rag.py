"""
From the graph, we make a rag
"""
import inspect
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


class Rag:
    def __init__(self,embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # GeoGPT-Research-Project/GeoEmbedding
            model_kwargs={"trust_remote_code": True},
        )
        self.strategy = Eager(
            k=12,
            start_k=4,
            select_k=12,
            max_depth=2,
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
        tracer = EvidenceTracer(graph_path)
        source_evidence = TextTransform().relations_to_evidence(tracer.structural_evidence())  # get graph in text
        content = self.prepare_content(source_evidence)
        vector_store = self.vector_store_from_rag(content)
        edges = {
            ("edge", "edge"),
            ("source", "source"),
            ("target", "target"),
        }
        return vector_store, edges

    def vector_store_from_rag(self,content):
        # get retrival
        vector_store = InMemoryVectorStore.from_documents(content,
                                                          embedding=self.embedding)
        return vector_store

    def format_docs(self,docs):
        return "\n".join(doc.page_content for doc in docs)

    def get_all(self,graph_path):
        tracer = EvidenceTracer(graph_path)
        source_evidence = TextTransform().relations_to_evidence(tracer.structural_evidence())
        content = self.prepare_content(source_evidence)
        return self.format_docs(content)

    @staticmethod
    def get_graph(graph_directory="graphs/properties_graph"):
        graph_paths = Path(graph_directory).resolve()
        return list(graph_paths.iterdir())

    @staticmethod
    def prepare_content(list_contents):
        prepared_content = []

        for content in list_contents:
            text_content = content.get("sentence")
            source = content.get("source")
            edge = content.get("edge")
            target = content.get("target")
            relation = content.get("relation")

            metadata = {
                "source": source,
                "edge": edge,
                "target": target,
                "relation": relation,
            }
            prepared_content.append(
                Document(
                    page_content=text_content,
                    metadata=metadata
                )
            )
        return prepared_content

if __name__ == "__main__":
    rag = Rag(embedding_model="all-MiniLM-L6-v2")
    selected_graph = rag.get_graph('graphs/properties_2d_graph')[0]
    print(rag.get_all(selected_graph))
