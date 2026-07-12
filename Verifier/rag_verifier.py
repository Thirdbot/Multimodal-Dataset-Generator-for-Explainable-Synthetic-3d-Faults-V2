def _clamp01(value):
    # cosine/NLI scores can round just past 1.0 in float32; keep them in [0, 1]
    return min(1.0, max(0.0, float(value)))

def serialize_docs(docs):
    return [
        {
            "text": doc.page_content,
            # answer docs carry trust_score; question docs only have retrieval _similarity_score
            "score": _clamp01(doc.metadata.get("trust_score",
                              doc.metadata.get("_similarity_score", 0.0))),
            "trace_type": doc.metadata.get("trace_type"),
            "source": doc.metadata.get("source"),
            "object_id": doc.metadata.get("object_id"),
            "edge": doc.metadata.get("edge"),
            "target": doc.metadata.get("target"),
            "relation": doc.metadata.get("relation"),
        }
        for doc in docs
    ]
