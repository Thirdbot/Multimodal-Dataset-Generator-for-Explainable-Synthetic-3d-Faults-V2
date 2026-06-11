# evidences must match
def score_qa_evidence(q_evidence,a_evidence):
    # check if relations and target are in the evidence
    q_evidence = {
        (doc.metadata.get("edge"), str(doc.metadata.get("target")))
        for doc in q_evidence
    }
    a_evidence = {
        (doc.metadata.get("edge"), str(doc.metadata.get("target")))
        for doc in a_evidence
    }

    if not q_evidence or not a_evidence:
        return 0.0

    overlap = q_evidence & a_evidence
    return len(overlap) / len(q_evidence)

def best_doc_score(docs):
    if not docs:
        return 0.0
    return max(float(doc.metadata.get("_similarity_score", 0.0)) for doc in docs)

def serialize_docs(docs):
    return [
        {
            "text": doc.page_content,
            "score": float(doc.metadata.get("_similarity_score", 0.0)),
            "source": doc.metadata.get("source"),
            "edge": doc.metadata.get("edge"),
            "target": doc.metadata.get("target"),
            "relation": doc.metadata.get("relation"),
        }
        for doc in docs
    ]
