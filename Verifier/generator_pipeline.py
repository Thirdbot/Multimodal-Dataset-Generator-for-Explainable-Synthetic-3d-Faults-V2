import json
import re
import sys
import hashlib
from pathlib import Path

from longtracer import LongTracer, check

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.images_generator import ensure_sample_images, validate_sample_images
from Verifier.create_rag import Rag
from Verifier.llm_machine import LLMMachine, parse_numbered_lines
from Verifier.rag_verifier import best_doc_score, score_qa_evidence, serialize_docs


DEFAULT_GRAPH_ROOT = ROOT / "graphs" / "properties_graph"
DEFAULT_OUTPUT = ROOT / "Dataset" / "hybrid_verified_qa.jsonl"


class HybridRagWorkflow(object):
    def __init__(self, graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT):
        LongTracer.init(verbose=False)
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.rag = Rag(embedding_model="all-MiniLM-L6-v2")
        self.llm = LLMMachine()

    def generate_dataset(self, max_graphs=None, questions_per_graph=5, candidates_per_question=5):
        rows = []
        for graph_path in self.graph_paths(max_graphs=max_graphs):
            rows.extend(self.generate_for_graph(
                graph_path,
                questions_per_graph=questions_per_graph,
                candidates_per_question=candidates_per_question,
            ))
        self.write_rows(rows)
        return rows

    def generate_for_graph(self, graph_path, questions_per_graph=5, candidates_per_question=5):
        graph_path = Path(graph_path)
        sample_id = sample_id_from_graph(graph_path)
        category = category_from_sample_id(sample_id)

        try:
            image_assets = ensure_sample_images(sample_id, graph_path=graph_path)
            valid, reason = validate_sample_images(image_assets)
            if not valid:
                print(f"[IMAGE SKIP] {sample_id}: {reason}")
                return []
        except Exception as exc:
            print(f"[IMAGE SKIP] {sample_id}: {exc}")
            return []

        vector_store, edges = self.rag.mapping_graph_rag(graph_path)
        retrieval = self.rag.graph_retrieval(vector_store, edges)
        retrieve_many = self.llm.retrieve_many(retrieval)
        evidence_text = self.rag.get_all(graph_path)
        used_questions = set()
        rows = []

        for _ in range(questions_per_graph):
            question = self.generate_question(evidence_text, used_questions)
            if not question:
                print(f"[QUESTION SKIP] {sample_id}: no parseable question")
                continue

            question_docs = retrieve_many(expand_query(question))
            if best_doc_score(question_docs) < 0.7:
                print(f"[QUESTION SKIP] {sample_id}: low retrieval score")
                continue

            used_questions.add(normalize_text(question))
            answer = self.best_answer(
                question=question,
                evidence_text=evidence_text,
                question_docs=question_docs,
                retrieve_many=retrieve_many,
                candidates=candidates_per_question,
            )
            if not answer:
                print(f"[ANSWER SKIP] {sample_id}: no supported answer")
                continue

            rows.append({
                "row_id": row_id(sample_id, question, answer["answer"]),
                "sample_id": sample_id,
                "category": category,
                "instruction": question,
                "question": question,
                "answer": answer["answer"],
                "evidence": serialize_docs(answer["docs"]),
                "verification": answer["verification"],
                "metadata": {
                    "graph_path": graph_path.as_posix(),
                    "category": category,
                    "image_asset_path": (ROOT / "Dataset" / "multimodal_images" / sample_id / "assets.json").as_posix(),
                },
                "trace": {
                    "question_evidence": serialize_docs(question_docs),
                    "answer_evidence": serialize_docs(answer["docs"]),
                    "graph_evidence": evidence_text.splitlines(),
                },
                "image_assets": image_assets,
            })

        return rows

    def generate_question(self, evidence_text, used_questions):
        prompt_evidence = evidence_text
        if used_questions:
            prompt_evidence = (
                f"{evidence_text}\n\n"
                "Already asked questions:\n"
                f"{chr(10).join(sorted(used_questions))}\n"
                "Ask about a different supported fact."
            )
        response = self.llm.question_generation().invoke({"evidence": prompt_evidence})
        questions = parse_numbered_lines(response)
        for question in questions:
            key = normalize_text(question)
            if key and key not in used_questions:
                return question
        return ""

    def best_answer(self, question, evidence_text, question_docs, retrieve_many, candidates=5):
        answers = []
        for _ in range(candidates):
            response = self.llm.answer_generation().invoke({
                "evidence": evidence_text,
                "question": question,
            })
            parsed = parse_numbered_lines(response)
            if not parsed:
                continue

            answer = parsed[0]
            answer_docs = retrieve_many(expand_query(answer))
            if score_qa_evidence(question_docs, answer_docs) < 1.0:
                continue

            verification = verify_answer(answer, evidence_text)
            if verification["score"] <= 0:
                continue

            answers.append({
                "answer": answer,
                "docs": answer_docs,
                "verification": verification,
            })

        answers.sort(key=lambda item: item["verification"]["score"], reverse=True)
        return answers[0] if answers else None

    def graph_paths(self, max_graphs=None):
        paths = sorted(self.graph_root.glob("*_properties_graph.json"))
        return paths[:max_graphs] if max_graphs else paths

    def write_rows(self, rows):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as file:
            for row in rows:
                file.write(json.dumps(row, default=str) + "\n")


def expand_query(text):
    text = str(text or "").strip()
    queries = [text]
    lowered = text.lower()

    # Keep expansion simple: add schema terms that match the wording.
    if "fault" in lowered:
        queries.extend(["number_faults", "fault_mode", "n_voxels_faults"])
    if "shear zone" in lowered:
        queries.append("shear_zone_width")
    if "throw" in lowered:
        queries.append("throw")
    if "gouge" in lowered:
        queries.append("gouge_pctile")
    if "salt" in lowered:
        queries.append("salt_inserted")
    if "closure" in lowered or "oil" in lowered or "gas" in lowered or "brine" in lowered:
        queries.extend(["fluid", "number_hc_closures", "closure_voxel_count"])
    if "onlap" in lowered:
        queries.append("number_onlap_episodes")
    if "fan" in lowered or "deposition" in lowered:
        queries.append("number_fan_episodes")

    return "\n".join(dedupe(queries))


def verify_answer(answer, evidence_text):
    result = check(answer, [evidence_text])
    return {
        "verdict": getattr(result, "verdict", ""),
        "score": float(getattr(result, "trust_score", 0.0) or 0.0),
    }


def sample_id_from_graph(graph_path):
    stem = Path(graph_path).stem
    return stem.removesuffix("_properties_graph").replace("_db_extract", "")


def category_from_sample_id(sample_id):
    match = re.search(r"recipe_\d+_(.+?)(?:_[0-9a-f]{32})?$", sample_id)
    return match.group(1) if match else "unknown"


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def row_id(sample_id, question, answer):
    payload = "|".join([sample_id, normalize_text(question), normalize_text(answer)])
    return hashlib.sha1(payload.encode()).hexdigest()


def dedupe(items):
    seen = set()
    unique = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def generate_hybrid_dataset(graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT, max_graphs=None):
    workflow = HybridRagWorkflow(graph_root=graph_root, output_path=output_path)
    return workflow.generate_dataset(max_graphs=max_graphs)


if __name__ == "__main__":
    rows = generate_hybrid_dataset()
    print(json.dumps({
        "rows": len(rows),
        "output": DEFAULT_OUTPUT.as_posix(),
    }, indent=2))
