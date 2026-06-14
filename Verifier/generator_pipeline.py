import json
import random
import re
import sys
import hashlib
from pathlib import Path

from longtracer import LongTracer, check

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Verifier.create_rag import Rag
from Verifier.llm_machine import LLMMachine
from Verifier.rag_verifier import best_doc_score, score_qa_evidence, serialize_docs


DEFAULT_GRAPH_ROOT = ROOT / "graphs" / "properties_2d_graph"
DEFAULT_OUTPUT = ROOT / "Dataset" / "hybrid_verified_qa.jsonl"


class RagWorkflow(object):
    def __init__(self, graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT):
        LongTracer.init(verbose=False)
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.output_started = False
        self.written_row_keys = set()
        self.rag = Rag(embedding_model="all-MiniLM-L6-v2")
        self.llm = LLMMachine()

    def generate_dataset(self, max_graphs=None,graph_views='inline', questions_per_graph=5, candidates_per_question=5):
        self.start_output(truncate=True)
        rows = []
        for graph_path in self.graph_paths(max_graphs=max_graphs,views=graph_views):
            rows.extend(self.generate_for_graph(
                graph_path,
                questions_per_graph=questions_per_graph,
                candidates_per_question=candidates_per_question,
            ))
        return rows

    def generate_for_graph(self, graph_path, questions_per_graph=5, candidates_per_question=5):
        graph_path = Path(graph_path)
        sample_id = sample_id_from_graph(graph_path)
        category = category_from_sample_id(sample_id)
        view = view_from_graph(graph_path)

        vector_store, edges = self.rag.mapping_graph_rag(graph_path)
        retrieval = self.rag.graph_retrieval(vector_store, edges) # graph retrieval is not deep enough
        retrieve_many = self.llm.retrieve_many(retrieval)
        all_docs = self.rag.evidence_documents(graph_path)
        evidence_text = self.rag.format_docs(all_docs)
        evidence_seeds = self.evidence_seeds(all_docs)
        seen_question_signatures = set()
        seen_question_texts = []
        repeated_questions = 0
        rows = []

        for seed_docs in evidence_seeds:
            if len(rows) >= questions_per_graph:
                break

            seed_text = self.rag.format_docs(seed_docs)
            question = self.generate_question(seed_text, seen_question_texts)
            if not question:
                print(f"[QUESTION SKIP] {sample_id}: no question generated")
                continue

            question_key = question_signature(question)
            if question_key in seen_question_signatures:
                repeated_questions += 1
                continue
            seen_question_signatures.add(question_key)
            seen_question_texts.append(question)

            question_docs = retrieve_many(question) # multiple question evidences
            if best_doc_score(question_docs) < 0.7:
                print(f"[QUESTION SKIP] {sample_id}: low retrieval score")
                continue

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

            row = {
                "row_id": row_id(sample_id, question, answer["answer"]),
                "sample_id": sample_id,
                "category": category,
                "view": view,
                "instruction": question,
                "question": question,
                "answer": answer["answer"],
                "evidence": serialize_docs(answer["docs"]),
                "verification": answer["verification"],
                "metadata": {
                    "graph_path": graph_path.as_posix(),
                    "category": category,
                    "view": view,
                },
                "trace": {
                    "reason": answer.get("reason", ""),
                    "question_evidence": serialize_docs(question_docs),
                    "answer_evidence": serialize_docs(answer["docs"]),
                    "graph_evidence": docs_to_text(dedupe_docs([*question_docs, *answer["docs"]])).splitlines(),
                },
            }
            if self.append_row(row):
                rows.append(row)

        if repeated_questions:
            print(f"[QUESTION SKIP] {sample_id}: {repeated_questions} repeated questions")

        return rows

    def evidence_seeds(self, docs, packet_size=3):
        docs = list(docs)
        random.shuffle(docs)
        by_object = {}
        for doc in docs:
            object_id = doc.metadata.get("object_id") or doc.metadata.get("source") or ""
            by_object.setdefault(object_id, []).append(doc)

        seeds = []
        for doc in docs:
            object_id = doc.metadata.get("object_id") or doc.metadata.get("source") or ""
            packet = [doc]
            for related in by_object.get(object_id, []):
                if related is doc:
                    continue
                packet.append(related)
                if len(packet) >= packet_size:
                    break
            seeds.append(packet)
        return seeds

    def generate_question(self, evidence_text, used_questions):
        prompt_evidence = evidence_text
        if used_questions:
            prompt_evidence = (
                f"{evidence_text}\n\n"
                "Already asked questions:\n"
                f"{chr(10).join(sorted(used_questions))}\n"
                "Ask about a different supported fact."
            )
        response = self.llm.question_generation().invoke({"evidences": prompt_evidence})
        return response.QUESTION if response else ""

    def best_answer(self, question, evidence_text, question_docs, retrieve_many, candidates=5):
        answers = []
        for _ in range(candidates):
            reason = self.llm.reason_generation().invoke({
                "evidences": evidence_text,
                "question": question,
            })
            reason_text = reason.REASON if reason else ""
            response = self.llm.answer_generation().invoke({
                "evidences": evidence_text,
                "question": question,
                "reason": reason_text,
            })
            answer = response.ANSWER if response else ""
            answer_docs = retrieve_many(answer)
            if score_qa_evidence(question_docs, answer_docs) < 1.0:
                continue

            verification_text = docs_to_text(dedupe_docs([*question_docs, *answer_docs]))
            verification = verify_answer(answer, verification_text)
            if verification["verdict"] != "PASS":
                continue

            answers.append({
                "answer": answer,
                "reason": reason_text,
                "docs": answer_docs,
                "verification": verification,
            })

        answers.sort(key=lambda item: item["verification"]["score"], reverse=True)
        return answers[0] if answers else None

    def graph_paths(self, max_graphs=None,views='inline'):
        paths = sorted(self.graph_root.glob(f"*_properties_graph_{views}*.json"))
        return paths[:max_graphs] if max_graphs else paths

    def write_rows(self, rows):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = dedupe_rows(rows)
        with open(self.output_path, "w") as file:
            for row in rows:
                file.write(json.dumps(row, default=str) + "\n")

    def start_output(self, truncate=False):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.written_row_keys = set()
        if truncate:
            self.output_path.write_text("")
        elif self.output_path.exists():
            for row in read_jsonl(self.output_path):
                self.written_row_keys.add(row_key(row))
        else:
            self.output_path.touch()
        self.output_started = True

    def append_row(self, row):
        if not self.output_started:
            self.start_output(truncate=False)

        key = row_key(row)
        if key in self.written_row_keys:
            return False

        self.written_row_keys.add(key)
        with open(self.output_path, "a") as file:
            file.write(json.dumps(row, default=str) + "\n")
            file.flush()
        print(f"[ROW SAVED] {row.get('sample_id')}: {row.get('question')}")
        return True


def dedupe_docs(docs):
    seen = set()
    output = []
    for doc in docs:
        key = (
            doc.metadata.get("object_id"),
            doc.metadata.get("edge"),
            json.dumps(doc.metadata.get("target"), sort_keys=True, default=str),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(doc)
    return output


def docs_to_text(docs):
    return "\n".join(doc.page_content for doc in docs)



def verify_answer(answer, evidence_text):
    result = check(answer, [evidence_text])
    return {
        "verdict": getattr(result, "verdict", ""),
        "score": float(getattr(result, "trust_score", 0.0) or 0.0),
    }


def sample_id_from_graph(graph_path):
    stem = Path(graph_path).stem
    suffixes = (
        "_db_extract_properties_graph_inline_properties_2d_graph",
        "_db_extract_properties_graph_crossline_properties_2d_graph",
        "_db_extract_properties_graph_timeslice_properties_2d_graph",
        "_db_extract_properties_graph",
        "_properties_graph",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem.replace("_db_extract", "")


def category_from_sample_id(sample_id):
    categories = (
        "salt_fault_mixed",
        "fault_complex",
        "fault_only",
        "full_mixed",
        "salt_only",
        "depositional",
        "boring",
        "onlap",
    )
    for category in categories:
        if f"_{category}_" in sample_id or sample_id.endswith(f"_{category}"):
            return category
    return "unknown"


def view_from_graph(graph_path):
    name = Path(graph_path).name
    if "_inline_properties_2d_graph" in name:
        return "inline"
    if "_crossline_properties_2d_graph" in name:
        return "crossline"
    if "_timeslice_properties_2d_graph" in name:
        return "timeslice"
    return "volume"


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def question_signature(question):
    question = normalize_text(question)
    question = re.sub(r"[^a-z0-9\s]", "", question)
    question = re.sub(r"\b(the|a|an)\b", " ", question)
    question = re.sub(r"\bdoes\s+", "", question)
    question = re.sub(r"\bis\s+", "", question)
    question = re.sub(r"\bare\s+", "", question)
    question = re.sub(r"\s+", " ", question).strip()
    question = re.sub(r"\bexists\b", "present", question)
    question = re.sub(r"\bexist\b", "present", question)
    question = re.sub(r"\bpresent in section\b", "present", question)
    question = re.sub(r"\bin section\b", "", question)
    question = re.sub(r"\s+", " ", question).strip()
    return question


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


def dedupe_rows(rows):
    seen = set()
    output = []
    for row in rows:
        key = (
            row.get("sample_id", ""),
            row.get("view", ""),
            question_signature(row.get("question", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def row_key(row):
    return (
        row.get("sample_id", ""),
        row.get("view", ""),
        question_signature(row.get("question", "")),
    )


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def generate_multimodal_dataset(graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT, max_graphs=None):
    workflow = RagWorkflow(graph_root=graph_root, output_path=output_path)
    return workflow.generate_dataset(max_graphs=max_graphs,graph_views='inline',
                                     candidates_per_question=100, questions_per_graph=100)

if __name__ == "__main__":
    rows = generate_multimodal_dataset()
    print(json.dumps({
        "rows": len(rows),
        "output": DEFAULT_OUTPUT.as_posix(),
    }, indent=2))
