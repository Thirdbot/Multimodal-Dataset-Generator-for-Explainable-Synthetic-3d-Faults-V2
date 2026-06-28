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
DEFAULT_OUTPUT = ROOT / "Dataset" / "verified_qa.jsonl"
MIN_RETRIEVAL_SCORE = 0.6
QUESTION_PER_GRAPH  = 10
CANDIDATE_PER_GRAPH = 5
MAX_ROWS_PER_EVIDENCE = 2
MAX_ATTEMPT = 2 * QUESTION_PER_GRAPH # max attempt for outer loop
INSTRUCTION = (
    "Inspect the seismic images, use the visible regions as visual evidence, "
    "and answer the question with concise geological reasoning."
)

class RagWorkflow(object):
    def __init__(self, graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT):
        LongTracer.init(verbose=False)
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.output_started = False
        self.rag = Rag(embedding_model="all-MiniLM-L6-v2")
        self.llm = LLMMachine()

    def generate_dataset(self, max_graphs=None, graph_views=("inline", "crossline"), questions_per_graph=5, candidates_per_question=5):
        self.start_output(truncate=True)
        rows = []
        for graph_path in self.graph_paths(max_graphs=max_graphs, views=graph_views):
            rows.extend(self.generate_for_graph(
                graph_path,
                questions_per_graph=questions_per_graph,
                candidates_per_question=candidates_per_question,
            ))
        return rows

    def generate_for_graph(self, graph_path, questions_per_graph=5, candidates_per_question=10):
        graph_path = Path(graph_path)
        sample_id = sample_id_from_graph(graph_path)
        category = category_from_sample_id(sample_id)
        view = view_from_graph(graph_path)

        vector_store, edges = self.rag.mapping_graph_rag(graph_path)
        retrieval = self.rag.graph_retrieval(vector_store, edges) # graph retrieval is not deep enough
        retrieve_many = self.llm.retrieve_many(retrieval)
        all_docs = self.rag.evidence_documents(graph_path)
        evidence_text = self.rag.format_docs(all_docs)
        rows = []

        # evidences seeds
        number_of_passes_questions = 0
        evidence_seeds = self.evidence_seeds(all_docs)

        seen_evidences = {} # same evidences lead to the same images and cause overfitting

        attempts = 0
        while number_of_passes_questions < questions_per_graph and attempts < MAX_ATTEMPT: # retry batches regenerations
            attempts += 1
            try:
                evidences_docs = next(evidence_seeds)
            except StopIteration:
                evidence_seeds = self.evidence_seeds(all_docs)
                evidences_docs = next(evidence_seeds)
            seed_text = self.rag.format_docs(evidences_docs)
            question_items = self.generate_question(seed_text, min(3,questions_per_graph - number_of_passes_questions)) # try 3 first, then try left
            if not question_items or question_items == []:
                print(f"[QUESTION SKIP] {sample_id}: no question generated")
                continue


            for question_item in question_items:
                q = question_item.get("question", "")
                retrieval_query = question_item.get("retrieval_query") or q
                question_docs = filter_docs_by_retrieval_score(
                    retrieve_many(retrieval_query),
                    MIN_RETRIEVAL_SCORE,
                ) # multiple question evidences
                if best_doc_score(question_docs) < MIN_RETRIEVAL_SCORE:
                    print("[REJECT] question:",q)
                    continue
                print("[ACCEPT] question:",q)

                answer = self.best_answer(
                    question=q,
                    evidence_text=evidence_text,
                    question_docs=question_docs,
                    retrieve_many=retrieve_many,
                    number_of_answer=candidates_per_question,
                ) # return 1 best answer

                if not answer:
                    print(f"[ANSWER SKIP] {sample_id}: no supported answer")
                    continue
                print("[ACCEPT] answer:", answer["answer"])

                answer_evidence_keys = tuple(sorted(evidence_key(doc) for doc in answer["docs"]))
                if answer_evidence_keys and seen_evidences.get(answer_evidence_keys, 0) >= MAX_ROWS_PER_EVIDENCE:
                    print(f"[ROW SKIP] {sample_id}: evidence already used")
                    continue

                seen_evidences[answer_evidence_keys] = seen_evidences.get(answer_evidence_keys, 0) + 1
                reason = self.generate_reason(
                    question=q,
                    answer=answer["answer"],
                    docs=dedupe_docs([*question_docs, *answer["docs"]]),
                )

                row = {
                    "row_id": row_id(sample_id, q, answer["answer"]),
                    "sample_id": sample_id,
                    "category": category,
                    "view": view,
                    "instruction":INSTRUCTION ,
                    "question": q,
                    "answer": answer["answer"],
                    "evidence": serialize_docs(answer["docs"]),
                    "verification": answer["verification"],
                    "metadata": {
                        "graph_path": graph_path.as_posix(),
                        "category": category,
                        "view": view,
                    },
                    "trace": {
                        "reason": reason,
                        "question_evidence": serialize_docs(question_docs),
                        "answer_evidence": serialize_docs(answer["docs"]),
                        "graph_evidence": docs_to_text(dedupe_docs([*question_docs, *answer["docs"]])).splitlines(),
                    },
                }
                if self.append_row(row):
                    rows.append(row)
                number_of_passes_questions += 1
        return rows

    def evidence_seeds(self, docs, packet_size=1):
        docs = list(docs)
        random.shuffle(docs)
        by_object = {}
        for doc in docs:
            object_id = doc.metadata.get("object_id") or doc.metadata.get("source") or ""
            by_object.setdefault(object_id, []).append(doc)

        for doc in docs:
            object_id = doc.metadata.get("object_id") or doc.metadata.get("source") or ""
            packet = [doc]
            for related in by_object.get(object_id, []):
                if related is doc:
                    continue
                packet.append(related)
                if len(packet) >= packet_size:
                    break
            yield packet

    def generate_question(self, evidence_text, number_of_questions):
        try:
            response = self.llm.question_batch_generation().invoke({
                "evidences": evidence_text,
                "count": number_of_questions,
            })
            if not response:
                return []
            return [
                {
                    "question": item.QUESTION.strip(),
                    "retrieval_query": item.RETRIEVAL_QUERY.strip(),
                }
                for item in response.QUESTIONS
                if item.QUESTION.strip()
            ]
        except Exception as error:
            print(f"[QUESTION ERROR] {error}")
            return []

    @staticmethod
    def filter_docs_by_trust(answer, docs, min_trust=0.7):
        kept = []
        for doc in docs:
            result = check(answer, [doc.page_content])
            if getattr(result, "verdict", "") == "PASS" and float(getattr(result, "trust_score", 0.0)) >= min_trust:
                kept.append(doc)
        return kept

    def best_answer(self, question,evidence_text, question_docs, retrieve_many, number_of_answer=5):
        answers = []
        try:
            response = self.llm.answer_batch_generation().invoke({
                "evidences": evidence_text,
                "question": question,
                "count": number_of_answer,
            })
        except Exception as error:
            print(f"[ANSWER ERROR] {question}: {error}")
            return None
        answer = response.ANSWERS if response else []

        for a in answer:
            a = a.strip()
            if not a:
                continue
            try:
                answer_docs = retrieve_many(a)

                if not answer_docs:
                    continue
                # this is not be use because the evidences is from question, and it is now chunking
                if score_qa_evidence(question_docs, answer_docs) < 0.7:
                    continue
                used_docs = dedupe_docs([*question_docs, *answer_docs])
                used_docs = filter_docs_by_retrieval_score(used_docs, MIN_RETRIEVAL_SCORE)
                filter_by_trust_docs = dedupe_docs(filter_docs_by_trust(a, used_docs))
                if not filter_by_trust_docs:
                    continue
                if not preserves_evidence_tags(a, filter_by_trust_docs):
                    continue
                verification_text = docs_to_text(filter_by_trust_docs)
                verification = verify_answer(a, verification_text) # answer verify evidences
            except Exception as error:
                print(f"\t[ANSWER CHECK ERROR] {a}: {error}")
                continue
            if verification["verdict"] != "PASS":
                print("\t[REJECT] answer:", a)
                continue

            answers.append({
                "answer": a,
                "docs": filter_by_trust_docs,
                "verification": verification,
            })

        answers.sort(key=lambda item: item["verification"]["score"], reverse=True)
        return answers[0] if answers else None

    def generate_reason(self, question, answer, docs):
        evidence_text = docs_to_text(docs)
        try:
            response = self.llm.reason_generation().invoke({
                "evidences": evidence_text,
                "question": question,
                "answer": answer,
            })
            return response.REASON if response else ""
        except Exception as error:
            print(f"[REASON SKIP] {question}: {error}")
            return ""

    def graph_paths(self, max_graphs=None, views=("inline", "crossline")):
        if isinstance(views, str):
            views = (views,)

        paths = []
        for view in views:
            paths.extend(self.graph_root.glob(f"*_properties_graph_{view}*.json"))
        paths = sorted(set(paths))
        return paths[:max_graphs] if max_graphs else paths

    def write_rows(self, rows):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as file:
            for row in rows:
                file.write(json.dumps(row, default=str) + "\n")

    def start_output(self, truncate=False):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            self.output_path.write_text("")
        else:
            self.output_path.touch()
        self.output_started = True

    def append_row(self, row):
        if not self.output_started:
            self.start_output(truncate=False)

        with open(self.output_path, "a") as file:
            file.write(json.dumps(row, default=str) + "\n")
            file.flush()
        print(f"""[ROW SAVED] {row.get('sample_id')}:
                Question: {row.get('question')}
                Answer:{row.get('answer')}
                Evidences:{row.get('evidence')}\n""")
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


def evidence_key(doc):
    return (
        doc.metadata.get("object_id"),
        doc.metadata.get("edge"),
        json.dumps(doc.metadata.get("target"), sort_keys=True, default=str),
        doc.page_content,
    )


def docs_to_text(docs):
    return "\n".join(doc.page_content for doc in docs)


def shared_docs(question_docs, answer_docs):
    question_keys = {evidence_key(doc) for doc in question_docs}
    return [
        doc for doc in answer_docs
        if evidence_key(doc) in question_keys
    ]


def filter_docs_by_retrieval_score(docs, min_score):
    return [
        doc for doc in docs
        if float(doc.metadata.get("_similarity_score", 0.0)) >= min_score
    ]


def filter_docs_by_trust(answer, docs, min_trust=0.7):
    kept = []
    for doc in docs:
        result = check(answer, [doc.page_content])
        trust_score = float(getattr(result, "trust_score", 0.0) or 0.0)

        if getattr(result, "verdict", "") == "PASS" and trust_score >= min_trust:
            doc.metadata['trust_score'] = trust_score
            kept.append(doc)
    return kept


def preserves_evidence_tags(answer, docs):
    evidence_text = docs_to_text(docs)
    required_spans = tagged_spans(evidence_text)
    if not required_spans:
        return True

    used_values = [
        span for span in required_spans
        if strip_tag(span) in answer
    ]
    if not used_values:
        return True

    return all(span in answer for span in used_values)


def tagged_spans(text):
    return re.findall(r"<(?:object|nums|center|bbox)>.*?</(?:object|nums|center|bbox)>", text)


def strip_tag(span):
    return re.sub(r"</?(?:object|nums|center|bbox)>", "", span)


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


def row_id(sample_id, question, answer):
    payload = "|".join([sample_id, normalize_text(question), normalize_text(answer)])
    return hashlib.sha1(payload.encode()).hexdigest()


def generate_multimodal_dataset(graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT, max_graphs=None):
    workflow = RagWorkflow(graph_root=graph_root, output_path=output_path)
    return workflow.generate_dataset(max_graphs=max_graphs, graph_views=("inline", "crossline"),
                                     candidates_per_question=CANDIDATE_PER_GRAPH, questions_per_graph=QUESTION_PER_GRAPH)

if __name__ == "__main__":
    rows = generate_multimodal_dataset()
    print(json.dumps({
        "rows": len(rows),
        "output": DEFAULT_OUTPUT.as_posix(),
    }, indent=2))
