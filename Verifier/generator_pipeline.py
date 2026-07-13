import json
import random
import re
import sys
import hashlib
from collections import Counter
from pathlib import Path

from longtracer import LongTracer, check, check_batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Verifier.create_rag import Rag
from Verifier.llm_machine import LLMMachine
from Verifier.rag_verifier import serialize_docs


DEFAULT_GRAPH_ROOT = ROOT / "graphs" / "properties_2d_graph"
DEFAULT_OUTPUT = ROOT / "Dataset" / "verified_qa.jsonl"
# MiniLM cosine rarely clears 0.9 for related-but-not-verbatim sentences, so 0.9
# starved the question/answer gates. Precision is still enforced downstream by the
# NLI trust filter (>=0.7) + entity guard; retrieval only decides candidate recall.
MIN_RETRIEVAL_SCORE = 0.7
# A single 2D section carries only a handful of evidence facts per object, so 100
# unique grounded questions is unreachable (MAX_ROWS_PER_EVIDENCE caps repeats) and
# the loop always ground out to MAX_ATTEMPT. Right-sized to what a section supports.
QUESTION_PER_GRAPH  = 12
# Only the single best answer is kept, so 100 candidates was ~20x wasted generation +
# retrieval + NLI. 5 gives enough spread. With count<=5 the JSON also fits max_tokens,
# which kills the truncated-JSON -> parser-retry(5x) storm the old count=100 caused.
CANDIDATE_PER_QUESTION = 5
MAX_ROWS_PER_EVIDENCE = 2
MAX_ATTEMPT = 3 * QUESTION_PER_GRAPH # max attempt for outer loop
# Rotated per batch so questions spread across angles instead of clustering on one
# phrasing. Evidence-gated in the prompt: an angle the Evidences cannot answer is skipped.
QUESTION_FACETS = (
    "whether a structure is present or the section is featureless",
    "how many of a structure there are",
    "where a structure sits in the section",
    "the orientation or geometry of a structure",
    "how two named structures relate",
    "the geological character or trap type of a structure",   # uses tier-2 readings
    "two related properties of one structure together",       # bundled/compound question
)
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
        self.rows = []
        self._seen_row_ids = set()  # dedup so a reprocess/regen never appends duplicates

    def generate_dataset(self, max_graphs=None, graph_views=("inline", "crossline"), questions_per_graph=5, candidates_per_question=5):
        self.start_output(truncate=True)

        for graph_path in self.graph_paths(max_graphs=max_graphs, views=graph_views):
            self.rows.extend(self.generate_for_graph(
                graph_path,
                questions_per_graph=questions_per_graph,
                candidates_per_question=candidates_per_question,
            ))
        return self.rows

    def generate_for_graph(self, graph_path, questions_per_graph=QUESTION_PER_GRAPH, candidates_per_question=CANDIDATE_PER_QUESTION):
        graph_path = Path(graph_path)
        sample_id = sample_id_from_graph(graph_path)
        category = category_from_sample_id(sample_id)
        view = view_from_graph(graph_path)
        scene = self._scene_metadata(graph_path)

        vector_store, edges = self.rag.mapping_graph_rag(graph_path)
        retrieval = self.rag.graph_retrieval(vector_store, edges) # graph retrieval is not deep enough
        retrieve_many = self.llm.retrieve_many(retrieval)
        all_docs = self.rag.evidence_documents(graph_path)

        # evidences seeds
        number_of_passes_questions = 0
        evidence_seeds = self.evidence_seeds(all_docs)
        seen_evidences = {} # same evidences lead to the same images and cause overfitting
        tally = Counter() # where attempts die, to decide if MAX_ATTEMPT is the bottleneck

        rows = []  # local; do not accumulate on self across every graph (unbounded in the watcher)
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
                    MIN_RETRIEVAL_SCORE
                ) # multiple question evidences
                if not question_docs:
                    print("[REJECT] question:",q)
                    tally["q_reject"] += 1
                    continue

                print("[ACCEPT] question:",q)

                answer = self.best_answer(
                    question=q,
                    evidence_text=seed_text,  # ground on retrieved evidence, not the whole graph (2048-tok context)
                    question_docs=question_docs,
                    retrieve_many=retrieve_many,
                    number_of_answer=candidates_per_question,
                ) # return 1 best answer

                if not answer:
                    print(f"[ANSWER SKIP] {sample_id}: no supported answer")
                    tally["a_reject"] += 1
                    continue
                print("[ACCEPT] answer:", answer["answer"])

                answer_evidence_keys = tuple(sorted(evidence_key(doc) for doc in answer["docs"]))
                if answer_evidence_keys and seen_evidences.get(answer_evidence_keys, 0) >= MAX_ROWS_PER_EVIDENCE:
                    print(f"[ROW SKIP] {sample_id}: evidence already used")
                    tally["row_skip"] += 1
                    continue

                seen_evidences[answer_evidence_keys] = seen_evidences.get(answer_evidence_keys, 0) + 1
                reason = self.generate_reason(
                    question=q,
                    answer=answer["answer"],
                    docs=dedupe_docs(shared_or_fallback_docs(question_docs,answer['docs'])),
                )

                row = {
                    "row_id": row_id(sample_id, q, answer["answer"]),
                    "sample_id": sample_id,
                    "category": category,
                    "view": view,
                    "instruction":INSTRUCTION ,
                    "question": q,
                    "answer": answer["answer"],
                    "evidence": serialize_docs(dedupe_docs(shared_or_fallback_docs(question_docs,answer['docs']))),
                    "verification": answer["verification"],
                    "metadata": {
                        "graph_path": graph_path.as_posix(),
                        "category": category,
                        "view": view,
                        # shared scene image; the row mask is composited downstream
                        # from only the retrieved objects (see DatasetMaker).
                        "image_path": scene.get("image_path", ""),
                        "overlay_path": scene.get("overlay_path", ""),
                        # mtime of the graph this row came from; the watcher reprocesses
                        # a graph whose file is newer than the row that recorded it.
                        "graph_mtime": _safe_mtime(graph_path),
                    },
                    "trace": {
                        "reason": reason,
                        "question_evidence": serialize_docs(question_docs),
                        "answer_evidence": serialize_docs(answer["docs"]),
                        "graph_evidence": docs_to_text(dedupe_docs(shared_or_fallback_docs(question_docs,answer['docs']))).splitlines(),
                    },
                }
                if self.append_row(row):
                    rows.append(row)
                tally["passed"] += 1
                number_of_passes_questions += 1
        print(f"[TALLY] {sample_id} {view}: attempts={attempts}/{MAX_ATTEMPT} "
              f"passed={tally['passed']} q_reject={tally['q_reject']} "
              f"a_reject={tally['a_reject']} row_skip={tally['row_skip']}")
        return rows

    @staticmethod
    def _scene_metadata(graph_path):
        # the 2d graph carries the shared scene's image/mask paths for its view
        try:
            data = json.loads(Path(graph_path).read_text())
        except Exception:
            return {}
        return data.get("scene") or {}

    def evidence_seeds(self, docs, packet_size=3):
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
        facets = ", ".join(random.sample(QUESTION_FACETS, len(QUESTION_FACETS)))
        try:
            response = self.llm.question_batch_generation().invoke({
                "evidences": evidence_text,
                "count": number_of_questions,
                "facets": facets,
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

    def best_answer(self, question, evidence_text, question_docs, retrieve_many, number_of_answer=5):
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

        for item in answer:
            a = item.ANSWER.strip()
            a_query = item.RETRIEVAL_QUERY.strip() or a  # concise claim for retrieval, not the verbose prose answer
            if not a:
                continue
            try:
                # One common evidence set: the question's grounding plus what the
                # answer's own claim pulls, kept only where it entails the answer.
                # Verify the natural answer itself, never the retrieval proxy.
                answer_docs = retrieve_many(a_query)
                answer_docs = filter_docs_by_retrieval_score(answer_docs,MIN_RETRIEVAL_SCORE)
                merge_docs = [*question_docs, *answer_docs]
                shared = dedupe_docs(merge_docs)
                grounding = filter_docs_by_trust(a, shared)
                if not grounding:
                    print("\t[REJECT] not supported by evidence:", a)
                    continue

                # The object(s) the answer names must actually appear in the grounding,
                # or it is an entity swap (asked Closure 10, answered Closure 8) that NLI
                # would wave through on near-duplicate wording.
                if not answer_objects_in_docs(a, grounding):
                    print("\t[REJECT] answer names an object not in evidence:", a)
                    continue

                verification = verify_answer(a, self.rag.format_docs(grounding))
            except Exception as error:
                print(f"\t[ANSWER CHECK ERROR] {a}: {error}")
                continue
            if verification["verdict"] != "PASS":
                print("\t[REJECT] answer:", a)
                continue

            answers.append({
                "answer": a,
                "docs": grounding,
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
            self._seen_row_ids = set()
        else:
            self.output_path.touch()
            self._seen_row_ids = _existing_row_ids(self.output_path)
        self.output_started = True

    def append_row(self, row):
        if not self.output_started:
            self.start_output(truncate=False)

        rid = row.get("row_id")
        if rid in self._seen_row_ids:
            return False
        self._seen_row_ids.add(rid)

        with open(self.output_path, "a") as file:
            file.write(json.dumps(row, default=str) + "\n")
            file.flush()
        print(f"""[ROW SAVED] {row.get('sample_id')}:
                Question: {row.get('question')}
                Answer:{row.get('answer')}
                Evidences:{row.get('evidence')}\n""")
        return True


def _safe_mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _existing_row_ids(output_path):
    ids = set()
    if Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                rid = json.loads(line).get("row_id")
            except json.JSONDecodeError:
                continue
            if rid:
                ids.add(rid)
    return ids


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


def shared_or_fallback_docs(question_docs, answer_docs):
    # Prefer the intersection (docs that are both question-relevant and
    # answer-supporting). When it is empty, fall back to the union so the row
    # keeps evidence instead of being dropped; downstream trust/verification
    # filters still discard docs that do not entail the answer.
    shared = shared_docs(question_docs, answer_docs)
    if shared:
        return shared
    return [*question_docs, *answer_docs]


def filter_docs_by_retrieval_score(docs, min_score):
    return [
        doc for doc in docs
        if float(doc.metadata.get("_similarity_score", 0.0)) >= min_score
    ]


def filter_docs_by_trust(answer, docs, min_trust=0.7):
    # One entailment check per doc (kept per-doc so we know which docs support the
    # answer), but run as a thread-pool batch instead of a serial CPU-NLI loop --
    # this is the hottest verification path once fan-out is reduced.
    if not docs:
        return []
    results = check_batch(
        [{"response": answer, "sources": [doc.page_content]} for doc in docs],
        max_workers=min(4, len(docs)),
    )
    kept = []
    for doc, result in zip(docs, results):
        trust_score = float(getattr(result, "trust_score", 0.0) or 0.0)
        if getattr(result, "verdict", "") == "PASS" and trust_score >= min_trust:
            doc.metadata['trust_score'] = trust_score
            kept.append(doc)
    return kept


def object_mentions(text):
    return {
        normalize_text(match)
        for match in re.findall(r"<object>(.*?)</object>", str(text or ""), flags=re.DOTALL)
    }


def entity_pairs(text):
    # A named object reference like "Fault 1" / "Closure 8" -> (type, id). Capitalised
    # leading word so it targets proper object names, not values ("about 62", "of 2").
    _ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z_-]*)\s*#?\s*(\d+)")
    return {(match.group(1).lower(), match.group(2)) for match in _ENTITY_RE.finditer(str(text or ""))}


def answer_objects_in_docs(answer, docs):
    # The object(s) the answer names must actually appear in the grounding
    # evidence. Blocks entity swaps (asked Closure 10, answered Closure 8) that NLI
    # would wave through on near-duplicate wording -- and, unlike the tag-only
    # check, still bites when the natural answer drops the <object> tag.
    evidence_text = docs_to_text(docs)
    evidence_objects = object_mentions(evidence_text)
    evidence_pairs = entity_pairs(evidence_text)
    evidence_types = {type_name for type_name, _ in evidence_pairs}

    # Tagged objects in the answer must be present verbatim in the evidence.
    tagged = object_mentions(answer)
    if tagged and not tagged <= evidence_objects:
        return False

    # Untagged "Type N": only judge a type the evidence actually enumerates (an
    # unknown type is left to NLI). A known type with an id the evidence never
    # names is a swap -- "Closure 8" when the grounding only holds "Closure 10".
    for type_name, obj_id in entity_pairs(answer):
        if type_name in evidence_types and (type_name, obj_id) not in evidence_pairs:
            return False
    return True


def verify_answer(answer, evidence_text):
    result = check(answer, [evidence_text])
    return {
        "verdict": getattr(result, "verdict", ""),
        "score": min(1.0, max(0.0, float(getattr(result, "trust_score", 0.0) or 0.0))),
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
                                     candidates_per_question=CANDIDATE_PER_QUESTION, questions_per_graph=QUESTION_PER_GRAPH)

if __name__ == "__main__":
    rows = generate_multimodal_dataset()
    print(json.dumps({
        "rows": len(rows),
        "output": DEFAULT_OUTPUT.as_posix(),
    }, indent=2))
