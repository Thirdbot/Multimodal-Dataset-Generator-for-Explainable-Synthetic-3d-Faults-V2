import json
import os
import random
import re
import sys
import hashlib
from collections import Counter
from pathlib import Path

from longtracer import LongTracer, check, check_batch
from langchain_core.documents import Document

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
_INSTANCE_ID_RE = re.compile(r"_\d+$")   # "fault_0" is an object; "fault_complex structure" is the section
MIN_RETRIEVAL_SCORE = 0.7
# Answer-coverage gate. Raised 0.7 -> 0.80: good answers ground verbatim-ish and score ~0.98
# (median), so a 0.80 floor trims only the low-confidence tail (~1% of good rows) where loosely
# grounded junk lives -- e.g. an attribute swap that scored 0.79 on shared-subject similarity.
# It is a blunt confidence floor, complementary to the structural attribute/count/topic guards;
# do not push past ~0.82 (yield falls off a cliff: ~15% of good rows gone at 0.85).
MIN_ANSWER_TRUST = 0.80
# A single 2D section carries only a handful of evidence facts per object, so 100
# unique grounded questions is unreachable (MAX_ROWS_PER_EVIDENCE caps repeats) and
# the loop always ground out to MAX_ATTEMPT. Right-sized to what a section supports.
QUESTION_PER_GRAPH  = 5   # sized to the plain-evidence rate (~350 rows/hr, 2.2x the tagged ~157):
                          # at that rate Q=5 covers ALL 408 graphs (204 scenes) in ~5.8h -> full
                          # diversity + max rows (~2040) using the whole 6h window. Q=3 would finish
                          # in ~3.5h (only ~1224 rows); Q=6 overruns 6h and drops to ~175 scenes.
# Only the single best answer is kept, so 100 candidates was ~20x wasted generation +
# retrieval + NLI. 5 gives enough spread. With count<=5 the JSON also fits max_tokens,
# which kills the truncated-JSON -> parser-retry(5x) storm the old count=100 caused.
CANDIDATE_PER_QUESTION = 3   # trimmed 5->3: fewer answer candidates to generate + NLI-verify per question
MAX_ROWS_PER_EVIDENCE = 2
MAX_ATTEMPT = 3 * QUESTION_PER_GRAPH # max attempt for outer loop
# How many individual objects each seed carries -> multi-object questions by default. Set 1 for
# single-object seeds (old behaviour). >1 lets the generator see several coordinate-named objects
# in one batch, so it can ask spatial/relational questions ("the fault at [x,y] vs the closure at
# [a,b]"); retrieval, the coord swap guard, union evidence, and masks/regions are all already
# multi-object. The section doc is always attached on top (counts/mode), same as before.
OBJECTS_PER_SEED = max(1, int(os.environ.get("OBJECTS_PER_SEED", "2")))
# Rotated per batch so questions spread across angles instead of clustering on one
# phrasing. Evidence-gated in the prompt: an angle the Evidences cannot answer is skipped.
QUESTION_FACETS = (
    "whether a structure is present or the section is featureless",
    "how many of a structure there are",
    "where a structure sits in the section",
    "the orientation or geometry of a structure",
    "how two named structures relate",
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

    def generate_dataset(self, max_graphs=None, graph_views=("inline", "crossline"), questions_per_graph=5, candidates_per_question=5, resume=False):
        # resume=True -> append to the existing output and SKIP graphs already turned into rows,
        # so a run can be stopped and continued later without redoing work (or truncating).
        self.start_output(truncate=not resume)
        done = _processed_graphs(self.output_path) if resume else set()
        if done:
            print(f"[RESUME] skipping {len(done)} already-processed graphs")

        for graph_path in self.graph_paths(max_graphs=max_graphs, views=graph_views):
            if graph_path.as_posix() in done:
                continue
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
                    question_query=retrieval_query,
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

                # Row evidence = UNION of the question's grounded facts and the answer's, both
                # per-fact -- masks/regions and stored evidence then cover every object the QA
                # pair rests on, not just the answer's. Dedup keys on the union so overfitting
                # control reflects the real fact set.
                evidence_docs = dedupe_docs([*answer.get("question_docs", []), *answer["docs"]])
                answer_evidence_keys = tuple(sorted(evidence_key(doc) for doc in evidence_docs))
                if answer_evidence_keys and seen_evidences.get(answer_evidence_keys, 0) >= MAX_ROWS_PER_EVIDENCE:
                    print(f"[ROW SKIP] {sample_id}: evidence already used")
                    tally["row_skip"] += 1
                    continue

                seen_evidences[answer_evidence_keys] = seen_evidences.get(answer_evidence_keys, 0) + 1
                row = {
                    "row_id": row_id(sample_id, q, answer["answer"]),
                    "sample_id": sample_id,
                    "category": category,
                    "view": view,
                    "instruction":INSTRUCTION ,
                    "question": q,
                    "answer": answer["answer"],
                    "evidence": serialize_docs(evidence_docs),
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
                        "question_evidence": serialize_docs(answer.get("question_docs", [])),
                        "answer_evidence": serialize_docs(answer["docs"]),
                        "graph_evidence": docs_to_text(evidence_docs).splitlines(),
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

    def evidence_seeds(self, docs):
        # A seed = ONE object's doc + the SECTION doc.
        #
        # Docs are object-level (create_rag.prepare_content), i.e. exactly one doc per object_id
        # and one "<category> structure" doc holding the model-level facts (number_faults,
        # fault_mode, number_fault_intersections).
        #
        # History worth keeping: this used to take a `packet_size` and gather "related" docs of
        # the same object. That made sense when docs were PER-FACT. Once docs went object-level
        # each object_id had exactly one doc, so that loop could never add anything (dead code)
        # and a seed silently became A SINGLE LONE OBJECT with no section facts in it. The
        # question generator still asked "how many faults?" and the answer generator, shown
        # exactly one fault, consistently answered "There is one fault present" -- then grounded
        # it on that same fault's own dip/position, so it was self-consistent and passed
        # verification. Counts were only ever correct when the section doc happened to be the
        # seed. Attaching the section doc to every seed makes counts answerable from real
        # evidence; it is 2 facts, so the 2048-token context is unaffected.
        #
        # A seed now carries OBJECTS_PER_SEED objects (default 2) + the section doc, so the
        # generator can relate/compare several coordinate-named objects in one batch. Objects
        # are shuffled and chunked; the section doc(s) ride on every seed (counts/mode). The
        # loop in generate_for_graph re-creates this generator on StopIteration, so each cycle
        # re-shuffles into fresh object groupings.
        docs = list(docs)

        # object ids are "<type>_<n>" (fault_0); the section doc is "<category> structure".
        section_docs = [
            doc for doc in docs
            if not _INSTANCE_ID_RE.search(str(doc.metadata.get("object_id") or ""))
        ]
        objects = [
            doc for doc in docs
            if _INSTANCE_ID_RE.search(str(doc.metadata.get("object_id") or ""))
        ]
        random.shuffle(objects)

        if not objects:                       # section-only graph (negatives) -> one section seed
            if section_docs:
                yield list(section_docs)
            return

        for i in range(0, len(objects), OBJECTS_PER_SEED):
            yield objects[i:i + OBJECTS_PER_SEED] + section_docs

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

    def best_answer(self, question, question_query, evidence_text, question_docs, retrieve_many, number_of_answer=5):
        answers = []
        # Ground the QUESTION to per-fact evidence (same coverage machinery, lenient). This
        # replaces the raw object-level question_docs -- which over-retrieve whole faults the
        # question never asks about -- with exactly the facts the question's RETRIEVAL_QUERY
        # rests on. The row's evidence is then the UNION of this and the answer's facts, so a
        # comparison question keeps the objects it compares while a single-fact question does
        # not drag in siblings. Uses question_docs as the pool (the question's own objects).
        q_cover = cover_answer(question, question_query, question_docs, require_all=False)
        question_grounding = q_cover["docs"] if q_cover else []
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
                # Retrieve OBJECT docs (question's + the answer-claim's), then verify by
                # COVERAGE: every fact the answer asserts must be entailed by some retrieved
                # object's fact set. cover_answer returns per-fact evidence Documents (one
                # per covered fact) so downstream evidence/masking stays per-fact, while the
                # NLI check runs against coherent objects (no fan-out, no partial-match noise).
                answer_docs = filter_docs_by_retrieval_score(retrieve_many(a_query), MIN_RETRIEVAL_SCORE)
                object_docs = dedupe_docs([*question_docs, *answer_docs])
                covered = cover_answer(a, a_query, object_docs, min_trust=MIN_ANSWER_TRUST)
                if not covered:
                    print("\t[REJECT] a fact not grounded:", a)
                    continue
                grounding = covered["docs"]

                # The object(s) the answer names must actually appear in the grounding,
                # or it is an entity swap (asked Closure 10, answered Closure 8).
                if not answer_objects_in_docs(a, grounding):
                    print("\t[REJECT] answer names an object not in evidence:", a)
                    continue

                # Question-coverage (edge gate): the answer's grounded facts must cover every
                # fact the QUESTION grounds to. Both sides are grounded to per-fact Documents
                # carrying metadata["edge"], so compare edge sets directly instead of the old
                # keyword-facet proxy. A compound question grounds to >1 edge (e.g. dip_deg +
                # number_faults); an answer that drops a clause is missing that edge -> reject.
                # When the question grounds to nothing (q_edges empty) this passes rather than
                # false-rejecting -- strictly no worse than before for un-grounded questions.
                q_edges = {d.metadata.get("edge") for d in question_grounding} - {None}
                a_edges = {d.metadata.get("edge") for d in grounding} - {None}
                if q_edges and not q_edges <= a_edges:
                    print("\t[REJECT] off-topic (question edges not covered):",
                          sorted(q_edges - a_edges), "|", a)
                    continue

                # Topic guard: the object TYPE(s) the question asks about and the answer's
                # object type(s) must not be disjoint -- "how many faults?" -> "salt is
                # present" is a cross-type swap the edge gate misses (empty q_edges).
                q_types, a_types = object_types_in(question), object_types_in(a)
                if q_types and a_types and q_types.isdisjoint(a_types):
                    print("\t[REJECT] topic mismatch (asked", sorted(q_types),
                          "answered", sorted(a_types), "):", a)
                    continue

                # Count guard: a "how many X" question must be answered FROM the matching
                # count edge (number_faults / number_hc_closures / ...); otherwise a count
                # question that grounds to no edge lets any grounded fact through.
                q_count_edge = count_edge_for_question(question)
                if q_count_edge and q_count_edge not in a_edges:
                    print("\t[REJECT] count question not answered with", q_count_edge, ":", a)
                    continue

                # Attribute-consistency guard: a dip / throw / coverage-% the answer asserts must
                # be grounded on the matching edge, else it claims a magnitude the object lacks.
                attr_missing = attr_edges_required(a) - a_edges
                if attr_missing:
                    print("\t[REJECT] attribute not grounded:", sorted(attr_missing), ":", a)
                    continue

                verification = {"verdict": "PASS", "score": covered["score"]}
            except Exception as error:
                print(f"\t[ANSWER CHECK ERROR] {a}: {error}")
                continue

            answers.append({
                "answer": a,
                "docs": grounding,
                "question_docs": question_grounding,
                "verification": verification,
            })

        answers.sort(key=lambda item: item["verification"]["score"], reverse=True)
        return answers[0] if answers else None

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


def _processed_graphs(output_path):
    # Graphs already turned into at least one row (by metadata.graph_path). Used by resume mode
    # to skip them. Graphs here are static symlinks, so presence is enough (no mtime check).
    done = set()
    if Path(output_path).exists():
        for line in Path(output_path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                gp = json.loads(line).get("metadata", {}).get("graph_path")
            except json.JSONDecodeError:
                continue
            if gp:
                done.add(gp)
    return done


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


def ground_per_fact(retrieval_query, answer, docs, min_trust=0.7):
    # A compound answer (e.g. throw AND dip) makes several claims, and each needs its
    # OWN supporting doc. filter_docs_by_trust checks a doc against the WHOLE answer, so
    # a single-fact doc only partially matches a multi-fact answer and hovers at the
    # threshold -> some facts silently lose their evidence (and its object_id, so they
    # also lose the mask). The RETRIEVAL_QUERY already lists one fact per line (the same
    # split retrieve_many uses), so trust-check each fact and union the docs that support
    # it -- every fact keeps its grounding. Falls back to the whole answer if the query
    # is a single line, so single-fact answers behave exactly as before.
    facts = [line.strip() for line in str(retrieval_query).splitlines() if line.strip()] or [answer]
    kept, seen = [], set()
    for fact in facts:
        for doc in filter_docs_by_trust(fact, docs, min_trust=min_trust):
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                kept.append(doc)
    return kept


_NLI_TAG_RE = re.compile(r"</?(?:object|nums|center|bbox|SEG)>")


def _untag(text):
    # Evidence tags (<object>,<nums>,<center>,<bbox>) are copy-slots for the VLM -- pure
    # noise to NLI. Left in, they drag a real fact's entailment under threshold (a true
    # throw scored 0.616 tagged vs 0.823 stripped), so verification must compare tag-free
    # text on both sides. Same reasoning as the embedder's _TagStrippingEmbeddings; the
    # stored evidence keeps its tags untouched.
    return _NLI_TAG_RE.sub("", str(text))


def cover_answer(answer, retrieval_query, object_docs, min_trust=0.7, require_all=True):
    # Coverage over the SHARED fact pool -- every fact of every retrieved object (the
    # question's AND the answer's). The RETRIEVAL_QUERY is already one fact per line, so
    # split it simply on "\n"; for each line keep the pool fact that best ENTAILS it by
    # NLI trust. Using BOTH evidences (old filter_docs_by_trust strength) means a fact the
    # QUESTION retrieved can ground the answer. Returns None if any line is uncovered;
    # else {"docs", "score"}.
    if not object_docs:
        return None
    pool = [(doc, fact)
            for doc in object_docs
            for fact in (doc.metadata.get("facts") or [])
            if fact.get("text")]
    if not pool:
        return None

    claims = [line.strip() for line in str(retrieval_query).splitlines() if line.strip()] or [str(answer)]
    pairs, items = [], []
    for claim in claims:
        for doc, fact in pool:
            pairs.append((claim, doc, fact))
            items.append({"response": _untag(claim), "sources": [_untag(fact["text"])]})
    results = check_batch(items, max_workers=min(4, max(1, len(items))))

    best = {}  # claim -> (score, doc, fact) : the pool fact that best entails the claim
    for (claim, doc, fact), result in zip(pairs, results):
        score = float(getattr(result, "trust_score", 0.0) or 0.0)
        if getattr(result, "verdict", "") == "PASS" and score >= min_trust:
            if claim not in best or score > best[claim][0]:
                best[claim] = (score, doc, fact)

    if require_all and len(best) < len(claims):
        return None  # a claim the answer makes is entailed by no fact -> reject
    # require_all=False (question side): keep whatever grounded -- the question's evidence is
    # ADDITIVE (it widens the row's fact set to the objects the question rests on), not a gate.

    # --- Compound completeness (reverse NLI) ---
    # The RETRIEVAL_QUERY can under-list a compound answer's facts (the 1.5B often copies one
    # line for a two-fact answer), leaving real facts the answer states ungrounded. Retrieval
    # already put every fact of the answer's objects in the pool, so instead of one-evidence-
    # per-query-line, ask each such fact whether the ANSWER entails it (response=fact,
    # sources=[answer]). Facts the answer actually asserts get attached; the rest are ignored.
    # NLI decides -- no answer-splitting, no number-matching. Restricted to the object(s) the
    # answer names to keep it cheap and block coincidental cross-object entailment. Objects are
    # named by COORDINATE now ("the fault at [x,y]"), so identity = the coord the fact/answer
    # cites, not a "Type N" token.
    named = coord_refs(answer)
    grounded_keys = {(doc.metadata.get("object_id"), fact.get("edge"), str(fact.get("target")))
                     for _, (_, doc, fact) in best.items()}
    rev_pool = [(doc, fact) for doc, fact in pool
                if named and (coord_refs(fact.get("text")) & named)]
    if rev_pool:
        rev_items = [{"response": _untag(fact["text"]), "sources": [_untag(answer)]} for _, fact in rev_pool]
        rev_results = check_batch(rev_items, max_workers=min(4, max(1, len(rev_items))))
        for (doc, fact), result in zip(rev_pool, rev_results):
            score = float(getattr(result, "trust_score", 0.0) or 0.0)
            if getattr(result, "verdict", "") == "PASS" and score >= min_trust:
                key = (doc.metadata.get("object_id"), fact.get("edge"), str(fact.get("target")))
                if key not in grounded_keys:
                    grounded_keys.add(key)
                    best[f"__ans::{fact['text']}"] = (score, doc, fact)

    evidence, scores, seen = [], [], set()
    for claim, (score, doc, fact) in best.items():
        key = (doc.metadata.get("object_id"), fact.get("edge"), str(fact.get("target")))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(Document(
            page_content=fact.get("text"),
            metadata={
                "trace_type": fact.get("trace_type"),
                "source": doc.metadata.get("object_id"),
                "object_id": doc.metadata.get("object_id"),
                "parent_id": doc.metadata.get("parent_id"),
                "category_id": doc.metadata.get("category_id"),
                "edge": fact.get("edge"),
                "target": fact.get("target"),
                "relation": fact.get("relation"),
                "trust_score": score,
            },
        ))
        scores.append(score)
    if not evidence:
        return None
    return {"docs": evidence, "score": sum(scores) / len(scores)}


def _match_fact_in_doc(fact_line, doc):
    # Pick which fact of the object the covered answer-fact refers to, so evidence + mask
    # route to the right (object_id, edge, target). Score each fact's SENTENCE against the
    # answer line by shared numbers first (values like 80.5 or [23,133.5]), then shared
    # words -- robust for dict-valued position/bbox facts and for value-free facts (fluid,
    # intersects) that the raw target string can't match.
    facts = doc.metadata.get("facts") or []
    if not facts:
        return None
    line_nums = set(re.findall(r"\d+\.?\d*", str(fact_line)))
    line_words = set(re.findall(r"[a-z]+", str(fact_line).lower()))
    best, best_score = facts[0], (-1, -1)
    for fact in facts:
        text = str(fact.get("text") or "")
        score = (len(line_nums & set(re.findall(r"\d+\.?\d*", text))),
                 len(line_words & set(re.findall(r"[a-z]+", text.lower()))))
        if score > best_score:
            best_score, best = score, fact
    return best


_COORD_GROUP_RE = re.compile(r"[\[\(]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*[\]\)]")


def _norm_coord_num(token):
    token = token.strip()
    try:
        value = float(token)
    except ValueError:
        return token
    return str(int(value)) if value.is_integer() else str(round(value, 4))


def coord_refs(text):
    # Every coordinate GROUP a text cites, as a normalized number tuple -- a center [x,y]
    # or a box [x1,y1,x2,y2], in any bracket style ([..] or (..)), number-normalized so
    # "289" and "289.0" match. Whole groups only, so a 4-number box is never mistaken for
    # two centers. This is the object's identity now that objects are named by coordinate.
    refs = set()
    for match in _COORD_GROUP_RE.finditer(str(text or "")):
        refs.add(tuple(_norm_coord_num(part) for part in match.group(1).split(",")))
    return refs


def answer_objects_in_docs(answer, docs):
    # Objects are named by COORDINATE ("the fault at [x,y]"), so an answer that refers to an
    # object must cite a coordinate that actually appears in the evidence. Every coordinate
    # group the answer states must be present in the docs -- a coordinate not in evidence is a
    # swap or a fabrication. NLI still judges the RELATION itself (e.g. "the fault at [..] is
    # left of the closure at [..]"); this only pins the object identities to real evidence.
    evidence_coords = coord_refs(docs_to_text(docs))
    answer_coords = coord_refs(answer)
    if answer_coords and not answer_coords <= evidence_coords:
        return False
    return True


# Topic guards (close the cross-type / count cross-wiring the edge gate misses: on a multi-type
# seed the 1.5B answers "how many faults?" with "salt is present", and since a count question
# often grounds to NO edge, q_edges is empty and the edge gate is skipped). Conservative --
# they only reject a CLEARLY off-topic answer, so legit compound/comparative answers pass.
_OBJECT_TYPE_WORDS = {"fault": "fault", "faults": "fault",
                      "closure": "closure", "closures": "closure",
                      "salt": "salt", "onlap": "onlap"}
_COUNT_Q_RE = re.compile(r"\b(how many|number of|how much)\b", re.I)


def object_types_in(text):
    """The set of object TYPES (fault/closure/salt/onlap) a text names."""
    t = str(text or "").lower()
    return {canon for word, canon in _OBJECT_TYPE_WORDS.items()
            if re.search(rf"\b{word}\b", t)}


def count_edge_for_question(question):
    """If the question is a count question, the number_* edge it asks for, else None."""
    q = str(question or "").lower()
    if not _COUNT_Q_RE.search(q):
        return None
    if "intersection" in q:
        return "number_fault_intersections"
    if re.search(r"\bonlap\b", q):
        return "number_onlap_episodes"
    if re.search(r"\bclosures?\b", q):
        return "number_hc_closures"
    if re.search(r"\bfaults?\b", q):
        return "number_faults"
    return None


# Attribute-consistency guard: a MAGNITUDE the answer's wording asserts (a dip, a throw, a
# coverage %) must be grounded on the matching edge -- else the answer claims a property the
# object doesn't actually have in evidence ("onlap appears steeply dipping" grounded only on
# area_pct; "onlap covers 20%" grounded only on extent). Patterns require a real value/keyword,
# not the ambiguous bare word "area" (which "occupies the area from [bbox]" -- an extent claim --
# would otherwise trip). Mirrors count_edge_for_question but for continuous measures.
_ATTR_CLAIM_EDGE = (
    (re.compile(r"\bdips?\b|\bdipping\b|\d\s*degrees?", re.I), "dip_deg"),
    (re.compile(r"throw of about|throw of \d|throw is|displacement of \d", re.I), "throw"),
    (re.compile(r"covers about \d|covers approximately \d|\d+\.?\d*\s*percent", re.I), "area_pct"),
)


def attr_edges_required(answer):
    """The measure edges the answer's wording claims -- each must be grounded, or it is asserting
    an attribute the evidence does not carry for that object."""
    a = str(answer or "")
    return {edge for pat, edge in _ATTR_CLAIM_EDGE if pat.search(a)}


# Question-coverage gate: the answer must address what the question ASKS, not just be grounded
# (catches "where is X?" answered with a dip). Enforced in best_answer by comparing the
# question's grounded edges to the answer's grounded edges (the edge gate), which replaced the
# old keyword-facet proxy. Mechanical, not a correctness check (that's the model's job).
# Question GENERATION still spreads across QUESTION_FACETS for variety -- separate, untouched.


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


def generate_multimodal_dataset(graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT, max_graphs=None, resume=False):
    workflow = RagWorkflow(graph_root=graph_root, output_path=output_path)
    return workflow.generate_dataset(max_graphs=max_graphs, graph_views=("inline", "crossline"),
                                     candidates_per_question=CANDIDATE_PER_QUESTION, questions_per_graph=QUESTION_PER_GRAPH,
                                     resume=resume)

if __name__ == "__main__":
    rows = generate_multimodal_dataset()
    print(json.dumps({
        "rows": len(rows),
        "output": DEFAULT_OUTPUT.as_posix(),
    }, indent=2))
