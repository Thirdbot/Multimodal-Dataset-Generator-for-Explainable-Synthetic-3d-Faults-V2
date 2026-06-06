import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LLM.vllm_client import DEFAULT_VLLM_ENDPOINT, generate_hypotheses_for_graph, generate_questions_for_graph
from NLI.verifier import (
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_NLI_MODEL,
    NliGraphVerifier,
)
from scripts.logger_color import logger


default_graph_root = ROOT / "traces" / "properties_graph"
default_output = ROOT / "Dataset" / "verified_hypotheses.jsonl"
llm_config_path = ROOT / "LLM" / "config.json"
nli_config_path = ROOT / "NLI" / "config.json"


class DatasetPipeline(object):
    def __init__(self, graph_root=default_graph_root, output_path=default_output):
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.nli_totals = {"entail": 0, "contradict": 0, "neutral": 0}
        self.llm_config = self.load_config(llm_config_path)
        self.nli_config = self.load_config(nli_config_path)
        self.rows = self.load_existing_rows()
        self.verifier = NliGraphVerifier(
            model_name=self.nli_config.get("model", DEFAULT_NLI_MODEL),
            top_k=self.nli_config.get("top_k", 8),
            entailment_threshold=self.nli_config.get("entailment_threshold", DEFAULT_ENTAILMENT_THRESHOLD),
            contradiction_threshold=self.nli_config.get("contradiction_threshold", DEFAULT_CONTRADICTION_THRESHOLD),
        )

    def build_sample_rows(self, graph_path, max_attempts=5, batch_size=5, evidence_limit=None, verbose=False):
        graph_path = Path(graph_path)
        sample_id = self.sample_id_from_graph(graph_path)
        seen = set()
        supported = []
        nli_counts = {"entail": 0, "contradict": 0, "neutral": 0}

        for attempt in range(1, max_attempts + 1):
            question_generation = generate_questions_for_graph(
                graph_path,
                endpoint=self.llm_config.get("vllm_endpoint", DEFAULT_VLLM_ENDPOINT),
                model=self.llm_config.get("model"),
                evidence_limit=evidence_limit,
                count=batch_size,
                seed=attempt,
            )

            for question_index, question in enumerate(question_generation.get("questions", []), start=1):
                question_key = self.normalize_text(question)
                if not question_key:
                    continue

                generation = generate_hypotheses_for_graph(
                    graph_path,
                    endpoint=self.llm_config.get("vllm_endpoint", DEFAULT_VLLM_ENDPOINT),
                    model=self.llm_config.get("model"),
                    evidence_limit=evidence_limit,
                    count=batch_size,
                    seed=(attempt * 1000) + question_index,
                    question=question,
                )

                for hypothesis in generation.get("hypotheses", []):
                    normalized = self.normalize_text(f"{question} {hypothesis}")
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)

                    verification = self.verifier.verify_graph_claim(graph_path, hypothesis)
                    bucket = self.nli_bucket(verification)
                    nli_counts[bucket] += 1

                    if verbose:
                        logger.info(f"[{sample_id}] {verification['status']} {verification['score']:.3f}: Q: {question} A: {hypothesis}")

                    if self.is_supported(verification):
                        supported.append({
                            "question": question,
                            "hypothesis": hypothesis,
                            "attempt": attempt,
                            "question_generation": question_generation,
                            "generation": generation,
                            "verification": verification,
                        })

        if not supported:
            return [], self.status(
                sample_id,
                "no_supported_answer",
                tested=len(seen),
                supported=0,
                counts=nli_counts,
            )

        rows = [
            self.row_from_supported_item(graph_path, sample_id, item, seen, supported)
            for item in supported
        ]
        return rows, self.status(
            sample_id,
            "saved",
            tested=len(seen),
            supported=len(supported),
            counts=nli_counts,
            answer=rows[0]["answer"] if rows else None,
            score=max(float(item["verification"].get("score", 0.0)) for item in supported),
        )

    def row_from_supported_item(self, graph_path, sample_id, item, seen, supported):
        evidence = self.selected_evidence(item["verification"])
        row_id = self.row_id(sample_id, item["question"], item["hypothesis"])
        return {
            "row_id": row_id,
            "sample_id": sample_id,
            "instruction": item["question"],
            "question": item["question"],
            "answer": item["hypothesis"],
            "evidence": evidence,
            "verification": {
                "status": item["verification"].get("status"),
                "score": item["verification"].get("score"),
            },
            "metadata": {
                "graph_path": graph_path.as_posix(),
                "category": self.category_from_graph(graph_path),
                "attempt": item["attempt"],
                "question_count": len(item["question_generation"].get("questions", [])),
                "tested_hypotheses": len(seen),
                "supported_candidates": len(supported),
                "evidence_count": len(evidence),
            },
            "trace": {
                "question_prompt": item["question_generation"].get("prompt", ""),
                "question_raw_output": item["question_generation"].get("raw_output", ""),
                "llm_prompt": item["generation"].get("prompt", ""),
                "llm_raw_output": item["generation"].get("raw_output", ""),
                "llm_question": item["question"],
                "llm_answer": item["hypothesis"],
                "graph_evidence": item["generation"].get("evidence", []),
                "retrieved_evidence": item["verification"].get("retrieved_evidence", []),
                "deciding_evidence": item["verification"].get("deciding_evidence", {}),
                "nli_model": item["verification"].get("model", ""),
            },
        }

    def process_graph(self, graph_path, max_attempts=5, batch_size=3, evidence_limit=None, verbose=False):
        graph_path = Path(graph_path)
        if not graph_path.exists():
            return None

        logger.info(f"[DATASET START] -> Graph: {graph_path.name}")
        rows, status = self.build_sample_rows(
            graph_path,
            max_attempts=max_attempts,
            batch_size=batch_size,
            evidence_limit=evidence_limit,
            verbose=verbose,
        )

        if rows:
            self.remove_rows_for_sample(self.sample_id_from_graph(graph_path))
            for row in rows:
                self.rows[row["row_id"]] = row
            self.write_rows()
            logger.info(f"[DATASET SAVE] -> Sample: {status['sample_id']} Rows: {len(rows)} Output: {self.output_path}")
        else:
            logger.warning(f"[DATASET SKIP] -> Sample: {status['sample_id']} Status: {status['status']}")

        self.accumulate_nli_counts(status["nli_counts"])
        self.log_summary(status)
        return status

    def is_supported(self, verification):
        return (
            verification.get("status") == "supported"
            and float(verification.get("score", 0.0)) >= float(self.verifier.entailment_threshold)
        )

    def selected_evidence(self, verification, limit=6):
        selected = []
        seen = set()

        def add(item):
            sentence = item.get("sentence")
            if not sentence or sentence in seen:
                return
            seen.add(sentence)
            selected.append({
                "source": item.get("source"),
                "fact_name": item.get("fact_name"),
                "sentence": sentence,
                "score": item.get("scores", {}).get("entailment"),
            })

        add(verification.get("deciding_evidence", {}))
        retrieved = verification.get("retrieved_evidence", [])
        retrieved = sorted(
            retrieved,
            key=lambda item: item.get("scores", {}).get("entailment", 0.0),
            reverse=True,
        )
        for item in retrieved:
            add(item)
            if len(selected) >= limit:
                break
        return selected

    def remove_graph(self, graph_path):
        sample_id = self.sample_id_from_graph(graph_path)
        removed = self.remove_rows_for_sample(sample_id)
        if not removed:
            return False
        self.write_rows()
        logger.info(f"[DATASET DELETE] -> Sample: {sample_id} Rows: {removed}")
        return True

    def remove_rows_for_sample(self, sample_id):
        keys = [key for key, row in self.rows.items() if row.get("sample_id") == sample_id]
        for key in keys:
            self.rows.pop(key, None)
        return len(keys)

    def watch(self, max_attempts=10, batch_size=5, evidence_limit=None, verbose=False):
        DatasetWatcher(
            pipeline=self,
            graph_root=self.graph_root,
            max_attempts=max_attempts,
            batch_size=batch_size,
            evidence_limit=evidence_limit,
            verbose=verbose,
        ).run()

    def load_existing_rows(self):
        rows = {}
        if not self.output_path.exists():
            return rows
        for line in self.output_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("row_id") or self.row_id(row.get("sample_id", ""), row.get("question", ""), row.get("answer", ""))
            if row_id:
                row["row_id"] = row_id
                rows[row_id] = row
        return rows

    def write_rows(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as file:
            for row_id in sorted(self.rows):
                file.write(json.dumps(self.rows[row_id]) + "\n")

    def accumulate_nli_counts(self, counts):
        for key in self.nli_totals:
            self.nli_totals[key] += int(counts.get(key, 0) or 0)

    def log_summary(self, status):
        counts = status["nli_counts"]
        logger.info(
            "[DATASET SUMMARY] -> "
            f"Sample: {status['sample_id']} "
            f"Status: {status['status']} "
            f"Tested: {status['tested_hypotheses']} "
            f"Supported: {status['supported_candidates']} "
            f"NLI(sample): entail={counts['entail']} contradict={counts['contradict']} neutral={counts['neutral']} "
            f"NLI(total): entail={self.nli_totals['entail']} contradict={self.nli_totals['contradict']} neutral={self.nli_totals['neutral']}"
        )
        if status.get("answer"):
            logger.info(f"[DATASET ANSWER] -> NLI: {float(status.get('nli_score', 0.0)):.3f} Text: {status['answer']}")

    @staticmethod
    def status(sample_id, status, tested, supported, counts, answer=None, score=None):
        return {
            "sample_id": sample_id,
            "status": status,
            "tested_hypotheses": tested,
            "supported_candidates": supported,
            "nli_counts": counts,
            "answer": answer,
            "nli_score": score,
        }

    @staticmethod
    def nli_bucket(verification):
        status = str(verification.get("status", "")).lower()
        if status == "supported":
            return "entail"
        if status == "contradicted":
            return "contradict"
        return "neutral"

    @staticmethod
    def category_from_graph(graph_path):
        name = Path(graph_path).stem.replace("_properties_graph", "").replace("_db_extract", "")
        match = re.search(r"recipe_\d+_(.+)", name)
        return match.group(1) if match else "unknown"

    @staticmethod
    def sample_id_from_graph(graph_path):
        return Path(graph_path).stem.replace("_properties_graph", "").replace("_db_extract", "")

    @staticmethod
    def normalize_text(text):
        return re.sub(r"\s+", " ", str(text).lower()).strip()

    @classmethod
    def row_id(cls, sample_id, question, answer):
        if not sample_id or not question or not answer:
            return ""
        payload = "|".join([
            str(sample_id),
            cls.normalize_text(question),
            cls.normalize_text(answer),
        ])
        return hashlib.sha1(payload.encode()).hexdigest()

    @staticmethod
    def load_config(path):
        path = Path(path)
        return json.loads(path.read_text()) if path.exists() else {}


class DatasetWatcher(FileSystemEventHandler):
    def __init__(self, pipeline, graph_root, max_attempts=5, batch_size=3, evidence_limit=None, verbose=False):
        self.pipeline = pipeline
        self.graph_root = Path(graph_root)
        self.max_attempts = max_attempts
        self.batch_size = batch_size
        self.evidence_limit = evidence_limit
        self.verbose = verbose
        self.processed_mtimes = {}

    def on_any_event(self, event):
        path = Path(getattr(event, "dest_path", event.src_path))
        if event.event_type == "deleted":
            self.handle_deleted(path)
        else:
            self.handle_path(path)

    def handle_deleted(self, path):
        if path.name.endswith("_properties_graph.json"):
            self.processed_mtimes.pop(path.as_posix(), None)
            self.pipeline.remove_graph(path)

    def handle_path(self, path):
        if not path.name.endswith("_properties_graph.json") or not path.exists():
            return
        key = path.as_posix()
        current_mtime = path.stat().st_mtime
        if self.processed_mtimes.get(key) == current_mtime:
            return
        time.sleep(0.2)
        self.processed_mtimes[key] = current_mtime
        self.pipeline.process_graph(
            path,
            max_attempts=self.max_attempts,
            batch_size=self.batch_size,
            evidence_limit=self.evidence_limit,
            verbose=self.verbose,
        )

    def process_existing(self):
        for graph_path in sorted(self.graph_root.glob("*_properties_graph.json")):
            self.handle_path(graph_path)

    def run(self):
        self.graph_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"[NOW MONITORING] -> Path: {self.graph_root}")
        observer = Observer()
        observer.schedule(self, self.graph_root, recursive=False)
        observer.start()
        try:
            self.process_existing()
            while True:
                self.process_existing()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.error("[STOPPING]")
        finally:
            observer.stop()
            observer.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch properties graphs and generate verified dataset rows.")
    parser.add_argument("--graph-root", default=str(default_graph_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--evidence-limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = DatasetPipeline(
        graph_root=args.graph_root,
        output_path=args.output,
    )
    pipeline.watch(
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        evidence_limit=args.evidence_limit,
        verbose=args.verbose,
    )
