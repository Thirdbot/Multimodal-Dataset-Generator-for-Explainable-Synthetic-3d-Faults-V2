import argparse
# watch properties graphs and generate verified dataset rows
import json
import re
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LLM.vllm_client import DEFAULT_VLLM_ENDPOINT, generate_hypotheses_for_graph
from NLI.verifier import (
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_NLI_MODEL,
    NliGraphVerifier,
)
from Tracer.tracer import EvidenceTracer
from scripts.logger_color import logger


default_graph_root = ROOT / "traces" / "properties_graph"
default_output = ROOT / "Dataset" / "verified_hypotheses.jsonl"
llm_config_path = ROOT / "LLM" / "config.json"
nli_config_path = ROOT / "NLI" / "config.json"


class DatasetPipeline(object):
    def __init__(self, graph_root=default_graph_root, output_path=default_output, task="structural_interpretation"):
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.task = task
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

    def build_sample_row(self, graph_path, max_attempts=5, batch_size=5, evidence_limit=12, verbose=False):
        seen = set()
        sample_id = self.sample_id_from_graph(graph_path)
        supported = []
        nli_counts = {"entail": 0, "contradict": 0, "neutral": 0}

        for attempt in range(1, max_attempts + 1):
            generation = generate_hypotheses_for_graph(
                graph_path,
                endpoint=self.llm_config.get("vllm_endpoint", DEFAULT_VLLM_ENDPOINT),
                model=self.llm_config.get("model"),
                evidence_limit=evidence_limit,
                count=batch_size,
                seed=attempt,
                task=self.task,
            )

            for hypothesis in generation["hypotheses"]:
                normalized = self.normalize_text(hypothesis)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)

                verification = self.verifier.verify_graph_claim(graph_path, hypothesis)
                nli_bucket = self.nli_bucket(verification)
                nli_counts[nli_bucket] += 1
                if verbose:
                    logger.info(
                        f"[{sample_id}] {verification['status']} {verification['score']:.3f}: {hypothesis}"
                    )

                if verification["status"] != "supported":
                    continue
                if not self.is_task_supported_hypothesis(hypothesis):
                    continue

                supported.append({
                    "hypothesis": hypothesis,
                    "attempt": attempt,
                    "verification": verification,
                    "generation": generation,
                })

        if supported:
            best = max(
                supported,
                key=lambda item: self.supported_candidate_score(item["hypothesis"], item["verification"]),
            )
            evidence = self.selected_evidence(best["verification"])
            row = {
                "sample_id": sample_id,
                "instruction": self.instruction_text(self.task),
                "input": "",
                "answer": best["hypothesis"],
                "evidence": evidence,
                "verification": {
                    "status": best["verification"]["status"],
                    "score": best["verification"]["score"],
                },
                "metadata": {
                    "graph_path": graph_path.as_posix(),
                    "category": self.category_from_graph(graph_path),
                    "task": self.task,
                    "attempt": best["attempt"],
                    "evidence_count": len(evidence),
                    "tested_hypotheses": len(seen),
                },
                "trace": {
                    "llm_prompt": best["generation"].get("prompt", ""),
                    "llm_raw_output": best["generation"].get("raw_output", ""),
                    "llm_answer": best["hypothesis"],
                    "graph_trace": best["generation"].get("evidence", []),
                    "retrieved_evidence": best["verification"].get("retrieved_evidence", []),
                    "deciding_evidence": best["verification"].get("deciding_evidence", {}),
                    "nli_model": best["verification"].get("model", ""),
                },
            }
            return row, {
                "sample_id": sample_id,
                "status": "saved",
                "task": self.task,
                "attempt": best["attempt"],
                "tested_hypotheses": len(seen),
                "supported_candidates": len(supported),
                "nli_score": best["verification"]["score"],
                "nli_counts": nli_counts,
                "answer": best["hypothesis"],
            }

        return None, {
            "sample_id": sample_id,
            "status": "no_entailed_hypothesis",
            "task": self.task,
            "attempts": max_attempts,
            "tested_hypotheses": len(seen),
            "supported_candidates": 0,
            "nli_counts": nli_counts,
        }

    def process_graph(self, graph_path, max_attempts=5, batch_size=3, evidence_limit=12, verbose=False):
        graph_path = Path(graph_path)
        if not graph_path.exists():
            return None

        logger.info(f"[DATASET START] -> Graph: {graph_path.name}")
        row, status = self.build_sample_row(
            graph_path,
            max_attempts=max_attempts,
            batch_size=batch_size,
            evidence_limit=evidence_limit,
            verbose=verbose,
        )

        sample_id = self.sample_id_from_graph(graph_path)
        if row is None:
            logger.warning(
                f"[DATASET SKIP] -> Sample: {sample_id} Status: {status['status']}"
            )
            self.accumulate_nli_counts(status.get("nli_counts", {}))
            self.log_summary(status)
            return status

        self.rows[sample_id] = row
        self.write_rows()
        logger.info(
            f"[DATASET SAVE] -> Sample: {sample_id} Output: {self.output_path}"
        )
        self.accumulate_nli_counts(status.get("nli_counts", {}))
        self.log_summary(status)
        return status

    def accumulate_nli_counts(self, counts):
        for key in self.nli_totals:
            self.nli_totals[key] += int(counts.get(key, 0) or 0)

    @staticmethod
    def nli_bucket(verification):
        status = str(verification.get("status", "")).lower()
        if status == "supported":
            return "entail"
        if status == "contradicted":
            return "contradict"
        return "neutral"

    def log_summary(self, status):
        counts = status.get("nli_counts", {})
        logger.info(
            "[DATASET SUMMARY] -> "
            f"Sample: {status.get('sample_id')} "
            f"Task: {status.get('task')} "
            f"Status: {status.get('status')} "
            f"Tested: {status.get('tested_hypotheses', 0)} "
            f"Supported: {status.get('supported_candidates', 0)} "
            f"NLI(sample): entail={counts.get('entail', 0)} "
            f"contradict={counts.get('contradict', 0)} "
            f"neutral={counts.get('neutral', 0)} "
            f"NLI(total): entail={self.nli_totals['entail']} "
            f"contradict={self.nli_totals['contradict']} "
            f"neutral={self.nli_totals['neutral']}"
        )
        if status.get("answer"):
            logger.info(
                "[DATASET ANSWER] -> "
                f"NLI: {float(status.get('nli_score', 0.0)):.3f} "
                f"Text: {status.get('answer')}"
            )

    def remove_graph(self, graph_path):
        sample_id = self.sample_id_from_graph(graph_path)
        if sample_id not in self.rows:
            return False
        self.rows.pop(sample_id, None)
        self.write_rows()
        logger.info(f"[DATASET DELETE] -> Sample: {sample_id}")
        return True

    def watch(self, max_attempts=10, batch_size=5, evidence_limit=100, verbose=False):
        watcher = DatasetWatcher(
            pipeline=self,
            graph_root=self.graph_root,
            max_attempts=max_attempts,
            batch_size=batch_size,
            evidence_limit=evidence_limit,
            verbose=verbose,
        )
        watcher.run()

    def load_existing_rows(self):
        rows = {}
        if not self.output_path.exists():
            return rows

        for line in self.output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id:
                rows[sample_id] = row
        return rows

    def write_rows(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_rows = [self.rows[key] for key in sorted(self.rows)]
        with open(self.output_path, "w") as file:
            for row in ordered_rows:
                file.write(json.dumps(row) + "\n")

    def selected_evidence(self, verification):
        deciding = verification["deciding_evidence"]
        retrieved = verification["retrieved_evidence"]
        selected = []
        seen = set()

        def add_item(item):
            sentence = item.get("sentence")
            if not sentence or sentence in seen:
                return
            seen.add(sentence)
            selected.append(item)

        add_item(deciding)

        def evidence_priority(item):
            source_rank = {
                "relation": 0,
                "property": 1,
                "sample": 2,
            }.get(item.get("source"), 9)
            return (source_rank, -item.get("scores", {}).get("entailment", 0.0))

        for item in sorted(retrieved, key=evidence_priority):
            if item.get("source") not in {"relation", "property"}:
                continue
            add_item(item)
            if len(selected) >= 4:
                break

        compact = []
        for item in selected[:4]:
            compact.append({
                "source": item.get("source"),
                "fact_name": item.get("fact_name"),
                "sentence": item.get("sentence"),
                "score": item.get("scores", {}).get("entailment"),
            })
        return compact

    @classmethod
    def supported_candidate_score(cls, hypothesis, verification):
        text = hypothesis.lower()
        score = float(verification["score"])
        bonus = 0.0

        if any(token in text for token in ("x=", "y=", "z=", "throw", "voxels")):
            bonus -= 0.20
        count_bonus = cls.count_specificity_bonus(text, verification)
        bonus += count_bonus
        if "category" in text or text.startswith("the sample belongs to"):
            bonus -= 0.30
        if text.startswith("has_category:"):
            bonus -= 0.50
        if not text.endswith((".", "!", "?")):
            bonus -= 0.50

        return score + bonus

    @classmethod
    def count_specificity_bonus(cls, text, verification):
        evidence_text = " ".join(
            item.get("sentence", "")
            for item in verification.get("retrieved_evidence", [])
        ).lower()
        bonus = 0.0
        generic_terms = ("multiple", "several", "many")

        mappings = [
            ("fault", cls.extract_count(evidence_text, r"with (\d+) faults?"), "fault"),
            ("closure", cls.extract_count(evidence_text, r"with (\d+) closures?"), "closure"),
        ]

        for _, count, noun in mappings:
            if count is None:
                continue
            if any(term in text for term in generic_terms):
                bonus -= 0.15
            words = cls.count_words(count)
            if count <= 12 and words and f"{words} {noun}" in text:
                bonus += 0.18
            elif re.search(rf"\b{count}\s+{noun}s?\b", text):
                bonus += 0.10

        return bonus

    @staticmethod
    def extract_count(text, pattern):
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def count_words(count):
        mapping = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
        }
        return mapping.get(count)

    def is_task_supported_hypothesis(self, hypothesis):
        if self.task != "fault_detection":
            return True
        text = hypothesis.lower()
        if "fault" not in text:
            return False
        blocked_terms = ("closure", "fluid", "hydrocarbon", "oil", "gas", "brine", "salt", "trap")
        return not any(term in text for term in blocked_terms)

    @staticmethod
    def instruction_text(task="structural_interpretation"):
        if task == "fault_detection":
            return "Detect and describe faults in the seismic sample. State if no fault evidence is present."
        return "Describe the structural interpretation of the seismic sample, focusing on faults, closures, fluids, and related structures."

    @staticmethod
    def category_from_graph(graph_path):
        name = Path(graph_path).stem.replace("_properties_graph", "").replace("_db_extract", "")
        match = re.search(r"recipe_\d+_(.+)", name)
        return match.group(1) if match else "unknown"

    @staticmethod
    def sample_id_from_graph(graph_path):
        name = Path(graph_path).stem
        return name.replace("_properties_graph", "").replace("_db_extract", "")

    @staticmethod
    def normalize_text(text):
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def load_config(path):
        path = Path(path)
        if not path.exists():
            return {}
        return json.loads(path.read_text())


class DatasetWatcher(FileSystemEventHandler):
    def __init__(self, pipeline, graph_root, max_attempts=5, batch_size=3, evidence_limit=100, verbose=False):
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
            return
        self.handle_path(path)

    def handle_deleted(self, path):
        if path.name.endswith("_properties_graph.json"):
            self.processed_mtimes.pop(path.as_posix(), None)
            self.pipeline.remove_graph(path)

    def handle_path(self, path):
        path = Path(path)
        if path.name.endswith("_properties_graph.json") and path.exists():
            current_mtime = path.stat().st_mtime
            key = path.as_posix()
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
    parser.add_argument("--task", default="structural_interpretation", choices=["structural_interpretation", "fault_detection"])
    parser.add_argument("--graph-root", default=str(default_graph_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--evidence-limit", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = DatasetPipeline(
        graph_root=args.graph_root,
        output_path=args.output,
        task=args.task,
    )
    pipeline.watch(
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        evidence_limit=args.evidence_limit,
        verbose=args.verbose,
    )
