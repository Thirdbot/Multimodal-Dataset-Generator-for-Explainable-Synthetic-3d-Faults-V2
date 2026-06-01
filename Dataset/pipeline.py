# automated dataset generation from properties graphs
import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

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


default_graph_root = ROOT / "traces" / "properties_graph"
default_output = ROOT / "Dataset" / "verified_hypotheses.jsonl"
llm_config_path = ROOT / "LLM" / "config.json"
nli_config_path = ROOT / "NLI" / "config.json"


class DatasetPipeline(object):
    def __init__(self, graph_root=default_graph_root, output_path=default_output):
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.llm_config = self.load_config(llm_config_path)
        self.nli_config = self.load_config(nli_config_path)

    def run(
        self,
        max_attempts=5,
        batch_size=5,
        evidence_limit=12,
        append=False,
        verbose=False,
    ):
        graph_paths = self.list_graph_paths()
        if not graph_paths:
            raise FileNotFoundError(f"no properties graph found in {self.graph_root}")

        verifier = NliGraphVerifier(
            model_name=self.nli_config.get("model", DEFAULT_NLI_MODEL),
            top_k=self.nli_config.get("top_k", 8),
            entailment_threshold=self.nli_config.get("entailment_threshold", DEFAULT_ENTAILMENT_THRESHOLD),
            contradiction_threshold=self.nli_config.get("contradiction_threshold", DEFAULT_CONTRADICTION_THRESHOLD),
        )

        mode = "a" if append else "w"
        rows = []
        report = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, mode) as file:
            for graph_path in tqdm(graph_paths, desc="building dataset"):
                row, status = self.build_sample_row(
                    graph_path,
                    verifier,
                    max_attempts=max_attempts,
                    batch_size=batch_size,
                    evidence_limit=evidence_limit,
                    verbose=verbose,
                )
                report.append(status)
                if row is None:
                    continue

                file.write(json.dumps(row) + "\n")
                file.flush()
                rows.append(row)

        return {
            "graph_count": len(graph_paths),
            "saved_rows": len(rows),
            "unsupported_graphs": len([item for item in report if item["status"] != "saved"]),
            "output_path": self.output_path.as_posix(),
            "report": report,
        }

    def build_sample_row(self, graph_path, verifier, max_attempts=5, batch_size=5, evidence_limit=12, verbose=False):
        seen = set()
        sample_id = self.sample_id_from_graph(graph_path)

        for attempt in range(1, max_attempts + 1):
            generation = generate_hypotheses_for_graph(
                graph_path,
                endpoint=self.llm_config.get("vllm_endpoint", DEFAULT_VLLM_ENDPOINT),
                model=self.llm_config.get("model"),
                evidence_limit=evidence_limit,
                count=batch_size,
                seed=attempt,
            )

            for hypothesis in generation["hypotheses"]:
                normalized = self.normalize_text(hypothesis)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)

                verification = verifier.verify_graph_claim(
                    graph_path,
                    hypothesis,
                )
                if verbose:
                    print(f"[{sample_id}] {verification['status']} {verification['score']:.3f}: {hypothesis}")

                if verification["status"] != "supported":
                    continue

                evidence = self.selected_evidence(verification["retrieved_evidence"])
                row = {
                    "sample_id": sample_id,
                    "instruction": self.instruction_text(),
                    "input": "",
                    "answer": hypothesis,
                    "evidence": evidence,
                    "verification": {
                        "status": verification["status"],
                        "score": verification["score"],
                    },
                    "metadata": {
                        "graph_path": graph_path.as_posix(),
                        "category": self.category_from_graph(graph_path),
                        "attempt": attempt,
                        "evidence_count": len(evidence),
                    },
                }
                return row, {
                    "sample_id": sample_id,
                    "status": "saved",
                    "attempt": attempt,
                    "tested_hypotheses": len(seen),
                }

        return None, {
            "sample_id": sample_id,
            "status": "no_entailed_hypothesis",
            "attempts": max_attempts,
            "tested_hypotheses": len(seen),
        }

    def check(self, evidence_limit=8):
        graph_paths = self.list_graph_paths()
        checks = []
        for graph_path in graph_paths:
            tracer = EvidenceTracer(graph_path)
            evidence = tracer.structural_evidence()
            checks.append({
                "sample_id": self.sample_id_from_graph(graph_path),
                "graph_path": graph_path.as_posix(),
                "evidence_count": len(evidence),
                "preview": evidence[:evidence_limit],
                "instruction": self.instruction_text(),
            })
        return checks

    def list_graph_paths(self):
        return sorted(self.graph_root.glob("*_properties_graph.json"))

    def selected_evidence(self, evidence):
        selected = [
            item for item in evidence
            if item.get("source") in {"relation", "property"}
        ]
        compact = []
        for item in selected[:4]:
            compact.append({
                "source": item.get("source"),
                "fact_name": item.get("fact_name"),
                "sentence": item.get("sentence"),
                "score": item.get("scores", {}).get("entailment"),
            })
        return compact

    @staticmethod
    def instruction_text():
        return "Describe the structural interpretation of the seismic sample, focusing on faults, closures, fluids, and related structures."

    @staticmethod
    def category_from_graph(graph_path):
        name = Path(graph_path).stem.replace("_properties_graph", "")
        match = re.search(r"recipe_\d+_(.+)", name)
        return match.group(1) if match else "unknown"

    @staticmethod
    def sample_id_from_graph(graph_path):
        name = Path(graph_path).stem
        return name.replace("_properties_graph", "")

    @staticmethod
    def normalize_text(text):
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def load_config(path):
        path = Path(path)
        if not path.exists():
            return {}
        return json.loads(path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one supported dataset row per properties graph.")
    parser.add_argument("--graph-root", default=str(default_graph_root))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--evidence-limit", type=int, default=12)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args.graph_root, args.output)
    if args.check:
        result = pipeline.check(evidence_limit=args.evidence_limit)
    else:
        result = pipeline.run(
            max_attempts=args.max_attempts,
            batch_size=args.batch_size,
            evidence_limit=args.evidence_limit,
            append=args.append,
            verbose=args.verbose,
        )
    print(json.dumps(result, indent=2, default=str))
