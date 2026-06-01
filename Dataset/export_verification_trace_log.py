import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Dataset.pipeline import DatasetPipeline
from LLM.vllm_client import DEFAULT_VLLM_ENDPOINT, generate_hypotheses_for_graph
from NLI.verifier import (
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_NLI_MODEL,
    NliGraphVerifier,
)
from Tracer.tracer import EvidenceTracer


DEFAULT_GRAPH_ROOT = ROOT / "traces" / "properties_graph"
DEFAULT_OUTPUT = ROOT / "Dataset" / "verification_trace_log.csv"
LLM_CONFIG_PATH = ROOT / "LLM" / "config.json"
NLI_CONFIG_PATH = ROOT / "NLI" / "config.json"


def load_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def list_graph_paths(graph_root, limit=None):
    graph_paths = sorted(Path(graph_root).glob("*_properties_graph.json"))
    return graph_paths[:limit] if limit else graph_paths


def csv_fieldnames():
    return [
        "logged_at",
        "sample_id",
        "graph_path",
        "instruction",
        "attempt",
        "batch_size",
        "evidence_limit",
        "llm_prompt",
        "llm_raw_output",
        "llm_answer",
        "nli_status",
        "nli_entailment",
        "nli_contradiction",
        "nli_neutral",
        "nli_deciding_sentence",
        "nli_deciding_fact_name",
        "nli_model",
        "graph_trace_json",
        "retrieved_evidence_json",
        "source_evidence_json",
    ]


def append_rows(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fieldnames())
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_log_rows_for_graph(
    graph_path,
    verifier,
    llm_config,
    max_attempts=5,
    batch_size=3,
    evidence_limit=12,
):
    tracer = EvidenceTracer(graph_path)
    source_evidence = tracer.structural_evidence()
    sample_id = DatasetPipeline.sample_id_from_graph(graph_path)
    instruction = DatasetPipeline.instruction_text()
    rows = []

    for attempt in range(1, max_attempts + 1):
        generation = generate_hypotheses_for_graph(
            graph_path,
            endpoint=llm_config.get("vllm_endpoint", DEFAULT_VLLM_ENDPOINT),
            model=llm_config.get("model"),
            evidence_limit=evidence_limit,
            count=batch_size,
            max_new_tokens=llm_config.get("max_new_tokens", 1024),
            temperature=llm_config.get("temperature", 0.7),
            top_p=llm_config.get("top_p", 0.9),
            timeout=llm_config.get("timeout", 120),
            seed=attempt,
        )

        graph_trace_json = json.dumps(generation["evidence"])
        llm_prompt = generation["prompt"]
        llm_raw_output = generation["raw_output"]

        for hypothesis in generation["hypotheses"]:
            verification = verifier.verify_graph_claim(graph_path, hypothesis)
            deciding = verification["deciding_evidence"]
            deciding_scores = deciding.get("scores", {})

            rows.append({
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "sample_id": sample_id,
                "graph_path": Path(graph_path).as_posix(),
                "instruction": instruction,
                "attempt": attempt,
                "batch_size": batch_size,
                "evidence_limit": evidence_limit,
                "llm_prompt": llm_prompt,
                "llm_raw_output": llm_raw_output,
                "llm_answer": hypothesis,
                "nli_status": verification["status"],
                "nli_entailment": deciding_scores.get("entailment", ""),
                "nli_contradiction": deciding_scores.get("contradiction", ""),
                "nli_neutral": deciding_scores.get("neutral", ""),
                "nli_deciding_sentence": deciding.get("sentence", ""),
                "nli_deciding_fact_name": deciding.get("fact_name", ""),
                "nli_model": verification.get("model", DEFAULT_NLI_MODEL),
                "graph_trace_json": graph_trace_json,
                "retrieved_evidence_json": json.dumps(verification["retrieved_evidence"]),
                "source_evidence_json": json.dumps(source_evidence),
            })

    return rows


def export_verification_trace_log(
    graph_root=DEFAULT_GRAPH_ROOT,
    output_path=DEFAULT_OUTPUT,
    max_attempts=5,
    batch_size=3,
    evidence_limit=12,
    limit=None,
):
    llm_config = load_config(LLM_CONFIG_PATH)
    nli_config = load_config(NLI_CONFIG_PATH)
    verifier = NliGraphVerifier(
        model_name=nli_config.get("model", DEFAULT_NLI_MODEL),
        top_k=nli_config.get("top_k", 8),
        entailment_threshold=nli_config.get("entailment_threshold", DEFAULT_ENTAILMENT_THRESHOLD),
        contradiction_threshold=nli_config.get("contradiction_threshold", DEFAULT_CONTRADICTION_THRESHOLD),
    )

    graph_paths = list_graph_paths(graph_root, limit=limit)
    all_rows = []
    for graph_path in graph_paths:
        rows = build_log_rows_for_graph(
            graph_path=graph_path,
            verifier=verifier,
            llm_config=llm_config,
            max_attempts=max_attempts,
            batch_size=batch_size,
            evidence_limit=evidence_limit,
        )
        all_rows.extend(rows)

    append_rows(output_path, all_rows)
    return {
        "graph_count": len(graph_paths),
        "logged_rows": len(all_rows),
        "output_path": Path(output_path).as_posix(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append LLM and NLI verification traces for each graph to a CSV log."
    )
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--evidence-limit", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = export_verification_trace_log(
        graph_root=args.graph_root,
        output_path=args.output,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        evidence_limit=args.evidence_limit,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
