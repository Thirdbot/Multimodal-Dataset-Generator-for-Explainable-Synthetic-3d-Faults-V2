import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Dataset.slice2d import create_images_for_row


DEFAULT_VERIFIED = ROOT / "Dataset" / "verified_hypotheses.jsonl"
DEFAULT_OUTPUT = ROOT / "Dataset" / "multimodal_verified_dataset.csv"
DEFAULT_IMAGE_DIR = ROOT / "Dataset" / "multimodal_images"


def load_verified_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"verified dataset not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def export_dataset(verified_path, output_path, image_dir, limit=None):
    rows = load_verified_rows(verified_path)
    source_rows = rows[:limit] if limit else rows
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = []
    skipped = []
    for row in source_rows:
        try:
            if not is_hybrid_verified_row(row):
                raise ValueError("row is not a hybrid verified row")
            image_info = create_images_for_row(row, image_dir)
            for view, view_info in views_for_row(row, image_info):
                exported.append(multimodal_row(row, image_info, view, view_info))
        except Exception as exc:
            skipped.append({
                "row_id": row.get("row_id", ""),
                "sample_id": row.get("sample_id", ""),
                "error": f"{exc.__class__.__name__}: {exc}",
            })

    if output_path.suffix.lower() == ".jsonl":
        with open(output_path, "w") as file:
            for row in exported:
                file.write(json.dumps(row) + "\n")
    else:
        write_csv(output_path, exported)

    return {
        "verified_rows": len(rows),
        "exported_rows": len(exported),
        "skipped_rows": len(skipped),
        "output_path": output_path.as_posix(),
        "image_dir": Path(image_dir).as_posix(),
        "skipped": skipped,
    }


def multimodal_row(source_row, image_info, view, view_info):
    instruction = source_row["instruction"]
    answer = source_row["answer"]
    sample_id = source_row["sample_id"]
    if not str(instruction).strip():
        raise ValueError("missing instruction")
    if not str(answer).strip():
        raise ValueError("missing answer")
    source_row_id = source_row.get("row_id", sample_id)
    row_id = f"{source_row_id}_{view}"
    trace = source_row.get("trace", {})
    verification = source_row.get("verification", {})
    deciding_evidence = trace.get("deciding_evidence") or trace.get("answer_evidence", [{}])[0] or {}
    if not deciding_evidence and source_row.get("evidence"):
        deciding_evidence = source_row["evidence"][0]

    image_path = view_info["image_path"].as_posix()
    if not image_path:
        raise ValueError(f"missing {view} image")
    overlay_image_path = view_info["overlay_image_path"].as_posix() if view_info["overlay_image_path"] else ""
    mask_image_path = view_info["mask_image_path"].as_posix() if view_info["mask_image_path"] else ""
    graph_path = source_row.get("metadata", {}).get("graph_path", "")
    view_graph_path = view_graph_for(graph_path, view)

    content = [
        {"type": "text", "text": instruction},
        {"type": "image", "image": image_path},
    ]
    if overlay_image_path:
        content.append({"type": "image", "image": overlay_image_path})
    if mask_image_path:
        content.append({"type": "image", "image": mask_image_path})

    return {
        "id": row_id,
        "row_id": row_id,
        "source_row_id": source_row_id,
        "sample_id": sample_id,
        "view": view,
        "instruction": instruction,
        "question": source_row.get("question", instruction),
        "answer": answer,
        "image": image_path,
        "overlay_image": overlay_image_path,
        "mask_image": mask_image_path,
        "overlay_kind": image_info.get("overlay_kind", ""),
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "metadata": {
            "graph_path": graph_path,
            "view_graph_path": view_graph_path,
            "view": view,
            "category": source_row.get("metadata", {}).get("category", ""),
            "seismic_array": image_info.get("seismic_relpath", ""),
            "overlay_array": image_info.get("overlay_relpath", ""),
            "overlay_arrays": image_info.get("overlay_arrays", []),
            "view_indices": image_info.get("slice_indices", {}),
            "fixed_axis": view_info.get("fixed_axis", view),
            "fixed_index": view_info.get("slice_index", ""),
            "selection_method": view_info.get("selection_method", ""),
            "render_type": "single_view_seismic_slice",
        },
        "trace": {
            "graph_trace": trace.get("graph_trace", trace.get("graph_evidence", source_row.get("evidence", []))),
            "question_prompt": trace.get("question_prompt", ""),
            "question_raw_output": trace.get("question_raw_output", ""),
            "llm_question": trace.get("llm_question", source_row.get("question", instruction)),
            "llm_prompt": trace.get("llm_prompt", ""),
            "llm_raw_output": trace.get("llm_raw_output", ""),
            "llm_answer": trace.get("llm_answer", answer),
            "nli_status": verification.get("status", verification.get("verdict", "")),
            "nli_score": verification.get("score", ""),
            "nli_model": trace.get("nli_model", ""),
            "nli_deciding_evidence": deciding_evidence,
            "nli_retrieved_evidence": trace.get("retrieved_evidence", trace.get("answer_evidence", source_row.get("evidence", []))),
        },
    }


def view_graph_for(graph_path, view):
    if not graph_path:
        return ""
    graph_path = Path(graph_path)
    if graph_path.name.endswith(f"_{view}_graph.json"):
        return graph_path.as_posix()
    name = graph_path.name.replace("_properties_graph.json", f"_{view}_graph.json")
    return (graph_path.parent.parent / "views_graph" / name).as_posix()


def views_for_row(row, image_info):
    view = row.get("view") or row.get("metadata", {}).get("view")
    graph_path = str(row.get("metadata", {}).get("graph_path", ""))
    if not view:
        if graph_path.endswith("_inline_graph.json"):
            view = "inline"
        elif graph_path.endswith("_crossline_graph.json"):
            view = "crossline"
    if view in image_info.get("views", {}):
        return [(view, image_info["views"][view])]
    return list(image_info.get("views", {}).items())


def is_hybrid_verified_row(row):
    trace = row.get("trace", {})
    verification = row.get("verification", {})
    return bool(
        row.get("question")
        and row.get("answer")
        and trace.get("question_evidence")
        and trace.get("answer_evidence")
        and verification.get("score") is not None
    )


def write_csv(output_path, rows):
    fieldnames = [
        "id",
        "row_id",
        "source_row_id",
        "sample_id",
        "view",
        "instruction",
        "question",
        "answer",
        "image",
        "overlay_image",
        "mask_image",
        "overlay_kind",
        "overlay_arrays",
        "view_inline",
        "view_crossline",
        "view_timeslice",
        "view_mode_inline",
        "view_mode_crossline",
        "view_mode_timeslice",
        "fixed_axis",
        "fixed_index",
        "selection_method",
        "graph_path",
        "view_graph_path",
        "seismic_array",
        "overlay_array",
        "llm_question",
        "llm_answer",
        "llm_raw_output",
        "question_raw_output",
        "nli_status",
        "nli_score",
        "nli_model",
        "nli_deciding_sentence",
        "nli_deciding_evidence_json",
        "llm_prompt",
        "graph_trace_json",
        "nli_retrieved_evidence_json",
        "messages_json",
        "metadata_json",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row.get("metadata", {})
            trace = row.get("trace", {})
            deciding = trace.get("nli_deciding_evidence", {}) or {}
            view_indices = metadata.get("view_indices", {})
            view_selection = view_indices.get("methods", {})
            writer.writerow({
                "id": row["id"],
                "row_id": row["row_id"],
                "source_row_id": row.get("source_row_id", ""),
                "sample_id": row["sample_id"],
                "view": row.get("view", ""),
                "instruction": row["instruction"],
                "question": row.get("question", row["instruction"]),
                "answer": row["answer"],
                "image": row["image"],
                "overlay_image": row.get("overlay_image", ""),
                "mask_image": row.get("mask_image", ""),
                "overlay_kind": row.get("overlay_kind", ""),
                "overlay_arrays": json.dumps(metadata.get("overlay_arrays", [])),
                "view_inline": view_indices.get("inline"),
                "view_crossline": view_indices.get("crossline"),
                "view_timeslice": view_indices.get("timeslice"),
                "view_mode_inline": view_selection.get("inline", ""),
                "view_mode_crossline": view_selection.get("crossline", ""),
                "view_mode_timeslice": view_selection.get("timeslice", ""),
                "fixed_axis": metadata.get("fixed_axis", ""),
                "fixed_index": metadata.get("fixed_index", ""),
                "selection_method": metadata.get("selection_method", ""),
                "graph_path": metadata.get("graph_path", ""),
                "view_graph_path": metadata.get("view_graph_path", ""),
                "seismic_array": metadata.get("seismic_array", ""),
                "overlay_array": metadata.get("overlay_array", ""),
                "llm_question": trace.get("llm_question", row.get("question", row["instruction"])),
                "llm_answer": trace.get("llm_answer", row["answer"]),
                "llm_raw_output": trace.get("llm_raw_output", ""),
                "question_raw_output": trace.get("question_raw_output", ""),
                "nli_status": trace.get("nli_status", ""),
                "nli_score": trace.get("nli_score", ""),
                "nli_model": trace.get("nli_model", ""),
                "nli_deciding_sentence": deciding.get("sentence", ""),
                "nli_deciding_evidence_json": json.dumps(deciding),
                "llm_prompt": trace.get("llm_prompt", ""),
                "graph_trace_json": json.dumps(trace.get("graph_trace", [])),
                "nli_retrieved_evidence_json": json.dumps(trace.get("nli_retrieved_evidence", [])),
                "messages_json": json.dumps(row.get("messages", [])),
                "metadata_json": json.dumps(metadata),
            })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert verified_hypotheses.jsonl into a 2D multimodal instruction-tuning dataset."
    )
    parser.add_argument("--verified", default=str(DEFAULT_VERIFIED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = export_dataset(
        verified_path=args.verified,
        output_path=args.output,
        image_dir=args.image_dir,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
