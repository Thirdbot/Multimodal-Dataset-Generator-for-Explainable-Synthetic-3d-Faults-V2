import argparse
import json
import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Verifier.generator_pipeline import HybridRagWorkflow
from Dataset.export_multimodal_dataset import export_dataset
from scripts.logger_color import logger


DEFAULT_GRAPH_ROOT = ROOT / "graphs" / "properties_graph"
DEFAULT_OUTPUT = ROOT / "Dataset" / "verified_hypotheses.jsonl"
DEFAULT_CSV_OUTPUT = ROOT / "Dataset" / "multimodal_verified_dataset.csv"
DEFAULT_IMAGE_DIR = ROOT / "Dataset" / "multimodal_images"


class DatasetPipeline(object):
    def __init__(self, graph_root=DEFAULT_GRAPH_ROOT, output_path=DEFAULT_OUTPUT):
        self.graph_root = Path(graph_root)
        self.output_path = Path(output_path)
        self.rows = self.load_existing_rows()
        self.totals = {"saved": 0, "skipped": 0}
        self.hybrid = HybridRagWorkflow(
            graph_root=self.graph_root,
            output_path=self.output_path,
        )

    def process_graph(self, graph_path, questions_per_graph=5, candidates_per_question=5, verbose=False):
        graph_path = Path(graph_path)
        if not graph_path.exists():
            return None

        sample_id = self.hybrid_sample_id(graph_path)
        logger.info(f"[DATASET START] -> Graph: {graph_path.name}")

        rows = self.hybrid.generate_for_graph(
            graph_path,
            questions_per_graph=questions_per_graph,
            candidates_per_question=candidates_per_question,
        )

        if rows:
            removed = self.remove_rows_for_graph(graph_path)
            for row in rows:
                self.rows[row["row_id"]] = row
            self.write_rows()
            self.totals["saved"] += len(rows)
            logger.info(
                f"[DATASET SAVE] -> Sample: {sample_id} "
                f"Rows: {len(rows)} Removed: {removed} Output: {self.output_path}"
            )
        else:
            self.totals["skipped"] += 1
            logger.warning(f"[DATASET SKIP] -> Sample: {sample_id}")

        self.log_summary(sample_id, len(rows))
        return {
            "sample_id": sample_id,
            "rows": len(rows),
            "output_path": self.output_path.as_posix(),
        }

    def watch(self, questions_per_graph=5, candidates_per_question=5, verbose=False):
        DatasetWatcher(
            pipeline=self,
            graph_root=self.graph_root,
            questions_per_graph=questions_per_graph,
            candidates_per_question=candidates_per_question,
            verbose=verbose,
        ).run()

    def run_once(self, questions_per_graph=5, candidates_per_question=5, limit=None, verbose=False):
        graph_paths = sorted(self.graph_root.glob("*_properties_graph.json"))
        if limit:
            graph_paths = graph_paths[:limit]

        statuses = []
        for graph_path in graph_paths:
            status = self.process_graph(
                graph_path,
                questions_per_graph=questions_per_graph,
                candidates_per_question=candidates_per_question,
                verbose=verbose,
            )
            if status:
                statuses.append(status)

        return {
            "graphs_seen": len(graph_paths),
            "graphs_processed": len(statuses),
            "verified_output": self.output_path.as_posix(),
            "rows_in_memory": len(self.rows),
        }

    def remove_graph(self, graph_path):
        sample_id = self.hybrid_sample_id(graph_path)
        removed = self.remove_rows_for_graph(graph_path)
        if not removed:
            return False
        self.write_rows()
        logger.info(f"[DATASET DELETE] -> Graph: {Path(graph_path).name} Sample: {sample_id} Rows: {removed}")
        return True

    def remove_rows_for_graph(self, graph_path):
        graph_path = Path(graph_path).as_posix()
        keys = [
            key
            for key, row in self.rows.items()
            if row.get("metadata", {}).get("graph_path") == graph_path
        ]
        for key in keys:
            self.rows.pop(key, None)
        return len(keys)

    def load_existing_rows(self):
        rows = {}
        if not self.output_path.exists():
            return rows
        for line in self.output_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("row_id")
            if row_id and self.is_hybrid_row(row):
                rows[row_id] = row
        return rows

    def write_rows(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as file:
            for row_id in sorted(self.rows):
                file.write(json.dumps(self.rows[row_id], default=str) + "\n")

    def log_summary(self, sample_id, row_count):
        logger.info(
            "[DATASET SUMMARY] -> "
            f"Sample: {sample_id} "
            f"Rows: {row_count} "
            f"Saved(total): {self.totals['saved']} "
            f"Skipped(total): {self.totals['skipped']}"
        )

    @staticmethod
    def hybrid_sample_id(graph_path):
        stem = Path(graph_path).stem
        return stem.removesuffix("_properties_graph").replace("_db_extract", "")

    @staticmethod
    def is_hybrid_row(row):
        trace = row.get("trace", {})
        return bool(trace.get("question_evidence") and trace.get("answer_evidence"))


class DatasetWatcher(FileSystemEventHandler):
    def __init__(self, pipeline, graph_root, questions_per_graph=5, candidates_per_question=5, verbose=False):
        self.pipeline = pipeline
        self.graph_root = Path(graph_root)
        self.questions_per_graph = questions_per_graph
        self.candidates_per_question = candidates_per_question
        self.verbose = verbose
        self.processed_mtimes = {}

    def on_any_event(self, event):
        path = Path(getattr(event, "dest_path", event.src_path))
        if event.event_type == "deleted":
            self.handle_deleted(path)
        else:
            self.handle_path(path)

    def handle_deleted(self, path):
        if self.is_graph_file(path):
            self.processed_mtimes.pop(path.as_posix(), None)
            self.pipeline.remove_graph(path)

    def handle_path(self, path):
        if not self.is_graph_file(path) or not path.exists():
            return

        key = path.as_posix()
        current_mtime = path.stat().st_mtime
        if self.processed_mtimes.get(key) == current_mtime:
            return

        # Let graph_generator.py finish writing before reading the graph.
        time.sleep(0.2)
        self.processed_mtimes[key] = current_mtime
        self.pipeline.process_graph(
            path,
            questions_per_graph=self.questions_per_graph,
            candidates_per_question=self.candidates_per_question,
            verbose=self.verbose,
        )

    def process_existing(self):
        for graph_path in sorted(self.graph_root.glob("*_properties_graph.json")):
            self.handle_path(graph_path)

    @staticmethod
    def is_graph_file(path):
        return path.name.endswith("_properties_graph.json")

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
    parser = argparse.ArgumentParser(description="Watch properties graphs and generate hybrid verified dataset rows.")
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--questions-per-graph", type=int, default=5)
    parser.add_argument("--candidates-per-question", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = DatasetPipeline(
        graph_root=args.graph_root,
        output_path=args.output,
    )
    questions_per_graph = args.max_attempts or args.questions_per_graph
    candidates_per_question = args.batch_size or args.candidates_per_question

    if args.once:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        result = pipeline.run_once(
            questions_per_graph=questions_per_graph,
            candidates_per_question=candidates_per_question,
            limit=args.limit,
            verbose=args.verbose,
        )
        if not args.no_export:
            result["export"] = export_dataset(
                verified_path=args.output,
                output_path=args.csv_output,
                image_dir=args.image_dir,
                limit=None,
            )
        print(json.dumps(result, indent=2, default=str))
    else:
        pipeline.watch(
            questions_per_graph=questions_per_graph,
            candidates_per_question=candidates_per_question,
            verbose=args.verbose,
        )
