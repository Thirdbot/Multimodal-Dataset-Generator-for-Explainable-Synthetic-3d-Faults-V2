"""Watch 2D properties graphs and generate verified QA JSONL rows."""

import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from logger_color import logger
from yaml_helper import YAMLHelper
from Verifier.generator_pipeline import generate_multimodal_dataset


class DatasetGenerationRunner(FileSystemEventHandler):
    """Convert 2D graph updates into verified_qa.jsonl."""

    def __init__(self):
        yaml_helper = YAMLHelper(ROOT / "settings.yaml")
        graphs_path = self._resolve(yaml_helper.get_data("graphs_path"))
        self.properties_2d_graph_path = graphs_path / "properties_2d_graph"
        self.last_run = 0

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    def _should_skip(self, seconds=2):
        now = time.time()
        if now - self.last_run < seconds:
            return True
        self.last_run = now
        return False

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def on_moved(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = Path(getattr(event, "dest_path", event.src_path))
        if path.suffix != ".json" or not path.name.endswith("_properties_2d_graph.json"):
            return
        if self._should_skip():
            return

        try:
            logger.info("[QA DATASET START]")
            generate_multimodal_dataset()
            logger.info("[QA DATASET DONE]")
        except Exception as exc:
            logger.error(f"[QA DATASET FAILED] -> Error: {exc}")


def watch_properties_2d_graph():
    runner = DatasetGenerationRunner()
    observer = Observer()
    path = runner.properties_2d_graph_path

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[NOW MONITORING] -> Path: {path}")
    observer.schedule(runner, path, recursive=False)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error("[STOPPING]")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    watch_properties_2d_graph()
