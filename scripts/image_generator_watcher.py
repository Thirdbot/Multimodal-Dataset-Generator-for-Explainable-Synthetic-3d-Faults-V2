"""Watch properties_graph files and generate 2D object images."""

import sys
import time
import os
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from images_generator import generate_images_for_graph
from logger_color import logger
from yaml_helper import YAMLHelper


class ImageGenerationRunner(FileSystemEventHandler):
    """Convert one properties graph update into image/mask extraction."""

    def __init__(self):
        self.root = ROOT
        yaml_helper = YAMLHelper(self.root / "settings.yaml")
        graphs_path = self._resolve(yaml_helper.get_data("graphs_path"))
        self.properties_graph_path = graphs_path / "properties_graph"
        self.samples_path = self._resolve(yaml_helper.get_data("samples_path"))
        self.last_event_time = {}

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _should_skip(self, path, seconds=2):
        now = time.time()
        last = self.last_event_time.get(path, 0)
        if now - last < seconds:
            return True
        self.last_event_time[path] = now
        return False

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def on_moved(self, event):
        self._handle(event)

    def process_existing(self):
        for path in sorted(self.properties_graph_path.glob("*_properties_graph.json")):
            self._generate(path)

    def _handle(self, event):
        if event.is_directory:
            return
        path = Path(getattr(event, "dest_path", event.src_path))
        if path.suffix != ".json" or not path.name.endswith("_properties_graph.json"):
            return
        if self._should_skip(path):
            return
        self._generate(path)

    def _generate(self, path):
        try:
            logger.info(f"[IMAGE START] -> Graph: {path.name}")
            generate_images_for_graph(path, self.samples_path)
            logger.info(f"[IMAGE DONE] -> Graph: {path.name}")
        except Exception as exc:
            logger.error(f"[IMAGE FAILED] -> Graph: {path} Error: {exc}")


def watch_properties_graph():
    runner = ImageGenerationRunner()
    observer = Observer()
    path = runner.properties_graph_path

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[NOW MONITORING] -> Path: {path}")
    observer.schedule(runner, path, recursive=False)

    observer.start()
    runner.process_existing()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error("[STOPPING]")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    watch_properties_graph()
