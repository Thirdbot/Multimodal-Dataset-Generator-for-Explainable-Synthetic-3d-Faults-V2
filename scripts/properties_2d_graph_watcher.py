"""Watch object position metadata and rebuild 2D properties graphs."""

import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from logger_color import logger
from properties_2d_graph import main as generate_2d_graphs


class Properties2DGraphRunner(FileSystemEventHandler):
    """Convert object_position updates into properties_2d_graph files."""

    def __init__(self):
        self.image_root = ROOT / "build_objects" / "images"
        self.last_run = 0

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
        if not path.name.endswith("_object_position.json"):
            return
        if self._should_skip():
            return

        try:
            logger.info("[2D GRAPH START]")
            generate_2d_graphs()
            logger.info("[2D GRAPH DONE]")
        except Exception as exc:
            logger.error(f"[2D GRAPH FAILED] -> Error: {exc}")


def watch_object_positions():
    runner = Properties2DGraphRunner()
    observer = Observer()
    path = runner.image_root

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[NOW MONITORING] -> Path: {path}")
    observer.schedule(runner, path, recursive=True)

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
    watch_object_positions()
