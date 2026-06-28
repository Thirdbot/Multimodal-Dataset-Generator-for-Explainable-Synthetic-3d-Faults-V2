"""Watch verified QA JSONL and rebuild the multimodal CSV."""

import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from Dataset.DatasetMaker import INPUT, main as build_dataset_csv
from logger_color import logger


class DatasetMakerRunner(FileSystemEventHandler):
    """Convert verified_qa.jsonl updates into multimodal CSV."""

    def __init__(self):
        self.input_path = INPUT.resolve()
        self.last_run = 0
        self.last_seen_state = None

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

    def on_closed(self, event):
        self._handle(event)

    def process_existing(self):
        if self.input_path.exists():
            self.last_seen_state = self._file_state()
            self._generate()

    def process_if_changed(self):
        state = self._file_state()
        if state is None or state == self.last_seen_state:
            return
        self.last_seen_state = state
        if self._should_skip():
            return
        self._generate()

    def _handle(self, event):
        if event.is_directory:
            return
        path = Path(getattr(event, "dest_path", event.src_path)).resolve()
        if not self._is_input_path(path):
            return
        self.last_seen_state = self._file_state()
        if self._should_skip():
            return
        self._generate()

    def _is_input_path(self, path):
        return path == self.input_path or (
            path.name == self.input_path.name
            and path.parent == self.input_path.parent
        )

    def _file_state(self):
        if not self.input_path.exists():
            return None
        stat = self.input_path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _generate(self):
        try:
            logger.info("[CSV START]")
            build_dataset_csv()
            logger.info("[CSV DONE]")
        except Exception as exc:
            logger.error(f"[CSV FAILED] -> Error: {exc}")


def watch_verified_qa():
    runner = DatasetMakerRunner()
    observer = Observer()
    path = runner.input_path.parent

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[NOW MONITORING] -> Path: {path}")
    observer.schedule(runner, path, recursive=False)

    observer.start()
    runner.process_existing()
    try:
        while True:
            time.sleep(1)
            runner.process_if_changed()
    except KeyboardInterrupt:
        logger.error("[STOPPING]")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    watch_verified_qa()
