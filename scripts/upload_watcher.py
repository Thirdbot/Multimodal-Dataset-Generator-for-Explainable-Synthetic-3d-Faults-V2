"""Watch multimodal CSV and upload it to HuggingFace."""

import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Dataset.upload_to_huggingface import DEFAULT_CSV, upload_dataset
from logger_color import logger


class HuggingFaceUploadRunner(FileSystemEventHandler):
    """Upload CSV dataset updates to HuggingFace."""

    def __init__(self):
        self.csv_path = DEFAULT_CSV
        self.last_run = 0

    def _should_skip(self, seconds=5):
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
        if path != self.csv_path:
            return
        if self._should_skip():
            return

        repo_id = os.getenv("HF_REPO_ID", "thirdExec/synthetic-seismic-vlm")
        token = os.getenv("HF_TOKEN")
        private = os.getenv("HF_PRIVATE", "0").lower() in {"1", "true", "yes"}
        dry_run = os.getenv("HF_DRY_RUN", "0").lower() in {"1", "true", "yes"}

        try:
            logger.info(f"[UPLOAD START] -> Repo: {repo_id}")
            result = upload_dataset(
                self.csv_path,
                repo_id,
                private=private,
                token=token,
                dry_run=dry_run,
            )
            logger.info(f"[UPLOAD DONE] -> {result}")
        except Exception as exc:
            logger.error(f"[UPLOAD FAILED] -> Error: {exc}")


def watch_dataset_csv():
    runner = HuggingFaceUploadRunner()
    observer = Observer()
    path = runner.csv_path.parent

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
    watch_dataset_csv()
