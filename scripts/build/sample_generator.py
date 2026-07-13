"""Run Synthoseis builds once for the configs referenced by recipe files.

Each recipe is treated as the orchestration unit. Every listed build config is
sent through the guarded Synthoseis build wrapper,
then success/failed YAML trackers are updated for downstream graph extraction.
"""

import fcntl
import os
import yaml
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.logger_color import logger

SYNTHOSEIS_ROOT = ROOT / "third_party" / "synthoseis"
sys.path.insert(0, str(SYNTHOSEIS_ROOT))
from main import build_model
from scripts.build.synthoseis_config_guard import guarded_build_model

# Builds run in separate processes (scripts/watcher/process.py launches each as its
# own subprocess), so a threading.Lock cannot serialize success.yaml/failed.yaml
# writes across them. fcntl.flock on a shared lock file serializes the whole
# read-modify-write across every build process on this machine, and each write is
# still atomic (temp file + os.replace). Together this makes BUILD_CONCURRENCY>1 safe:
# no torn reads, no lost success/failed entries. A single lock covers both trackers.
@contextmanager
def _tracker_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".tracker.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)   # blocks until every other process releases
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _merge_tracker(path, key, new_entries):
    with _tracker_lock(path.parent):
        existing = {}
        if path.exists():
            with open(path, "r") as file:
                existing = yaml.safe_load(file) or {}
        merged = list(dict.fromkeys(existing.get(key, []) + list(new_entries)))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as file:
            yaml.dump({key: merged}, file)
        os.replace(tmp, path)  # atomic: no torn reads, no lost writes
    return merged


def _drop_from_tracker(path, key, drop_entries):
    drop = {str(entry) for entry in drop_entries}
    with _tracker_lock(path.parent):
        if not path.exists():
            return
        with open(path, "r") as file:
            existing = yaml.safe_load(file) or {}
        current = existing.get(key, [])
        kept = [entry for entry in current if str(entry) not in drop]
        if kept == current:
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as file:
            yaml.dump({key: kept}, file)
        os.replace(tmp, path)

class BuildGenerator:
    def __init__(self):
        self.failed = []
        self.success = []
        self.samples_path = (ROOT / "builds").resolve()
    def build_sample(self,build_config_path, run_id):
        if Path(build_config_path).exists():
            """Run one Synthoseis config and track the resulting build folder."""
            logger.info(f"[BUILD START] -> Run: {run_id}")
            try:
                guarded_build_model(
                    build_model,
                    user_json=str(build_config_path),
                    run_id=run_id,
                    test_mode=None,
                    seed=None,
                )
                logger.info(f"[BUILD DONE] -> Run: {run_id}")
                success_tracker_path = self.samples_path / 'success.yaml'
                completed_build_folders = list(Path(self.samples_path).glob(f'seismic__*_{run_id}'))
                new_success = [folder.as_posix() for folder in completed_build_folders if folder.exists()]
                _merge_tracker(success_tracker_path, "success_build_obj", new_success)
                # A build that now succeeds must not linger in failed.yaml, or a later
                # failed.yaml rewrite would rmtree this good build folder.
                _drop_from_tracker(self.samples_path / 'failed.yaml', "failed_build_config", [str(build_config_path)])
                logger.info(f"[TRACK SUCCESS] -> Path: {success_tracker_path}")

                return True
            except BaseException as exc:
                logger.error(f"[BUILD FAILED] -> Run: {run_id} Error: {exc}")

                failed_tracker_path = self.samples_path / 'failed.yaml'
                _merge_tracker(failed_tracker_path, "failed_build_config", [str(build_config_path)])
                logger.info(f"[TRACK FAILED] -> Path: {failed_tracker_path}")

                return False
        else:
            return False


def _main():
    """CLI entry so the watcher can run each build in its own subprocess.

    Synthoseis monkeypatches process-global state and is CPU-bound; isolating each
    build in a fresh process keeps those patches out of the watcher and stops the
    build from holding the event loop's GIL. Exit 0 on success, 1 on failure.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run one guarded Synthoseis build.")
    parser.add_argument("--config", required=True, help="Path to the build config JSON.")
    parser.add_argument("--run-id", required=True, help="Run id (build config stem).")
    args = parser.parse_args()

    ok = BuildGenerator().build_sample(args.config, args.run_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _main()