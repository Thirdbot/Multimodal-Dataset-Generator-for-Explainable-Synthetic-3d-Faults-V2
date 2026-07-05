"""Run Synthoseis builds once for the configs referenced by recipe files.

Each recipe is treated as the orchestration unit. Every listed build config is
sent through the guarded Synthoseis build wrapper,
then success/failed YAML trackers are updated for downstream graph extraction.
"""

import os
import threading
import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.logger_color import logger

SYNTHOSEIS_ROOT = ROOT / "third_party" / "synthoseis"
sys.path.insert(0, str(SYNTHOSEIS_ROOT))
from main import build_model
from scripts.build.synthoseis_config_guard import guarded_build_model

# Builds run concurrently in threads; success.yaml/failed.yaml are shared state.
# One process-wide lock serializes read-modify-write, and every write is atomic
# (temp file + os.replace) so a reader never sees a truncated file mid-write.
_TRACKER_LOCK = threading.Lock()


def _merge_tracker(path, key, new_entries):
    with _TRACKER_LOCK:
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