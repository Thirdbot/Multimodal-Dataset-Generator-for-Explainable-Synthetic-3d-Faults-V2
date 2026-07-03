"""Run Synthoseis builds once for the configs referenced by recipe files.

Each recipe is treated as the orchestration unit. Every listed build config is
sent through the guarded Synthoseis build wrapper,
then success/failed YAML trackers are updated for downstream graph extraction.
"""

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
                existing = {}
                if success_tracker_path.exists():
                    with open(success_tracker_path, 'r') as file:
                        existing = yaml.safe_load(file) or {}

                completed_build_folders = list(Path(self.samples_path).glob(f'seismic__*_{run_id}'))
                old_success = existing.get("success_build_obj", [])
                new_success = [folder.as_posix() for folder in completed_build_folders if folder.exists()]
                success = list(dict.fromkeys(old_success + new_success))
                success = {"success_build_obj":success}
                with open(success_tracker_path,'w') as file:
                    yaml.dump(success, file)
                logger.info(f"[TRACK SUCCESS] -> Path: {success_tracker_path}")

                return True
            except BaseException as exc:
                logger.error(f"[BUILD FAILED] -> Run: {run_id} Error: {exc}")

                failed_tracker_path = self.samples_path / 'failed.yaml'
                failed_tracker_path.touch(exist_ok=True)

                self.failed.append(str(build_config_path))
                failed = {"failed_build_config": self.failed}

                with open(failed_tracker_path,'w') as file:
                    yaml.dump(failed, file)
                logger.info(f"[TRACK FAILED] -> Path: {failed_tracker_path}")

                return False
        else:
            return False