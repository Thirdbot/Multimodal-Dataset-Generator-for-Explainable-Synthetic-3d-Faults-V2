"""Watch recipe files and run Synthoseis builds for their referenced configs.

The watcher treats each recipe as the orchestration unit. When a recipe appears,
each listed build config is sent through the guarded Synthoseis build wrapper,
then success/failed YAML trackers are updated for downstream graph extraction.
"""

import json
import shutil
import hashlib

import yaml
from watchdog.events import FileSystemEventHandler
import time
from logger_color import logger
import sys
from pathlib import Path


from yaml_helper import YAMLHelper
from watchdog.observers import Observer

SYNTHOSEIS_ROOT = Path(__file__).parent.parent / "third_party" / "synthoseis"
sys.path.insert(0, str(SYNTHOSEIS_ROOT))
from main import build_model
from synthoseis_config_guard import guarded_build_model


class SampleBuildRunner(FileSystemEventHandler):
    """Watchdog handler that converts recipe file events into build jobs."""

    def __init__(self, recipes_path):
        self.root = Path(__file__).parent.parent
        self.recipes_path = self._resolve_path(recipes_path)
        self.recipes_name_path = None
        self.parent_path = self.recipes_path.parent
        self.build_configs_path = self.parent_path.joinpath('build_configs')
        setting_path = self.root / "settings.yaml"
        yaml_helper = YAMLHelper(setting_path)
        self.samples_path = self._resolve_path(yaml_helper.get_data("samples_path"))
        self.temp_builds_path = self._resolve_path(yaml_helper.get_data("temp_builds_path"))
        self.graphs_path = self._resolve_path(yaml_helper.get_data("graphs_path"))

        self.success = []
        self.failed = []

        self.recipe_cache = {}

        self.last_event_time = {}

    def _resolve_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    # de-bouncing
    def _should_skip(self, path, seconds=2):
        now = time.time()
        last = self.last_event_time.get(path, 0)

        if now - last < seconds:
            return True

        self.last_event_time[path] = now
        return False

    def _build_sample(self, build_config_path, run_id):
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
            self.success = list(dict.fromkeys(old_success + new_success))
            success = {"success_build_obj":self.success}
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

    # event only create
    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".yaml":
            return
        time.sleep(0.2)
        yaml_helper = YAMLHelper(path)
        population = yaml_helper.get_data("population")
        samples = population['samples']

        logger.info(f"[CREATE] -> from: {path.stem}")
        self.recipe_cache[path] = {
            "samples": samples,
            "build_configs": {},
        }
        for sample in samples:
            build_config_path = self.build_configs_path.joinpath(f"{sample}.json")
            logger.info(f"[GENERATE] -> Config: {sample}")

            with open(str(build_config_path), "r") as file:
                build_config = json.load(file)
                self.recipe_cache[path]["build_configs"][sample] = build_config
            self._build_sample(build_config_path, f"{path.stem}_{sample}")

        self.last_event_time[path] = time.time()

def watch_recipes(recipes_path):
    """Start a persistent watchdog observer over the recipe directory."""
    observer = Observer()
    recipe_build_runner = SampleBuildRunner(recipes_path)
    path = recipe_build_runner.recipes_path

    if path.exists():
        logger.info(f"[NOW MONITORING] -> Path: {path}")
        observer.schedule(
            recipe_build_runner,
            path,
            recursive=True,
        )
    else:
        logger.warning(f"[SKIPPING] -> Path: {path}")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error(f"[STOPPING]")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    # Entry point used during live generation: watch recipes and build samples.
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    recipes_path = yaml_helper.get_data('recipes_path')

    # watch over update in recipes
    watch_recipes(recipes_path)
