"""Run Synthoseis builds once for the configs referenced by recipe files.

Each recipe is treated as the orchestration unit. Every listed build config is
sent through the guarded Synthoseis build wrapper,
then success/failed YAML trackers are updated for downstream graph extraction.
"""

import json

import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.logger_color import logger
from scripts.common.yaml_helper import YAMLHelper

SYNTHOSEIS_ROOT = ROOT / "third_party" / "synthoseis"
sys.path.insert(0, str(SYNTHOSEIS_ROOT))
from main import build_model
from scripts.build.synthoseis_config_guard import guarded_build_model


class SampleBuildRunner:
    """Convert recipe files into build jobs."""

    def __init__(self, recipes_path):
        self.root = ROOT
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

    def _resolve_path(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

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

    def process_recipe(self, path):
        """Build every sample listed in one recipe YAML."""
        path = Path(path)
        if path.suffix != ".yaml":
            return

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

    def process_existing_recipes(self):
        """Run all current recipe YAML files once."""
        if not self.recipes_path.exists():
            logger.warning(f"[SKIPPING] -> Path: {self.recipes_path}")
            return

        logger.info(f"[RUN ONCE] -> Recipes: {self.recipes_path}")
        for recipe_path in sorted(self.recipes_path.glob("*.yaml")):
            self.process_recipe(recipe_path)


def run_recipes_once(recipes_path):
    """Build samples for all existing recipes and exit."""
    SampleBuildRunner(recipes_path).process_existing_recipes()

if __name__ == "__main__":
    # One-shot entry point: build samples for existing recipes and exit.
    setting_path = ROOT.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    recipes_path = yaml_helper.get_data('recipes_path')

    run_recipes_once(recipes_path)
