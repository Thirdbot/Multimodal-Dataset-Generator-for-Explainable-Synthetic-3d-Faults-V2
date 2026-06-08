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


class ConfigRunner(FileSystemEventHandler):
    def __init__(self, recipes_path):
        self.root = Path(__file__).parent.parent
        self.recipes_path = self._resolve_path(recipes_path)
        self.recipes_name_path = None
        self.parent_path = self.recipes_path.parent
        self.configs_path = self.parent_path.joinpath('configs')
        setting_path = self.root / "settings.yaml"
        yaml_helper = YAMLHelper(setting_path)
        self.project_folder = self._resolve_path(yaml_helper.get_data("output_path"))
        self.work_folder = self._resolve_path(yaml_helper.get_data("work_path"))
        self.traces_path = self._resolve_path(yaml_helper.get_data("traces_path"))

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
    def should_skip(self, path, seconds=2):
        now = time.time()
        last = self.last_event_time.get(path, 0)

        if now - last < seconds:
            return True

        self.last_event_time[path] = now
        return False

    def delete_sample_artifacts(self, recipe_name, sample):
        deleted_outputs = []

        for folder in Path(self.project_folder).glob(f"seismic__*_{recipe_name}_{sample}"):
            deleted_outputs.append(folder.as_posix())
            shutil.rmtree(folder, ignore_errors=True)

        for folder in Path(self.work_folder).glob(f"temp_folder__*_{recipe_name}_{sample}"):
            shutil.rmtree(folder, ignore_errors=True)

        for output_folder in deleted_outputs:
            output_name = Path(output_folder).name
            db_extract = self.traces_path / f"{output_name}_db_extract.json"
            properties_graph = self.traces_path / "properties_graph" / f"{output_name}_db_extract_properties_graph.json"

            if db_extract.exists():
                db_extract.unlink()
                logger.info(f"[TRACE DELETE] -> {db_extract}")

            if properties_graph.exists():
                properties_graph.unlink()
                logger.info(f"[GRAPH DELETE] -> {properties_graph}")

        self.prune_success_tracker(deleted_outputs)

    def prune_success_tracker(self, deleted_outputs):
        success_configs = self.project_folder / "success.yaml"
        if not success_configs.exists():
            return

        with open(success_configs, "r") as file:
            data = yaml.safe_load(file) or {}

        old_success = data.get("success_build_obj", [])
        new_success = [path for path in old_success if path not in deleted_outputs]

        if new_success == old_success:
            return

        self.success = new_success
        with open(success_configs, "w") as file:
            yaml.dump({"success_build_obj": new_success}, file)

        logger.info(f"[TRACK PRUNE] -> Path: {success_configs}")

    def build_sample(self, sample_path, run_id, seed):
        logger.info(f"[BUILD START] -> Run: {run_id}")
        try:
            sample_seed = self.sample_seed(seed, sample_path.stem)
            guarded_build_model(
                build_model,
                user_json=str(sample_path),
                run_id=run_id,
                test_mode=None,
                seed=sample_seed,
            )
            logger.info(f"[BUILD DONE] -> Run: {run_id}")
            success_configs = self.project_folder / 'success.yaml'
            existing = {}
            if success_configs.exists():
                with open(success_configs, 'r') as file:
                    existing = yaml.safe_load(file) or {}

            list_folder = list(Path(self.project_folder).glob(f'seismic__*_{run_id}'))
            old_success = existing.get("success_build_obj", [])
            new_success = [folder.as_posix() for folder in list_folder if folder.exists()]
            self.success = list(dict.fromkeys(old_success + new_success))
            success = {"success_build_obj":self.success}
            with open(success_configs,'w') as file:
                yaml.dump(success, file)
            logger.info(f"[TRACK SUCCESS] -> Path: {success_configs}")

            return True
        except BaseException as exc:
            logger.error(f"[BUILD FAILED] -> Run: {run_id} Error: {exc}")

            failed_configs = self.project_folder / 'failed.yaml'
            failed_configs.touch(exist_ok=True)

            self.failed.append(str(sample_path))
            failed = {"failed_build_config": self.failed}

            with open(failed_configs,'w') as file:
                yaml.dump(failed, file)
            logger.info(f"[TRACK FAILED] -> Path: {failed_configs}")

            return False

    def sample_seed(self, seed, sample_name):
        digest = hashlib.sha256(f"{seed}:{sample_name}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    # events like delete ,create ,update recipes
    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".yaml":
            return
        time.sleep(0.2)
        yaml_helper = YAMLHelper(path)
        population = yaml_helper.get_data("population")
        seed = population['seed']
        samples = population['samples']

        logger.info(f"[CREATE] -> from: {path.stem}")
        self.recipe_cache[path] = {
            "seed": seed,
            "samples": samples,
            "configs": {},
        }
        for sample in samples:
            sample_path = self.configs_path.joinpath(f"{sample}.json")
            logger.info(f"[GENERATE] -> Config: {sample}")

            with open(str(sample_path), "r") as file:
                recipe = json.load(file)
                self.recipe_cache[path]["configs"][sample] = recipe
            self.build_sample(sample_path, f"{path.stem}_{sample}", seed)

        self.last_event_time[path] = time.time()


    def on_modified(self, event):
        if event.is_directory:
            return


        path = Path(event.src_path)
        if path.suffix != ".yaml":
            return

        if self.should_skip(path):
            return

        logger.debug(f"[UPDATE] -> Config changed: {path.stem}")
        time.sleep(0.2)
        yaml_helper = YAMLHelper(path)
        population = yaml_helper.get_data("population")
        seed = population['seed']
        samples = population['samples']

        logger.info(f"[MODIFIED] -> from: {path.stem}")
        old_recipe = self.recipe_cache.get(path)
        if old_recipe is None:
            old_recipe = {
                "seed": seed,
                "samples": [],
                "configs": {},
            }

        old_samples = set(old_recipe["samples"])
        new_samples = set(samples)

        for sample in old_samples - new_samples:
            sample_path = self.configs_path.joinpath(f"{sample}.json")
            if sample_path.exists():
                sample_path.unlink()

            self.delete_sample_artifacts(path.stem, sample)

        new_configs = {}
        for sample in new_samples:
            sample_path = self.configs_path.joinpath(f"{sample}.json")

            # read new recipes
            with open(str(sample_path), "r") as file:
                new_recipes = json.load(file)

            old_recipes = old_recipe["configs"].get(sample)
            should_rebuild = (
                sample in new_samples - old_samples
                or old_recipe["seed"] != seed
                or old_recipes != new_recipes
            )

            if should_rebuild:
                logger.warning(f"[MODIFIED] -> Config: {sample}")
                self.delete_sample_artifacts(path.stem, sample)
                self.build_sample(sample_path, f"{path.stem}_{sample}", seed)

            new_configs[sample] = new_recipes

        self.recipe_cache[path] = {
            "seed": seed,
            "samples": samples,
            "configs": new_configs,
        }

    def on_deleted(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".yaml":
            return
        recipe = self.recipe_cache.pop(path, None)

        logger.warning(f"[DELETE] -> Config deleted: {path}")

        if recipe:
            seed = recipe['seed']
            samples = recipe['samples']
        else:
            seed = 0
            samples = []
        logger.info(f"[CREATE] -> from: {path.stem}")
        for sample in samples:
            sample_path = self.configs_path.joinpath(f"{sample}.json")
            if sample_path.exists():
                sample_path.unlink()

            logger.warning(f"[MODIFIED] -> Config: {sample}")
            self.delete_sample_artifacts(path.stem, sample)



def watch_over_files(recipes_path):
    observer = Observer()
    file_watcher = ConfigRunner(recipes_path)
    path = file_watcher.recipes_path

    if path.exists():
        logger.info(f"[NOW MONITORING] -> Path: {path}")
        observer.schedule(
            file_watcher,
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
    # get configs parameters
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    recipes_path = yaml_helper.get_data('recipes_path')

    # watch over update in recipes
    watch_over_files(recipes_path)
