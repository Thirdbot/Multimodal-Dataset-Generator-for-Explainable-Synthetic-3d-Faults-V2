import json
import shutil

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


class ConfigRunner(FileSystemEventHandler):
    def __init__(self, recipes_path):
        self.recipes_path = Path(recipes_path)
        self.recipes_name_path = None
        self.parent_path = self.recipes_path.parent
        self.configs_path = self.parent_path.joinpath('configs')
        setting_path = Path(__file__).parent.parent / "settings.yaml"
        yaml_helper = YAMLHelper(setting_path)
        self.project_folder = Path(yaml_helper.get_data("output_path"))
        self.work_folder = Path(yaml_helper.get_data("work_path"))

        self.recipe_cache = {}

        self.last_event_time = {}

    # de-bouncing
    def should_skip(self, path, seconds=2):
        now = time.time()
        last = self.last_event_time.get(path, 0)

        if now - last < seconds:
            return True

        self.last_event_time[path] = now
        return False

    def build_sample(self, sample_path, run_id, seed):
        try:
            build_model(
                user_json=str(sample_path),
                run_id=run_id,
                test_mode=None,
                seed=seed,
            )
            return True
        except BaseException as exc:
            logger.error(f"[BUILD FAILED] -> Run: {run_id} Error: {exc}")
            return False

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

            for folder in Path(self.project_folder).glob(f"seismic__*_{path.stem}_{sample}"):
                shutil.rmtree(folder, ignore_errors=True)

            for folder in Path(self.work_folder).glob(f"temp_folder__*_{path.stem}_{sample}"):
                shutil.rmtree(folder, ignore_errors=True)

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
                for folder in Path(self.project_folder).glob(f"seismic__*_{path.stem}_{sample}"):
                    shutil.rmtree(folder, ignore_errors=True)

                for folder in Path(self.work_folder).glob(f"temp_folder__*_{path.stem}_{sample}"):
                    shutil.rmtree(folder, ignore_errors=True)

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
            # sample_path.
            for folder in Path(self.project_folder).glob(f"seismic__*_{path.stem}_{sample}"):
                shutil.rmtree(folder, ignore_errors=True)

            for folder in Path(self.work_folder).glob(f"temp_folder__*_{path.stem}_{sample}"):
                shutil.rmtree(folder, ignore_errors=True)



def watch_over_files(recipes_path):
    observer = Observer()
    file_watcher = ConfigRunner(recipes_path)
    path = Path(recipes_path)

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
