import logging
import time
from pathlib import Path

from logger_color import SilentWatcher,logger
from yaml_helper import YAMLHelper
from watchdog.observers import Observer

logging.getLogger("watchdog.events").setLevel(logging.INFO)


def silent_bg_watcher(paths:list):
    colorful_loger = SilentWatcher()

    observer = Observer()
    for path in paths:
        path = Path(path)
        if path.exists():
            logger.info(f"[NOW MONITORING] -> Path: {path}")
            observer.schedule(
                colorful_loger,
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

if __name__ == '__main__':
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)

    output_path = yaml_helper.get_data('output_path')

    config_path = yaml_helper.get_data('config_path')
    recipe_path = yaml_helper.get_data('recipes_path')

    paths = [output_path,config_path,recipe_path]

    silent_bg_watcher(paths)
