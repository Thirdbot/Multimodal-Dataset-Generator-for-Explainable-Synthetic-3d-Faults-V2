"""Generic colored filesystem watcher for configured runtime folders."""

import logging
import time
from pathlib import Path

from logger_color import SilentWatcher,logger
from yaml_helper import YAMLHelper
from watchdog.observers import Observer

logging.getLogger("watchdog.events").setLevel(logging.INFO)


def watch_runtime_paths(paths:list):
    """Watch every existing path in paths and report events with colors."""
    color_event_reporter = SilentWatcher()

    observer = Observer()
    for path in paths:
        path = Path(path)
        if path.exists():
            logger.info(f"[NOW MONITORING] -> Path: {path}")
            observer.schedule(
                color_event_reporter,
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
    # Developer utility entry point for watching outputs/configs/recipes.
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)

    samples_path = yaml_helper.get_data('samples_path')

    build_configs_path = yaml_helper.get_data('build_configs_path')
    recipes_path = yaml_helper.get_data('recipes_path')

    paths = [samples_path,build_configs_path,recipes_path]

    watch_runtime_paths(paths)
