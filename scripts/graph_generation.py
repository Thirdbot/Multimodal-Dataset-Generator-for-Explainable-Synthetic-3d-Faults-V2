# run watchdog over outputs of complete built only --extract db and array relation
# watch dog each extract.json to make graph using graph_generation.py

import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from logger_color import logger
import yaml
from yaml_helper import YAMLHelper
from low_level_tracer import LowLevelTracer
from high_level_tracer import HighLevelTracer
from graph_system import GraphSystem, ArrayRelationsGraph
from semantic_graph import SemanticGraph

class GraphGeneration(FileSystemEventHandler):
    def __init__(self):
        self.already_traced = set()

    def on_any_event(self, event):
        self.trace_success_file(event)

    def on_created(self, event):
        self.trace_success_file(event)

    def on_modified(self, event):
        self.trace_success_file(event)

    def on_moved(self, event):
        self.trace_success_file(event)

    def trace_success_file(self, event):
        if event.is_directory:
            return

        path = Path(getattr(event, "dest_path", event.src_path))

        self.trace_success_path(path)

    def trace_success_path(self, path):
        path = Path(path)

        if path.name != 'success.yaml':
            return

        if not path.exists():
            return

        time.sleep(0.2)
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        for folder in data.get("success_build_obj", []):
            folder = Path(folder)
            if not folder.exists():
                logger.warning(f"[TRACE SKIP] -> Missing folder: {folder}")
                continue

            if folder in self.already_traced:
                continue

            if self.trace_sample(folder):
                self.already_traced.add(folder)

    def trace_sample(self, folder):
        try:
            logger.info(f"[TRACE START] -> Sample: {folder.name}")

            low_tracker = LowLevelTracer(folder)
            low_tracker.save_to_json()
            low_trace_path = low_tracker.trace_sample_path

            high_tracker = HighLevelTracer(folder)
            high_tracker.save_to_json()
            high_trace_path = high_tracker.trace_sample_path

            properties_graph = GraphSystem()
            properties_graph.build(low_trace_path)
            properties_graph_path = properties_graph.save_to_json()

            array_graph = ArrayRelationsGraph()
            array_graph.build(high_trace_path)
            array_graph_path = array_graph.save_to_json()

            semantic_graph = SemanticGraph()
            semantic_graph.build(properties_graph_path, array_graph_path)
            semantic_graph_path = semantic_graph.save_to_json()

            logger.info(f"[TRACE DONE] -> Graph: {semantic_graph_path}")
            return True
        except Exception as exc:
            logger.error(f"[TRACE FAILED] -> Sample: {folder} Error: {exc}")
            return False

    def on_deleted(self, event):

        if not event.is_directory:
            return

        # delete extracted


def watch_over_outputs(successful_path):
    observe = Observer()
    successful_path = Path(successful_path)

    if successful_path.parent.exists():
        logger.debug(f"Watching {successful_path}")

        graph = GraphGeneration()
        if successful_path.exists():
            graph.trace_success_path(successful_path)
        observe.schedule(graph, successful_path.parent, recursive=False)
    else:
        logger.warning(f"file {successful_path} does not exist")

    observe.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error(f"[STOPPING]")
    finally:
        observe.stop()
    observe.join()


if __name__ == "__main__":
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    output_path = yaml_helper.get_data('output_path')
    successful_path = Path(output_path) / 'success.yaml'

    # watch over update in recipes
    watch_over_outputs(successful_path)
