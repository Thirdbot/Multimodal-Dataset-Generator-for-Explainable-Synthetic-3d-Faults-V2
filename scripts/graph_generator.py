"""Extract DB metadata from completed builds and convert it into graph JSON.

This script watches success.yaml, traces each successful Synthoseis build's
parameters.db, filters the extracted tables by sample category, and writes a
properties graph for downstream evidence/RAG/dataset generation.
"""

import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from logger_color import logger
import yaml
from yaml_helper import YAMLHelper
from low_level_tracer import ParameterDbTracer
from graph_system import GraphSystem

MODEL_KEYS = {
            "number_faults",
            "fault_mode",
            "salt_inserted",
            # "number_onlap_episodes",  # too broad/noisy for current QA generation
            # "number_fan_episodes",  # maps to broad lithology/depositional evidence
            "number_hc_closures",
            "number_fault_intersections",
            "fault_voxel_count_list",
            }
CLOSURE_KEYS = {
            "fluid",
            "intersects_fault",
            "intersects_onlap",
            "intersects_salt",

            }
FAULT_KEYS = {
            "throw",
            "tilt_pct",
            "shear_zone_width",
            "gouge_pctile"
}

# Per-category table/key filters for DB-derived graph content of what can exist
CATEGORY_FILTERS = {
    "boring": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys":MODEL_KEYS,
        "closure_keys": CLOSURE_KEYS
        },
    "fault_only": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "fault_keys": FAULT_KEYS,
        "closure_keys": CLOSURE_KEYS
    },
    "fault_complex": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "fault_keys": FAULT_KEYS,
        "closure_keys": CLOSURE_KEYS
    },
    "salt_only": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "closure_keys": CLOSURE_KEYS,
    },
    "salt_fault_mixed": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "fault_keys": FAULT_KEYS,
        "closure_keys": CLOSURE_KEYS,
    },
    "onlap": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "closure_keys": CLOSURE_KEYS,
    },
    "depositional": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "closure_keys": CLOSURE_KEYS,
    },
    "full_mixed": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": MODEL_KEYS,
        "fault_keys": FAULT_KEYS,
        "closure_keys": CLOSURE_KEYS,
    },
}

class BuildGraphGenerator(FileSystemEventHandler):
    """Watchdog handler for success.yaml updates emitted by the build stage."""

    def __init__(self):
        self.already_traced = set()
        self.last_success_mtime = None

    def on_any_event(self, event):
        self._trace_success_file(event)

    def on_created(self, event):
        self._trace_success_file(event)

    def on_modified(self, event):
        self._trace_success_file(event)

    def on_moved(self, event):
        self._trace_success_file(event)

    def _trace_success_file(self, event):
        if event.is_directory:
            return

        path = Path(getattr(event, "dest_path", event.src_path))

        self._trace_success_path(path)

    def _trace_success_path(self, path):
        """Read success.yaml and trace any newly completed build folders."""
        path = Path(path)

        if path.name != 'success.yaml':
            return

        if not path.exists():
            return

        current_mtime = path.stat().st_mtime
        if self.last_success_mtime is not None and current_mtime == self.last_success_mtime:
            return

        time.sleep(0.2)
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self.last_success_mtime = current_mtime

        for build_folder in data.get("success_build_obj", []):
            build_folder = Path(build_folder)
            if not build_folder.exists():
                logger.warning(f"[TRACE SKIP] -> Missing folder: {build_folder}")
                continue

            if build_folder in self.already_traced:
                continue

            if self._trace_sample(build_folder):
                self.already_traced.add(build_folder)

    def _category_from_sample_name(self,sample_name):
        for category in sorted(CATEGORY_FILTERS, key=len, reverse=True):
            if f"_{category}_" in sample_name or sample_name.startswith(f"{category}_"):
                return category
        return "unknown"

    def _trace_sample(self, build_folder):
        """Extract one build's DB tables and save its filtered properties graph."""
        try:
            logger.info(f"[TRACE START] -> Sample: {build_folder.name}")

            # create graphs
            db_tracer = ParameterDbTracer(build_folder)
            db_tracer.save_to_json()
            db_extract_path = db_tracer.db_extract_path

            properties_graph = GraphSystem()
            category = self._category_from_sample_name(build_folder.name)
            category_filter = CATEGORY_FILTERS.get(category, {})
            properties_graph.build(db_extract_path,category_filter=category_filter) # filtering by
            # create property graph
            properties_graph_path = properties_graph.save_to_json()
            # generate 2d images from 3d for each object in graphs that has position slice an individual one, and it corresponds mask and one with all objects in it with three angles (inline, crossline, horizontal)
            pass
            logger.info(f"[TRACE DONE] -> Graph: {properties_graph_path}")
            return True
        except Exception as exc:
            logger.error(f"[TRACE FAILED] -> Sample: {build_folder} Error: {exc}")
            return False

    def on_deleted(self, event):

        if not event.is_directory:
            return


def watch_success_tracker(success_tracker_path):
    """Run the graph-generation watcher around the build success tracker."""
    observer = Observer()
    success_tracker_path = Path(success_tracker_path)

    if success_tracker_path.parent.exists():
        logger.info(f"[NOW MONITORING] -> Path: {success_tracker_path}")

        graph_generator = BuildGraphGenerator()
        if success_tracker_path.exists():
            graph_generator._trace_success_path(success_tracker_path)
        observer.schedule(graph_generator, success_tracker_path.parent, recursive=False)
    else:
        logger.warning(f"file {success_tracker_path} does not exist")

    observer.start()
    try:
        while True:
            graph_generator._trace_success_path(success_tracker_path)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error(f"[STOPPING]")
    finally:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    setting_path = root.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    samples_path = yaml_helper.get_data('samples_path')
    samples_path = Path(samples_path)
    if not samples_path.is_absolute():
        samples_path = root / samples_path
    success_tracker_path = samples_path / 'success.yaml'

    # watch over update in recipes
    watch_success_tracker(success_tracker_path)
