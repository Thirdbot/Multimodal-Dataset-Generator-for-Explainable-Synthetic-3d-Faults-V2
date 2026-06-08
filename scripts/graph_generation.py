# run watchdog over outputs of complete built only --extract db and build properties graph
# watch dog each extract.json to make graph using graph_generation.py

import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from logger_color import logger
import yaml
from yaml_helper import YAMLHelper
from low_level_tracer import LowLevelTracer
from graph_system import GraphSystem
from sample_image_generation import create_sample_images

CATEGORY_FILTERS = {
    "boring": {
        "tables": {"model_parameters"},
        "model_keys": {
            "number_faults",
            "fault_mode",
            "salt_inserted",
            "number_onlap_episodes",
            "number_fan_episodes",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
            "sand_voxel_pct",
            "sand_layer_percent_a_posteriori",
        },
    },
    "fault_only": {
        "tables": {"model_parameters", "fault_parameters"},
        "model_keys": {
            "number_faults",
            "fault_mode",
            "n_voxels_faults",
            "n_voxels_fault_intersections",
            "number_fault_intersections",
            "fault_voxel_count_list",
        },
        "fault_keys": {
            "x0",
            "y0",
            "z0",
            "throw",
            "tilt_pct",
            "shear_zone_width",
            "gouge_pctile",
        },
    },
    "fault_complex": {
        "tables": {"model_parameters", "fault_parameters"},
        "model_keys": {
            "number_faults",
            "fault_mode",
            "n_voxels_faults",
            "n_voxels_fault_intersections",
            "number_fault_intersections",
            "fault_voxel_count_list",
        },
        "fault_keys": {
            "x0",
            "y0",
            "z0",
            "throw",
            "tilt_pct",
            "shear_zone_width",
            "gouge_pctile",
        },
    },
    "salt_only": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": {
            "salt_inserted",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
            "closure_voxel_count_brine",
            "closure_voxel_count_oil",
            "closure_voxel_count_gas",
            "closure_voxel_pct_brine",
            "closure_voxel_pct_oil",
            "closure_voxel_pct_gas",
        },
        "closure_keys": {
            "fluid",
            "n_voxels",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "intersects_salt",
        },
    },
    "salt_fault_mixed": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": {
            "salt_inserted",
            "number_faults",
            "fault_mode",
            "n_voxels_faults",
            "n_voxels_fault_intersections",
            "number_fault_intersections",
            "fault_voxel_count_list",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
        },
        "fault_keys": {
            "x0",
            "y0",
            "z0",
            "throw",
            "tilt_pct",
            "shear_zone_width",
            "gouge_pctile",
        },
        "closure_keys": {
            "fluid",
            "n_voxels",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "intersects_fault",
            "intersects_salt",
        },
    },
    "onlap": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": {
            "number_onlap_episodes",
            "onlaps_horizon_list",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
        },
        "closure_keys": {
            "fluid",
            "n_voxels",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "intersects_onlap",
        },
    },
    "depositional": {
        "tables": {"model_parameters", "closure_parameters"},
        "model_keys": {
            "number_fan_episodes",
            "fan_horizon_list",
            "sand_voxel_pct",
            "sand_layer_percent_a_posteriori",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
        },
        "closure_keys": {
            "fluid",
            "n_voxels",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
        },
    },
    "full_mixed": {
        "tables": {"model_parameters", "fault_parameters", "closure_parameters"},
        "model_keys": {
            "number_faults",
            "fault_mode",
            "n_voxels_faults",
            "n_voxels_fault_intersections",
            "number_fault_intersections",
            "fault_voxel_count_list",
            "salt_inserted",
            "number_onlap_episodes",
            "onlaps_horizon_list",
            "number_fan_episodes",
            "fan_horizon_list",
            "sand_voxel_pct",
            "sand_layer_percent_a_posteriori",
            "number_hc_closures",
            "closure_voxel_count",
            "closure_voxel_pct",
            "closure_voxel_count_brine",
            "closure_voxel_count_oil",
            "closure_voxel_count_gas",
            "closure_voxel_pct_brine",
            "closure_voxel_pct_oil",
            "closure_voxel_pct_gas",
        },
        "fault_keys": {
            "x0",
            "y0",
            "z0",
            "throw",
            "tilt_pct",
            "shear_zone_width",
            "gouge_pctile",
        },
        "closure_keys": {
            "fluid",
            "n_voxels",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "intersects_fault",
            "intersects_salt",
            "intersects_onlap",
        },
    },
}

class GraphGeneration(FileSystemEventHandler):
    def __init__(self):
        self.already_traced = set()
        self.last_success_mtime = None

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

        current_mtime = path.stat().st_mtime
        if self.last_success_mtime is not None and current_mtime == self.last_success_mtime:
            return

        time.sleep(0.2)
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        self.last_success_mtime = current_mtime

        for folder in data.get("success_build_obj", []):
            folder = Path(folder)
            if not folder.exists():
                logger.warning(f"[TRACE SKIP] -> Missing folder: {folder}")
                continue

            if folder in self.already_traced:
                continue

            if self.trace_sample(folder):
                self.already_traced.add(folder)

    def category_from_sample_name(self,sample_name):
        for category in sorted(CATEGORY_FILTERS, key=len, reverse=True):
            if f"_{category}_" in sample_name or sample_name.startswith(f"{category}_"):
                return category
        return "unknown"

    def trace_sample(self, folder):
        try:
            logger.info(f"[TRACE START] -> Sample: {folder.name}")

            low_tracker = LowLevelTracer(folder)
            low_tracker.save_to_json()
            low_trace_path = low_tracker.trace_sample_path

            properties_graph = GraphSystem()
            category = self.category_from_sample_name(folder.name)
            the_great_filter = CATEGORY_FILTERS.get(category, {})
            properties_graph.build(low_trace_path,the_great_filter=the_great_filter) # filtering by
            properties_graph_path = properties_graph.save_to_json()
            # generate 2d images from 3d
            image_assets = create_sample_images(folder, graph_path=properties_graph_path)
            for view in ["inline", "crossline"]:
                graph_2d = properties_graph.change_build(view, image_assets=image_assets)
                graph_2d.save_to_json(sub_folder="views_graph", suffix=f"{view}_graph")

            logger.info(f"[TRACE DONE] -> Graph: {properties_graph_path}")
            return True
        except Exception as exc:
            logger.error(f"[TRACE FAILED] -> Sample: {folder} Error: {exc}")
            return False

    def on_deleted(self, event):

        if not event.is_directory:
            return


def watch_over_outputs(successful_path):
    observe = Observer()
    successful_path = Path(successful_path)

    if successful_path.parent.exists():
        logger.info(f"[NOW MONITORING] -> Path: {successful_path}")

        graph = GraphGeneration()
        if successful_path.exists():
            graph.trace_success_path(successful_path)
        observe.schedule(graph, successful_path.parent, recursive=False)
    else:
        logger.warning(f"file {successful_path} does not exist")

    observe.start()
    try:
        while True:
            graph.trace_success_path(successful_path)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.error(f"[STOPPING]")
    finally:
        observe.stop()
    observe.join()


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    setting_path = root.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    output_path = yaml_helper.get_data('output_path')
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    successful_path = output_path / 'success.yaml'

    # watch over update in recipes
    watch_over_outputs(successful_path)
