"""Prototype helpers for linking graph objects back to build output arrays.

The class reads a properties graph, identifies graph nodes that represent
objects such as faults/closures/salt, and resolves the matching Synthoseis build
folder from the graph filename. Image extraction is intentionally still a stub.
"""

import json
from pathlib import Path
from yaml_helper import YAMLHelper
import re
import xarray
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

GRAPH_OBJECT_TYPES = [
    "closure",
    "fault",
]

ENTITY_OBJECT_TYPES = {
    "fault",
    "closure",
    "salt",
    "onlap",
}

CLASS_IDS = {
    "fault": 1,
    "closure": 2,
    "salt": 3,
    "onlap": 4,
    "lithology": 5,
    "age_depth": 6,
}

CLASS_RGB_COLORS = {
    1: [255, 0, 0],
    2: [0, 128, 255],
    3: [180, 0, 255],
    4: [255, 220, 0],
    5: [0, 200, 100],
    6: [255, 140, 0],
}

CLASS_COLOR_NAMES = {
    1: "red",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "green",
    6: "orange",
}
FLUID_CLOSURE_PATTERNS = {
    "oil": "closures/oil.zarr",
    "gas": "closures/gas.zarr",
    "brine": "closures/brine.zarr",
}

OBJECT_ZARR_CANDIDATES = {
      "fault": [
          "faults/fault_*.zarr",               # per-fault masks written by wrapper
          "fault_segments_*.zarr",              # best fault object mask
          "fault_intersection_segments_*.zarr", # fault intersections
      ],
      "closure": [
          "closures/oil.zarr",
          "closures/gas.zarr",
          "closures/brine.zarr",
      ],
      "onlap": [
          "onlap_segments_*.zarr",
      ],
      "salt": [
          # only if Synthoseis writes a salt label in that build.
          # If none exists, infer from closures/intersects_salt graph data, not image mask.
          "salt_[0-9]*.zarr",
      ],
      "lithology": [
          "geology/faulted_lithology.zarr",
          "faulted_lithology_*.zarr",
      ],
      "age_depth": [
          "geology/geologic_age.zarr",
          "geologic_age_*.zarr",
          "faulted_age_*.zarr",
          "depth_maps.zarr",
          "faulted_depth_*.zarr",
      ],
  }

OBJECT_ZARR_BASIC = {
"full_model": [
        "seismicCubes_RFC_fullstack_*.zarr" , # first priority
        "seismicCubes_cumsum_fullstack_*.zarr" # fall-back
    ]
}

class GraphImageExtractor:
    """Inspect a graph and locate the build folder with the same sample id."""

    def __init__(self,graph_path,samples_path):
        self.graph_path = Path(graph_path)

        self.image2d_path = self.graph_path.parent.parent.parent.joinpath('build_objects','images') # 2d slicing batch of images
        self.image3d_path = self.graph_path.parent.parent.parent.joinpath('build_objects','objects') # original 3d object from generation

        self.build_folder_path = samples_path / self._sample_id_from_graph_path(self.graph_path) # sample folder name

        self.graph_json = self._read_json() # json object

    @staticmethod
    def _sample_id_from_graph_path(graph_path):
        """Convert a graph filename back to its generated sample folder name."""
        stem = Path(graph_path).stem
        suffix = "_properties_graph"
        if stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
        return stem.replace("_db_extract", "")

    def _read_json(self):
        with open(self.graph_path) as json_file:
            return json.loads(json_file.read())

    def _get_nodes(self):
        return self.graph_json['nodes']

    def _get_edges(self):
        return self.graph_json['edges']

    def _get_object_nodes(self):
        """Return graph nodes whose ids look like per-object nodes."""
        objects = []

        for item in self._get_nodes():
            for types in GRAPH_OBJECT_TYPES:
                if re.findall(rf"^{types}_\d",item['id']):
                    objects.append(item)
        return objects

    def extract_object_images(self):
        # Extract global and per-entity 2D views from generated 3D arrays.
        #
        # Global arrays are still useful for broad context. Individual entities
        # are only emitted when the array/graph gives enough information to map
        # one graph object to one mask component.
        sample_name = self.build_folder_path.stem
        basic_objects = self._extract_basic()

        sample_object_properties = self._get_object_nodes()

        if not basic_objects:
            return []

        base_volume = self._prepare_3d_array(self._as_array(basic_objects[0]))

        selected_objects = {}
        for object_type,patterns in OBJECT_ZARR_CANDIDATES.items():
            if object_type == "fault":
                self._warn_missing_wrapper_faults(sample_name)
            self.image2d_path.joinpath(sample_name,object_type).mkdir(parents=True, exist_ok=True) # create folder for each object type and sample for 2d images
            self.image3d_path.joinpath(sample_name,object_type).mkdir(parents=True, exist_ok=True) # create folder for each object type and sample for 3d objects
            type_sliced_objects = {}
            type_global_mask = None
            for pattern in patterns:
                for match_sample in self.build_folder_path.glob(pattern):
                    if self._is_empty_zarr_store(match_sample):
                        continue
                    if object_type == "fault" and self._is_fault_property_volume(match_sample):
                        continue
                    # Load one generated object volume and split it into useful
                    # global/individual masks depending on object type.
                    property_object = self._load_sample_object(match_sample)
                    property_array = self._prepare_3d_array(self._as_array(property_object))
                    if object_type == "closure":
                        type_global_mask = self._merge_mask_arrays(type_global_mask, property_array)
                        slices = self._closure_individual_slices(
                            base_volume,
                            property_array,
                            match_sample,
                            sample_object_properties,
                        )
                    else:
                        slices = self._entity_slices(
                            base_volume,
                            property_array,
                            object_type,
                            match_sample,
                            sample_object_properties,
                        )
                    self._merge_slices(type_sliced_objects, slices) # store all slices both closure and faults and etc.
            if object_type == "closure" and type_global_mask is not None and type_global_mask.any():
                self._merge_slices(
                    type_sliced_objects,
                    {"closure": self._slice_by_mask(base_volume, type_global_mask)},
                )
            selected_type_slices = self._select_average_mask_slices(
                type_sliced_objects,
                object_type,
                sample_object_properties,
            )
            self.make_image(sample_name, object_type, selected_type_slices)
            self._save_object_positions(
                sample_name,
                object_type,
                self._position_records(sample_name, object_type, selected_type_slices),
            )
            selected_objects.update(selected_type_slices)
        return selected_objects

    def _warn_missing_wrapper_faults(self,sample_name):
        expected_faults = self._expected_visible_fault_count()
        if expected_faults <= 0:
            return

        wrapper_faults = list(self.build_folder_path.glob("faults/fault_*.zarr"))
        if len(wrapper_faults) >= expected_faults:
            return

        print(
            "[MISSING WRAPPER FAULTS] -> "
            f"Sample: {sample_name} expected {expected_faults}, found {len(wrapper_faults)}. "
            "Rebuild this sample through guarded_build_model; existing global fault_segments cannot be split reliably."
        )

    def _expected_visible_fault_count(self):
        fault_counts = self._fault_voxel_count_list()
        if fault_counts:
            return len([count for count in fault_counts if count > 0])
        return self._expected_fault_count()

    def _expected_fault_count(self):
        for node in self._get_nodes():
            value = node.get("number_faults")
            if value is None:
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0
        return len([node for node in self._get_object_nodes() if node.get("id", "").startswith("fault_")])

    @staticmethod
    def _is_fault_property_volume(path):
        name = Path(path).name
        return "fault_segments_throw_" in name or "fault_segments_azimuth_" in name

    @staticmethod
    def _is_empty_zarr_store(path):
        # Some Synthoseis outputs are metadata-only Zarr stores with all-zero
        # fill. They cannot produce a mask, so skip them before xarray opens.
        path = Path(path)
        if path.suffix != ".zarr" or not path.is_dir():
            return False
        for item in path.rglob("*"):
            if item.is_file() and item.name != "zarr.json":
                return False
        return True

    def _merge_slices(self,target,new_slices):
        for object_id, sliced in new_slices.items():
            if object_id not in target:
                target[object_id] = sliced
                continue
            if target[object_id].get("wrapper_source") and not sliced.get("wrapper_source"):
                continue
            if sliced.get("wrapper_source") and not target[object_id].get("wrapper_source"):
                target[object_id] = sliced
                continue
            if self._slice_mask_area(sliced) > self._slice_mask_area(target[object_id]):
                target[object_id] = sliced

    def _merge_mask_arrays(self,current,new_array):
        new_mask = np.nan_to_num(new_array, nan=0.0) != 0
        if current is None:
            return new_mask

        common_shape = tuple(min(a, b) for a, b in zip(current.shape, new_mask.shape))
        merged = np.zeros(common_shape, dtype=bool)
        merged |= current[:common_shape[0], :common_shape[1], :common_shape[2]]
        merged |= new_mask[:common_shape[0], :common_shape[1], :common_shape[2]]
        return merged

    def _extract_basic(self):
        # Extract original 3D object arrays from Synthoseis build folder
        all_patterns = self._find_objects(OBJECT_ZARR_BASIC)
        store_objs = []
        for pattern in all_patterns:
            for match_sample in self.build_folder_path.glob(pattern):
                if self._is_empty_zarr_store(match_sample):
                    continue
                data = self._load_sample_object(match_sample)
                store_objs.append(data)
            if store_objs:
                break
        return store_objs

    @staticmethod
    def _find_objects(objects_filename):
        all_patterns = []
        for patterns in objects_filename.values():
            all_patterns.extend(patterns)
        return all_patterns

    def _entity_slices(self,base_array,property_array,object_type,source_path,object_properties):
        # Always keep the global object-type view. Individual entity views are
        # added beside it when graph/object masks can separate them reliably.
        slices = self._global_slices(base_array, property_array, object_type)

        if object_type == "closure":
            individual = self._closure_individual_slices(base_array, property_array, source_path, object_properties)
            slices.update(individual)
            return slices

        if object_type == "fault":
            individual = self._direct_fault_slices(base_array, property_array, source_path)
            if individual:
                slices.update(individual)
                return slices
            individual = self._fault_individual_slices(base_array, property_array, object_properties)
            slices.update(individual)
            return slices

        if object_type in {"salt", "onlap"}:
            individual = self._component_individual_slices(base_array, property_array, object_type)
            slices.update(individual)
            return slices

        return slices

    def _object_slicing(self,basic_obj,prop_obj):
        # Slice 3D object arrays and create 2D object crops/masks
        basic_array = self._prepare_3d_array(self._as_array(basic_obj))
        prop_array = self._prepare_3d_array(self._as_array(prop_obj))
        return list(self._global_slices(basic_array, prop_array, "object").values())

    def _global_slices(self,basic_array,prop_array,object_type):
        # One mask for the whole generated entity class.
        common_shape = tuple(min(a, b) for a, b in zip(basic_array.shape, prop_array.shape))
        basic_array = basic_array[:common_shape[0], :common_shape[1], :common_shape[2]]
        prop_array = prop_array[:common_shape[0], :common_shape[1], :common_shape[2]]

        prop_mask = np.nan_to_num(prop_array, nan=0.0) != 0

        if prop_mask.any():
            return {object_type: self._slice_by_mask(basic_array, prop_mask)}
        return {}

    def _fault_individual_slices(self,base_array,property_array,object_properties):
        # Synthoseis final fault_segments is often a combined binary mask. Do
        # not turn that into fake fault_0/fault_1 masks. Individual faults are
        # expected from the wrapper-generated faults/fault_*.zarr files. This
        # fallback is only for true label-map volumes.
        fault_objects = [obj for obj in object_properties if obj.get("id", "").startswith("fault_")]
        if not fault_objects:
            return {}

        prop_mask = np.nan_to_num(property_array, nan=0.0)
        if not np.any(prop_mask):
            return {}

        label_masks = self._label_masks(prop_mask)
        if not label_masks:
            return {}

        selected = {}
        used = set()
        for fault in sorted(fault_objects, key=lambda item: self._object_index(item.get("id")) or 0):
            expected_count = self._expected_voxel_count(fault.get("id"), "fault", object_properties)
            best_index, best_mask = self._best_mask_for_object(label_masks, fault, expected_count, used)
            if best_mask is None:
                continue
            used.add(best_index)
            selected[fault["id"]] = self._slice_by_mask(base_array, best_mask)
        return selected

    def _direct_fault_slices(self,base_array,property_array,source_path):
        # Wrapper-generated fault volumes are already one file per fault.
        match = re.match(r"fault_(\d+)\.zarr$", Path(source_path).name)
        if not match or Path(source_path).parent.name != "faults":
            return {}

        fault_id = self._fault_id_from_original_index(int(match.group(1)))
        fault_mask = np.nan_to_num(property_array, nan=0.0) != 0
        if not fault_mask.any():
            return {}
        sliced = self._slice_by_mask(base_array, fault_mask)
        sliced["wrapper_source"] = True
        return {fault_id: sliced}

    def _fault_id_from_original_index(self,original_index):
        for node in self._get_object_nodes():
            if not node.get("id", "").startswith("fault_"):
                continue
            if self._as_float(node.get("original_fault_index")) == float(original_index):
                return node["id"]
        return f"fault_{original_index}"

    def _closure_individual_slices(self,base_array,property_array,source_path,object_properties):
        # Closure positions are derived from the generated closure mask itself.
        # The graph only needs closure row identity and fluid type.
        source_fluid = self._closure_fluid_from_path(source_path)
        if source_fluid is None:
            return {}

        closure_mask = np.nan_to_num(property_array, nan=0.0) != 0
        if not closure_mask.any():
            return {}

        selected = {}
        closure_objects = [
            obj for obj in object_properties
            if obj.get("id", "").startswith("closure_")
            and str(obj.get("fluid", "")).lower() == source_fluid
        ]
        closure_objects = sorted(closure_objects, key=lambda item: self._object_index(item.get("id")) or 0)
        component_masks = self._connected_component_masks(closure_mask)

        for index, component_mask in enumerate(component_masks):
            object_id = closure_objects[index]["id"] if index < len(closure_objects) else f"closure_{index}"
            selected[object_id] = self._slice_by_mask(base_array, component_mask)
        return selected

    @staticmethod
    def _closure_fluid_from_path(path):
        path = Path(path)
        if path.parent.name != "closures":
            return None
        if path.name in {"oil.zarr", "gas.zarr", "brine.zarr"}:
            return path.stem
        return None

    def _component_individual_slices(self,base_array,property_array,object_type):
        # Salt and onlap do not currently have stable per-object graph ids.
        # Split disconnected components and name them by component order.
        prop_mask = np.nan_to_num(property_array, nan=0.0) != 0
        selected = {}
        for index, component_mask in enumerate(self._connected_component_masks(prop_mask)):
            selected[f"{object_type}_{index}"] = self._slice_by_mask(base_array, component_mask)
        return selected

    def _slice_by_mask(self,basic_array,mask):
        common_shape = tuple(min(a, b) for a, b in zip(basic_array.shape, mask.shape))
        basic_array = basic_array[:common_shape[0], :common_shape[1], :common_shape[2]]
        mask = mask[:common_shape[0], :common_shape[1], :common_shape[2]]

        inline_index = int(mask.sum(axis=(1, 2)).argmax())
        crossline_index = int(mask.sum(axis=(0, 2)).argmax())
        timeslice_index = int(mask.sum(axis=(0, 1)).argmax())
        return {
            "basic": {
                "inline": self._display_slice("inline", basic_array[inline_index, :, :]),
                "crossline": self._display_slice("crossline", basic_array[:, crossline_index, :]),
                "timeslice": self._display_slice("timeslice", basic_array[:, :, timeslice_index]),
            },
            "mask": {
                "inline": self._display_slice("inline", mask[inline_index, :, :]),
                "crossline": self._display_slice("crossline", mask[:, crossline_index, :]),
                "timeslice": self._display_slice("timeslice", mask[:, :, timeslice_index]),
            },
        }

    @staticmethod
    def _display_slice(view,slice_2d):
        # Vertical sections are displayed as depth-by-lateral images, so the
        # shorter lateral axis stays horizontal and depth reads vertically.
        if view in {"inline", "crossline"}:
            return np.asarray(slice_2d).T
        return np.asarray(slice_2d)

    def _select_average_mask_slices(self,slices,object_type,object_properties):
        # Global slices and individual slices are already keyed. Empty masks are
        # dropped, but object-level global views remain as fallback.
        candidates = {
            object_id: candidate
            for object_id, candidate in slices.items()
            if self._slice_mask_area(candidate) > 0
        }
        return candidates

    def _expected_voxel_count(self,object_id,object_type,object_properties):
        if object_type == "fault":
            fault_counts = self._fault_voxel_count_list()
            object_index = self._object_index(object_id)
            if fault_counts and object_index is not None and object_index < len(fault_counts):
                return fault_counts[object_index]

        for graph_object in object_properties:
            if graph_object.get("id") == object_id and graph_object.get("n_voxels") is not None:
                return float(graph_object["n_voxels"])

        return None

    def _fault_voxel_count_list(self):
        for node in self._get_nodes():
            value = node.get("fault_voxel_count_list")
            if value is None:
                continue
            if isinstance(value, list):
                return [float(item) for item in value]
            if isinstance(value, str):
                return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]
        return []

    @staticmethod
    def _prepare_3d_array(array):
        array = np.asarray(array)
        if array.ndim > 3:
            array = np.squeeze(array)
        if array.ndim != 3:
            raise ValueError("object array must resolve to a 3D array")
        return array

    @staticmethod
    def _label_masks(array):
        # Use integer labels only when the volume looks like an actual label map.
        # Float-valued volumes such as throw/azimuth should not become labels.
        values = np.unique(array[np.isfinite(array)])
        values = values[values != 0]
        if values.size == 0:
            return []
        if values.size == 1 and np.isclose(values[0], 1):
            return []
        if values.size > 256:
            return []
        if not np.allclose(values, np.round(values)):
            return []
        return [array == value for value in values]

    @staticmethod
    def _connected_component_masks(mask):
        labels, number_labels = ndimage.label(mask)
        masks = []
        for label_id in range(1, number_labels + 1):
            component = labels == label_id
            if component.any():
                masks.append(component)
        return masks

    def _best_mask_for_object(self,masks,graph_object,expected_count,used_indexes):
        best_index = None
        best_mask = None
        best_score = None
        expected_count = self._as_float(expected_count)
        expected_index = self._object_center_index(graph_object, masks[0].shape) if masks else None

        for index, mask in enumerate(masks):
            if index in used_indexes:
                continue

            count = float(mask.sum())
            count_score = 0.0
            if expected_count and expected_count > 0:
                count_score = abs(count - expected_count) / expected_count

            distance_score = 0.0
            if expected_index is not None:
                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                distance_score = float(np.linalg.norm(coords.mean(axis=0) - expected_index)) / max(mask.shape)

            score = count_score + distance_score
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
                best_mask = mask

        return best_index, best_mask

    def _bbox_mask(self,mask,graph_object):
        # Restrict closure masks to the DB bounding box before component search.
        x_min = self._axis_value(graph_object, "x_min", 0)
        x_max = self._axis_value(graph_object, "x_max", mask.shape[0] - 1)
        y_min = self._axis_value(graph_object, "y_min", 0)
        y_max = self._axis_value(graph_object, "y_max", mask.shape[1] - 1)
        z_min = self._axis_value(graph_object, "z_min", 0)
        z_max = self._axis_value(graph_object, "z_max", mask.shape[2] - 1)

        bounded = np.zeros_like(mask, dtype=bool)
        x0, x1 = self._clip_range(x_min, x_max, mask.shape[0])
        y0, y1 = self._clip_range(y_min, y_max, mask.shape[1])
        z0, z1 = self._clip_range(z_min, z_max, mask.shape[2])
        bounded[x0:x1, y0:y1, z0:z1] = mask[x0:x1, y0:y1, z0:z1]
        return bounded

    def _object_center_index(self,graph_object,shape):
        if all(key in graph_object for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
            return np.array([
                sum(self._clip_range(graph_object["x_min"], graph_object["x_max"], shape[0])) / 2.0,
                sum(self._clip_range(graph_object["y_min"], graph_object["y_max"], shape[1])) / 2.0,
                sum(self._clip_range(graph_object["z_min"], graph_object["z_max"], shape[2])) / 2.0,
            ])

        if all(key in graph_object for key in ("x0", "y0", "z0")):
            return np.array(self._coordinate_to_index(
                {"x": graph_object["x0"], "y": graph_object["y0"], "z": graph_object["z0"]},
                shape,
            ))

        return None

    @staticmethod
    def _axis_value(graph_object,key,default):
        value = graph_object.get(key, default)
        if value is None:
            return default
        return value

    @staticmethod
    def _clip_range(low,high,size):
        low = int(np.clip(round(float(low)), 0, size - 1))
        high = int(np.clip(round(float(high)), 0, size - 1))
        if high < low:
            low, high = high, low
        return low, min(high + 1, size)

    @staticmethod
    def _as_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _object_index(object_id):
        match = re.search(r"_(\d+)$", str(object_id))
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _slice_mask_area(sliced):
        return sum(int(np.asarray(mask_slice).sum()) for mask_slice in sliced.get("mask", {}).values())

    def make_image(self,sample_name,object_type,selected_slices):
        output_folder = self.image2d_path.joinpath(sample_name, object_type)
        output_folder.mkdir(parents=True, exist_ok=True)

        for object_id, sliced in selected_slices.items():
            object_folder = output_folder / self._safe_filename(object_id)
            object_folder.mkdir(parents=True, exist_ok=True)
            overlay_color = self._class_rgb_color(object_type)
            for view in ("inline", "crossline", "timeslice"):
                basic_slice = self._normalize_image(sliced["basic"][view])
                mask_slice = self._display_mask(object_type, sliced["mask"][view])
                self._save_png(object_folder / f"{view}.png", basic_slice, cmap="gray")
                self._save_mask_png(object_folder / f"{view}_mask.png", mask_slice)
                self._save_overlay(object_folder / f"{view}_overlay.png", basic_slice, mask_slice, overlay_color)

    def _position_records(self,sample_name,object_type,selected_slices):
        records = []
        for object_id, sliced in selected_slices.items():
            object_folder = Path(object_type) / self._safe_filename(object_id)
            for view in ("inline", "crossline", "timeslice"):
                mask_slice = self._display_mask(object_type, sliced["mask"][view])
                bbox = self._mask_bbox(mask_slice)
                if bbox is None:
                    continue
                records.append({
                    "sample_id": sample_name,
                    "object_type": object_type,
                    "object_id": str(object_id),
                    "class_color": self._class_color(object_type),
                    "view": view,
                    "image_path": (object_folder / f"{view}.png").as_posix(),
                    "mask_path": (object_folder / f"{view}_mask.png").as_posix(),
                    "overlay_path": (object_folder / f"{view}_overlay.png").as_posix(),
                    "bbox": bbox,
                    "center": {
                        "x": (bbox["x_min"] + bbox["x_max"]) / 2,
                        "y": (bbox["y_min"] + bbox["y_max"]) / 2,
                    },
                })
        return records

    @staticmethod
    def _class_id(object_type):
        return CLASS_IDS.get(str(object_type), 0)

    @staticmethod
    def _class_color(object_type):
        return CLASS_COLOR_NAMES.get(CLASS_IDS.get(str(object_type), 0), "white")
    @staticmethod
    def _class_rgb_color(object_type):
        return CLASS_RGB_COLORS.get(CLASS_IDS.get(str(object_type), 0), [255,255,255])

    def _save_object_positions(self,sample_name,object_type,records):
        if not records:
            return
        output_path = self.image2d_path / sample_name / f"{object_type}_object_position.json"
        payload = {
            "sample_id": sample_name,
            "object_type": object_type,
            "objects": records,
        }
        output_path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _display_mask(object_type,mask_slice):
        mask = np.asarray(mask_slice, dtype=bool)
        if object_type != "fault" or not mask.any():
            return mask

        # Synthoseis fault masks are fault-affected zones. For visual training
        # targets, reduce the zone to a trace-like mask.
        if skeletonize is not None:
            traced = skeletonize(mask)
            if traced.any():
                return traced

        eroded = ndimage.binary_erosion(mask)
        traced = mask ^ eroded
        return traced if traced.any() else mask

    @staticmethod
    def _mask_bbox(mask_slice):
        coords = np.argwhere(mask_slice)
        if coords.size == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
        }

    @staticmethod
    def _normalize_image(image):
        image = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0)
        low, high = np.percentile(image, [1, 99])
        if high <= low:
            high = image.max()
            low = image.min()
        if high <= low:
            return np.zeros_like(image, dtype=float)
        return np.clip((image - low) / (high - low), 0.0, 1.0)

    @staticmethod
    def _save_png(path,image,cmap):
        plt.imsave(path, image, cmap=cmap)

    @staticmethod
    def _save_mask_png(path,mask_slice):
        mask = np.asarray(mask_slice, dtype=bool).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(path)

    @staticmethod
    def _save_overlay(path,basic_slice,mask_slice,color):
        overlay = np.dstack([basic_slice, basic_slice, basic_slice])
        color = np.asarray(color, dtype=float) / 255.0
        overlay[mask_slice] = (overlay[mask_slice] * 0.35) + (color * 0.65)
        plt.imsave(path, overlay)

    @staticmethod
    def _safe_filename(value):
        return re.sub(r"[^0-9A-Za-z_.-]", "_", str(value))

    def _map_coordinate_to_mask(self,prop_mask,coordinate,search_radius=20):
        # Map graph coordinate -> voxel seed -> nearest mask voxel -> connected mask component.
        seed_index = self._coordinate_to_index(coordinate, prop_mask.shape)
        seed_index = self._nearest_mask_index(prop_mask, seed_index, search_radius)
        if seed_index is None:
            return np.zeros_like(prop_mask, dtype=bool)

        labels, _ = ndimage.label(prop_mask)
        component_label = int(labels[seed_index])
        if component_label == 0:
            return np.zeros_like(prop_mask, dtype=bool)
        return labels == component_label

    @staticmethod
    def _coordinate_to_index(coordinate,shape):
        x = float(coordinate["x"])
        y = float(coordinate["y"])
        z = float(coordinate["z"])
        return (
            int(np.clip(round(x + shape[0] / 2), 0, shape[0] - 1)),
            int(np.clip(round(y + shape[1] / 2), 0, shape[1] - 1)),
            int(np.clip(round(abs(z)), 0, shape[2] - 1)),
        )

    @staticmethod
    def _nearest_mask_index(mask,seed_index,search_radius):
        if mask[seed_index]:
            return seed_index

        i, j, k = seed_index
        i0 = max(i - search_radius, 0)
        i1 = min(i + search_radius + 1, mask.shape[0])
        j0 = max(j - search_radius, 0)
        j1 = min(j + search_radius + 1, mask.shape[1])
        k0 = max(k - search_radius, 0)
        k1 = min(k + search_radius + 1, mask.shape[2])

        window = mask[i0:i1, j0:j1, k0:k1]
        hits = np.argwhere(window)
        if hits.size == 0:
            return None

        local_seed = np.array([i - i0, j - j0, k - k0])
        nearest = hits[np.linalg.norm(hits - local_seed, axis=1).argmin()]
        return tuple((nearest + np.array([i0, j0, k0])).astype(int))

    @staticmethod
    def _as_array(obj):
        if isinstance(obj, xarray.Dataset):
            if not obj.data_vars:
                raise ValueError("xarray Dataset has no data variables")
            first_var = next(iter(obj.data_vars))
            return obj[first_var].values
        if isinstance(obj, xarray.DataArray):
            return obj.values
        return np.asarray(obj)

    @staticmethod
    def _load_sample_object(sample_path):
        data = xarray.open_dataset(sample_path,engine='zarr')
        return data

def generate_images_for_graph(graph_path, samples_path=None):
    root = Path(__file__).parent.parent.absolute()
    if samples_path is None:
        setting_path = root.joinpath('settings.yaml')
        yaml_helper = YAMLHelper(setting_path)
        samples_path = yaml_helper.get_data('samples_path')
    samples_path = Path(samples_path)
    if not samples_path.is_absolute():
        samples_path = root / samples_path

    image_extractor = GraphImageExtractor(graph_path, samples_path)
    return image_extractor.extract_object_images()


def generate_images_for_all():
    root = Path(__file__).parent.parent.absolute()
    setting_path = root.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    graphs_path = yaml_helper.get_data('graphs_path')
    properties_graph_path = root / graphs_path / 'properties_graph'
    samples_path = yaml_helper.get_data('samples_path')
    samples_path = root / samples_path

    generated = []
    for selected_graph_path in sorted(properties_graph_path.glob("*.json")):
        generated.append(generate_images_for_graph(selected_graph_path, samples_path))
    print('done')
    return generated


if __name__ == "__main__":
    generate_images_for_all()
