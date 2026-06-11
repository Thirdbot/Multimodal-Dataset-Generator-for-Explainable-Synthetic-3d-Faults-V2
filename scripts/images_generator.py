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

GRAPH_OBJECT_TYPES = [  "closure",
                 "fault",
                ]

OBJECT_ZARR_CANDIDATES = {
      "fault": [
          "fault_segments_*.zarr",              # best fault object mask
          "fault_intersection_segments_*.zarr", # fault intersections
      ],
      "closure": [
          "closures/oil.zarr",
          "closures/gas.zarr",
          "closures/brine.zarr",
          "closures/hc_labels.zarr",
          "all_closure_segments_*.zarr",
          "hc_closures_augmented_*.zarr",
      ],
      "onlap": [
          "onlap_segments_*.zarr",
          "depth_maps_onlaps.zarr",
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
        # Future hook: extract 3D object arrays, create 2D object crops/masks,
        # and write view-specific graph coordinates for VLM dataset rows.
        sample_name = self.build_folder_path.stem
        basic_objects = self._extract_basic()

        sample_object_properties = self._get_object_nodes()

        if not basic_objects:
            return []

        base_volume = basic_objects[0]

        selected_objects = {}
        for object_type,patterns in OBJECT_ZARR_CANDIDATES.items():
            self.image2d_path.joinpath(sample_name,object_type).mkdir(parents=True, exist_ok=True) # create folder for each object type and sample for 2d images
            self.image3d_path.joinpath(sample_name,object_type).mkdir(parents=True, exist_ok=True) # create folder for each object type and sample for 3d objects
            type_sliced_objects = []
            for pattern in patterns:
                for match_sample in self.build_folder_path.glob(pattern):
                    if object_type == "fault" and self._is_fault_property_volume(match_sample):
                        continue
                    # load 1 object and slice it
                    property_object = self._load_sample_object(match_sample)
                    slices = self._object_slicing(base_volume,property_object)
                    type_sliced_objects.extend(slices)
            selected_type_slices = self._select_average_mask_slices(
                type_sliced_objects,
                object_type,
                sample_object_properties,
            )
            self.make_image(sample_name, object_type, selected_type_slices)
            selected_objects.update(selected_type_slices)
        return selected_objects

    @staticmethod
    def _is_fault_property_volume(path):
        name = Path(path).name
        return "fault_segments_throw_" in name or "fault_segments_azimuth_" in name

    def _extract_basic(self):
        # Extract original 3D object arrays from Synthoseis build folder
        all_patterns = self._find_objects(OBJECT_ZARR_BASIC)
        store_objs = []
        for pattern in all_patterns:
            for match_sample in self.build_folder_path.glob(pattern):
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

    def _object_slicing(self,basic_obj,prop_obj):
        # Slice 3D object arrays and create 2D object crops/masks
        basic_array = self._as_array(basic_obj)
        prop_array = self._as_array(prop_obj)
        sliced_objects = []

        if basic_array.ndim > 3:
            basic_array = np.squeeze(basic_array)
        if prop_array.ndim > 3:
            prop_array = np.squeeze(prop_array)
        if basic_array.ndim != 3 or prop_array.ndim != 3:
            raise ValueError("basic_obj and prop_obj must resolve to 3D arrays")

        common_shape = tuple(min(a, b) for a, b in zip(basic_array.shape, prop_array.shape))
        basic_array = basic_array[:common_shape[0], :common_shape[1], :common_shape[2]]
        prop_array = prop_array[:common_shape[0], :common_shape[1], :common_shape[2]]

        prop_mask = np.nan_to_num(prop_array, nan=0.0) != 0


        if prop_mask.any():
            inline_index = int(prop_mask.sum(axis=(1, 2)).argmax())
            crossline_index = int(prop_mask.sum(axis=(0, 2)).argmax())
            timeslice_index = int(prop_mask.sum(axis=(0, 1)).argmax())
            sliced = {
                "basic": {
                    "inline": basic_array[inline_index, :, :],
                    "crossline": basic_array[:, crossline_index, :],
                    "timeslice": basic_array[:, :, timeslice_index],
                },
                "mask": {
                    "inline": prop_mask[inline_index, :, :],
                    "crossline": prop_mask[:, crossline_index, :],
                    "timeslice": prop_mask[:, :, timeslice_index],
                },
            }
            sliced_objects.append(sliced)
        return sliced_objects

    def _select_average_mask_slices(self,slices,object_type,object_properties):
        # Group slice candidates by object id and prefer candidates close to expected object size.
        candidates = [
            candidate
            for candidate in slices
            if self._slice_mask_area(candidate) > 0
        ]
        if not candidates:
            return {}

        return {
            object_type: max(
                candidates,
                key=self._slice_mask_area,
            )
        }

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
            for view in ("inline", "crossline", "timeslice"):
                basic_slice = self._normalize_image(sliced["basic"][view])
                mask_slice = np.asarray(sliced["mask"][view], dtype=bool)
                self._save_png(object_folder / f"{view}.png", basic_slice, cmap="gray")
                self._save_png(object_folder / f"{view}_mask.png", mask_slice.astype(float), cmap="Reds")
                self._save_overlay(object_folder / f"{view}_overlay.png", basic_slice, mask_slice)

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
    def _save_overlay(path,basic_slice,mask_slice):
        overlay = np.dstack([basic_slice, basic_slice, basic_slice])
        overlay[mask_slice, 0] = 1.0
        overlay[mask_slice, 1] *= 0.25
        overlay[mask_slice, 2] *= 0.25
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

if __name__ == "__main__":
    root = Path(__file__).parent.parent.absolute()
    setting_path = root.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)
    graphs_path = yaml_helper.get_data('graphs_path')
    properties_graph_path = root / graphs_path / 'properties_graph'
    samples_path = yaml_helper.get_data('samples_path')
    samples_path = root / samples_path
    graph_list =list(properties_graph_path.iterdir())
    # selected_graph_path = graph_list[0]
    for selected_graph_path in graph_list:
        image_extractor = GraphImageExtractor(selected_graph_path,samples_path)
        image_extractor.extract_object_images()
    print('done')
