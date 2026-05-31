# obj-to-obj relationship in 1 sample
import json

# volume roles
volume_roles = {
    'label_mask': None, # integer/object labels such as closure ids
    'binary_mask': None, # 0/1 mask for one object class
    'continuous_volume': None, # measured value per voxel
    'surface_stack': None, # horizon/depth map surfaces
    'seismic_volume': None, # seismic amplitude/RFC volume
    'unknown': None, # fallback for unclassified zarr
}

# object types
object_types = {
    'closure': None,
    'fluid_oil': None,
    'fluid_gas': None,
    'fluid_brine': None,
    'hydrocarbon_closure': None,
    'trap': None,
    'reservoir': None,
    'seal': None,
    'fault': None,
    'salt': None,
    'lithology': None,
    'geologic_age': None,
    'horizon': None,
    'depth': None,
    'depth_map': None,
    'depth_map_gaps': None,
    'onlap_depth_map': None,
    'horizon_depth_map': None,
    'seismic_angle': None,
    'seismic_rfc_fullstack': None,
    'seismic_cumsum_fullstack': None,
    'unknown': None,
}

# relation types
relation_types = {
    'MASK_OVERLAP': None, # mask/label compared with mask/label
    'MASK_VOLUME_STATS': None, # mask/label used to sample continuous volume
    'VOLUME_COMPARISON': None, # continuous volume compared with continuous volume
    'SURFACE_COMPARISON': None, # surface stack compared with surface stack
}

# metric families
metric_families = {
    'spatial_overlap': None,
    'object_statistics': None,
    'seismic_response': None,
    'stratigraphic_position': None,
    'surface_offset': None,
}

# filename classification
volume_classification = {
    'all_closure_segments': {
        'role': 'label_mask',
        'object_type': 'closure',
    },
    'closure_segments_hc_voxelcount': {
        'role': 'label_mask',
        'object_type': 'closure',
    },
    'closures/oil': {
        'role': 'binary_mask',
        'object_type': 'fluid_oil',
    },
    'closures/gas': {
        'role': 'binary_mask',
        'object_type': 'fluid_gas',
    },
    'closures/brine': {
        'role': 'binary_mask',
        'object_type': 'fluid_brine',
    },
    'closures/hc_labels': {
        'role': 'label_mask',
        'object_type': 'hydrocarbon_closure',
    },
    'trap_label': {
        'role': 'binary_mask',
        'object_type': 'trap',
    },
    'reservoir_label': {
        'role': 'binary_mask',
        'object_type': 'reservoir',
    },
    'sealed_label': {
        'role': 'binary_mask',
        'object_type': 'seal',
    },
    'geology/geologic_age': {
        'role': 'continuous_volume',
        'object_type': 'geologic_age',
    },
    'geology/faulted_lithology': {
        'role': 'label_mask',
        'object_type': 'lithology',
    },
    'depth_maps': {
        'role': 'surface_stack',
        'object_type': 'depth_map',
    },
    'depth_maps_gaps': {
        'role': 'surface_stack',
        'object_type': 'depth_map_gaps',
    },
    'depth_maps_onlaps': {
        'role': 'surface_stack',
        'object_type': 'onlap_depth_map',
    },
    'horizons/depth_maps': {
        'role': 'surface_stack',
        'object_type': 'horizon_depth_map',
    },
    'seismic/': {
        'role': 'seismic_volume',
        'object_type': 'seismic_angle',
    },
    'seismicCubes_RFC_fullstack': {
        'role': 'seismic_volume',
        'object_type': 'seismic_rfc_fullstack',
    },
    'seismicCubes_cumsum_fullstack': {
        'role': 'seismic_volume',
        'object_type': 'seismic_cumsum_fullstack',
    },
}


from yaml_helper import YAMLHelper
from pathlib import Path
from itertools import combinations, product
import numpy as np
from numcodecs import blosc

def find_zarr(path):
    return Path(path).rglob('*.zarr')

def devide_into_groups(path_list, sample_path):
    sample_path = Path(sample_path)
    volume_class_keys = sorted(volume_classification.keys(), key=len, reverse=True)
    groups = list()
    for path in path_list:
        name = Path(path).relative_to(sample_path).as_posix().replace('.zarr', '')
        for keys in volume_class_keys:
            if keys in name:
                groups.append({
                    "path": f"{name}.zarr",
                    "role": volume_classification[keys]['role'],
                    "object_type": volume_classification[keys]['object_type'],
                })
                break
    return groups

def category_groups(groups:list):
    label_volume = []
    binary_volume = []
    fluid_masks = []
    geology_volumes = []
    depth = []
    seismic_volumes = []
    fault_volumes = []
    salt_volumes = []

    for group in groups:
        role = group['role']
        object_type = group['object_type']
        path = group['path']

        # label volume
        if role == 'label_mask':
            label_volume.append(group)
        # binary volume
        if role =='binary_mask':
            binary_volume.append(group)
         # fluid
        if 'fluid' in object_type:
             fluid_masks.append(group)
        #geology
        if 'geology' in path:
            geology_volumes.append(group)
        # depth
        if 'depth' in path:
            depth.append(group)
        # seismic
        if 'seismic' in path:
            seismic_volumes.append(group)
        # fault
        if 'fault' in path:
            fault_volumes.append(group)
        # salt
        if 'salt' in path:
            salt_volumes.append(group)
    return label_volume,binary_volume,fluid_masks,geology_volumes,depth,seismic_volumes,fault_volumes,salt_volumes

def pair_group(category_lists):
    label_volume, binary_volume, fluid_masks, geology_volumes, depth, seismic_volumes, fault_volumes, salt_volumes = category_lists

    mask_like = label_volume + binary_volume + fault_volumes + salt_volumes
    value_like = geology_volumes + seismic_volumes

    mask_overlap = [mask_like,mask_like]
    mask_volume_stats = [mask_like,value_like]
    volume_comparison = [seismic_volumes,seismic_volumes]
    surface_comparison = [depth,depth]

    return {
        'MASK_OVERLAP': mask_overlap,
        'MASK_VOLUME_STATS': mask_volume_stats,
        'VOLUME_COMPARISON': volume_comparison,
        'SURFACE_COMPARISON': surface_comparison,
    }

def permute_pair_group(pair_groups):
    pairs = []
    for relation_type, groups in pair_groups.items():
        left_group, right_group = groups
        if left_group is right_group:
            iterator = combinations(left_group, 2)
        else:
            iterator = product(left_group, right_group)

        for source, target in iterator:
            if source['path'] == target['path']:
                continue
            pairs.append({
                'source': source,
                'target': target,
                'relation_type': relation_type,
            })
    return pairs


class HighLevelTracer(object):
    def __init__(self,path):
        self.path = Path(path)
        self.sample_name = self.path.stem
        self._array_cache = {}

        setting_path = Path(__file__).parent.parent.joinpath("settings.yaml")
        yaml_helper = YAMLHelper(setting_path)
        traces_path = yaml_helper.get_data('traces_path')
        self.traces_path = Path(traces_path)
        Path(traces_path).mkdir(parents=True, exist_ok=True)

        self.groups = devide_into_groups(find_zarr(self.path), self.path)
        self.category_lists = category_groups(self.groups)
        self.pair_groups = pair_group(self.category_lists)
        self.pairs = permute_pair_group(self.pair_groups)
        self.properties_data = self.get_relation_data()

    def save_to_json(self):
        self.trace_sample_path = Path(self.traces_path.joinpath(f"{self.sample_name}_array_relations.json"))
        print(f"extracting array relations from sample: {self.sample_name}")
        self.trace_sample_path.write_text(json.dumps(self.properties_data, indent=2, default=self._json_default))

    def get_properties_data(self):
        return self.properties_data

    def get_relation_data(self):
        relations = []
        print(f"groups: {len(self.groups)}")
        print(f"pairs: {len(self.pairs)}")
        for index, pair in enumerate(self.pairs, start=1):
            if index == 1 or index % 25 == 0 or index == len(self.pairs):
                print(f"measuring relation {index}/{len(self.pairs)}", flush=True)
            relation = self.measure_pair(pair)
            if relation["status"] == "ok":
                relations.append(relation)

        return {
            "sample": self.sample_name,
            "sample_path": self.path.as_posix(),
            "volume_count": len(self.groups),
            "candidate_relation_count": len(self.pairs),
            "relation_count": len(relations),
            "catalog": self.groups,
            "relations": relations,
        }

    def measure_pair(self, pair):
        source = pair["source"]
        target = pair["target"]
        relation_type = pair["relation_type"]
        relation = {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "metric_family": self.metric_family_for_pair(relation_type, source, target),
        }

        try:
            source_array = self.load_array(source["path"])
            target_array = self.load_array(target["path"])
            relation["source_shape"] = list(source_array.shape)
            relation["target_shape"] = list(target_array.shape)

            if source_array.shape != target_array.shape:
                relation["status"] = "shape_mismatch"
                relation["metrics"] = {}
                return relation

            if relation_type == "MASK_OVERLAP":
                relation["metrics"] = self.mask_overlap_metrics(source_array, target_array)
            elif relation_type == "MASK_VOLUME_STATS":
                relation["metrics"] = self.mask_volume_stats(source_array, target_array)
            elif relation_type == "VOLUME_COMPARISON":
                relation["metrics"] = self.volume_comparison_metrics(source_array, target_array)
            elif relation_type == "SURFACE_COMPARISON":
                relation["metrics"] = self.surface_comparison_metrics(source_array, target_array)
            else:
                relation["metrics"] = {}

            relation["status"] = "ok"
            return relation
        except Exception as exc:
            relation["status"] = "error"
            relation["error"] = f"{exc.__class__.__name__}: {exc}"
            relation["metrics"] = {}
            return relation

    def load_array(self, relative_path):
        relative_path = Path(relative_path)
        if relative_path.as_posix() not in self._array_cache:
            zarr_path = self.path / relative_path
            self._array_cache[relative_path.as_posix()] = self.read_zarr_v3_array(zarr_path)
        return self._array_cache[relative_path.as_posix()]

    def read_zarr_v3_array(self, zarr_path):
        array_path = self.find_array_path(Path(zarr_path))
        metadata = json.loads(array_path.joinpath("zarr.json").read_text())
        shape = tuple(metadata["shape"])
        chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
        dtype = np.dtype(metadata["data_type"]).newbyteorder("<")
        fill_value = self.parse_fill_value(metadata.get("fill_value"), dtype)
        array = np.full(shape, fill_value, dtype=dtype)
        chunk_root = array_path / "c"

        if not chunk_root.exists():
            return array

        for chunk_path in chunk_root.rglob("*"):
            if not chunk_path.is_file():
                continue
            chunk_index = tuple(int(part) for part in chunk_path.relative_to(chunk_root).parts)
            chunk_data = self.read_chunk(chunk_path, chunk_shape, dtype)
            slices = []
            chunk_slices = []
            for axis, chunk_number in enumerate(chunk_index):
                start = chunk_number * chunk_shape[axis]
                stop = min(start + chunk_shape[axis], shape[axis])
                width = stop - start
                slices.append(slice(start, stop))
                chunk_slices.append(slice(0, width))
            array[tuple(slices)] = chunk_data[tuple(chunk_slices)]

        return array

    @staticmethod
    def find_array_path(zarr_path):
        metadata_path = zarr_path / "zarr.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing zarr metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text())
        if metadata.get("node_type") == "array":
            return zarr_path

        for child in zarr_path.iterdir():
            child_metadata = child / "zarr.json"
            if child_metadata.exists():
                child_data = json.loads(child_metadata.read_text())
                if child_data.get("node_type") == "array":
                    return child

        raise ValueError(f"zarr group has no array child: {zarr_path}")

    @staticmethod
    def parse_fill_value(fill_value, dtype):
        if fill_value is None:
            return dtype.type(0)
        if isinstance(fill_value, str) and fill_value.lower() == "nan":
            return np.nan
        return dtype.type(fill_value)

    @staticmethod
    def read_chunk(chunk_path, chunk_shape, dtype):
        raw = chunk_path.read_bytes()
        try:
            raw = blosc.decompress(raw)
        except Exception:
            pass
        return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)

    @staticmethod
    def finite_values(array):
        array = np.asarray(array)
        return array[np.isfinite(array)]

    @staticmethod
    def mask_from_array(array):
        array = np.asarray(array)
        return np.isfinite(array) & (array != 0)

    def mask_overlap_metrics(self, source_array, target_array):
        source_mask = self.mask_from_array(source_array)
        target_mask = self.mask_from_array(target_array)
        overlap = source_mask & target_mask
        union = source_mask | target_mask

        source_voxels = int(source_mask.sum())
        target_voxels = int(target_mask.sum())
        overlap_voxels = int(overlap.sum())
        union_voxels = int(union.sum())

        return {
            "source_voxels": source_voxels,
            "target_voxels": target_voxels,
            "overlap_voxels": overlap_voxels,
            "union_voxels": union_voxels,
            "overlap_fraction_source": self.safe_divide(overlap_voxels, source_voxels),
            "overlap_fraction_target": self.safe_divide(overlap_voxels, target_voxels),
            "iou": self.safe_divide(overlap_voxels, union_voxels),
        }

    def mask_volume_stats(self, mask_array, volume_array):
        mask = self.mask_from_array(mask_array)
        values = np.asarray(volume_array)[mask]
        values = self.finite_values(values)

        if values.size == 0:
            return {
                "sampled_voxels": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p25": None,
                "median": None,
                "p75": None,
                "nonzero_fraction": None,
            }

        return {
            "sampled_voxels": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "nonzero_fraction": self.safe_divide(int(np.count_nonzero(values)), int(values.size)),
        }

    def volume_comparison_metrics(self, source_array, target_array):
        source_values = np.asarray(source_array)
        target_values = np.asarray(target_array)
        valid = np.isfinite(source_values) & np.isfinite(target_values)
        source_values = source_values[valid]
        target_values = target_values[valid]

        if source_values.size == 0:
            return {
                "compared_voxels": 0,
                "mean_difference": None,
                "std_difference": None,
                "min_difference": None,
                "max_difference": None,
                "rmse": None,
                "correlation": None,
                "source_mean": None,
                "target_mean": None,
                "source_std": None,
                "target_std": None,
            }

        difference = target_values - source_values
        source_std = float(np.std(source_values))
        target_std = float(np.std(target_values))

        return {
            "compared_voxels": int(source_values.size),
            "mean_difference": float(np.mean(difference)),
            "std_difference": float(np.std(difference)),
            "min_difference": float(np.min(difference)),
            "max_difference": float(np.max(difference)),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "correlation": self.correlation(source_values, target_values, source_std, target_std),
            "source_mean": float(np.mean(source_values)),
            "target_mean": float(np.mean(target_values)),
            "source_std": source_std,
            "target_std": target_std,
        }

    def surface_comparison_metrics(self, source_array, target_array):
        source_values = np.asarray(source_array)
        target_values = np.asarray(target_array)
        source_valid = np.isfinite(source_values)
        target_valid = np.isfinite(target_values)
        valid = source_valid & target_valid

        if valid.sum() == 0:
            return {
                "compared_cells": 0,
                "mean_offset": None,
                "std_offset": None,
                "min_offset": None,
                "max_offset": None,
                "missing_fraction_source": self.safe_divide(source_values.size - int(source_valid.sum()), source_values.size),
                "missing_fraction_target": self.safe_divide(target_values.size - int(target_valid.sum()), target_values.size),
            }

        offset = target_values[valid] - source_values[valid]
        return {
            "compared_cells": int(valid.sum()),
            "mean_offset": float(np.mean(offset)),
            "std_offset": float(np.std(offset)),
            "min_offset": float(np.min(offset)),
            "max_offset": float(np.max(offset)),
            "missing_fraction_source": self.safe_divide(source_values.size - int(source_valid.sum()), source_values.size),
            "missing_fraction_target": self.safe_divide(target_values.size - int(target_valid.sum()), target_values.size),
        }

    @staticmethod
    def metric_family_for_pair(relation_type, source, target):
        if relation_type == "MASK_OVERLAP":
            return "spatial_overlap"
        if relation_type == "MASK_VOLUME_STATS":
            target_type = target["object_type"]
            if "seismic" in target_type:
                return "seismic_response"
            if target_type in {"geologic_age", "lithology"}:
                return "stratigraphic_position"
            return "object_statistics"
        if relation_type == "VOLUME_COMPARISON":
            return "seismic_response"
        if relation_type == "SURFACE_COMPARISON":
            return "surface_offset"
        return "object_statistics"

    @staticmethod
    def correlation(source_values, target_values, source_std, target_std):
        if source_values.size < 2 or source_std == 0 or target_std == 0:
            return None
        source_centered = source_values - np.mean(source_values)
        target_centered = target_values - np.mean(target_values)
        numerator = np.mean(source_centered * target_centered)
        denominator = source_std * target_std
        if denominator == 0:
            return None
        return float(numerator / denominator)

    @staticmethod
    def safe_divide(numerator, denominator):
        if denominator == 0:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _json_default(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return value.as_posix()
        return str(value)


if __name__ == '__main__':
    root = Path(__file__).parent.parent
    settings = root / 'settings.yaml'
    yaml_helper = YAMLHelper(settings)
    output_path = yaml_helper.get_data('output_path')
    all_seismic = sorted(Path(output_path).glob("seismic__*/"))
    select_one = all_seismic[0] # outputs/seismic__0531
    high_tracker = HighLevelTracer(select_one)
    high_tracker.save_to_json()
