# obj-to-obj relationship in 1 sample

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


if __name__ == '__main__':
    root = Path(__file__).parent.parent
    settings = root / 'settings.yaml'
    yaml_helper = YAMLHelper(settings)
    output_path = yaml_helper.get_data('output_path')
    all_seismic = list(Path(output_path).iterdir())
    select_one = all_seismic[0] # outputs/seismic__0531
    groups = devide_into_groups(find_zarr(select_one), select_one)
    print(f"groups: {len(groups)}")
    grouped = category_groups(groups)
    pair_groups = pair_group(grouped)
    pairs = permute_pair_group(pair_groups)
    print(f"pairs: {len(pairs)}")
