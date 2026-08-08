"""Build high-level recipes and low-level Synthoseis config JSON files.

This script is the first stage of the local generation workflow:
settings.yaml -> category controls -> build_configs/*.json + recipes/*.yaml.
"""

from pathlib import Path
import random
import uuid
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.logger_color import logger
from scripts.common.yaml_helper import YAMLHelper
from typing import Literal
import json

# low-level controls
# Structural
structural = {
    'project': None, # rock name
    'project_folder': None, # output folder
    'work_folder': None,  # working folder .default is /tmp
    'cube_shape': None, # 3 dimensions x,y,z
    'incident_angles': None, # angles
    'digi': None, # Vertical sampling
}

# structural properties
structural_properties = {
    'infill_factor': None, # horizon sampling
    'initial_layer_stdev': None, # horizonal roughness
    'thickness_min': None, # min layer thickness
    'thickness_max': None, # max layer thickness
    'seabed_min_depth': None, # seabed top layers
    'dip_factor_max': None, # layer dip amount
    'pad_samples': None,  # filling Nan in vertical
}

# fault controls
fault_controls = {
    'min_number_faults': None, # minimum for faults
    'max_number_faults': None, # maximum for faults
}
# style and displacement are randoms

# Geo_body controls
geo_body_controls = {
    'sand_layer_thickness': None, # sand
    'sand_layer_fraction': {
        'min':None,
        'max':None,
    }, # minimum sand friction
    'variable_shale_ng': False, # heterogeneity
    'basin_floor_fans': None, # geomorphology
    'include_channels': False, # inactive
    'include_salt': None, # salt body complexity
    'partial_voxels': False, # mixing layers (true for realism; false for speed)
}

# trap controls
trap_controls = {
    'max_column_height': None, # maximum trapped-fluid height
    'closure_types': None, # simple / faulted / onlap all combine
    'min_closure_voxels_simple': None, # minimum simple closure size
    'min_closure_voxels_faulted': None, # minimum faulted closure size
    'min_closure_voxels_onlap': None, # minimum onlap closure size
}

# seismic signal controls (realism vs. legibility)
seismic_signal_controls = {
    # Signal:Noise in dB as a triangular distribution [min, mode, max] -> per-scene VARIETY
    # (noisy .. clean), realistic centre. Higher dB = cleaner. ~6 dB is a realistically hard
    # section, ~18 dB is clean; mode 12 keeps most sections legible while training on a spread.
    # (Masks/attributes come from the geometry, so noise only varies the IMAGE, not the labels.)
    'signal_to_noise_ratio_db': [6.0, 12.0, 18.0],
    'bandwidth_low':  [3.0, 6.0],    # Hz, low-cut range (per-scene uniform pick)
    'bandwidth_high': [20.0, 35.0],  # Hz, high-cut range -> vertical resolution / wavelet
    'bandwidth_ord':  4,             # Butterworth filter order
    'broadband_qc_volume': False,    # false is for speed
}

# Quality check output such as images, logs , in-memory storage

quality_check_output = {
    'extra_qc_plots': None, # false for batch generation
    'verbose': None, # CLI logging while dev
    'model_qc_volumes': None,  # false for low storage
    'model_store_in_memory': None, # false; large volume can eat rams
    'cleanup_intermediates': None, # no clean-up for labels
}

# Holds the mutable Synthoseis parameter template and category-specific presets.
class CategoricalParameter:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set(self,categories:Literal["boring","fault_only","fault_complex","salt_only","salt_fault_mixed","onlap","depositional","full_mixed"],**extra):
        """
        Selecting categories methods for sample
        :param categories:
        :param extra:
        :return: None
        """
        method = getattr(self, categories)
        # print(f"category selected: {categories}")
        method(**extra)


    def initialize(self,project,
                   project_folder,
                   work_folder,
                   cube_shape,
                   incident_angles,
                   digi,
                   infill_factor,
                   initial_layer_stdev,
                   thickness_min,
                   thickness_max,
                   seabed_min_depth,
                   dip_factor_max,
                   pad_samples,
                   sand_layer_thickness,
                   sand_layer_fraction,
                   include_channels,
                   bandwidth_ord,
                   broadband_qc_volume,
                   extra_qc_plots,
                   verbose,
                   model_qc_volumes,
                   model_store_in_memory,
                   cleanup_intermediates,
                   max_column_height,
                   min_closure_voxels_simple,
                   min_closure_voxels_faulted,
                   min_closure_voxels_onlap,
                   signal_to_noise_ratio_db,
                   bandwidth_low,
                   bandwidth_high,
                   closure_types,
                   include_salt,
                   basin_floor_fans,
                   min_number_faults,
                   max_number_faults
                   ):

        self.kwargs["project"] = project
        self.kwargs["project_folder"] = project_folder
        self.kwargs["work_folder"] = work_folder
        self.kwargs["cube_shape"] = cube_shape
        self.kwargs["incident_angles"] = incident_angles
        self.kwargs["digi"] = digi
        self.kwargs["infill_factor"] = infill_factor
        self.kwargs["initial_layer_stdev"] = initial_layer_stdev
        self.kwargs["thickness_min"] = thickness_min
        self.kwargs["thickness_max"] = thickness_max
        self.kwargs["seabed_min_depth"] = seabed_min_depth
        self.kwargs["dip_factor_max"] = dip_factor_max
        self.kwargs["pad_samples"] = pad_samples
        self.kwargs["sand_layer_thickness"] = sand_layer_thickness
        self.kwargs["sand_layer_fraction"] = sand_layer_fraction

        self.kwargs["include_channels"] = include_channels
        self.kwargs["bandwidth_ord"] = bandwidth_ord
        self.kwargs["broadband_qc_volume"] = broadband_qc_volume
        self.kwargs["extra_qc_plots"] = extra_qc_plots
        self.kwargs["verbose"] = verbose
        self.kwargs["model_qc_volumes"] = model_qc_volumes
        self.kwargs["model_store_in_memory"] = model_store_in_memory
        self.kwargs["cleanup_intermediates"] = cleanup_intermediates

        self.kwargs["max_column_height"] = max_column_height
        self.kwargs["min_closure_voxels_simple"] = min_closure_voxels_simple
        self.kwargs["min_closure_voxels_faulted"] = min_closure_voxels_faulted
        self.kwargs["min_closure_voxels_onlap"] = min_closure_voxels_onlap
        self.kwargs["signal_to_noise_ratio_db"] = signal_to_noise_ratio_db
        self.kwargs["bandwidth_low"] = bandwidth_low
        self.kwargs["bandwidth_high"] = bandwidth_high

        self.kwargs["closure_types"] = closure_types
        self.kwargs["include_salt"] = include_salt
        self.kwargs["basin_floor_fans"] = basin_floor_fans
        self.kwargs["min_number_faults"] = min_number_faults
        self.kwargs["max_number_faults"] = max_number_faults

        # None-guard: fail fast if any parameter was left unset before serialization.
        self._check_value()

    def _check_value(self):
        all_none = {k for k,v in self.kwargs.items() if v is None}
        if all_none:
            raise Exception(f"{all_none} are None")
        else:
            print("config checking passed!!")

    def _expose(self):
        """
        print out whole parameter for 1 sample
        :return: Dictionary
        """
        return dict(self.kwargs)

    def boring(self):
        """
        basically, have nothing except stratigraphy
        :return: None
        """
        self.kwargs["include_salt"] = False
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] = 0
        self.kwargs["max_number_faults"] = 0
        self.kwargs["closure_types"] = ["simple"]

    def fault_only(self,f_min=1,f_max=9):
        """
        basically, only fault
        :param f_min:
        :param f_max:
        :return: None
        """
        self.kwargs["include_salt"] = False
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] = f_min
        self.kwargs["max_number_faults"] = f_max
        self.kwargs["closure_types"] = ["faulted"]

    def fault_complex(self,f_min=10,f_max=20):
        """
        basically, fault but more complex
        :param f_min:
        :param f_max:
        :return: None
        """
        self.kwargs["include_salt"] = False
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] =  f_min
        self.kwargs["max_number_faults"] = f_max
        self.kwargs["closure_types"] = ["faulted"]

    def salt_only(self):
        """
        basically, only salt
        :return: None
        """
        self.kwargs["include_salt"] = True
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] = 0
        self.kwargs["max_number_faults"] = 0
        self.kwargs["closure_types"] = ["simple"]

    def salt_fault_mixed(self,f_min=1,f_max=4):
        """
        basically, only salt mixed with faults
        :param f_min:
        :param f_max:
        :return: None
        """
        self.kwargs["include_salt"] = True
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] = f_min
        self.kwargs["max_number_faults"] = f_max
        self.kwargs["closure_types"] = ["faulted","simple"]

    def onlap(self):
        """
        basically, stratigraphy
        :return: None
        """
        self.kwargs["include_salt"] = False
        self.kwargs["basin_floor_fans"] = False
        self.kwargs["min_number_faults"] = 0
        self.kwargs["max_number_faults"] = 0
        self.kwargs["closure_types"] = ["onlap"]

    def depositional(self):
        """
        basically, deposition of GeoBody
        :return: None
        """
        self.kwargs["include_salt"] = False
        self.kwargs["basin_floor_fans"] = True
        self.kwargs["min_number_faults"] = 0
        self.kwargs["max_number_faults"] = 0
        self.kwargs["closure_types"] = ["simple", "onlap"]

    def full_mixed(self,f_min=2,f_max=6):
        """
        basically, mixed all
        :return: None
        """
        self.kwargs["include_salt"] = True
        self.kwargs["basin_floor_fans"] = True
        self.kwargs["min_number_faults"] = f_min
        self.kwargs["max_number_faults"] = f_max
        self.kwargs["closure_types"] = ["simple", "faulted", "onlap"]

# Turns population/ratio intent into concrete recipe files and per-sample configs.
class SampleControl:
    def __init__(self,categorical_parameter, **kwargs):
        self.kwargs = kwargs
        self.categorical_parameter = categorical_parameter
        self.population_amount,self.ratio_configs = self._manage_population()


    def _run_category(self,category):
        """
        set sample name to build's id and its category
        :param name:
        :param category:
        :return: parameters of Synthoseis build_configs
        """
        self.categorical_parameter.set(category)
        return self.categorical_parameter._expose()

    def _manage_population(self):
        """
        ratio sampling of each type
        :return: ratio of each types
        """
        max_ratio = 1.0
        min_ratio = 0.0

        population = self.kwargs["sample_population"]
        ratio_per_types = self.kwargs["ratio_per_types"]
        types = self.kwargs["sample_types"]
        types_ratio = {}
        if population is None or types is None:
            raise Exception("population or types is None")

        # by default, for all samples, we only distribute each type equally
        ratio = 1.0 / len(types)
        # distribute sample by same ratio
        for t in types:
            types_ratio[t] = ratio

        # case that there are ratio_per_types, use ratio per types
        if ratio_per_types:
            common_keys = types_ratio.keys() & ratio_per_types.keys()
            intersection = {k: ratio_per_types[k] for k in common_keys}
            combine_types = types_ratio | intersection # combine types (result in the same types as types_ratio or replaced by ratio_per_types)

            for rt in intersection:
                min_ratio += intersection[rt]
            left_ratio = max_ratio - min_ratio # can be 0 if all are ratio

            # ratio not exceeding max
            if left_ratio < 0:
                raise Exception(f"sum of all ratio is greater than {max_ratio}")

            left_types = dict(combine_types.items() - intersection.items())

            # no left for distribution
            if len(left_types) <= 0:
                distribute_ratio = 0.0
            else:
                distribute_ratio = left_ratio / len(left_types)

            for t in left_types:
                left_types[t] = distribute_ratio
            final_types = combine_types | left_types
            return population ,final_types

        return population ,types_ratio

    def populate(self,recipes_path,build_configs_path):
        recipes_path = Path(recipes_path)
        build_configs_path = Path(build_configs_path)

        # Derive the next index from existing recipe_*.yaml files so a deleted
        # recipe or a stray non-recipe file cannot cause an index collision.
        existing_indices = []
        for recipe_file in recipes_path.glob("recipe_*.yaml"):
            suffix = recipe_file.stem[len("recipe_"):]
            if suffix.isdigit():
                existing_indices.append(int(suffix))
        run_number = max(existing_indices, default=-1) + 1

        # Weighted-random category per sample. The old `int(population * ratio)` floored every
        # ratio to 0 for a small population (0.22*4 -> 0), so the ratios were ignored and the
        # leftover loop just filled the first N categories in dict order -- only 4 ever appeared.
        # Sampling each sample's category by its ratio converges to the target mix across the
        # many small recipes the driver populates, and every category can appear.
        categories = list(self.ratio_configs.keys())
        weights = list(self.ratio_configs.values())
        counts = {category: 0 for category in categories}
        for _ in range(self.population_amount):
            counts[random.choices(categories, weights=weights, k=1)[0]] += 1
        # saved recipe config
        recipe_name = f"recipe_{run_number}"
        recipe_config = {
            'population': {
                'amount':self.population_amount},
            'category_ratio': self.ratio_configs,
            'category_counts': counts,
            'category_order': list(counts.keys())
        }

        recipe_name_path = recipes_path / f"{recipe_name}.yaml"
        sample_names = []

        logger.info(f"[Populating] from file {recipe_name} At {recipe_name_path}")
        for category,amount in counts.items():
            # loop in amount
            for _ in range(amount):
                name = f"{category}_{uuid.uuid4().hex}"
                build_config_path = build_configs_path / f"{name}.json"
                config = self._run_category(category) # generate sample with unique id with its type
                with open(build_config_path,'w') as f:
                    json.dump(config,f,indent=2)
                sample_names.append(name)
            logger.debug(f"[Populating]: Category {category} Amount {amount}")

        recipe_config["population"].update({
            "samples": sample_names,
        })
        with open(recipe_name_path, 'w') as f:
            yaml.dump(recipe_config, f)


# low-level controls
low_level_controls = structural |structural_properties |fault_controls |geo_body_controls |trap_controls |seismic_signal_controls |quality_check_output

# high-level controls
# rule is simple, assign types that will be in dataset in sample_types
# set ratio for each types, what that not mentioned will be ratio
# (you can set some type to not exists by set 0.0)

high_level_controls = {
    'sample_population': 4, # amount of sample that will be populated
    # each sample that is randomly created or mixed category will be ratio
    # all-faulted has different fault-line that it will be ratio, salt-fault will be ratio
    'sample_types': [
                     "fault_only",     # 1-9 faults + faulted closures: clean, countable masks
                     "fault_complex",  # 10-20 faults: structural density diversity
                     "boring",         # featureless -> empty-mask NEGATIVES (no fault to segment)
                     # "salt_only",
                     # "salt_fault_mixed",
                     # "onlap",
                     # "depositional",
                     # "full_mixed",
    ], # for dataset generations each generation will be ratio in same amount
    # Two-class dataset: FAULT (class 1) + CLOSURE (class 2) only. include_salt=False and
    # closure_types=["faulted"] in both fault recipes -> no salt/onlap masks ever appear.
    # `boring` contributes ~25% negatives (no-fault scenes); absent classes also surface as
    # natural empty-mask "nothing" rows (see NaturalTransform ABSENCE_TEMPLATES). Sums to 1.0.
    'ratio_per_types':{
        "fault_only":       0.45,   # lean here: cleaner, well-separated, countable fault masks
        "fault_complex":    0.30,   # denser faulting for coverage
        "boring":           0.25,   # negatives: object-absent -> empty mask, no <SEG>
        # "salt_fault_mixed": 0.0,
        # "salt_only":        0.0,
        # "onlap":            0.0,
        # "depositional":     0.0,
        # "full_mixed":       0.0,
    }
}


if __name__ == "__main__":
    # CLI entry point: load settings, initialize the parameter template, then
    # populate a recipe plus the referenced JSON build configs.
    setting_path = ROOT.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)

    recipes_path = yaml_helper.get_data('recipes_path' )# store all high-level configuration (type-sample)
    build_configs_path = yaml_helper.get_data('build_configs_path') # store all low-level configuration (samples)

    samples_path = yaml_helper.get_data('samples_path') # store all generated samples
    temp_builds_path = yaml_helper.get_data('temp_builds_path') # store as tmp

    ## initialize values
    # NOTE: cube_shape is not read from yaml here; it is set from the resolution
    # preset (LOW/MEDIUM/HIGH) below.
    initial_layer_stdev = yaml_helper.get_data('initial_layer_stdev')
    incident_angles = yaml_helper.get_data('incident_angles')
    digi = yaml_helper.get_data('digi')
    infill_factor = yaml_helper.get_data('infill_factor')
    thickness_min = yaml_helper.get_data('thickness_min')
    thickness_max = yaml_helper.get_data('thickness_max')
    seabed_min_depth = yaml_helper.get_data('seabed_min_depth')
    dip_factor_max = yaml_helper.get_data('dip_factor_max')
    pad_samples = yaml_helper.get_data('pad_samples')
    sand_layer_thickness = yaml_helper.get_data('sand_layer_thickness')
    sand_layer_fraction = {'min': yaml_helper.get_data('sand_layer_fraction_min'),
                           'max': yaml_helper.get_data('sand_layer_fraction_max')}
    include_channels = yaml_helper.get_data('include_channels')
    bandwidth_ord = yaml_helper.get_data('bandwidth_ord')
    broadband_qc_volume = yaml_helper.get_data('broadband_qc_volume')
    extra_qc_plots = yaml_helper.get_data('extra_qc_plots')
    verbose = yaml_helper.get_data('verbose')
    model_qc_volumes = yaml_helper.get_data('model_qc_volumes')
    model_store_in_memory = yaml_helper.get_data('model_store_in_memory')
    cleanup_intermediates = yaml_helper.get_data('cleanup_intermediates')

    max_column_height = yaml_helper.get_data('max_column_height')
    min_closure_voxels_simple =yaml_helper.get_data('min_closure_voxels_simple')
    min_closure_voxels_faulted = yaml_helper.get_data('min_closure_voxels_faulted')
    min_closure_voxels_onlap = yaml_helper.get_data('min_closure_voxels_onlap')
    signal_to_noise_ratio_db = yaml_helper.get_data('signal_to_noise_ratio_db')
    bandwidth_low = yaml_helper.get_data('bandwidth_low')
    bandwidth_high = yaml_helper.get_data('bandwidth_high')

    closure_types = yaml_helper.get_data('closure_types')
    include_salt = yaml_helper.get_data('include_salt')
    basin_floor_fans = yaml_helper.get_data('basin_floor_fans')
    min_number_faults = yaml_helper.get_data('min_number_faults')
    max_number_faults = yaml_helper.get_data('max_number_faults')

    categorical_parameter = CategoricalParameter(**low_level_controls)

    # Resolution presets. Only LOW is currently wired up; MEDIUM and HIGH are
    # kept as documented tiers but are intentionally unused for now.
    LOW = {
        "cube_shape": [100, 100, 500],
    }

    MEDIUM = {
        "cube_shape": [150, 150, 750],
    }

    HIGH = {
        "cube_shape": [300, 300, 1250],
    }
    # set cube shape
    cube_shape = LOW['cube_shape']

    categorical_parameter.initialize(
        project="example",
        project_folder=samples_path,
        work_folder=temp_builds_path,
        cube_shape=cube_shape,
        initial_layer_stdev=initial_layer_stdev,
        incident_angles=incident_angles,
        digi=digi,
        infill_factor=infill_factor,
        thickness_min=thickness_min,
        thickness_max=thickness_max,
        seabed_min_depth=seabed_min_depth,
        dip_factor_max=dip_factor_max,
        pad_samples=pad_samples,
        sand_layer_thickness=sand_layer_thickness,
        sand_layer_fraction=sand_layer_fraction,
        include_channels=include_channels,
        bandwidth_ord=bandwidth_ord,
        broadband_qc_volume=broadband_qc_volume,
        extra_qc_plots=extra_qc_plots,
        verbose=verbose,
        model_qc_volumes = model_qc_volumes,
        model_store_in_memory = model_store_in_memory,
        cleanup_intermediates = cleanup_intermediates,
        max_column_height = max_column_height,
        min_closure_voxels_simple = min_closure_voxels_simple,
        min_closure_voxels_faulted = min_closure_voxels_faulted,
        min_closure_voxels_onlap = min_closure_voxels_onlap,
        signal_to_noise_ratio_db = signal_to_noise_ratio_db,
        bandwidth_low = bandwidth_low,
        bandwidth_high = bandwidth_high,
        closure_types = closure_types,
        include_salt = include_salt,
        basin_floor_fans = basin_floor_fans,
        min_number_faults = min_number_faults,
        max_number_faults = max_number_faults
    )


    # control initialized template
    sample_control = SampleControl(categorical_parameter,**high_level_controls)
    # sample_control.load_recipe(Path(recipes_path) / "f0e5bfe6bb074c74b9b9617aaa5d9e60.yaml")
    sample_control.populate(recipes_path,build_configs_path)
