from pathlib import Path
import uuid

import yaml

from logger_color import logger
from yaml_helper import YAMLHelper
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

# seismic signal controls
seismic_signal_controls = {
    'signal_to_noise_ratio_db': None, # Noise level
    'bandwidth_low': None, # low-cut frequency range
    'bandwidth_high': None, # high-cut frequency range
    'broadband_qc_volume': False, # false is for speed
}

# Quality check output such as images, logs , in-memory storage

quality_check_output = {
    'extra_qc_plots': None, # false for batch generation
    'verbose': None, # CLI logging while dev
    'model_qc_volumes': None,  # false for low storage
    'model_store_in_memory': None, # false; large volume can eat rams
    'cleanup_intermediates': None, # no clean-up for labels
}

# read only
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

    def check_value(self):
        all_none = {k for k,v in self.kwargs.items() if v is None}
        if all_none:
            raise Exception(f"{all_none} are None")
        else:
            print("config checking passed!!")

    def expose(self):
        """
        print out whole parameter for 1 sample
        :return: Dictionary
        """
        return self.kwargs

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

    def fault_only(self,f_min=3,f_max=7):
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

    def fault_complex(self,f_min=5,f_max=9):
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

# read-write only
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
        :return: parameters of Synthoseis configs
        """
        self.categorical_parameter.set(category)
        return self.categorical_parameter.expose()

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

            for rt in ratio_per_types:
                min_ratio = ratio_per_types[rt] + min_ratio
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

    def populate(self,recipe_path,config_path,seed=42):
        recipe_path = Path(recipe_path)
        config_path = Path(config_path)

        run_number = len(list(recipe_path.iterdir()))

        counts = {}

        for category, ratio in self.ratio_configs.items():
            counts[category] = int(self.population_amount * ratio)

        remaining = self.population_amount - sum(counts.values())

        # Add leftover samples caused by rounding
        for category in self.ratio_configs:
            if remaining <= 0:
                break
            counts[category] += 1
            remaining -= 1
        # saved recipe config
        recipe_name = f"recipe_{run_number}"
        recipe_config = {
            'population': {
                'seed': seed,
                'amount':self.population_amount},
            'category_ratio': self.ratio_configs,
            'category_counts': counts,
            'category_order': list(counts.keys())
        }

        recipe_name_path = recipe_path / f"{recipe_name}.yaml"
        recipe_name_path.touch(exist_ok=True)
        configs_list = []

        logger.info(f"[Populating] from file {recipe_name} At {recipe_name_path}")
        for category,amount in counts.items():
            # loop in amount
            for _ in range(amount):
                name = f"{category}_{uuid.uuid4().hex}"
                config_name_path = config_path / f"{name}.json"
                config = self._run_category(category) # generate sample with unique id with its type
                with open(config_name_path,'w') as f:
                    json.dump(config,f,indent=2)
                configs_list.append(name)
            logger.debug(f"[Populating]: Category {category} Amount {amount}")

        recipe_config["population"].update({
            "samples": configs_list,
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
    'sample_population': 6, # amount of sample that will be populated
    # each sample that is randomly created or mixed category will be ratio
    # all-faulted has different fault-line that it will be ratio, salt-fault will be ratio
    'sample_types': [
                     "boring",
                     "fault_only",
                     "fault_complex",
                     # "salt_only",
                     # "salt_fault_mixed",
                     # "onlap",
                     # "depositional",
                     # "full_mixed"
    ], # for dataset generations each generation will be ratio in same amount
    'ratio_per_types':{
        # "boring":0.0,
        # "fault_only":1.0,
        # "fault_complex":0.4,
        # "salt_only":0.0,
        # "salt_fault_mixed":0.0,
        # "onlap":0.0,
        # "depositional":0.0,
        # "full_mixed":0.0
    }
}


if __name__ == "__main__":
    setting_path = Path(__file__).parent.parent.joinpath('settings.yaml')
    yaml_helper = YAMLHelper(setting_path)

    recipes_path = yaml_helper.get_data('recipes_path' )# store all high-level configuration (type-sample)
    config_path = yaml_helper.get_data('config_path') # store all low-level configuration (samples)

    output_path = yaml_helper.get_data('output_path') # store all generated samples
    work_path = yaml_helper.get_data('work_path') # store as tmp

    ## initialize values
    cube_shape = yaml_helper.get_data('cube_shape')
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
        project_folder=output_path,
        work_folder=work_path,
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
    sample_control.populate(recipes_path,config_path,seed=42)
