from pathlib import Path

from scripts.control_parameter import CategoricalParameter, low_level_controls, high_level_controls, SampleControl
from scripts.yaml_helper import YAMLHelper




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