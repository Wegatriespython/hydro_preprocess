import netCDF4 as nc
import os

# Configuration for test data generation
TEST_CONFIGS = [
    {
        "variable": "qtot",
        "temporal_resolution": "monthly",
        "source_path": "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc",
        "time_samples": 2,  # 2 months
    },
    {
        "variable": "qtot",
        "temporal_resolution": "daily",
        "source_path": "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_daily_2015_2020.nc",
        "time_samples": 30,  # 1 month of daily data
    },
    {
        "variable": "qr",
        "temporal_resolution": "monthly",
        "source_path": "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qr_global_monthly_2015_2100.nc",
        "time_samples": 2,  # 2 months
    },
]

# Fixed configuration parameters (non-varying)
FIXED_CONFIG = {
    "isimip_version": "3b",
    "climate_model": "gfdl-esm4",
    "scenario": "ssp126",
    "data_period": "future",
    "region": "R12",
    "iso3": "ZMB",
    "spatial_method": "sum",
    "basin_shapefile": "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp",
    "area_file": "/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc",
}


def create_test_file(config):
    """Create a test NetCDF file for given configuration."""
    dest_dir = "tests/data"
    dest_file = os.path.join(
        dest_dir, f"{config['variable']}_{config['temporal_resolution']}_test_data.nc"
    )

    # Create directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    source_file = config["source_path"]

    if not os.path.exists(source_file):
        print(f"WARNING: Source file not found: {source_file}")
        return None

    try:
        # Open the source NetCDF file
        with nc.Dataset(source_file, "r") as src:
            # Create the destination NetCDF file
            with nc.Dataset(dest_file, "w", format=src.file_format) as dst:
                # Copy global attributes
                dst.setncatts(src.__dict__)

                # Copy dimensions, adjusting time dimension for sample size
                for name, dimension in src.dimensions.items():
                    if name == "time":
                        dst.createDimension(name, config["time_samples"])
                    else:
                        dst.createDimension(
                            name,
                            (len(dimension) if not dimension.isunlimited() else None),
                        )

                # Copy variables and their attributes
                for name, variable in src.variables.items():
                    # Create the variable in the destination file
                    dst.createVariable(name, variable.datatype, variable.dimensions)
                    # Copy variable attributes, skipping _FillValue
                    att_dict = {
                        k: v for k, v in variable.__dict__.items() if k != "_FillValue"
                    }
                    dst.variables[name].setncatts(att_dict)

                    # Copy data, slicing the time dimension
                    if name == "time":
                        dst.variables[name][:] = variable[: config["time_samples"]]
                    elif "time" in variable.dimensions:
                        # Create a slice object
                        slicing = [slice(None)] * len(variable.dimensions)
                        time_index = variable.dimensions.index("time")
                        slicing[time_index] = slice(0, config["time_samples"])
                        dst.variables[name][:] = variable[tuple(slicing)]
                    else:
                        dst.variables[name][:] = variable[:]

        print(f"Successfully created {dest_file}")
        return dest_file

    except Exception as e:
        print(f"Error creating {dest_file}: {e}")
        return None


def create_config_file():
    """Create test configuration documentation file."""
    config_file = "tests/test_data_config.txt"

    with open(config_file, "w") as f:
        f.write("Test Data Configuration\n")
        f.write("======================\n\n")

        f.write("Fixed Parameters (non-varying across all test files):\n")
        f.write("---------------------------------------------------\n")
        for key, value in FIXED_CONFIG.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTest File Configurations:\n")
        f.write("------------------------\n")
        for i, config in enumerate(TEST_CONFIGS, 1):
            f.write(
                f"\n{i}. {config['variable']}_{config['temporal_resolution']}_test_data.nc\n"
            )
            f.write(f"   Variable: {config['variable']}\n")
            f.write(f"   Temporal Resolution: {config['temporal_resolution']}\n")
            f.write(f"   Time Samples: {config['time_samples']}\n")
            f.write(f"   Source: {os.path.basename(config['source_path'])}\n")

            # Expected output filename
            expected_output = f"{config['variable']}_{config['temporal_resolution']}_test_expected_output.csv"
            f.write(f"   Expected Output: {expected_output}\n")

            # Test command
            test_cmd = f"julia hydro_agg_raster.jl --test-file tests/data/{config['variable']}_{config['temporal_resolution']}_test_data.nc --variable {config['variable']} --temporal-resolution {config['temporal_resolution']} --output-dir test_output"
            f.write(f"   Test Command: {test_cmd}\n")

    print(f"Configuration file created: {config_file}")


# Main execution
if __name__ == "__main__":
    print("Creating test NetCDF files...")

    created_files = []
    for config in TEST_CONFIGS:
        result = create_test_file(config)
        if result:
            created_files.append(result)

    print(f"\nCreated {len(created_files)} test files:")
    for file in created_files:
        print(f"  - {file}")

    # Create configuration documentation
    create_config_file()

    print("\nTest data generation complete!")

