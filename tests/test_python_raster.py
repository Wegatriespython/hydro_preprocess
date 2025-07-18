#!/usr/bin/env python3
"""
Minimal Python raster aggregation test
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds


def test_python_aggregation(output_file="python_test_output.csv"):
    # Test parameters
    netcdf_file = "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
    basin_file = (
        "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"
    )

    # Load data
    da = xr.open_dataset(netcdf_file)
    da_subset = da.isel(time=slice(0, 12))

    # Clean extreme values
    da_subset = da_subset.where(np.abs(da_subset) < 1e15, np.nan)

    # Load basins (first 10 for testing)
    shapes = gpd.read_file(basin_file)
    test_basins = shapes.iloc[:10]

    # Get coordinates
    lat = da_subset.lat.values
    lon = da_subset.lon.values

    # Create transform and basin mask
    transform = from_bounds(
        lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat)
    )

    basin_mask = rasterize(
        [
            (geom, basin_id)
            for geom, basin_id in zip(test_basins.geometry, test_basins.BASIN_ID)
        ],
        out_shape=(len(lat), len(lon)),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    # Load data into memory
    da_loaded = da_subset.compute()

    # Aggregate
    results = {}
    basin_ids = test_basins["BASIN_ID"].unique()

    for t, time_val in enumerate(da_loaded["qtot"].time.values):
        timestep_data = da_loaded["qtot"].isel(time=t)

        year_results = []
        for basin_id in basin_ids:
            basin_pixels = timestep_data.values[basin_mask == basin_id]
            basin_value = np.sum(basin_pixels[~np.isnan(basin_pixels)])
            year_results.append(basin_value)

        results[f"timestep_{t + 1}"] = year_results

    # Create output DataFrame
    output_df = pd.DataFrame()
    output_df["BASIN_ID"] = test_basins["BASIN_ID"].values
    output_df["BCU_name"] = test_basins["BCU_name"].values

    # Add timesteps in correct order
    for i in range(1, len(results) + 1):
        key = f"timestep_{i}"
        output_df[key] = results[key]

    # Save
    output_df.to_csv(output_file, index=False)

    return True


if __name__ == "__main__":
    test_python_aggregation()

