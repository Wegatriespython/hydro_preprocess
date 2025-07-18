"""
This script aggregates the global gridded data to any scale. The following
script specifically aggregates global gridded hydrological data onto the basin
 mapping used in the nexus module.
"""

import os
import sys

from shapely import to_ragged_array

print(sys.executable)
#  Import packages
import glob
from datetime import datetime as dt

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

start = dt.now()
import rasterstats
from dask.diagnostics import ProgressBar

# --- Configuration ---

# variable, for detailed symbols, refer to ISIMIP2b documentation
variables = [
    "qtot",  # total runoff
    "dis",  # discharge
    "qg",  # groundwater runoff
    "qr",
]  # groudnwater recharge
var = "qtot"

isimip = "3b"
data = "future"  # else future

# Define if use all touched raster for file naming
all_touched = True

# define lat and long chunk for reducing computational load
latchunk = 120
lonchunk = 640

if var == "dis":
    # define a spatial method to aggregate
    spatialmethod = "max"
else:
    # define a spatial method to aggregate
    spatialmethod = "sum"

# define quantile for statistical aggregation (if used)
quant = 0.1

# --- Paths ---
# The hydrological data can be accessed in watxene p drive. For accessing
# particular drive, seek permission from Edward Byers (byers@iiasa.ac.at)
# The files should be copied on to local drive
if isimip == "2b":
    climmodels = ["gfdl-esm2m", "hadgem2-es", "ipsl-cm5a-lr", "miroc5"]
    scenarios = ["rcp26", "rcp60"]
    scen = "rcp26"
    wd1 = "/mnt/p/ene.model/NEST/Hydrology"
    wd_base = "/mnt/p/watxene/ISIMIP/ISIMIP2b/OutputData/CWatM"
    wd2 = "/mnt/p/ene.model/NEST/hydrology/processed_nc4"
else:
    climmodels = [
        "gfdl-esm4",
    ]
    scenarios = ["ssp126", "ssp370", "ssp585"]
    scen = "ssp126"
    wd1 = "/mnt/p/ene.model/NEST/Hydrology"
    wd_base = "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM"
    wd2 = "/mnt/p/ene.model/NEST/hydrology/processed_nc4"

# Open raster area file
# The file landareamaskmap0.nc can be found under
# P:\ene.model\NEST\delineation\data\delineated_basins_new
area_path = os.path.join(wd1, "landareamaskmap0.nc")
if os.path.exists(area_path):
    area = xr.open_dataarray(area_path)
else:
    print(f"Area file not found at {area_path}")
    area = 1  # Fallback to 1, but results will be incorrect if area is needed.

# Basin shapefile
basin_path = "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"

# TO AVOID ERROR WHEN OPENING AND SLICING INPUT DATA - CHECK!
dask.config.set({"array.slicing.split-large-chunks": False})

# --- Main Processing Loop ---
for cl in climmodels:
    print(f"Processing model: {cl}, scenario: {scen}, variable: {var}")
    loop_start = dt.now()

    wd = f"{wd_base}/{scen}/{cl.upper()}"

    if data == "historical":
        hydro_data_path = os.path.join(wd, f"*{cl}*{var}*monthly*.nc")
    elif data == "future":
        hydro_data_path = os.path.join(wd, f"*{cl}*{scen}*{var}*monthly*.nc")

    files = glob.glob(hydro_data_path)
    if not files:
        print(f"No files found for {cl} at {hydro_data_path}. Skipping.")
        continue

    # Open hydrological data as a combined dataset
    da = xr.open_mfdataset(files)

    # Apply unit conversion directly
    da = da.fillna(0)

    if var == "dis":
        # Discharge: m³/s → km³/year
        da[var] = da[var] * 0.031556952
        da[var].attrs["unit"] = "km3/year"
    elif var in ["qtot", "qr"]:
        # Total runoff / Groundwater recharge: kg/m²/sec → km³/year
        da[var] = da[var] * 86400 * area * 3.65e-16 * 1000000
        da[var].attrs["unit"] = "km3/year"

    # Apply chunking for memory efficiency
    da[var] = da[var].chunk(
        {"lat": latchunk, "lon": lonchunk, "time": len(da[var].time)}
    )

    print(f"Processed {var} data with unit conversion only")
    print(f"Data shape: {da[var].shape}")
    print(f"Time range: {da.time.values[0]} to {da.time.values[-1]}")

    # --- Raster to Basin Aggregation ---
    print("Starting basin aggregation...")

    # Read shapefile of basins
    shapes = gpd.read_file(basin_path)

    # Create basin raster mask once
    print("Creating basin raster mask...")
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Get coordinates from data
    lat = da.lat.values
    lon = da.lon.values

    # Create transform
    transform = from_bounds(
        lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat)
    )

    # Create basin mask - this is the key optimization
    basin_mask = rasterize(
        [(geom, basin_id) for geom, basin_id in zip(shapes.geometry, shapes.BASIN_ID)],
        out_shape=(len(lat), len(lon)),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    print("Basin mask created successfully")

    # Load data into memory to speed up computation
    print("Loading data into memory...")
    with ProgressBar():
        da_loaded = da.compute()

    # Group data by year and apply vectorized aggregation
    print(f"Aggregating data using method: {spatialmethod}")

    # Initialize results dictionary
    results = {}

    # Get unique basin IDs
    basin_ids = shapes["BASIN_ID"].unique()

    # Process each year
    for year in da_loaded[var].time.dt.year.values:
        print(f"Processing year: {year}")

        # Select data for this year
        year_data = da_loaded[var].sel(time=da_loaded.time.dt.year == year)

        match spatialmethod:
            case "sum":
                # Sum over time (months) first, then aggregate by basin
                year_total = year_data.sum(dim="time")

                year_results = []
                for basin_id in basin_ids:
                    basin_pixels = year_total.values[basin_mask == basin_id]
                    basin_value = np.sum(basin_pixels[~np.isnan(basin_pixels)])
                    year_results.append(basin_value)

            case "max":
                # Max over time (months) first, then aggregate by basin
                year_max = year_data.max(dim="time")

                year_results = []
                for basin_id in basin_ids:
                    basin_pixels = year_max.values[basin_mask == basin_id]
                    basin_value = (
                        np.max(basin_pixels[~np.isnan(basin_pixels)])
                        if len(basin_pixels) > 0
                        else 0
                    )
                    year_results.append(basin_value)

            case "mean":
                # Mean over time (months) first, then aggregate by basin
                year_mean = year_data.mean(dim="time")

                year_results = []
                for basin_id in basin_ids:
                    basin_pixels = year_mean.values[basin_mask == basin_id]
                    basin_value = (
                        np.mean(basin_pixels[~np.isnan(basin_pixels)])
                        if len(basin_pixels) > 0
                        else 0
                    )
                    year_results.append(basin_value)

            case "quantile":
                # Quantile over time (months) first, then aggregate by basin
                year_quantile = year_data.quantile(quant, dim="time")

                year_results = []
                for basin_id in basin_ids:
                    basin_pixels = year_quantile.values[basin_mask == basin_id]
                    basin_value = (
                        np.mean(basin_pixels[~np.isnan(basin_pixels)])
                        if len(basin_pixels) > 0
                        else 0
                    )
                    year_results.append(basin_value)

            case "meansd":
                # Mean and std over time (months) first, then aggregate by basin
                year_mean = year_data.mean(dim="time")
                year_std = year_data.std(dim="time")

                year_results = []
                for basin_id in basin_ids:
                    basin_mean_pixels = year_mean.values[basin_mask == basin_id]
                    basin_std_pixels = year_std.values[basin_mask == basin_id]

                    mean_val = (
                        np.mean(basin_mean_pixels[~np.isnan(basin_mean_pixels)])
                        if len(basin_mean_pixels) > 0
                        else 0
                    )
                    std_val = (
                        np.mean(basin_std_pixels[~np.isnan(basin_std_pixels)])
                        if len(basin_std_pixels) > 0
                        else 0
                    )
                    year_results.append(mean_val + std_val)

        results[year] = year_results

    # --- Format Output ---
    print("Formatting output dataframe...")
    # Create dataframe from results
    df_pivot = pd.DataFrame(results, index=basin_ids)
    df_pivot.columns.name = "year"

    # Create the final dataframe with metadata
    df_out = shapes[["BASIN_ID", "BCU_name"]].set_index("BASIN_ID")
    df_out = df_out.join(df_pivot)

    df_out["Model"] = "CWatM"
    df_out["Scenario"] = f"{cl}_{scen}"
    df_out["Variable"] = f"{var}"
    df_out["Region"] = df_out["BCU_name"]
    df_out["Unit"] = da[var].attrs.get("unit", "km3/year")

    # --- Save Output ---
    output_dir = "/home/raghunathan/hydro_preprocess/hydro_output_py"
    os.makedirs(output_dir, exist_ok=True)

    at_suffix = "_at" if all_touched else "_nt"
    output_filename = f"CWatM_{var}_{cl}_{scen}_{spatialmethod}{at_suffix}.csv"
    output_path = os.path.join(output_dir, output_filename)

    df_out.to_csv(output_path)
    print(f"Successfully saved output to {output_path}")
    print(f"Time taken for model {cl}: {dt.now() - loop_start}")

print(f"Total script execution time: {dt.now() - start}")
