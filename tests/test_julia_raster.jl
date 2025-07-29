"""
Minimal Julia raster aggregation test
"""

using DataFrames, CSV

# Import shared basin aggregation functions
include("../src/basin_aggregation_concept.jl")

function test_julia_aggregation(output_file::String="julia_test_output.csv")
  # Test parameters
  netcdf_file = "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
  basin_file = "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"

  # Use shared processing pipeline
  output_df = process_basin_aggregation(netcdf_file, basin_file;
    variable="qtot",
    n_basins=10,
    n_timesteps=12,
    agg_method="sum")

  # Save
  CSV.write(output_file, output_df)

  return true
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
  test_julia_aggregation()
end
