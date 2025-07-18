"""
Minimal Julia raster aggregation test
"""

using NetCDF, DataFrames, CSV
using ArchGDAL, GeoDataFrames, Rasters

function test_julia_aggregation(output_file::String="julia_test_output.csv")
  # Test parameters
  netcdf_file = "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc"
  basin_file = "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"

  # Load data
  lons = NetCDF.ncread(netcdf_file, "lon")
  lats = NetCDF.ncread(netcdf_file, "lat")
  data_slice = NetCDF.ncread(netcdf_file, "qtot", start=[1, 1, 1], count=[length(lons), length(lats), 12])

  # Clean extreme values
  data_slice[abs.(data_slice).>=1e15] .= NaN

  # Load basins (first 10 for testing)
  basins_gdf = GeoDataFrames.read(basin_file)
  test_basins = basins_gdf[1:10, :]

  # Create coordinate ranges
  x_range = range(minimum(lons), maximum(lons), length=length(lons))
  y_range = range(maximum(lats), minimum(lats), length=length(lats))

  # Aggregate
  n_timesteps = size(data_slice, 3)
  n_basins = nrow(test_basins)
  results = zeros(Float64, n_basins, n_timesteps)

  for t in 1:n_timesteps
    spatial_slice = data_slice[:, :, t]
    raster_obj = Raster(spatial_slice, (X(x_range), Y(y_range)))
    basin_values = Rasters.zonal(sum, raster_obj; of=test_basins)
    results[:, t] = basin_values
  end

  # Create output DataFrame
  output_df = DataFrame()
  output_df.BASIN_ID = test_basins.BASIN_ID
  output_df.BCU_name = test_basins.BCU_name

  # Add timesteps in correct order
  for i in 1:n_timesteps
    col_name = "timestep_$(i)"
    output_df[!, col_name] = results[:, i]
  end

  # Save
  CSV.write(output_file, output_df)

  return true
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
  test_julia_aggregation()
end
