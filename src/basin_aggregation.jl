using NetCDF, Rasters, GeoDataFrames, Dates, DataFrames, ProgressMeter, Statistics

include("definitions.jl")
include("utils.jl")


"""
Process a single NetCDF file using chunked processing for memory efficiency.
Performs spatial aggregation to basins and applies unit conversion.
"""
function process_netcdf_file(filepath::String, config::ProcessingConfig,
  basins_gdf::DataFrame, area_grid::Union{Array,Nothing}, time_chunk_size::Int=100)

  println("Processing NetCDF file: $(basename(filepath))")

  # Open NetCDF file
  ncfile = NetCDF.open(filepath)
  try
    # Read coordinate variables
    lons = NetCDF.ncread(filepath, "lon")
    lats = NetCDF.ncread(filepath, "lat")
    times = NetCDF.ncread(filepath, "time")

    # Get data dimensions
    n_lon, n_lat, n_time = length(lons), length(lats), length(times)

    println("Data dimensions: $(n_lon) × $(n_lat) × $(n_time)")
    println("Processing in chunks of $(time_chunk_size) timesteps")

    # Initialize output array: basins × timesteps
    n_basins = nrow(basins_gdf)
    basin_data = zeros(Float64, n_basins, n_time)

    # Create proper coordinate ranges for Rasters.jl
    x_range = range(minimum(lons), maximum(lons), length=length(lons))
    y_range = range(maximum(lats), minimum(lats), length=length(lats))

    # Define aggregation function based on method
    # Note: NaN handling is done at the data level, not aggregation level
    agg_func = config.spatial_method == "sum" ? sum : mean

    # Get unit conversion factor once
    conversion_factor, needs_area = get_unit_conversion_factor(config.variable)

    # Process data in temporal chunks
    @showprogress "Processing temporal chunks..." for chunk_start in 1:time_chunk_size:n_time
      chunk_end = min(chunk_start + time_chunk_size - 1, n_time)
      chunk_length = chunk_end - chunk_start + 1

      # Read only the current temporal chunk
      var_chunk = NetCDF.ncread(filepath, config.variable,
        start=[1, 1, chunk_start],
        count=[n_lon, n_lat, chunk_length])

      # Process each timestep in current chunk
      for t in 1:chunk_length
        global_t = chunk_start + t - 1

        # Extract spatial slice for this timestep
        spatial_slice = @view var_chunk[:, :, t]

        # Create working copy for unit conversion
        working_slice = copy(spatial_slice)

        # Apply unit conversion
        if needs_area && area_grid !== nothing
          working_slice .*= area_grid
        end
        working_slice .*= conversion_factor

        # Create Raster object from working slice
        raster_slice = Raster(working_slice, (X(x_range), Y(y_range)))

        # Perform zonal aggregation for all basins at once
        basin_values = Rasters.zonal(agg_func, raster_slice; of=basins_gdf)

        # Post-process extreme values (likely from missing data flags like 1e20)
        # Replace extreme values with NaN to indicate missing/invalid data
        extreme_threshold = 1e15  # Much smaller than 1e20 but still very large
        basin_values[abs.(basin_values).>=extreme_threshold] .= NaN

        # Store results
        basin_data[:, global_t] = basin_values
      end

      # Force garbage collection after each chunk
      GC.gc()
    end

    # Create time index
    time_index = create_time_index(times, filepath)

    return basin_data, time_index

  finally
    NetCDF.close(ncfile)
  end
end
