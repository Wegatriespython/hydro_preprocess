"""
Julia implementation of NetCDF to CSV pipeline for MESSAGEix water module.

This script replicates the functionality of hydro_agg_raster.py + hydro_agg_spatial.R,
converting global gridded NetCDF hydrological data to basin-aggregated CSV files
ready for processing by hydro_agg_basin.py.

"""

using NetCDF, DataFrames, CSV, Dates, Statistics
using ArchGDAL, GeoDataFrames
using ProgressMeter, ArgParse, YAML
using PyCall, Rasters

iam_units = pyimport("iam_units")
registry = iam_units.registry

# Import GC for garbage collection
import Base: GC

"""
Expand path with OS-neutral handling and environment variable substitution.
"""
function expand_path(path_str::String)
  # Handle environment variable substitution: ${VAR_NAME}
  expanded = path_str
  for m in eachmatch(r"\$\{([^}]+)\}", path_str)
    var_name = m.captures[1]
    var_value = get(ENV, var_name, "")
    if isempty(var_value)
      @warn "Environment variable $(var_name) not found, using empty string"
    end
    expanded = replace(expanded, m.match => var_value)
  end

  # Convert to OS-appropriate path separators and expand home directory
  expanded = expanduser(expanded)

  # Normalize path (handles forward/backward slashes)
  return normpath(expanded)
end

# Configuration structure matching the original scripts
Base.@kwdef struct ProcessingConfig
  # Variable selection
  variable::String  # qtot, dis, qr

  # Climate model configuration  
  isimip_version::String  # "2b" or "3b"
  climate_model::String
  scenario::String  # ssp126, ssp370, ssp585
  data_period::String  # "historical" or "future"

  # Regional configuration
  region::String  # R11, R12, or country code
  iso3::String

  # Processing parameters
  temporal_resolution::String  # "monthly" or "annual"
  spatial_method::String  # "sum" or "mean"

  # Paths (configurable)
  input_dir::String
  area_file::String
  basin_shapefile::String
  output_dir::String
end

"""
Load and validate basin polygons from shapefile.
Returns GeoDataFrame with basin geometries and metadata.
"""
function load_basin_polygons(config::ProcessingConfig)
  println("Loading basin polygons...")

  shapefile_path = config.basin_shapefile

  # Load shapefile using ArchGDAL
  basins_gdf = GeoDataFrames.read(shapefile_path)

  println("Loaded $(nrow(basins_gdf)) basins for region $(config.region)")

  # Validate required columns
  required_cols = ["BASIN_ID", "BCU_name", "NAME", "REGION", "area_km2"]
  missing_cols = [col for col in required_cols if !(col in names(basins_gdf))]

  if !isempty(missing_cols)
    error("Missing required columns in basin shapefile: $(missing_cols)")
  end

  return basins_gdf
end

"""
Find and validate NetCDF input files based on configuration.
"""
function find_netcdf_files(config::ProcessingConfig)
  println("Searching for NetCDF files...")

  # Construct ISIMIP path: base_dir/hydro_model/scenario/climate_model
  isimip_dir = joinpath(config.input_dir, "CWatM", config.scenario, uppercase(config.climate_model))

  files = filter(f -> endswith(f, ".nc"), readdir(isimip_dir, join=true))
  matching_files = filter(f -> occursin(config.variable, f) && occursin(config.temporal_resolution, f), files)

  if isempty(matching_files)
    error("No NetCDF files found matching pattern in: $(isimip_dir)")
  end

  println("Found $(length(matching_files)) NetCDF files")
  return matching_files
end

"""
Get unit conversion factor for the specified variable.
Returns the conversion factor and whether area multiplication is needed.
"""
function get_unit_conversion_factor(variable::String)
  if variable == "dis"
    # Discharge: m³/s → km³/year
    source_unit = registry.meter^3 / registry.second
    target_unit = registry.kilometer^3 / registry.year
    return registry.convert(1, source_unit, target_unit), false
  elseif variable in ["qtot", "qr"]
    # Total runoff / groundwater recharge: kg/m²/sec → km³/year
    flux_unit = registry.kilogram / registry.meter^2 / registry.second
    area_factor = registry.meter^2
    water_density = registry.water
    target_unit = registry.kilometer^3 / registry.year

    # Convert: kg/m²/s * m² / water_density = km³/s → km³/year
    volume_per_second = (flux_unit * area_factor) / water_density
    return registry.convert(1.0, volume_per_second, target_unit), true
  else
    error("Unknown variable for unit conversion: $(variable)")
  end
end

"""
Load area grid for unit conversion (landareamaskmap0.nc equivalent).
"""
function load_area_grid(config::ProcessingConfig)
  area_file = config.area_file

  if !isfile(area_file)
    @warn "Area grid file not found: $(area_file). Using unit area."
    return nothing
  end

  println("Loading area grid...")
  area_data = NetCDF.ncread(area_file, "land area")
  return area_data
end

"""
Create time index from NetCDF time variable with proper unit handling.
"""
function create_time_index(times::Vector, filepath::String)
  # Read time units attribute from NetCDF file
  time_units = NetCDF.ncgetatt(filepath, "time", "units")

  # Parse time units with flexible regex patterns
  # Handle both padded (YYYY-MM-DD) and unpadded (YYYY-M-D) date formats
  patterns = [
    r"^(days?|months?|years?|hours?|minutes?|seconds?)\s+since\s+(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
    r"^(days?|months?|years?|hours?|minutes?|seconds?)\s+since\s+(\d{4}-\d{1,2}-\d{1,2})"  # YYYY-M-D
  ]

  match_result = nothing
  for pattern in patterns
    match_result = match(pattern, time_units)
    if match_result !== nothing
      break
    end
  end

  if match_result === nothing
    error("Could not parse time units: $(time_units)")
  end

  time_unit = match_result.captures[1]
  date_string = match_result.captures[2]

  # Parse date with flexible format
  if occursin(r"\d{4}-\d{2}-\d{2}", date_string)
    reference_date = Date(date_string, "yyyy-mm-dd")
  else
    # Handle unpadded format by padding zeros
    parts = split(date_string, "-")
    padded_date = parts[1] * "-" * lpad(parts[2], 2, "0") * "-" * lpad(parts[3], 2, "0")
    reference_date = Date(padded_date, "yyyy-mm-dd")
  end

  # Convert time values to dates
  if time_unit in ["day", "days"]
    dates = reference_date .+ Day.(round.(Int, times))
  elseif time_unit in ["month", "months"]
    dates = reference_date .+ Month.(round.(Int, times))
  elseif time_unit in ["year", "years"]
    dates = reference_date .+ Year.(round.(Int, times))
  elseif time_unit in ["hour", "hours"]
    dates = reference_date .+ Day.(round.(Int, times ./ 24))
  elseif time_unit in ["minute", "minutes"]
    dates = reference_date .+ Day.(round.(Int, times ./ (24 * 60)))
  elseif time_unit in ["second", "seconds"]
    dates = reference_date .+ Day.(round.(Int, times ./ (24 * 60 * 60)))
  else
    error("Unsupported time unit: $(time_unit)")
  end

  return dates
end

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
    agg_func = config.spatial_method == "sum" ? sum : mean #Sum and mean are incorrect for dis.

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

"""
Format and save basin-aggregated data as CSV.
Output format matches the expected input for hydro_agg_basin.py.
"""
function save_basin_csv(aggregated_data::Matrix, time_index::Vector{Date},
  basins_gdf::DataFrame, config::ProcessingConfig)

  println("Formatting and saving CSV output...")

  n_basins, n_timesteps = size(aggregated_data)

  # Create output DataFrame
  output_df = DataFrame()

  # Add basin metadata columns (matching hydro_agg_basin.py expectations)
  output_df.BASIN_ID = basins_gdf.BASIN_ID
  output_df.BCU_name = basins_gdf.BCU_name
  output_df.NAME = basins_gdf.NAME
  output_df.REGION = basins_gdf.REGION
  output_df.area_km2 = basins_gdf.area_km2

  # Add time series data as columns
  for (i, date) in enumerate(time_index)
    col_name = string(date)
    output_df[!, col_name] = aggregated_data[:, i]
  end

  # Generate output filename
  output_filename = "$(config.variable)_$(config.temporal_resolution)_$(config.climate_model)_$(config.scenario)_$(config.data_period).csv"
  output_path = joinpath(config.output_dir, output_filename)

  # Save CSV
  CSV.write(output_path, output_df)

  println("Saved basin-aggregated data: $(output_path)")
  println("Output dimensions: $(nrow(output_df)) basins × $(ncol(output_df)-5) timesteps")

  return output_path
end

"""
Main processing pipeline function.
"""
function process_hydro_data(config::ProcessingConfig)
  println("Starting NetCDF to CSV pipeline...")
  println("Configuration: $(config.variable), $(config.climate_model), $(config.scenario)")

  # Load basin polygons
  basins_gdf = load_basin_polygons(config)

  # Load area grid for unit conversion
  area_grid = load_area_grid(config)

  # Find input NetCDF files
  netcdf_files = find_netcdf_files(config)

  # Process NetCDF files
  all_outputs = String[]

  if config.temporal_resolution == "daily" && length(netcdf_files) > 1
    # Handle multiple daily files - process sequentially and concatenate
    println("Processing $(length(netcdf_files)) daily files sequentially...")

    # Store results
    successful_results = Tuple{Matrix{Float64},Vector{Date},String}[]

    # Process files sequentially
    for netcdf_file in netcdf_files
      try
        # Process single file
        aggregated_data, time_index = process_netcdf_file(
          netcdf_file, config, basins_gdf, area_grid
        )

        # Store result
        push!(successful_results, (aggregated_data, time_index, basename(netcdf_file)))

      catch e
        @error "Failed to process $(basename(netcdf_file)): $(e)"
        continue
      end
    end

    # Sort by filename to ensure temporal order
    sort!(successful_results, by=x -> x[3])

    # Concatenate all daily chunks sequentially
    if !isempty(successful_results)
      all_aggregated_data = [r[1] for r in successful_results]
      all_time_indices = [r[2] for r in successful_results]

      concatenated_data = hcat(all_aggregated_data...)
      concatenated_time_index = vcat(all_time_indices...)

      println("Concatenated $(length(successful_results)) daily files into single dataset")
      println("Final dimensions: $(size(concatenated_data))")

      # Save single CSV output
      output_path = save_basin_csv(
        concatenated_data, concatenated_time_index, basins_gdf, config
      )

      push!(all_outputs, output_path)
    else
      @warn "No daily files were successfully processed"
    end

  else
    # Handle single file (monthly) or single daily file
    for netcdf_file in netcdf_files
      try
        # Process single file
        aggregated_data, time_index = process_netcdf_file(
          netcdf_file, config, basins_gdf, area_grid
        )

        # Save CSV output
        output_path = save_basin_csv(
          aggregated_data, time_index, basins_gdf, config
        )

        push!(all_outputs, output_path)

      catch e
        @error "Failed to process $(basename(netcdf_file)): $(e)"
        continue
      end
    end
  end

  println("\nPipeline completed successfully!")
  println("Generated $(length(all_outputs)) CSV files:")
  for output in all_outputs
    println("  - $(basename(output))")
  end

  return all_outputs
end

"""
Command-line interface for the pipeline.
"""
function main()
  # Parse command line arguments
  s = ArgParseSettings(description="NetCDF to CSV pipeline for MESSAGEix water module")

  @add_arg_table! s begin
    "--variable", "-v"
    help = "Hydrological variable (qtot, dis, qr)"
    default = "qtot"
    "--isimip-version"
    help = "ISIMIP version (2b or 3b)"
    default = "3b"
    "--climate-model", "-m"
    help = "Climate model name"
    default = "gfdl-esm4"
    "--scenario", "-s"
    help = "Climate scenario"
    default = "ssp370"
    "--region", "-r"
    help = "Regional configuration (R11, R12)"
    default = "R12"
    "--iso3"
    help = "ISO3 country code"
    default = "ZMB"
    "--data-period"
    help = "Data period (historical or future)"
    default = "future"
    "--temporal-resolution"
    help = "Temporal resolution (monthly or daily)"
    default = "daily"
    "--spatial-method"
    help = "Spatial aggregation method (sum or mean)"
    default = "sum"
    "--input-dir", "-i"
    help = "Input directory containing NetCDF files"
    default = "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData"
    "--area-file", "-a"
    help = "Areas for grid cells"
    default = "/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc"
    "--basin-shapefile", "-b"
    help = "Basin shapefile directory"
    default = "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"
    "--output-dir", "-o"
    help = "Output directory for CSV files"
    default = "./hydro_output"
    "--config", "-c"
    help = "YAML configuration file"
    default = ""
  end

  args = parse_args(s)

  # Load configuration
  if !isempty(args["config"]) && isfile(args["config"])
    config_dict = YAML.load_file(args["config"])
    config = ProcessingConfig(;
      variable=get(config_dict, "variable", args["variable"]),
      isimip_version=get(config_dict, "isimip_version", args["isimip-version"]),
      climate_model=get(config_dict, "climate_model", args["climate-model"]),
      scenario=get(config_dict, "scenario", args["scenario"]),
      region=get(config_dict, "region", args["region"]),
      iso3=get(config_dict, "iso3", args["iso3"]),
      data_period=get(config_dict, "data_period", args["data-period"]),
      temporal_resolution=get(config_dict, "temporal_resolution", args["temporal-resolution"]),
      spatial_method=get(config_dict, "spatial_method", args["spatial-method"]),
      input_dir=expand_path(get(config_dict, "input_dir", args["input-dir"])),
      area_file=expand_path(get(config_dict, "area_file", args["area-file"])),
      basin_shapefile=expand_path(get(config_dict, "basin_shapefile", args["basin-shapefile"])),
      output_dir=expand_path(get(config_dict, "output_dir", args["output-dir"])),
    )
  else
    config = ProcessingConfig(
      variable=args["variable"],
      isimip_version=args["isimip-version"],
      climate_model=args["climate-model"],
      scenario=args["scenario"],
      region=args["region"],
      iso3=args["iso3"],
      data_period=args["data-period"],
      temporal_resolution=args["temporal-resolution"],
      spatial_method=args["spatial-method"],
      input_dir=expand_path(args["input-dir"]),
      area_file=expand_path(args["area-file"]),
      basin_shapefile=expand_path(args["basin-shapefile"]),
      output_dir=expand_path(args["output-dir"]),
    )
  end

  # Create output directory if it doesn't exist
  mkpath(config.output_dir)

  # Run the pipeline
  process_hydro_data(config)
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
