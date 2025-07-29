using PyCall, CSV, DataFrames, Dates, NetCDF, GeoDataFrames

iam_units = pyimport("iam_units")
registry = iam_units.registry

include("definitions.jl")

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
function find_netcdf_files(config::ProcessingConfig; test_file::String="")
  # If test file is provided, use it directly
  if !isempty(test_file)
    if !isfile(test_file)
      error("Test file not found: $(test_file)")
    end
    println("Using test file: $(test_file)")
    return [test_file]
  end

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

  # Check for NaN values in area grid
  nan_count = sum(isnan.(area_data))
  total_count = length(area_data)

  if nan_count > 0
    println("WARNING: Area grid contains $(nan_count)/$(total_count) NaN values ($(round(nan_count/total_count*100, digits=1))%)")

    # Replace NaN values with zero for areas (no contribution to aggregation)
    area_data[isnan.(area_data)] .= 0.0
    println("Replaced NaN values with 0.0 for proper aggregation")
  end

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


