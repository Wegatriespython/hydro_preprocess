"""
Test script for discharge outlet-pixel movement tracking.

Modified from test_dis_out_ext.jl to track movement of outlet pixels over time.
This allows comparison with Python package results to validate consistency.
"""

# Import required packages
using NetCDF, DataFrames, CSV, Dates, Statistics
using ArchGDAL, GeoDataFrames
using ProgressMeter, YAML
using PyCall, Rasters, JSON

# Import the main processing module
include("hydro_agg_raster.jl")

"""
Find the maximum value and its location within a basin using Rasters.mask (fixed version).
"""
function find_max_in_basin(dis_data::Matrix, basin_geom, lons::Vector, lats::Vector, transform)
    # Get dimensions - dis_data comes as [n_lon, n_lat] from NetCDF
    n_lon, n_lat = size(dis_data)
    
    # Verify array dimensions match coordinate arrays
    if n_lon != length(lons) || n_lat != length(lats)
        @error "Dimension mismatch: data($n_lon, $n_lat) vs coords($(length(lons)), $(length(lats)))"
        return nothing, nothing
    end
    
    try
        # Create coordinate ranges - CORRECTED to match data dimensions
        x_range = range(minimum(lons), maximum(lons), length=n_lon)
        y_range = range(maximum(lats), minimum(lats), length=n_lat)
        
        # Create Raster with correct dimension mapping: data is [n_lon, n_lat]
        raster = Rasters.Raster(dis_data, (Rasters.X(x_range), Rasters.Y(y_range)))
        
        # Apply mask to extract values within basin
        basin_mask = Rasters.mask(raster; with=basin_geom, missingval=missing)
        
        # Find all non-missing values and their indices
        valid_indices = findall(!ismissing, basin_mask)
        if isempty(valid_indices)
            return nothing, nothing
        end
        
        # Get valid values, filtering out NaN and extreme values
        valid_values = Float64[]
        valid_index_map = Tuple{Int,Int}[]
        
        for idx in valid_indices
            val = basin_mask[idx]
            if !ismissing(val) && !isnan(val) && abs(val) < 1e15
                push!(valid_values, Float64(val))
                push!(valid_index_map, idx)
            end
        end
        
        if isempty(valid_values)
            return nothing, nothing
        end
        
        # Find maximum value
        max_val = maximum(valid_values)
        max_val_idx = findfirst(==(max_val), valid_values)
        
        if max_val_idx === nothing
            return nothing, nothing
        end
        
        # Get the raster index of the maximum
        raster_idx = valid_index_map[max_val_idx]
        
        # Convert raster index to grid coordinates
        # raster_idx is (x_idx, y_idx) where x=lon, y=lat
        lon_idx = raster_idx[1]  # x index in raster corresponds to longitude
        lat_idx = raster_idx[2]  # y index in raster corresponds to latitude
        
        # Return as (lat_idx, lon_idx) for consistency with Python format
        max_loc = (lat_idx, lon_idx)
        
        return max_loc, Float64(max_val)
        
    catch e
        @warn "Error in Rasters.mask operation for basin: $e"
        return nothing, nothing
    end
end


"""
Track outlet pixel movement over time using a robust rasterization approach.
"""
function track_outlet_movement_julia(filepath::String, config::ProcessingConfig,
  basins_gdf::DataFrame, area_grid::Union{Array,Nothing},
  n_basins::Int=10, max_timesteps::Int=30)

  println("Tracking outlet pixel movement using Julia with rasterization...")
  println("File: $(basename(filepath))")

  ncfile = NetCDF.open(filepath)
  try
    lons = NetCDF.ncread(filepath, "lon")
    lats = NetCDF.ncread(filepath, "lat")
    times = NetCDF.ncread(filepath, "time")

    n_time_test = min(length(times), max_timesteps)
    println("Testing with $(n_time_test) timesteps (limited from $(length(times)))")

    n_lon, n_lat = length(lons), length(lats)
    n_basins_test = min(n_basins, nrow(basins_gdf))

    # Create transform for rasterization (matches Python version)
    lon_res = abs(lons[2] - lons[1])
    lat_res = abs(lats[2] - lats[1])
    transform = [minimum(lons) - lon_res/2, lon_res, 0.0, maximum(lats) + lat_res/2, 0.0, -lat_res]

    basin_outlet_locations = [Vector{Tuple{Int,Int}}() for _ in 1:n_basins_test]
    basin_outlet_values = [Vector{Float64}() for _ in 1:n_basins_test]

    conversion_factor, needs_area = get_unit_conversion_factor(config.variable)
    timesteps_to_process = collect(1:5:n_time_test)

    @showprogress "Tracking outlet movement..." for t in timesteps_to_process
      spatial_slice = NetCDF.ncread(filepath, config.variable, start=[1, 1, t], count=[n_lon, n_lat, 1])[:, :, 1]
      
      working_slice = Float32.(spatial_slice)
      if needs_area && area_grid !== nothing
        working_slice .*= area_grid
      end
      working_slice .*= conversion_factor
      working_slice[abs.(working_slice) .>= 1e15] .= NaN32

      for basin_idx in 1:n_basins_test
        basin_geom = basins_gdf.geometry[basin_idx]
        
        max_loc, max_val = find_max_in_basin(working_slice, basin_geom, lons, lats, transform)
        
        if max_loc !== nothing
          # Verify outlet is actually inside basin (like Python version)
          outlet_lat = lats[max_loc[1]]
          outlet_lon = lons[max_loc[2]]
          point_wkt = "POINT($outlet_lon $outlet_lat)"
          point_geom = ArchGDAL.fromWKT(point_wkt)
          
          if ArchGDAL.contains(basin_geom, point_geom)
            push!(basin_outlet_locations[basin_idx], max_loc)
            push!(basin_outlet_values[basin_idx], max_val)
          else
            @warn "Outlet at $(max_loc) (lon: $outlet_lon, lat: $outlet_lat) is outside basin $(basins_gdf.BASIN_ID[basin_idx])"
          end
        end
      end
    end

    results = analyze_outlet_movement_julia(basin_outlet_locations, basin_outlet_values, basins_gdf, n_basins_test)
    return results

  finally
    NetCDF.close(ncfile)
  end
end

"""
Analyze outlet movement results from Julia implementation.
"""
function analyze_outlet_movement_julia(basin_outlet_locations::Vector{Vector{Tuple{Int,Int}}},
  basin_outlet_values::Vector{Vector{Float64}},
  basins_gdf::DataFrame, n_basins::Int)

  println("\n=== JULIA OUTLET PIXEL MOVEMENT ANALYSIS ===")

  # Initialize counters
  fixed_outlet_basins = 0
  moving_outlet_basins = 0
  empty_basins = 0

  # Analyze each basin
  basin_analysis = Dict()

  for basin_idx in 1:n_basins
    # Get basin ID from BASIN_ID column to match Python indexing
    basin_id = basins_gdf.BASIN_ID[basin_idx]
    locations = basin_outlet_locations[basin_idx]
    values = basin_outlet_values[basin_idx]

    if isempty(locations)
      empty_basins += 1
      basin_analysis[string(basin_id)] = Dict(
        "status" => "empty",
        "n_unique_locations" => 0,
        "unique_locations" => [],
        "n_observations" => 0
      )
    else
      unique_locations = unique(locations)
      n_unique = length(unique_locations)

      if n_unique == 1
        fixed_outlet_basins += 1
        status = "fixed"
      else
        moving_outlet_basins += 1
        status = "moving"
      end

      basin_analysis[string(basin_id)] = Dict(
        "status" => status,
        "n_unique_locations" => n_unique,
        "unique_locations" => unique_locations,
        "max_value_range" => [minimum(values), maximum(values)],
        "n_observations" => length(locations)
      )
    end
  end

  # Calculate percentages
  fixed_percentage = fixed_outlet_basins / n_basins * 100
  moving_percentage = moving_outlet_basins / n_basins * 100

  println("Total basins analyzed: $n_basins")
  println("Basins with FIXED outlet: $fixed_outlet_basins ($(round(fixed_percentage, digits=1))%)")
  println("Basins with MOVING outlet: $moving_outlet_basins ($(round(moving_percentage, digits=1))%)")
  println("Empty basins: $empty_basins")

  # Show examples of moving outlets
  println("\n=== Examples of basins with moving outlets ===")
  moving_count = 0
  for (basin_id, info) in basin_analysis
    if info["status"] == "moving" && moving_count < 5
      println("\nBasin $basin_id:")
      println("  Number of unique outlet locations: $(info["n_unique_locations"])")
      locations_str = join(["($(loc[1]),$(loc[2]))" for loc in info["unique_locations"][1:min(3, end)]], ", ")
      println("  Locations: $locations_str...")
      val_range = info["max_value_range"]
      println("  Discharge range: $(round(val_range[1], digits=2)) - $(round(val_range[2], digits=2)) m³/s")
      moving_count += 1
    end
  end

  return Dict(
    "method" => "julia_rasters",
    "basin_analysis" => basin_analysis,
    "summary" => Dict(
      "method" => "julia_rasters",
      "total_basins" => n_basins,
      "fixed_outlet_basins" => fixed_outlet_basins,
      "moving_outlet_basins" => moving_outlet_basins,
      "empty_basins" => empty_basins,
      "fixed_outlet_percentage" => fixed_percentage,
      "moving_outlet_percentage" => moving_percentage
    )
  )
end
"""
Main test function for outlet movement tracking.
"""
function main_test()
  println("=== JULIA OUTLET PIXEL MOVEMENT TEST ===\n")

  # Test configuration - match Python test parameters
  config = ProcessingConfig(
    variable="dis",
    isimip_version="3b",
    climate_model="gfdl-esm4",
    scenario="ssp126",
    region="R12",
    iso3="ZMB",
    data_period="future",
    temporal_resolution="monthly",
    spatial_method="sum",  # This will be ignored for discharge
    input_dir="/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData",
    area_file="/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc",
    basin_shapefile="/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp",
    output_dir="./hydro_output"
  )

  println("Test configuration:")
  println("  Variable: $(config.variable)")
  println("  Climate model: $(config.climate_model)")
  println("  Scenario: $(config.scenario)")
  println("  Basin shapefile: $(basename(config.basin_shapefile))")

  try
    # Create output directory
    mkpath(config.output_dir)

    # Load basin polygons
    basins_gdf = load_basin_polygons(config)

    # Load area grid
    area_grid = load_area_grid(config)

    # Find NetCDF files
    netcdf_files = find_netcdf_files(config)

    if isempty(netcdf_files)
      error("No NetCDF files found for testing")
    end

    # Test with first file
    test_file = netcdf_files[1]
    println("Testing with file: $(basename(test_file))")

    # Run outlet movement tracking - match Python test parameters
    results = track_outlet_movement_julia(
      test_file, config, basins_gdf, area_grid, 10, 30  # 10 basins, 30 timesteps
    )

    # Save results to JSON for comparison with Python
    output_file = "julia_outlet_movement_results.json"
    open(output_file, "w") do io
      JSON.print(io, results, 2)
    end

    println("\nResults saved to $output_file for comparison with Python results")

    # Overall conclusion
    fixed_pct = results["summary"]["fixed_outlet_percentage"]
    if fixed_pct > 90
      println("\n✓ CONCLUSION: Outlet pixels are predominantly FIXED - outlet-pixel approach is valid!")
    elseif fixed_pct > 70
      println("\n⚠ CONCLUSION: Most outlets are fixed but some move - approach needs refinement")
    else
      println("\n✗ CONCLUSION: Many outlets move over time - outlet-pixel approach may be problematic")
    end

  catch e
    println("ERROR during testing: $(e)")
    rethrow(e)
  end
end

# Run test if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  main_test()
end
