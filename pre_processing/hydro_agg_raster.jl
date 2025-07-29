"""
Julia implementation of NetCDF to CSV pipeline for MESSAGEix water module.

This script replicates the functionality of hydro_agg_raster.py + hydro_agg_spatial.R,
converting global gridded NetCDF hydrological data to basin-aggregated CSV files
ready for processing by hydro_agg_basin.py.

"""

using NetCDF, DataFrames, CSV, Dates, Statistics
using ArchGDAL, GeoDataFrames
using ProgressMeter, ArgParse, YAML
using Rasters

# Import GC for garbage collection
import Base: GC

include("../src/utils.jl")
include("../src/basin_aggregation.jl")

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
  output_filename = "$(config.variable)_$(config.temporal_resolution)_$(config.hydro_model)_$(config.climate_model)_$(config.scenario)_$(config.data_period).csv"
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
function process_hydro_data(config::ProcessingConfig; test_file::String="")
  println("Starting NetCDF to CSV pipeline...")
  println("Configuration: $(config.variable), $(config.climate_model), $(config.scenario)")

  # Load basin polygons
  basins_gdf = load_basin_polygons(config)

  # Load area grid for unit conversion
  area_grid = load_area_grid(config)

  # Find input NetCDF files
  netcdf_files = find_netcdf_files(config; test_file=test_file)

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
    "--test-file"
    help = "Direct NetCDF file path for testing (bypasses file discovery)"
    default = ""
    "--variable", "-v"
    help = "Hydrological variable (qtot, dis, qr)"
    default = "qtot"
    "--isimip-version"
    help = "ISIMIP version (2b or 3b)"
    default = "3b"
    "--hydro-model"
    help = "Hydro model name"
    default = "CWatM"
    "--climate-model", "-m"
    help = "Climate model name"
    default = "gfdl-esm4"
    "--scenario", "-s"
    help = "Climate scenario"
    default = "ssp126"
    "--region", "-r"
    help = "Regional configuration (R11, R12)"
    default = "R12"
    "--iso3"
    help = "ISO3 country code"
    default = "ZMB"
    "--data-period"
    help = "Data period (historical or future)"
    default = "historical"
    "--temporal-resolution"
    help = "Temporal resolution (monthly or daily)"
    default = "monthly"
    "--spatial-method"
    help = "Spatial aggregation method (sum or mean)"
    default = "sum"
    "--input-dir", "-i"
    help = "Input directory containing NetCDF files"
    default = "\${WATXENE_DATA_PATH}/ISIMIP/ISIMIP3b/OutputData"
    "--area-file", "-a"
    help = "Areas for grid cells"
    default = "\${ENE_MODEL_DATA_PATH}/NEST/Hydrology/landareamaskmap0.nc"
    "--basin-shapefile", "-b"
    help = "Basin shapefile directory"
    default = "\${BASIN_SHAPEFILE_PATH}/basins_delineated/basins_by_region_simpl_R12.shp"
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
      hydro_model=get(config_dict, "hydro_model", args["hydro-model"]),
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
      hydro_model=args["hydro-model"],
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

  # Run the pipeline (pass test file if provided)
  test_file = !isempty(args["test-file"]) ? expand_path(args["test-file"]) : ""
  process_hydro_data(config; test_file=test_file)
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
