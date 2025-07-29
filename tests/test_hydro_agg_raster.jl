
using Test
using CSV, DataFrames

# Disable verbose test output and stack traces (pytest-like behavior)  
ENV["JULIA_TEST_FAILFAST"] = "true"

# Include the script to be tested
include("../pre_processing/hydro_agg_raster.jl")

# Test configurations - matches our pre-generated test data
const TEST_CONFIGS = [
  (
    name="qtot_monthly",
    test_file="/home/raghunathan/hydro_preprocess/tests/data/qtot_monthly_test_data.nc",
    expected_file="/home/raghunathan/hydro_preprocess/tests/test_output/qtot_monthly_test_expected_output.csv",
    variable="qtot",
    temporal_resolution="monthly"
  ),
  (
    name="qr_monthly",
    test_file="/home/raghunathan/hydro_preprocess/tests/data/qr_monthly_test_data.nc",
    expected_file="/home/raghunathan/hydro_preprocess/tests/test_output/qr_monthly_test_expected_output.csv",
    variable="qr",
    temporal_resolution="monthly"
  ),
  (
    name="qtot_daily",
    test_file="/home/raghunathan/hydro_preprocess/tests/data/qtot_daily_test_data.nc",
    expected_file="/home/raghunathan/hydro_preprocess/tests/test_output/qtot_daily_test_expected_output.csv",
    variable="qtot",
    temporal_resolution="daily"
  )
]

function run_processing_test(test_config)
  """Run hydro_agg_raster processing and compare with expected output."""

  # Store current directory and change to match expected output generation
  original_dir = pwd()
  cd("../pre_processing")
  try
    # Create test configuration 
    config = ProcessingConfig(
      variable=test_config.variable,
      isimip_version="3b",
      hydro_model="CWatM",
      climate_model="gfdl-esm4",
      scenario="ssp126",
      region="R12",
      iso3="ZMB",
      data_period="future",
      temporal_resolution=test_config.temporal_resolution,
      spatial_method="sum",
      input_dir="TEST_MODE",  # Not used in test mode
      area_file="/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc",
      basin_shapefile="/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp",
      output_dir="../test_temp_output"
    )

    # Create temp output directory
    mkpath(config.output_dir)

    # Run processing with test file (expand path like CLI does)
    expanded_test_file = expanduser(normpath(test_config.test_file))
    output_files = process_hydro_data(config; test_file=expanded_test_file)

    # Load results
    test_output_df = CSV.read(output_files[1], DataFrame)
    expected_df = CSV.read(test_config.expected_file, DataFrame)

    # Compare results
    @test names(expected_df) == names(test_output_df)
    @test size(expected_df) == size(test_output_df)

    # Compare each column with cleaner error reporting, dropping NaNs
    failed_cols = String[]
    for col in names(expected_df)
      if eltype(expected_df[!, col]) <: Number
        # Drop NaNs before comparison
        expected_clean = filter(!isnan, expected_df[!, col])
        actual_clean = filter(!isnan, test_output_df[!, col])

        # Compare non-NaN values with tolerance
        if length(expected_clean) != length(actual_clean) || !all(isapprox.(expected_clean, actual_clean, rtol=1e-6))
          push!(failed_cols, col)
        end
      else
        # Exact match for strings/metadata
        if !all(expected_df[!, col] .== test_output_df[!, col])
          push!(failed_cols, "$(col) (metadata)")
        end
      end
    end

    # Single assertion to avoid multiple stack traces
    if !isempty(failed_cols)
      println("✗ $(test_config.name) test FAILED - columns differ: $(join(failed_cols, ", "))")
      @test false
    else
      println("✓ $(test_config.name) test PASSED")
      @test true
    end

  finally
    # Cleanup temp directory and restore working directory
    rm("../test_temp_output", recursive=true, force=true)
    cd(original_dir)
  end
end

@testset "Hydro Agg Raster Regression Tests" begin

  @testset "Processing Test: $(config.name)" for config in TEST_CONFIGS
    # Check that expected output file exists
    @test isfile(config.expected_file)

    # Check that test data file exists  
    @test isfile(config.test_file)

    # Run the processing test
    run_processing_test(config)
  end

end
