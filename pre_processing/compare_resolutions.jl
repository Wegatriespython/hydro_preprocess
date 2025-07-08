#!/usr/bin/env julia

"""
Variance analysis of daily vs monthly temporal resolution across all basins and years
Both datasets have units of km³/year (annual rates)

Refactored Julia version with helper functions and parallelization over basins
"""

using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots
using Dates
using Base.Threads
using SharedArrays
using Printf
using Random

# Configuration structure
struct AnalysisConfig
  daily_file::String
  monthly_file::String
  meta_cols::Vector{String}
  output_dir::String
end

# Results structure for type safety
struct BasinVarianceResult
  basin_id::String
  basin_name::String
  year::Int
  daily_annual_variance::Float64
  monthly_annual_variance::Float64
  avg_monthly_variance_from_daily::Float64
  variance_ratio::Float64
  daily_mean::Float64
  monthly_mean::Float64
  daily_std::Float64
  monthly_std::Float64
end

# Results structure for extreme events analysis
struct BasinExtremeResult
  basin_id::String
  basin_name::String
  year::Int
  daily_flood_excess::Float64
  monthly_flood_excess::Float64
  daily_drought_excess::Float64
  monthly_drought_excess::Float64
  flood_threshold::Float64
  drought_threshold::Float64
  daily_flood_days::Int
  monthly_flood_months::Int
  daily_drought_days::Int
  monthly_drought_months::Int
end

# Helper function: Extract year from column name
function get_year_from_col(col::String)::Int
  return parse(Int, split(col, "-")[1])
end

# Helper function: Load and validate data
function load_data(config::AnalysisConfig)
  println("Loading data...")

  daily_df = CSV.read(config.daily_file, DataFrame)
  monthly_df = CSV.read(config.monthly_file, DataFrame)

  # Get time columns
  daily_time_cols = [col for col in names(daily_df) if !(col in config.meta_cols)]
  monthly_time_cols = [col for col in names(monthly_df) if !(col in config.meta_cols)]

  println("Data loaded: $(nrow(daily_df)) basins")
  println("Daily time series: $(length(daily_time_cols)) days ($(daily_time_cols[1]) to $(daily_time_cols[end]))")
  println("Monthly time series: $(length(monthly_time_cols)) months ($(monthly_time_cols[1]) to $(monthly_time_cols[end]))")

  return daily_df, monthly_df, daily_time_cols, monthly_time_cols
end

# Helper function: Get common years between datasets
function get_common_years(daily_time_cols::Vector{String}, monthly_time_cols::Vector{String})
  daily_years = Set([get_year_from_col(col) for col in daily_time_cols])
  monthly_years = Set([get_year_from_col(col) for col in monthly_time_cols])
  common_years = sort(collect(intersect(daily_years, monthly_years)))

  println("\nAnalyzing $(length(common_years)) years: $(common_years[1]) to $(common_years[end])")
  return common_years
end

# Helper function: Calculate monthly variance from daily data
function calculate_monthly_variance_from_daily(daily_row::DataFrameRow, year::Int, month::Int, daily_time_cols::Vector{String})::Float64
  month_str = @sprintf("%d-%02d", year, month)
  month_cols = [col for col in daily_time_cols if startswith(col, month_str)]

  if isempty(month_cols)
    return NaN
  end

  month_values = [daily_row[col] for col in month_cols]
  return var(month_values)
end

# Helper function: Calculate thresholds for extreme events
function calculate_extreme_thresholds(all_values::Vector{Float64})
  valid_values = filter(!isnan, all_values)
  if isempty(valid_values)
    return NaN, NaN
  end

  drought_threshold = quantile(valid_values, 0.05)  # 5th percentile for droughts
  flood_threshold = quantile(valid_values, 0.95)   # 95th percentile for floods

  return drought_threshold, flood_threshold
end

# Helper function: Calculate excess mass for extreme events
function calculate_excess_mass(values::Vector{Float64}, flood_threshold::Float64, drought_threshold::Float64)
  # Flood excess: sum of (Q - flood_threshold) for Q > flood_threshold
  flood_excess = sum(max(0, q - flood_threshold) for q in values)
  flood_days = count(q -> q > flood_threshold, values)

  # Drought excess: sum of (drought_threshold - Q) for Q < drought_threshold  
  drought_excess = sum(max(0, drought_threshold - q) for q in values)
  drought_days = count(q -> q < drought_threshold, values)

  return flood_excess, drought_excess, flood_days, drought_days
end

# Generic basin processor that can call arbitrary analysis functions
function process_basin(basin_idx::Int, daily_df::DataFrame, monthly_df::DataFrame,
  daily_time_cols::Vector{String}, monthly_time_cols::Vector{String},
  common_years::Vector{Int}, analysis_func::Function, precompute_func::Union{Function,Nothing}=nothing)

  basin_id = string(daily_df[basin_idx, :BASIN_ID])
  basin_name = string(daily_df[basin_idx, :NAME])

  # Precompute basin-wide data if needed (e.g., for thresholds)
  precomputed_data = nothing
  if precompute_func !== nothing
    precomputed_data = precompute_func(basin_idx, daily_df, monthly_df, daily_time_cols, monthly_time_cols)
  end

  results = []
  for year in common_years
    # Get columns for the year
    year_daily_cols = [col for col in daily_time_cols if startswith(col, string(year))]
    year_monthly_cols = [col for col in monthly_time_cols if startswith(col, string(year))]

    if isempty(year_daily_cols) || isempty(year_monthly_cols)
      continue
    end

    # Extract data for the year
    daily_year_values = [daily_df[basin_idx, col] for col in year_daily_cols]
    monthly_year_values = [monthly_df[basin_idx, col] for col in year_monthly_cols]

    # Call the analysis function
    result = analysis_func(basin_id, basin_name, year, daily_year_values, monthly_year_values,
      daily_df[basin_idx, :], daily_time_cols, precomputed_data)

    if result !== nothing
      push!(results, result)
    end
  end

  return results
end

# Analysis function for variance calculations
function analyze_variance(basin_id::String, basin_name::String, year::Int,
  daily_year_values::Vector{Float64}, monthly_year_values::Vector{Float64},
  basin_row::DataFrameRow, daily_time_cols::Vector{String}, precomputed_data)

  # Calculate annual variances
  daily_annual_var = var(daily_year_values)
  monthly_annual_var = var(monthly_year_values)

  # Calculate monthly variances from daily data
  monthly_vars_from_daily = Float64[]
  for month in 1:12
    month_var = calculate_monthly_variance_from_daily(basin_row, year, month, daily_time_cols)
    if !isnan(month_var)
      push!(monthly_vars_from_daily, month_var)
    end
  end

  avg_monthly_var_from_daily = isempty(monthly_vars_from_daily) ? NaN : mean(monthly_vars_from_daily)

  # Calculate variance ratio
  var_ratio = monthly_annual_var > 0 ? daily_annual_var / monthly_annual_var : NaN

  return BasinVarianceResult(
    basin_id,
    basin_name,
    year,
    daily_annual_var,
    monthly_annual_var,
    avg_monthly_var_from_daily,
    var_ratio,
    mean(daily_year_values),
    mean(monthly_year_values),
    std(daily_year_values),
    std(monthly_year_values)
  )
end

# Precompute function for extreme events analysis - calculates thresholds using annual data
function precompute_extreme_thresholds(basin_idx::Int, daily_df::DataFrame, monthly_df::DataFrame,
  daily_time_cols::Vector{String}, monthly_time_cols::Vector{String})

  # Get all annual values for the basin to calculate thresholds
  # Use annual aggregation to ensure consistent time periods for mean and median excess
  common_years = get_common_years(daily_time_cols, monthly_time_cols)

  annual_daily_values = Float64[]
  annual_monthly_values = Float64[]

  for year in common_years
    year_daily_cols = [col for col in daily_time_cols if startswith(col, string(year))]
    year_monthly_cols = [col for col in monthly_time_cols if startswith(col, string(year))]

    if !isempty(year_daily_cols) && !isempty(year_monthly_cols)
      # Calculate annual means for threshold calculation
      daily_annual_mean = mean([daily_df[basin_idx, col] for col in year_daily_cols])
      monthly_annual_mean = mean([monthly_df[basin_idx, col] for col in year_monthly_cols])

      push!(annual_daily_values, daily_annual_mean)
      push!(annual_monthly_values, monthly_annual_mean)
    end
  end

  # Calculate thresholds from annual data
  daily_drought_thresh, daily_flood_thresh = calculate_extreme_thresholds(annual_daily_values)
  monthly_drought_thresh, monthly_flood_thresh = calculate_extreme_thresholds(annual_monthly_values)

  return (daily_drought_thresh, daily_flood_thresh, monthly_drought_thresh, monthly_flood_thresh)
end

# Analysis function for extreme events calculations
function analyze_extreme_events(basin_id::String, basin_name::String, year::Int,
  daily_year_values::Vector{Float64}, monthly_year_values::Vector{Float64},
  basin_row::DataFrameRow, daily_time_cols::Vector{String}, precomputed_data)

  if precomputed_data === nothing
    return nothing
  end

  daily_drought_thresh, daily_flood_thresh, monthly_drought_thresh, monthly_flood_thresh = precomputed_data

  # Calculate extreme event metrics at native frequency then aggregate
  daily_flood_excess, daily_drought_excess, daily_flood_days, daily_drought_days =
    calculate_excess_mass(daily_year_values, daily_flood_thresh, daily_drought_thresh)

  monthly_flood_excess, monthly_drought_excess, monthly_flood_months, monthly_drought_months =
    calculate_excess_mass(monthly_year_values, monthly_flood_thresh, monthly_drought_thresh)

  return BasinExtremeResult(
    basin_id,
    basin_name,
    year,
    daily_flood_excess,
    monthly_flood_excess,
    daily_drought_excess,
    monthly_drought_excess,
    daily_flood_thresh,
    daily_drought_thresh,
    daily_flood_days,
    monthly_flood_months,
    daily_drought_days,
    monthly_drought_months
  )
end

# Helper function: Process a single basin for extreme events analysis (backwards compatibility)
function process_basin_extremes(basin_idx::Int, daily_df::DataFrame, monthly_df::DataFrame,
  daily_time_cols::Vector{String}, monthly_time_cols::Vector{String},
  common_years::Vector{Int})::Vector{BasinExtremeResult}

  return process_basin(basin_idx, daily_df, monthly_df, daily_time_cols,
    monthly_time_cols, common_years, analyze_extreme_events,
    precompute_extreme_thresholds)
end

# Generic parallelized analysis function
function calculate_analysis_parallel(daily_df::DataFrame, monthly_df::DataFrame,
  daily_time_cols::Vector{String}, monthly_time_cols::Vector{String},
  analysis_func::Function, precompute_func::Union{Function,Nothing},
  result_to_dataframe_func::Function, analysis_name::String)::DataFrame

  common_years = get_common_years(daily_time_cols, monthly_time_cols)
  n_basins = nrow(daily_df)

  println("$(analysis_name) using $(nthreads()) threads for parallel processing...")

  # Parallel processing over basins
  all_results = []

  @threads for basin_idx in 1:n_basins
    if basin_idx % 100 == 0
      basin_name = daily_df[basin_idx, :NAME]
      println("Processing $(analysis_name) for basin $basin_idx/$n_basins: $basin_name")
    end

    basin_results = process_basin(basin_idx, daily_df, monthly_df, daily_time_cols,
      monthly_time_cols, common_years, analysis_func, precompute_func)
    push!(all_results, basin_results)
  end

  # Flatten results
  flattened_results = vcat(all_results...)

  # Convert to DataFrame using the provided function
  return result_to_dataframe_func(flattened_results)
end

# Convert variance results to DataFrame
function variance_results_to_dataframe(results::Vector)::DataFrame
  return DataFrame(
    basin_id=[r.basin_id for r in results],
    basin_name=[r.basin_name for r in results],
    year=[r.year for r in results],
    daily_annual_variance=[r.daily_annual_variance for r in results],
    monthly_annual_variance=[r.monthly_annual_variance for r in results],
    avg_monthly_variance_from_daily=[r.avg_monthly_variance_from_daily for r in results],
    variance_ratio=[r.variance_ratio for r in results],
    daily_mean=[r.daily_mean for r in results],
    monthly_mean=[r.monthly_mean for r in results],
    daily_std=[r.daily_std for r in results],
    monthly_std=[r.monthly_std for r in results]
  )
end

# Convert extreme results to DataFrame
function extreme_results_to_dataframe(results::Vector)::DataFrame
  return DataFrame(
    basin_id=[r.basin_id for r in results],
    basin_name=[r.basin_name for r in results],
    year=[r.year for r in results],
    daily_flood_excess=[r.daily_flood_excess for r in results],
    monthly_flood_excess=[r.monthly_flood_excess for r in results],
    daily_drought_excess=[r.daily_drought_excess for r in results],
    monthly_drought_excess=[r.monthly_drought_excess for r in results],
    flood_threshold=[r.flood_threshold for r in results],
    drought_threshold=[r.drought_threshold for r in results],
    daily_flood_days=[r.daily_flood_days for r in results],
    monthly_flood_months=[r.monthly_flood_months for r in results],
    daily_drought_days=[r.daily_drought_days for r in results],
    monthly_drought_months=[r.monthly_drought_months for r in results]
  )
end


# Helper function: Clean data for analysis
function clean_data_for_analysis(results_df::DataFrame)::DataFrame
  # Remove rows with NaN or infinite variance ratios
  clean_df = filter(row -> !isnan(row.variance_ratio) && isfinite(row.variance_ratio), results_df)
  return clean_df
end

# Analysis function: Generate summary statistics
function analyze_variance_patterns(results_df::DataFrame)
  println("\n=== VARIANCE ANALYSIS RESULTS ===")
  println("Total basin-years analyzed: $(nrow(results_df))")

  clean_df = clean_data_for_analysis(results_df)
  println("Valid variance ratios: $(nrow(clean_df))")

  if nrow(clean_df) == 0
    println("No valid variance ratios found!")
    return
  end

  # Overall statistics
  println("\n=== VARIANCE RATIO STATISTICS ===")
  variance_ratios = clean_df.variance_ratio
  @printf("Mean variance ratio (disaggregated/aggregated): %.4f\n", mean(variance_ratios))
  @printf("Median variance ratio: %.4f\n", median(variance_ratios))
  @printf("Std variance ratio: %.4f\n", std(variance_ratios))
  @printf("Min variance ratio: %.4f\n", minimum(variance_ratios))
  @printf("Max variance ratio: %.4f\n", maximum(variance_ratios))

  # Quantiles
  quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
  println("\n=== VARIANCE RATIO QUANTILES ===")
  for q in quantiles
    val = quantile(variance_ratios, q)
    @printf("%3.0f%%: %.4f\n", q * 100, val)
  end

  # Analysis by year
  println("\n=== VARIANCE BY YEAR ===")
  year_stats = combine(groupby(clean_df, :year),
    :variance_ratio => mean => :mean_variance_ratio,
    :variance_ratio => median => :median_variance_ratio,
    :variance_ratio => std => :std_variance_ratio,
    :variance_ratio => length => :count,
    :daily_annual_variance => mean => :mean_disaggregated_variance,
    :monthly_annual_variance => mean => :mean_aggregated_variance)

  println(first(year_stats, 10))

  # High variance ratio cases
  high_threshold = quantile(variance_ratios, 0.95)
  high_var_cases = filter(row -> row.variance_ratio > high_threshold, clean_df)
  println("\n=== HIGH VARIANCE RATIO CASES (top 5%) ===")
  println("Number of cases: $(nrow(high_var_cases))")

  if nrow(high_var_cases) > 0
    println("Sample cases:")
    top_cases = sort(high_var_cases, :variance_ratio, rev=true)[1:min(5, nrow(high_var_cases)),
      [:basin_name, :year, :variance_ratio, :daily_annual_variance, :monthly_annual_variance]]
    println(top_cases)
  end

  # Low variance ratio cases  
  low_threshold = quantile(variance_ratios, 0.05)
  low_var_cases = filter(row -> row.variance_ratio < low_threshold, clean_df)
  println("\n=== LOW VARIANCE RATIO CASES (bottom 5%) ===")
  println("Number of cases: $(nrow(low_var_cases))")

  if nrow(low_var_cases) > 0
    println("Sample cases:")
    bottom_cases = sort(low_var_cases, :variance_ratio)[1:min(5, nrow(low_var_cases)),
      [:basin_name, :year, :variance_ratio, :daily_annual_variance, :monthly_annual_variance]]
    println(bottom_cases)
  end

  # Basin-level analysis
  println("\n=== BASIN-LEVEL VARIANCE PATTERNS ===")
  basin_stats = combine(groupby(clean_df, :basin_name),
    :variance_ratio => mean => :mean_variance_ratio,
    :variance_ratio => std => :std_variance_ratio,
    :daily_annual_variance => mean => :mean_disaggregated_variance,
    :monthly_annual_variance => mean => :mean_aggregated_variance)

  # Top 5 basins with highest average variance ratio
  top_basins = sort(basin_stats, :mean_variance_ratio, rev=true)[1:min(5, nrow(basin_stats)), :]
  println("Top 5 basins with highest average variance ratio:")
  println(top_basins)
end

# Analysis function: Generate extreme events statistics
function analyze_extreme_events_patterns(extreme_results_df::DataFrame)
  println("\n=== EXTREME EVENTS ANALYSIS RESULTS ===")
  println("Total basin-years analyzed: $(nrow(extreme_results_df))")

  # Filter out invalid results
  valid_df = filter(row -> !isnan(row.daily_flood_excess) && !isnan(row.monthly_flood_excess) &&
                             !isnan(row.daily_drought_excess) && !isnan(row.monthly_drought_excess), extreme_results_df)
  println("Valid extreme event calculations: $(nrow(valid_df))")

  if nrow(valid_df) == 0
    println("No valid extreme event data found!")
    return
  end

  # Overall flood statistics
  println("\n=== FLOOD EXCESS MASS STATISTICS ===")
  @printf("Mean disaggregated (daily) flood excess: %.4f km³/year\n", mean(valid_df.daily_flood_excess))
  @printf("Mean aggregated (monthly) flood excess: %.4f km³/year\n", mean(valid_df.monthly_flood_excess))
  @printf("Median disaggregated (daily) flood excess: %.4f km³/year\n", median(valid_df.daily_flood_excess))
  @printf("Median aggregated (monthly) flood excess: %.4f km³/year\n", median(valid_df.monthly_flood_excess))

  # Calculate flood excess ratio (disaggregated/aggregated)
  valid_flood_df = filter(row -> row.monthly_flood_excess > 0, valid_df)
  if nrow(valid_flood_df) > 0
    flood_ratios = valid_flood_df.daily_flood_excess ./ valid_flood_df.monthly_flood_excess
    @printf("Mean flood excess ratio (disaggregated/aggregated): %.4f\n", mean(flood_ratios))
    @printf("Median flood excess ratio: %.4f\n", median(flood_ratios))
    @printf("Std flood excess ratio: %.4f\n", std(flood_ratios))
  end

  # Overall drought statistics
  println("\n=== DROUGHT EXCESS MASS STATISTICS ===")
  @printf("Mean disaggregated (daily) drought excess: %.4f km³/year\n", mean(valid_df.daily_drought_excess))
  @printf("Mean aggregated (monthly) drought excess: %.4f km³/year\n", mean(valid_df.monthly_drought_excess))
  @printf("Median disaggregated (daily) drought excess: %.4f km³/year\n", median(valid_df.daily_drought_excess))
  @printf("Median aggregated (monthly) drought excess: %.4f km³/year\n", median(valid_df.monthly_drought_excess))

  # Calculate drought excess ratio (disaggregated/aggregated)
  valid_drought_df = filter(row -> row.monthly_drought_excess > 0, valid_df)
  if nrow(valid_drought_df) > 0
    drought_ratios = valid_drought_df.daily_drought_excess ./ valid_drought_df.monthly_drought_excess
    @printf("Mean drought excess ratio (disaggregated/aggregated): %.4f\n", mean(drought_ratios))
    @printf("Median drought excess ratio: %.4f\n", median(drought_ratios))
    @printf("Std drought excess ratio: %.4f\n", std(drought_ratios))
  end

  # Frequency statistics
  println("\n=== EVENT FREQUENCY STATISTICS ===")
  @printf("Mean disaggregated flood days per year: %.2f\n", mean(valid_df.daily_flood_days))
  @printf("Mean aggregated flood months per year: %.2f\n", mean(valid_df.monthly_flood_months))
  @printf("Mean disaggregated drought days per year: %.2f\n", mean(valid_df.daily_drought_days))
  @printf("Mean aggregated drought months per year: %.2f\n", mean(valid_df.monthly_drought_months))

  # Analysis by year
  println("\n=== EXTREME EVENTS BY YEAR ===")
  year_stats = combine(groupby(valid_df, :year),
    :daily_flood_excess => mean => :mean_disaggregated_flood,
    :monthly_flood_excess => mean => :mean_aggregated_flood,
    :daily_drought_excess => mean => :mean_disaggregated_drought,
    :monthly_drought_excess => mean => :mean_aggregated_drought,
    :daily_flood_days => mean => :mean_flood_days,
    :daily_drought_days => mean => :mean_drought_days)

  println("Sample yearly statistics:")
  println(first(year_stats, 5))

  # Top extreme cases
  println("\n=== TOP FLOOD EXCESS CASES ===")
  top_floods = sort(valid_df, :daily_flood_excess, rev=true)[1:min(5, nrow(valid_df)),
    [:basin_name, :year, :daily_flood_excess, :monthly_flood_excess, :daily_flood_days]]
  println(top_floods)

  println("\n=== TOP DROUGHT EXCESS CASES ===")
  top_droughts = sort(valid_df, :daily_drought_excess, rev=true)[1:min(5, nrow(valid_df)),
    [:basin_name, :year, :daily_drought_excess, :monthly_drought_excess, :daily_drought_days]]
  println(top_droughts)

  # Basin-level patterns
  println("\n=== BASIN-LEVEL EXTREME PATTERNS ===")
  basin_stats = combine(groupby(valid_df, :basin_name),
    :daily_flood_excess => mean => :mean_disaggregated_flood,
    :monthly_flood_excess => mean => :mean_aggregated_flood,
    :daily_drought_excess => mean => :mean_disaggregated_drought,
    :monthly_drought_excess => mean => :mean_aggregated_drought)

  # Basins with highest average flood excess
  top_flood_basins = sort(basin_stats, :mean_disaggregated_flood, rev=true)[1:min(5, nrow(basin_stats)), :]
  println("Top 5 basins with highest average disaggregated flood excess:")
  println(top_flood_basins)

  # Basins with highest average drought excess
  top_drought_basins = sort(basin_stats, :mean_disaggregated_drought, rev=true)[1:min(5, nrow(basin_stats)), :]
  println("\nTop 5 basins with highest average disaggregated drought excess:")
  println(top_drought_basins)
end

# Plotting function
function create_summary_plots(results_df::DataFrame, output_dir::String)
  clean_df = clean_data_for_analysis(results_df)

  if nrow(clean_df) == 0
    println("No valid data for plotting!")
    return
  end

  # Create plots
  p1 = histogram(clean_df.variance_ratio, bins=50, alpha=0.7,
    title="Distribution of Variance Ratios",
    xlabel="Variance Ratio (Daily/Monthly)", ylabel="Frequency")
  vline!(p1, [mean(clean_df.variance_ratio)], label="Mean", color=:red, linestyle=:dash)
  vline!(p1, [median(clean_df.variance_ratio)], label="Median", color=:orange, linestyle=:dash)

  # Variance ratio by year
  year_means = combine(groupby(clean_df, :year), :variance_ratio => mean => :mean_ratio)
  p2 = plot(year_means.year, year_means.mean_ratio, marker=:o, linewidth=2,
    title="Variance Ratio Trend Over Years",
    xlabel="Year", ylabel="Mean Variance Ratio")

  # Daily vs Monthly variance scatter (sample for readability)
  sample_size = min(1000, nrow(clean_df))
  sample_df = clean_df[randperm(nrow(clean_df))[1:sample_size], :]
  p3 = scatter(sample_df.monthly_annual_variance, sample_df.daily_annual_variance,
    alpha=0.6, title="Daily vs Monthly Variance",
    xlabel="Monthly Annual Variance", ylabel="Daily Annual Variance")
  max_val = max(maximum(sample_df.monthly_annual_variance), maximum(sample_df.daily_annual_variance))
  plot!(p3, [0, max_val], [0, max_val], color=:red, linestyle=:dash, label="y=x")

  # Boxplot by decade
  clean_df[!, :decade] = (clean_df.year .÷ 10) .* 10
  decades = sort(unique(clean_df.decade))
  decade_data = [clean_df[clean_df.decade.==d, :variance_ratio] for d in decades]
  p4 = boxplot([string(d) * "s" for d in decades], decade_data,
    title="Variance Ratio Distribution by Decade",
    xlabel="Decade", ylabel="Variance Ratio")

  # Combine plots
  combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))

  # Save plot
  output_file = joinpath(output_dir, "variance_analysis_summary.png")
  savefig(combined_plot, output_file)
  println("\nSummary plots saved as '$output_file'")
end

# Main execution function
function main()
  # Configuration
  config = AnalysisConfig(
    "/home/raghunathan/hydro_preprocess/pre_processing/hydro_output/qtot_daily_gfdl-esm4_ssp126_future.csv",
    "/home/raghunathan/hydro_preprocess/pre_processing/hydro_output/qtot_monthly_gfdl-esm4_ssp126_future.csv",
    ["BASIN_ID", "BCU_name", "NAME", "REGION", "area_km2"],
    "."
  )

  println("Starting comprehensive variance analysis...")
  println("Julia version with $(nthreads()) threads")

  # Load data
  daily_df, monthly_df, daily_time_cols, monthly_time_cols = load_data(config)

  results_df = calculate_analysis_parallel(daily_df, monthly_df, daily_time_cols, monthly_time_cols,
    analyze_variance, nothing, variance_results_to_dataframe,
    "Variance analysis")
  # Save detailed results
  output_file = joinpath(config.output_dir, "variance_analysis_results.csv")
  CSV.write(output_file, results_df)
  println("\nDetailed results saved to '$output_file'")

  # Analyze patterns
  analyze_variance_patterns(results_df)


  extreme_results_df = calculate_analysis_parallel(daily_df, monthly_df, daily_time_cols, monthly_time_cols,
    analyze_extreme_events, precompute_extreme_thresholds,
    extreme_results_to_dataframe, "Extreme events analysis")


  # Save extreme events results
  extreme_output_file = joinpath(config.output_dir, "extreme_events_analysis_results.csv")
  CSV.write(extreme_output_file, extreme_results_df)
  println("\nExtreme events results saved to '$extreme_output_file'")

  # Analyze extreme events patterns
  analyze_extreme_events_patterns(extreme_results_df)

  # Create summary plots (variance only, no plots for extremes as requested)
  create_summary_plots(results_df, config.output_dir)

  println("\n=== ANALYSIS COMPLETE ===")
  println("The analysis quantifies temporal aggregation loss by comparing:")
  println("1. Variance differences between disaggregated (daily) and aggregated (monthly) data")
  println("2. Extreme events excess mass at both temporal resolutions")
  println("   - Floods: 95th percentile threshold")
  println("   - Droughts: 5th percentile threshold")
  println("   - Excess mass: ∑max[0, Q-threshold] for floods, ∑max[0, threshold-Q] for droughts")
  println("Higher ratios (disaggregated/aggregated) indicate information loss from temporal aggregation.")

  return results_df, extreme_results_df
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
