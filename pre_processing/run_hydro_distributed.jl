"""
Distributed runner script for hydro_agg_raster.jl using Julia's Distributed computing.

This script dispatches workers for all combinations of climate models, scenarios, 
and variables as defined in submit_hydro_jobs.sh, but using Julia's distributed 
computing instead of SLURM.
"""

using Distributed, YAML, ProgressMeter, Dates, ArgParse
using Logging, LoggingExtras

# Function to add workers with thread configuration
function add_workers_with_threads(num_workers::Int, threads_per_worker::Int=2)
  if nworkers() == 1
    println("Adding $(num_workers) workers with $(threads_per_worker) threads each...")

    # Set environment variable for new workers
    ENV["JULIA_NUM_THREADS"] = string(threads_per_worker)

    # Add workers
    addprocs(num_workers; env=["JULIA_NUM_THREADS" => string(threads_per_worker)])

    println("Added $(nworkers()) workers, each with $(threads_per_worker) threads")

    # Verify thread configuration on workers
    @everywhere println("Worker $(myid()) has $(Threads.nthreads()) threads")
  end
end

# Calculate optimal worker/thread allocation
function calculate_worker_allocation(total_cores::Int=Sys.CPU_THREADS)
  # Rule of thumb: 2-4 threads per worker for I/O bound tasks
  threads_per_worker = min(4, max(2, total_cores ÷ 8))
  num_workers = max(1, total_cores ÷ threads_per_worker)

  return num_workers, threads_per_worker
end

# Add workers with optimal configuration
num_workers, threads_per_worker = calculate_worker_allocation()
add_workers_with_threads(num_workers, threads_per_worker)

# Load the main processing script on all workers
@everywhere begin
  include("hydro_agg_raster.jl")
  # Functions are now available in Main scope after include
end

# Job configuration matching submit_hydro_jobs.sh
const CLIMATE_MODELS = ["gfdl-esm4", "gfd-cm6a-lr", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
const SCENARIOS = ["ssp126", "ssp370", "ssp585"]
const VARIABLES_MONTHLY = ["qtot", "qr"]
const VARIABLES_DAILY = ["qtot"]

"""
Job configuration structure for distributed processing.
"""
struct JobConfig
  model::String
  scenario::String
  variable::String
  temporal_resolution::String
  job_id::String

  function JobConfig(model, scenario, variable, temporal_resolution)
    job_id = "hydro_$(model)_$(scenario)_$(variable)_$(temporal_resolution)"
    new(model, scenario, variable, temporal_resolution, job_id)
  end
end

"""
Create base ProcessingConfig with common settings.
"""
function create_base_config(;
  input_dir="/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData",
  basin_shapefile="/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp",
  area_file="/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc",
  output_dir="./hydro_output",
  data_period="future",
  region="R12",
  iso3="ZMB",
  spatial_method="sum",
  isimip_version="3b"
)
  return ProcessingConfig(
    variable="qtot",  # Will be overridden
    isimip_version=isimip_version,
    climate_model="gfdl-esm4",  # Will be overridden
    scenario="ssp126",  # Will be overridden
    data_period=data_period,
    region=region,
    iso3=iso3,
    temporal_resolution="monthly",  # Will be overridden
    spatial_method=spatial_method,
    input_dir=expand_path(input_dir),
    area_file=expand_path(area_file),
    basin_shapefile=expand_path(basin_shapefile),
    output_dir=expand_path(output_dir)
  )
end

"""
Create ProcessingConfig for a specific job.
"""
function create_job_config(job::JobConfig, base_config::ProcessingConfig)
  return ProcessingConfig(
    variable=job.variable,
    isimip_version=base_config.isimip_version,
    climate_model=job.model,
    scenario=job.scenario,
    data_period=base_config.data_period,
    region=base_config.region,
    iso3=base_config.iso3,
    temporal_resolution=job.temporal_resolution,
    spatial_method=base_config.spatial_method,
    input_dir=base_config.input_dir,
    area_file=base_config.area_file,
    basin_shapefile=base_config.basin_shapefile,
    output_dir=base_config.output_dir
  )
end

# Process a single job configuration.
@everywhere function process_job(job_config::ProcessingConfig, job_id::String)
  try
    println("[$(now())] Worker $(myid()) starting job: $(job_id)")

    # Run the processing pipeline
    output_files = process_hydro_data(job_config)

    println("[$(now())] Worker $(myid()) completed job: $(job_id)")
    println("Generated $(length(output_files)) output files")

    return (job_id=job_id, status="success", output_files=output_files, worker_id=myid())

  catch e
    error_msg = "Error in job $(job_id): $(e)"
    println("[$(now())] Worker $(myid()) failed job: $(job_id)")
    println("Error: $(error_msg)")

    return (job_id=job_id, status="failed", error=error_msg, worker_id=myid())
  end
end

"""
Generate all job combinations based on the patterns in submit_hydro_jobs.sh.
"""
function generate_job_combinations()
  jobs = JobConfig[]

  # Monthly jobs for all variables
  for model in CLIMATE_MODELS
    for scenario in SCENARIOS
      for variable in VARIABLES_MONTHLY
        push!(jobs, JobConfig(model, scenario, variable, "monthly"))
      end
    end
  end

  # Daily jobs for qtot and dis only
  for model in CLIMATE_MODELS
    for scenario in SCENARIOS
      for variable in VARIABLES_DAILY
        push!(jobs, JobConfig(model, scenario, variable, "daily"))
      end
    end
  end

  return jobs
end

"""
Setup logging for distributed processing.
"""
function setup_logging(log_dir::String)
  mkpath(log_dir)

  # Create timestamped log file
  timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
  log_file = joinpath(log_dir, "hydro_distributed_$(timestamp).log")

  # Setup logger
  logger = TeeLogger(
    ConsoleLogger(stderr, Logging.Info),
    FileLogger(log_file)
  )

  global_logger(logger)

  return log_file
end

"""
Print job statistics and summary.
"""
function print_job_summary(jobs::Vector{JobConfig})
  println("\n" * "="^60)
  println("JOB SUMMARY")
  println("="^60)

  # Count by temporal resolution
  monthly_count = count(j -> j.temporal_resolution == "monthly", jobs)
  daily_count = count(j -> j.temporal_resolution == "daily", jobs)

  println("Total jobs: $(length(jobs))")
  println("  Monthly jobs: $(monthly_count)")
  println("  Daily jobs: $(daily_count)")

  # Count by variable
  println("\nJobs by variable:")
  for var in unique([j.variable for j in jobs])
    count_var = count(j -> j.variable == var, jobs)
    println("  $(var): $(count_var)")
  end

  # Count by scenario
  println("\nJobs by scenario:")
  for scenario in unique([j.scenario for j in jobs])
    count_scenario = count(j -> j.scenario == scenario, jobs)
    println("  $(scenario): $(count_scenario)")
  end

  println("\nUsing $(nworkers()) workers on $(nprocs()) processes")
  println("="^60)
end

"""
Main distributed processing function.
"""
function run_distributed_processing(;
  config_file::String="",
  max_concurrent_jobs::Int=nworkers(),
  log_dir::String="./logs",
  dry_run::Bool=false
)
  # Setup logging
  log_file = setup_logging(log_dir)
  @info "Starting distributed hydro processing" log_file = log_file

  # Load base configuration
  base_config = if !isempty(config_file) && isfile(config_file)
    @info "Loading configuration from: $(config_file)"
    config_dict = YAML.load_file(config_file)
    create_base_config(;
      input_dir=get(config_dict, "input_dir", "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData"),
      basin_shapefile=get(config_dict, "basin_shapefile", "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"),
      area_file=get(config_dict, "area_file", "/mnt/p/ene.model/NEST/Hydrology/landareamaskmap0.nc"),
      output_dir=get(config_dict, "output_dir", "./hydro_output"),
      data_period=get(config_dict, "data_period", "future"),
      region=get(config_dict, "region", "R12"),
      spatial_method=get(config_dict, "spatial_method", "sum"),
      isimip_version=get(config_dict, "isimip_version", "3b")
    )
  else
    @info "Using default configuration"
    create_base_config()
  end

  # Create output directory
  mkpath(base_config.output_dir)

  # Generate job combinations
  jobs = generate_job_combinations()
  print_job_summary(jobs)

  if dry_run
    @info "Dry run mode - no jobs will be executed"
    return
  end

  # Process jobs in parallel with progress tracking
  @info "Starting parallel processing of $(length(jobs)) jobs"

  # Use pmap for distributed processing with progress tracking
  results = Vector{Any}(undef, length(jobs))

  # Process jobs in batches to avoid overwhelming workers
  batch_size = max_concurrent_jobs
  n_batches = ceil(Int, length(jobs) / batch_size)

  @showprogress "Processing job batches..." for batch_idx in 1:n_batches
    start_idx = (batch_idx - 1) * batch_size + 1
    end_idx = min(batch_idx * batch_size, length(jobs))
    batch_jobs = jobs[start_idx:end_idx]

    # Process batch in parallel
    batch_results = pmap(batch_jobs) do job
      job_config = create_job_config(job, base_config)
      process_job(job_config, job.job_id)
    end

    # Store results
    results[start_idx:end_idx] = batch_results

    # Print batch summary
    batch_success = count(r -> r.status == "success", batch_results)
    batch_failed = count(r -> r.status == "failed", batch_results)
    @info "Batch $(batch_idx)/$(n_batches) completed: $(batch_success) success, $(batch_failed) failed"
  end

  # Print final summary
  successful_jobs = count(r -> r.status == "success", results)
  failed_jobs = count(r -> r.status == "failed", results)

  println("\n" * "="^60)
  println("PROCESSING COMPLETE")
  println("="^60)
  println("Total jobs: $(length(jobs))")
  println("Successful: $(successful_jobs)")
  println("Failed: $(failed_jobs)")
  println("Success rate: $(round(successful_jobs/length(jobs)*100, digits=1))%")

  # List failed jobs
  if failed_jobs > 0
    println("\nFailed jobs:")
    for result in results
      if result.status == "failed"
        println("  - $(result.job_id): $(result.error)")
      end
    end
  end

  # Count total output files
  total_outputs = sum(length(r.output_files) for r in results if r.status == "success")
  println("\nTotal output files generated: $(total_outputs)")
  println("Output directory: $(base_config.output_dir)")
  println("Log file: $(log_file)")
  println("="^60)

  return results
end

"""
Command-line interface for distributed processing.
"""
function main()
  s = ArgParseSettings(description="Distributed runner for hydro_agg_raster.jl")

  @add_arg_table! s begin
    "--config", "-c"
    help = "YAML configuration file"
    default = ""

    "--max-concurrent-jobs", "-j"
    help = "Maximum number of concurrent jobs"
    arg_type = Int
    default = nworkers()

    "--log-dir", "-l"
    help = "Directory for log files"
    default = "./logs"

    "--dry-run", "-n"
    help = "Print job summary without executing"
    action = :store_true

    "--workers", "-w"
    help = "Number of worker processes to add"
    arg_type = Int
    default = 0

    "--threads-per-worker", "-t"
    help = "Number of threads per worker process"
    arg_type = Int
    default = 2
  end

  args = parse_args(s)

  # Add additional workers if requested
  if args["workers"] > 0
    current_workers = nworkers()
    if current_workers < args["workers"]
      additional_workers = args["workers"] - current_workers
      println("Adding $(additional_workers) additional workers with $(args["threads-per-worker"]) threads each...")
      addprocs(additional_workers; env=["JULIA_NUM_THREADS" => string(args["threads-per-worker"])])

      # Verify thread configuration on new workers
      @everywhere println("Worker $(myid()) has $(Threads.nthreads()) threads")
    end
  end

  # Run distributed processing
  results = run_distributed_processing(
    config_file=args["config"],
    max_concurrent_jobs=args["max-concurrent-jobs"],
    log_dir=args["log-dir"],
    dry_run=args["dry-run"]
  )

  # Exit with error code if any jobs failed
  if !args["dry-run"] && any(r -> r.status == "failed", results)
    exit(1)
  end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
