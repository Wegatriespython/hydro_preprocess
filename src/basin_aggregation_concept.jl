"""
Core basin aggregation functions for hydrological data processing.
Shared by both production and test code to avoid duplication.
"""

using NetCDF, DataFrames, CSV, Dates, Statistics
using ArchGDAL, GeoDataFrames, Rasters

"""
Load and clean hydrological data from NetCDF file.
Returns cleaned data slice with extreme values replaced by NaN.
"""
function load_data(netcdf_file::String, variable::String="qtot", 
                  start_indices::Vector{Int}=[1, 1, 1], 
                  count_indices::Union{Vector{Int}, Nothing}=nothing)
    
    # Read coordinate variables
    lons = NetCDF.ncread(netcdf_file, "lon")
    lats = NetCDF.ncread(netcdf_file, "lat")
    
    # Determine count if not provided
    if count_indices === nothing
        count_indices = [length(lons), length(lats), 12]  # Default to 12 timesteps
    end
    
    # Read data slice
    data_slice = NetCDF.ncread(netcdf_file, variable, 
                              start=start_indices, 
                              count=count_indices)
    
    # Clean extreme values (NetCDF missing data flags like 1e20)
    data_slice[abs.(data_slice) .>= 1e15] .= NaN
    
    return data_slice, lons, lats
end

"""
Load basin polygons from shapefile.
Returns GeoDataFrame with basin geometries and metadata.
"""
function load_basins(basin_file::String, n_basins::Union{Int, Nothing}=nothing)
    basins_gdf = GeoDataFrames.read(basin_file)
    
    if n_basins !== nothing
        basins_gdf = basins_gdf[1:n_basins, :]
    end
    
    return basins_gdf
end

"""
Perform zonal aggregation of raster data over basin polygons.
Returns aggregated values for each basin and timestep.
"""
function aggregate_basins(data_slice::Array, lons::Vector, lats::Vector, 
                         basins_gdf::DataFrame, agg_method::String="sum")
    
    # Create coordinate ranges for Rasters.jl
    x_range = range(minimum(lons), maximum(lons), length=length(lons))
    y_range = range(maximum(lats), minimum(lats), length=length(lats))
    
    # Determine aggregation function
    agg_func = agg_method == "sum" ? sum : mean
    
    # Get dimensions
    n_timesteps = size(data_slice, 3)
    n_basins = nrow(basins_gdf)
    results = zeros(Float64, n_basins, n_timesteps)
    
    # Aggregate each timestep
    for t in 1:n_timesteps
        spatial_slice = data_slice[:, :, t]
        raster_obj = Raster(spatial_slice, (X(x_range), Y(y_range)))
        basin_values = Rasters.zonal(agg_func, raster_obj; of=basins_gdf)
        results[:, t] = basin_values
    end
    
    return results
end

"""
Format aggregated data as DataFrame with proper column names.
"""
function format_output(results::Matrix, basins_gdf::DataFrame, 
                      timestep_names::Union{Vector{String}, Nothing}=nothing)
    
    n_basins, n_timesteps = size(results)
    
    # Create output DataFrame
    output_df = DataFrame()
    output_df.BASIN_ID = basins_gdf.BASIN_ID
    output_df.BCU_name = basins_gdf.BCU_name
    
    # Add timestep columns
    for i in 1:n_timesteps
        if timestep_names !== nothing && i <= length(timestep_names)
            col_name = timestep_names[i]
        else
            col_name = "timestep_$(i)"
        end
        output_df[!, col_name] = results[:, i]
    end
    
    return output_df
end

"""
Complete processing pipeline: load data, aggregate, and format output.
"""
function process_basin_aggregation(netcdf_file::String, basin_file::String;
                                  variable::String="qtot",
                                  n_basins::Union{Int, Nothing}=nothing,
                                  n_timesteps::Int=12,
                                  agg_method::String="sum")
    
    # Load data
    data_slice, lons, lats = load_data(netcdf_file, variable, [1, 1, 1], 
                                      [length(NetCDF.ncread(netcdf_file, "lon")), 
                                       length(NetCDF.ncread(netcdf_file, "lat")), 
                                       n_timesteps])
    
    # Load basins
    basins_gdf = load_basins(basin_file, n_basins)
    
    # Aggregate
    results = aggregate_basins(data_slice, lons, lats, basins_gdf, agg_method)
    
    # Format output
    output_df = format_output(results, basins_gdf)
    
    return output_df
end
