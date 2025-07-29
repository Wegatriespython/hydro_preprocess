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


