#!/usr/bin/env python3
"""
Data Preparation Module for Hydro Model Output Analysis
=======================================================

This module handles data loading, cleaning, reshaping, and preparation
for downstream analysis (ANOVA, etc.). Designed to be reusable and configurable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import warnings
import gc
import sys

warnings.filterwarnings("ignore")


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract components.

    Expected format: {variable}_{frequency}_{hydro_model}_{climate_model}_{scenario}_{time_period}.csv

    Args:
        filename: Name of the CSV file

    Returns:
        Dict with components or None if parsing fails
    """
    if not filename.endswith(".csv"):
        return None

    # Remove .csv extension
    basename = filename[:-4]

    # Split by underscore
    parts = basename.split("_")

    if len(parts) < 6:
        return None

    # Handle cases where hydro model names have hyphens (e.g., MIROC-INTEG-LAND)
    # We know the structure is: variable_frequency_hydro_model_climate_model_scenario_timeperiod
    # Climate models and scenarios are known patterns

    variable = parts[0]
    frequency = parts[1]

    # Find the climate model and scenario by looking for known patterns
    climate_models = [
        "gfdl-esm4",
        "ipsl-cm6a-lr",
        "mpi-esm1-2-hr",
        "mri-esm2-0",
        "ukesm1-0-ll",
    ]
    ssp_scenarios = ["historical"]

    climate_model_idx = None
    scenario_idx = None

    # Find climate model position
    for i, part in enumerate(parts):
        if part.lower() in climate_models:
            climate_model_idx = i
            break

    # Find scenario position
    for i, part in enumerate(parts):
        if part.lower() in ssp_scenarios:
            scenario_idx = i
            break

    if climate_model_idx is None or scenario_idx is None:
        return None

    # Extract components
    climate_model = parts[climate_model_idx]
    scenario = parts[scenario_idx]
    time_period = parts[-1]  # Last part is always time period

    # Hydro model is everything between frequency and climate model
    hydro_model_parts = parts[2:climate_model_idx]
    hydro_model = "-".join(hydro_model_parts)

    return {
        "variable": variable,
        "frequency": frequency,
        "hydro_model": hydro_model,
        "climate_model": climate_model,
        "scenario": scenario,
        "time_period": time_period,
    }


def discover_available_models(
    data_dir: Union[str, Path], variable: str, frequency: str
) -> Tuple[List[str], List[str]]:
    """
    Auto-discover available hydro and climate models from directory.

    Args:
        data_dir: Directory containing CSV files
        variable: Variable to look for
        frequency: Frequency to look for

    Returns:
        Tuple of (hydro_models, climate_models) found
    """
    data_dir = Path(data_dir)

    hydro_models = set()
    climate_models = set()

    # Look for files matching the pattern
    for file_path in data_dir.glob(f"{variable}_{frequency}_*.csv"):
        parsed = parse_filename(file_path.name)
        if parsed:
            hydro_models.add(parsed["hydro_model"])
            climate_models.add(parsed["climate_model"])

    return sorted(list(hydro_models)), sorted(list(climate_models))


def load_var_data(
    data_dir: Union[str, Path],
    variable: str,
    frequency: str,
    ssp_scenarios: List[str],
    time_period: str,
    hydro_models: Optional[List[str]] = None,
    climate_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all variable CSV files and merge them into a single dataframe.

    Args:
        data_dir: Directory containing the CSV files
        variable: Variable name (default: "qtot")
        frequency: Frequency ("daily" or "monthly")
        hydro_models: List of hydro models (auto-discover if None)
        climate_models: List of climate models (auto-discover if None)
        ssp_scenarios: List of SSP scenarios (default: ssp126, ssp370, ssp585)
        time_period: Time period (default: "future")

    Returns:
        pd.DataFrame: Combined dataset with values for all model/scenario combinations
    """
    data_dir = Path(data_dir)

    # Auto-discover models if not specified
    if hydro_models is None or climate_models is None:
        discovered_hydro, discovered_climate = discover_available_models(
            data_dir, variable, frequency
        )
        if hydro_models is None:
            hydro_models = discovered_hydro
        if climate_models is None:
            climate_models = discovered_climate

    if ssp_scenarios is None:
        ssp_scenarios = ["historical"]

    print(f"Loading {variable} {frequency} data files from {data_dir}...")
    print(f"  Hydro models: {hydro_models}")
    print(f"  Climate models: {climate_models}")

    all_data = []
    file_count = 0

    for hydro_model in hydro_models:
        for climate_model in climate_models:
            for scenario in ssp_scenarios:
                filename = f"{variable}_{frequency}_{hydro_model}_{climate_model}_{scenario}_{time_period}.csv"
                filepath = data_dir / filename

                if not filepath.exists():
                    print(f"Warning: File not found: {filename}")
                    continue

                print(f"  Loading: {filename}")

                df = pd.read_csv(filepath)
                df["hydro_model"] = hydro_model
                df["climate_model"] = climate_model
                df["ssp_scenario"] = scenario

                all_data.append(df)
                file_count += 1

                # Force garbage collection every 10 files to manage memory
                if file_count % 10 == 0:
                    gc.collect()
                    print(f"    Loaded {file_count} files, memory cleaned")

    if not all_data:
        raise FileNotFoundError(
            f"No {variable} {frequency} data files found in {data_dir}!"
        )

    print(f"Concatenating {len(all_data)} files...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Clean up individual dataframes
    del all_data
    gc.collect()

    print(f"Loaded {file_count} files with {len(combined_df)} total rows")

    return combined_df


def reshape_to_long_format(
    df: pd.DataFrame,
    value_name: str,
    id_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reshape from wide format (months as columns) to long format.

    Args:
        df: Wide format dataframe
        value_name: Name for the value column in long format
        id_columns: List of columns to keep as identifiers

    Returns:
        pd.DataFrame: Long format dataframe
    """
    print("Reshaping data to long format...")

    date_columns = [col for col in df.columns if "-" in col and len(col) == 10]
    date_columns = sorted(date_columns)
    print(
        f"Found {len(date_columns)} date columns from {date_columns[0]} to {date_columns[-1]}"
    )

    if id_columns is None:
        id_columns = [
            "BASIN_ID",
            "BCU_name",
            "NAME",
            "REGION",
            "area_km2",
            "hydro_model",
            "climate_model",
            "ssp_scenario",
        ]

    available_id_columns = [col for col in id_columns if col in df.columns]

    # Process in chunks to avoid memory issues
    chunk_size = 1000
    chunk_results = []
    num_chunks = (len(date_columns) + chunk_size - 1) // chunk_size
    print(
        f"Processing {len(date_columns)} date columns in {num_chunks} chunks of {chunk_size}"
    )

    try:
        for i in range(0, len(date_columns), chunk_size):
            chunk_end = min(i + chunk_size, len(date_columns))
            chunk_date_cols = date_columns[i:chunk_end]
            chunk_num = (i // chunk_size) + 1

            print(
                f"Processing chunk {chunk_num}/{num_chunks} ({len(chunk_date_cols)} columns)..."
            )

            # Create subset with ID columns and current chunk of date columns
            chunk_cols = available_id_columns + chunk_date_cols
            chunk_df = df[chunk_cols].copy()

            # Melt this chunk
            long_chunk = pd.melt(
                chunk_df,
                id_vars=available_id_columns,
                value_vars=chunk_date_cols,
                var_name="date",
                value_name=value_name,
            )

            # Process dates for this chunk
            long_chunk["date"] = pd.to_datetime(long_chunk["date"])
            long_chunk["year"] = long_chunk["date"].dt.year
            long_chunk["month"] = long_chunk["date"].dt.month

            # Remove NAs
            long_chunk = long_chunk.dropna(subset=[value_name])

            chunk_results.append(long_chunk)

            # Clean up
            del chunk_df, long_chunk
            gc.collect()

            print(f"Chunk {chunk_num} completed")

        print("Combining all chunks...")
        long_df = pd.concat(chunk_results, ignore_index=True)

        # Clean up chunk results
        del chunk_results
        gc.collect()

        print(f"Final dataset: {len(long_df)} rows")
        return long_df

    except Exception as e:
        print(f"ERROR in reshape_to_long_format: {e}")
        print(f"Available ID columns: {available_id_columns}")
        print(f"Number of date columns: {len(date_columns)}")
        print(f"DataFrame shape: {df.shape}")
        raise


def prepare_data(
    data_dir: str | Path,
    long_gen: bool,
    long_path: str | Path,
    output_file: str | Path,
    variable: str,
    frequency: str,  # Changed default to daily
    ssp_scenarios: List[str],
    time_period: str,
    hydro_models: Optional[List[str]] = None,
    climate_models: Optional[List[str]] = None,
    rolling_windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Main data preparation pipeline.

    Args:
        data_dir: Directory containing the CSV files
        output_file: Optional path to save the prepared data
        variable: Variable name (default: "qtot")
        frequency: Data frequency ("daily" or "monthly")
        hydro_models: List of hydro models (auto-discover if None)
        climate_models: List of climate models (auto-discover if None)
        ssp_scenarios: List of SSP scenarios (default: ssp126, ssp370, ssp585)
        time_period: Time period (default: "future")
        drop_percentile: Percentile threshold for dropping low-value basins
        outlier_threshold: Upper threshold for outlier removal
        rolling_windows: Dictionary of rolling window sizes (auto-adjusted by frequency if None)

    Returns:
        pd.DataFrame: Prepared long-format dataframe
    """
    if long_gen:
        combined_data = load_var_data(
            data_dir,
            variable,
            frequency,
            ssp_scenarios,
            time_period,
            hydro_models,
            climate_models,
        )

        long_data = reshape_to_long_format(combined_data, value_name=variable)
    else:
        long_data = pd.read_csv(long_path)
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        long_data.to_csv(output_file, index=False)
        print(f"Saved prepared data to {output_file}")

    return long_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare hydro model data for analysis"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/raghunathan/hydro_preprocess/pre_processing/unicc_output_deux",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--long_gen",
        type=bool,
        default=True,
        help="Generate long data from hydro_data",
    )
    parser.add_argument("--long_path", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="/home/raghunathan/hydro_preprocess/anova_results/combined_long_data_new.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--variable", type=str, default="qtot", help="Variable name to process"
    )
    parser.add_argument(
        "--ssp_scenarios", default=["historical"], help="scenarios to process"
    )
    parser.add_argument(
        "--time_period", type=str, default="historical", help="Time period to process"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="daily",  # Changed default to daily
        choices=["daily", "monthly"],
        help="Data frequency (daily or monthly)",
    )
    args = parser.parse_args()

    # Run the full data preparation pipeline
    prepare_data(
        data_dir=args.data_dir,
        long_gen=args.long_gen,
        long_path=args.long_path,
        output_file=args.output,
        variable=args.variable,
        ssp_scenarios=args.ssp_scenarios,
        time_period=args.time_period,
        frequency=args.frequency,
    )
