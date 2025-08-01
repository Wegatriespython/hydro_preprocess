#!/usr/bin/env python3
"""
Combined Historical-Future Rolling Average Calculator
===================================================

This module computes rolling averages using both historical (1850-2014) and
future scenario data (2015-2100) from pre-existing long-format CSV files
to ensure proper 30-year rolling averages are available from the start
of the model horizon (2015 onwards).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
import warnings
import gc

warnings.filterwarnings("ignore")


def load_combined_long_data(
    historical_csv: Union[str, Path],
    future_csv: Union[str, Path],
    value_column: str = "qtot_mean",
) -> pd.DataFrame:
    """
    Load and combine historical and future long-format CSV files.

    Args:
        historical_csv: Path to historical long-format CSV file
        future_csv: Path to future long-format CSV file
        value_column: Name of the value column

    Returns:
        pd.DataFrame: Combined historical + future dataset in long format
    """
    historical_csv = Path(historical_csv)
    future_csv = Path(future_csv)

    print(f"Loading historical data from {historical_csv}...")
    hist_df = pd.read_csv(historical_csv)
    hist_df["data_period"] = "historical"

    print(f"Loading future data from {future_csv}...")
    future_df = pd.read_csv(future_csv)
    future_df["data_period"] = "future"

    # Ensure date columns are datetime
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    future_df["date"] = pd.to_datetime(future_df["date"])

    # Add year/month if not present
    if "year" not in hist_df.columns:
        hist_df["year"] = hist_df["date"].dt.year
    if "month" not in hist_df.columns:
        hist_df["month"] = hist_df["date"].dt.month

    if "year" not in future_df.columns:
        future_df["year"] = future_df["date"].dt.year
    if "month" not in future_df.columns:
        future_df["month"] = future_df["date"].dt.month

    # Standardize climate model names to lowercase for consistent grouping
    hist_df["climate_model"] = hist_df["climate_model"].str.lower()
    future_df["climate_model"] = future_df["climate_model"].str.lower()

    print(f"Combining datasets...")
    print(
        f"  Historical: {len(hist_df):,} rows ({hist_df['year'].min()}-{hist_df['year'].max()})"
    )
    print(
        f"  Future: {len(future_df):,} rows ({future_df['year'].min()}-{future_df['year'].max()})"
    )

    # Replicate historical data for each future SSP scenario
    future_scenarios = future_df['ssp_scenario'].unique()
    hist_replicated = []
    
    for scenario in future_scenarios:
        hist_copy = hist_df.copy()
        hist_copy['ssp_scenario'] = scenario
        hist_replicated.append(hist_copy)
    
    if hist_replicated:
        hist_df_expanded = pd.concat(hist_replicated, ignore_index=True)
        print(f"  Replicated historical data for {len(future_scenarios)} scenarios")
    else:
        hist_df_expanded = hist_df
    
    combined_df = pd.concat([hist_df_expanded, future_df], ignore_index=True)

    # Clean up
    del hist_df, future_df, hist_df_expanded
    gc.collect()

    print(
        f"Combined dataset: {len(combined_df):,} rows ({combined_df['year'].min()}-{combined_df['year'].max()})"
    )
    return combined_df


def compute_combined_rolling_averages(
    df: pd.DataFrame,
    value_column: str = "qtot_mean",
    frequency: Optional[str] = None,
    windows: Optional[Dict[str, int]] = None,
    future_start_year: int = 2015,
) -> pd.DataFrame:
    """
    Compute rolling averages using combined historical + future data.

    This function ensures that rolling averages for future scenarios (2015+)
    use the full historical context, avoiding gaps at the start of projections.

    Args:
        df: Long format dataframe with combined historical + future data
        value_column: Column containing values to average
        frequency: Data frequency ("daily" or "monthly"). Auto-detected if None.
        windows: Dictionary of window names and sizes. If None, defaults based on frequency.
        future_start_year: Year when future projections begin (default: 2015)

    Returns:
        pd.DataFrame: Dataframe with rolling average columns added
    """
    # Auto-detect frequency if not provided
    if frequency is None:
        sample_dates = df["date"].sort_values().head(3)
        if len(sample_dates) >= 2:
            diff_days = (sample_dates.iloc[1] - sample_dates.iloc[0]).days
            frequency = "daily" if diff_days == 1 else "monthly"
        else:
            frequency = "monthly"

    # Set default windows based on frequency
    if windows is None:
        if frequency == "daily":
            windows = {"5yr": 1825, "10yr": 3650, "30yr": 10950}
        else:
            windows = {"5yr": 60, "10yr": 120, "30yr": 360}

    print(
        f"Computing combined rolling averages for {value_column} ({frequency} frequency)..."
    )
    print(f"  Window sizes: {windows}")
    print(f"  Future start year: {future_start_year}")

    def compute_rolling_for_scenario_group(group_data):
        """
        Compute rolling averages for a basin-hydro_model-climate_model-scenario combination.
        Uses historical data to provide context for future rolling averages.
        """
        # Separate historical and future data
        historical_data = group_data[group_data["data_period"] == "historical"].copy()
        future_data = group_data[group_data["data_period"] == "future"].copy()

        if len(historical_data) == 0:
            # No historical data, just compute rolling on future data
            future_data = future_data.sort_values("date").reset_index(drop=True)
            for window_name, window_size in windows.items():
                col_name = f"{value_column}_{window_name}"
                rolling_values = (
                    future_data[value_column]
                    .rolling(window=window_size, min_periods=window_size, center=False)
                    .mean()
                )
                future_data[col_name] = rolling_values
            return future_data

        if len(future_data) == 0:
            # No future data, just compute rolling on historical data
            historical_data = historical_data.sort_values("date").reset_index(drop=True)
            for window_name, window_size in windows.items():
                col_name = f"{value_column}_{window_name}"
                rolling_values = (
                    historical_data[value_column]
                    .rolling(window=window_size, min_periods=window_size, center=False)
                    .mean()
                )
                historical_data[col_name] = rolling_values
            return historical_data

        # Combine historical + future for continuous time series
        combined_data = pd.concat([historical_data, future_data], ignore_index=True)
        combined_data = combined_data.sort_values("date").reset_index(drop=True)

        # Compute rolling averages on the combined series
        for window_name, window_size in windows.items():
            col_name = f"{value_column}_{window_name}"

            # Compute rolling average on combined data
            rolling_values = (
                combined_data[value_column]
                .rolling(window=window_size, min_periods=window_size, center=False)
                .mean()
            )

            combined_data[col_name] = rolling_values

        return combined_data

    # Apply rolling averages to each basin-hydro_model-climate_model-scenario combination
    # Note: We group by scenario too because each scenario needs its own rolling calculation
    grouping_cols = ["BASIN_ID", "hydro_model", "climate_model", "ssp_scenario"]

    print("  Computing rolling averages for each group...")
    df_with_rolling = df.groupby(grouping_cols, group_keys=False).apply(
        compute_rolling_for_scenario_group
    )

    # Add time period indicators
    df_with_rolling["period_2015_2030"] = (df_with_rolling["year"] >= 2015) & (
        df_with_rolling["year"] <= 2030
    )
    df_with_rolling["period_2040_2055"] = (df_with_rolling["year"] >= 2040) & (
        df_with_rolling["year"] <= 2055
    )
    df_with_rolling["period_2070_2085"] = (df_with_rolling["year"] >= 2070) & (
        df_with_rolling["year"] <= 2085
    )

    df_with_rolling["decade"] = (df_with_rolling["year"] // 10) * 10

    # Report effective date ranges for rolling averages
    print("  Rolling average data availability:")
    for window_name in windows.keys():
        col_name = f"{value_column}_{window_name}"

        # Overall statistics
        valid_data = df_with_rolling.dropna(subset=[col_name])
        if len(valid_data) > 0:
            min_year = valid_data["year"].min()
            max_year = valid_data["year"].max()
            total_points = len(valid_data)
            print(
                f"    {col_name}: {min_year}-{max_year} ({total_points:,} valid data points)"
            )

            # Future period availability
            future_valid = valid_data[valid_data["year"] >= future_start_year]
            if len(future_valid) > 0:
                future_min_year = future_valid["year"].min()
                future_max_year = future_valid["year"].max()
                future_points = len(future_valid)
                print(
                    f"      Future ({future_start_year}+): {future_min_year}-{future_max_year} ({future_points:,} points)"
                )
        else:
            print(f"    {col_name}: No valid data points")

    return df_with_rolling


def filter_future_data(
    df: pd.DataFrame,
    future_start_year: int = 2015,
    keep_scenarios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter to keep only future data (2015+) for specified scenarios.

    Args:
        df: Combined dataframe with rolling averages
        future_start_year: Year when future projections begin
        keep_scenarios: List of scenarios to keep (default: all non-historical scenarios)

    Returns:
        pd.DataFrame: Filtered dataframe with only future data
    """
    if keep_scenarios is None:
        # Keep all scenarios except historical
        keep_scenarios = (
            df[df["ssp_scenario"] != "historical"]["ssp_scenario"].unique().tolist()
        )

    print(
        f"Filtering to future data ({future_start_year}+) for scenarios: {keep_scenarios}"
    )

    # Filter to future years and specified scenarios
    future_df = df[
        (df["year"] >= future_start_year) & (df["ssp_scenario"].isin(keep_scenarios))
    ].copy()

    print(f"Filtered dataset: {len(future_df):,} rows")
    return future_df


def create_combined_rolling_averages(
    historical_csv: Union[str, Path],
    future_csv: Union[str, Path],
    output_file: Union[str, Path],
    value_column: str = "qtot_mean",
    frequency: Optional[str] = None,
    windows: Optional[Dict[str, int]] = None,
    future_only: bool = True,
    future_start_year: int = 2015,
) -> pd.DataFrame:
    """
    Main function to create rolling averages using combined historical + future long-format CSV files.

    Args:
        historical_csv: Path to historical long-format CSV file
        future_csv: Path to future long-format CSV file
        output_file: Path to save the result
        value_column: Name of the value column (default: "qtot_mean")
        frequency: Data frequency ("daily" or "monthly"). Auto-detected if None.
        windows: Dictionary of rolling window sizes (auto-set by frequency if None)
        future_only: If True, output only contains future data (default: True)
        future_start_year: Year when future projections begin (default: 2015)

    Returns:
        pd.DataFrame: Final dataset with rolling averages
    """
    print("=" * 60)
    print("COMBINED HISTORICAL-FUTURE ROLLING AVERAGE CALCULATOR")
    print("=" * 60)

    # Step 1: Load and combine historical + future long-format data
    combined_long = load_combined_long_data(historical_csv, future_csv, value_column)

    # Step 2: Compute rolling averages using full historical context
    data_with_rolling = compute_combined_rolling_averages(
        combined_long,
        value_column=value_column,
        frequency=frequency,
        windows=windows,
        future_start_year=future_start_year,
    )

    # Clean up intermediate data
    del combined_long
    gc.collect()

    # Step 3: Filter to future data only if requested
    if future_only:
        final_data = filter_future_data(
            data_with_rolling,
            future_start_year=future_start_year,
        )

        # Clean up full data
        del data_with_rolling
        gc.collect()
    else:
        final_data = data_with_rolling

    # Step 4: Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(output_file, index=False)

    print("=" * 60)
    print(f"COMPLETED: Saved {len(final_data):,} rows to {output_file}")
    print("=" * 60)

    return final_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create rolling averages using combined historical + future long-format CSV files"
    )
    parser.add_argument(
        "--historical-csv",
        type=str,
        required=True,
        help="Path to historical long-format CSV file",
    )
    parser.add_argument(
        "--future-csv",
        type=str,
        required=True,
        help="Path to future long-format CSV file",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for results"
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default="qtot_mean",
        help="Value column name (default: qtot_mean)",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default=None,
        choices=["daily", "monthly"],
        help="Data frequency (auto-detected if not specified)",
    )
    parser.add_argument(
        "--future-start-year",
        type=int,
        default=2015,
        help="Year when future projections begin (default: 2015)",
    )
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Include historical data in output (default: future only)",
    )

    args = parser.parse_args()

    # Run the main function
    create_combined_rolling_averages(
        historical_csv=args.historical_csv,
        future_csv=args.future_csv,
        output_file=args.output,
        value_column=args.value_column,
        frequency=args.frequency,
        future_only=not args.include_historical,
        future_start_year=args.future_start_year,
    )

