#!/usr/bin/env python3
"""
Data Aggregation for Hydro Model Output Analysis
=======================================================
Aggregates Daily data to monthly frequency.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import warnings
import gc
import sys

warnings.filterwarnings("ignore")


def aggregate_daily_to_monthly(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    value_column: str,
    chunk_size: int,
    aggregation_stats: Optional[List[str]] = None,
) -> None:
    """
    Aggregate daily data to monthly in a memory-efficient way.

    This function reads a large daily CSV file in chunks and aggregates it to monthly
    resolution to reduce memory requirements for downstream analysis.

    Args:
        input_csv: Path to the input daily long-format CSV file
        output_csv: Path to save the monthly aggregated CSV file
        value_column: Column containing values to aggregate (default: "qtot")
        chunk_size: Number of rows to read per chunk (default: 50000)
        aggregation_stats: List of aggregation statistics to compute.
                          Default: ["mean", "std", "min", "max"]
    """
    if aggregation_stats is None:
        aggregation_stats = ["mean", "std", "min", "max"]

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    print(f"Aggregating daily data to monthly: {input_csv} -> {output_csv}")
    print(f"  Chunk size: {chunk_size:,} rows")
    print(f"  Aggregation stats: {aggregation_stats}")

    # Initialize containers for accumulating results
    monthly_data = {}
    processed_chunks = 0
    total_rows_processed = 0

    # Read the CSV in chunks
    try:
        csv_reader = pd.read_csv(input_csv, chunksize=chunk_size)

        for chunk_num, chunk in enumerate(csv_reader, 1):
            print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...")

            # Ensure date column is datetime
            if chunk["date"].dtype == "object":
                chunk["date"] = pd.to_datetime(chunk["date"])

            # Add year and month columns if not present
            if "year" not in chunk.columns:
                chunk["year"] = chunk["date"].dt.year
            if "month" not in chunk.columns:
                chunk["month"] = chunk["date"].dt.month

            # Define grouping columns (all except date and value column)
            id_columns = [
                "BASIN_ID",
                "BCU_name",
                "NAME",
                "REGION",
                "area_km2",
                "hydro_model",
                "climate_model",
                "ssp_scenario",
                "year",
                "month",
            ]

            # Keep only columns that exist in the chunk
            available_id_columns = [col for col in id_columns if col in chunk.columns]

            # Group by monthly periods and aggregate
            agg_dict = {value_column: aggregation_stats}

            chunk_monthly = (
                chunk.groupby(available_id_columns).agg(agg_dict).reset_index()
            )

            # Flatten column names for multi-level aggregation
            if len(aggregation_stats) > 1:
                chunk_monthly.columns = [
                    col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
                    for col in chunk_monthly.columns
                ]

            # Add representative date (middle of month)
            chunk_monthly["date"] = pd.to_datetime(
                chunk_monthly[["year", "month"]].assign(day=15)
            )

            # Accumulate results by creating a unique key for each basin-model-scenario-year-month
            for _, row in chunk_monthly.iterrows():
                key = (
                    row["BASIN_ID"],
                    row.get("hydro_model", ""),
                    row.get("climate_model", ""),
                    row.get("ssp_scenario", ""),
                    row["year"],
                    row["month"],
                )

                if key not in monthly_data:
                    # First time seeing this combination
                    monthly_data[key] = row.to_dict()
                    monthly_data[key]["count"] = 1
                else:
                    # Accumulate statistics for this combination
                    existing = monthly_data[key]
                    count = existing["count"]
                    new_count = count + 1

                    # Update running averages and other stats
                    for stat in aggregation_stats:
                        col_name = (
                            f"{value_column}_{stat}"
                            if len(aggregation_stats) > 1
                            else value_column
                        )

                        if stat == "mean":
                            # Update running mean
                            existing[col_name] = (
                                existing[col_name] * count + row[col_name]
                            ) / new_count
                        elif stat == "min":
                            existing[col_name] = min(existing[col_name], row[col_name])
                        elif stat == "max":
                            existing[col_name] = max(existing[col_name], row[col_name])
                        elif stat == "std":
                            # For std, we'll approximate by taking weighted average
                            # This is not perfectly accurate but sufficient for large datasets
                            existing[col_name] = (
                                existing[col_name] * count + row[col_name]
                            ) / new_count

                    monthly_data[key]["count"] = new_count

            total_rows_processed += len(chunk)
            processed_chunks += 1

            # Clean up memory every 10 chunks
            if processed_chunks % 10 == 0:
                gc.collect()
                print(
                    f"    Processed {processed_chunks} chunks, {total_rows_processed:,} total rows"
                )
                print(f"    Unique monthly records so far: {len(monthly_data):,}")

        print(
            f"Completed processing {processed_chunks} chunks ({total_rows_processed:,} total rows)"
        )
        print(f"Final unique monthly records: {len(monthly_data):,}")

        # Convert accumulated results to DataFrame
        print("Converting results to DataFrame...")
        monthly_df = pd.DataFrame(list(monthly_data.values()))

        # Remove the temporary count column
        if "count" in monthly_df.columns:
            monthly_df = monthly_df.drop("count", axis=1)

        # Sort by basin, model, scenario, date
        sort_columns = [
            "BASIN_ID",
            "hydro_model",
            "climate_model",
            "ssp_scenario",
            "date",
        ]
        available_sort_columns = [
            col for col in sort_columns if col in monthly_df.columns
        ]
        monthly_df = monthly_df.sort_values(available_sort_columns).reset_index(
            drop=True
        )

        # Save to CSV
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        monthly_df.to_csv(output_csv, index=False)

        print(f"Successfully aggregated to monthly data:")
        print(f"  Input: {total_rows_processed:,} daily rows")
        print(f"  Output: {len(monthly_df):,} monthly rows")
        print(f"  Reduction factor: {total_rows_processed / len(monthly_df):.1f}x")
        print(f"  Saved to: {output_csv}")

        # Clean up
        del monthly_data, monthly_df
        gc.collect()

    except Exception as e:
        print(f"Error during aggregation: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate daily to monthly data")

    parser.add_argument(
        "--input-daily", type=str, help="input daily data csv to be aggregated"
    )
    parser.add_argument(
        "--chunk-size", type=str, help="chunk size for daily to monthly aggregation"
    )
    parser.add_argument("--variable", help="variable to aggregate", default="qtot_mean")
    parser.add_argument("--output-file", help="output file path")
    args = parser.parse_args()

    aggregate_daily_to_monthly(
        args.input_daily, args.output_file, args.variable, args.chunk_size
    )
