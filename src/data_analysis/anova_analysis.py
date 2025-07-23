#!/usr/bin/env python3
"""
ANOVA Analysis for Hydro Model Output
=====================================

This script performs variance decomposition analysis to quantify how much
variance in qtot (total runoff) is explained by:
1. SSP scenarios (ssp126, ssp370, ssp585)
2. Climate models (5 models)
3. Their interaction
4. Residual variance

For each of the 217 basins, we calculate the percentage contribution of each factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
import sys

# Import data preparation module
from data_prep import prepare_data

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/raghunathan/hydro_preprocess/pre_processing/unicc_output_deux")
OUTPUT_DIR = Path("/home/raghunathan/hydro_preprocess/anova_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model and scenario definitions
CLIMATE_MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
]
SSP_SCENARIOS = ["ssp126", "ssp370", "ssp585"]


def perform_anova_comparison(df):
    """
    Compare different temporal representations in ANOVA models.
    Tests: year, 5-year, 10-year, and 30-year rolling averages.

    Args:
        df (pd.DataFrame): Long format dataframe with rolling averages

    Returns:
        dict: Results for different temporal representations
    """
    print("Performing ANOVA comparison with different temporal representations...")

    # Define different temporal models to test (all include hydro_model)
    temporal_models = {
        "year": {
            "formula": """qtot ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):year + 
                          C(climate_model):year + C(hydro_model):year""",
            "description": "Annual temporal trend with hydro model",
        },
        "5yr_rolling": {
            "formula": """qtot_5yr ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):year + 
                          C(climate_model):year + C(hydro_model):year""",
            "description": "5-year rolling average with hydro model",
        },
        "10yr_rolling": {
            "formula": """qtot_10yr ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):year + 
                          C(climate_model):year + C(hydro_model):year""",
            "description": "10-year rolling average with hydro model",
        },
        "30yr_rolling": {
            "formula": """qtot_30yr ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):year + 
                          C(climate_model):year + C(hydro_model):year""",
            "description": "30-year rolling average with hydro model",
        },
        "decade": {
            "formula": """qtot ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + C(decade) + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):C(decade) + 
                          C(climate_model):C(decade) + C(hydro_model):C(decade)""",
            "description": "Decade-based temporal grouping with hydro model",
        },
    }

    all_results = {}
    basin_ids = df["BASIN_ID"].unique()

    for model_name, model_config in temporal_models.items():
        print(f"\n  Testing {model_name}: {model_config['description']}")
        results = []

        for i, basin_id in enumerate(basin_ids):
            if i % 50 == 0:
                print(f"    Processing basin {i + 1}/{len(basin_ids)}: {basin_id}")

            # Filter data for this basin
            basin_data = df[df["BASIN_ID"] == basin_id].copy()

            # For rolling averages, remove rows with NaN values
            if model_name.endswith("_rolling"):
                target_var = model_config["formula"].split("~")[0].strip()
                basin_data = basin_data.dropna(subset=[target_var])

            # Skip if insufficient data
            min_obs = 150  # Need reasonable sample size
            if len(basin_data) < min_obs:
                continue

            try:
                # Fit the model
                model = ols(model_config["formula"], data=basin_data).fit()
                anova_table = anova_lm(model, typ=2)

                # Extract variance components
                total_ss = anova_table["sum_sq"].sum()

                # Initialize results dictionary
                result = {
                    "basin_id": basin_id,
                    "basin_name": basin_data["NAME"].iloc[0],
                    "region": basin_data["REGION"].iloc[0],
                    "n_observations": len(basin_data),
                    "r_squared": model.rsquared,
                    "temporal_model": model_name,
                }

                # Extract variance percentages for available terms
                for term in anova_table.index:
                    if term != "Residual":
                        var_name = f"{term.lower().replace('c(', '').replace(')', '').replace(':', '_x_')}_pct"
                        var_name = var_name.replace(" ", "_")
                        result[var_name] = (
                            anova_table.loc[term, "sum_sq"] / total_ss
                        ) * 100

                        # P-values
                        pval_name = var_name.replace("_pct", "_pvalue")
                        result[pval_name] = anova_table.loc[term, "PR(>F)"]

                # Residual variance
                result["residual_pct"] = (
                    anova_table.loc["Residual", "sum_sq"] / total_ss
                ) * 100

                results.append(result)

            except Exception as e:
                continue

        all_results[model_name] = pd.DataFrame(results)
        if len(results) > 0:
            print(f"    Successfully analyzed {len(results)} basins with {model_name}")
        else:
            print(f"    No successful analyses with {model_name}")

    return all_results


def perform_anova_by_basin(df):
    """
    Perform ANOVA analysis for each basin separately.
    This is the main analysis function using annual year.

    Args:
        df (pd.DataFrame): Long format dataframe

    Returns:
        pd.DataFrame: Variance decomposition results by basin
    """
    print("Performing ANOVA analysis by basin...")

    results = []
    basin_ids = df["BASIN_ID"].unique()

    for i, basin_id in enumerate(basin_ids):
        if i % 50 == 0:
            print(f"  Processing basin {i + 1}/{len(basin_ids)}: {basin_id}")

        # Filter data for this basin
        basin_data = df[df["BASIN_ID"] == basin_id].copy()

        # Skip if insufficient data for reliable ANOVA
        # Need minimum observations for each model-scenario combination
        min_obs_per_group = 10
        min_total_obs = len(CLIMATE_MODELS) * len(SSP_SCENARIOS) * min_obs_per_group

        if len(basin_data) < min_total_obs:
            print(
                f"    Skipping basin {basin_id}: insufficient data ({len(basin_data)} < {min_total_obs})"
            )
            continue

        # Check if we have data for all model-scenario combinations
        combo_counts = basin_data.groupby(["climate_model", "ssp_scenario"]).size()
        if len(combo_counts) < len(CLIMATE_MODELS) * len(SSP_SCENARIOS):
            print(f"    Skipping basin {basin_id}: missing model-scenario combinations")
            continue

        # Check if any combination has too few observations
        if combo_counts.min() < min_obs_per_group:
            print(
                f"    Skipping basin {basin_id}: some combinations have <{min_obs_per_group} observations"
            )
            continue

        try:
            # Fit comprehensive ANOVA model with temporal controls and hydro model
            # Include month, year, and hydro model as controls, plus interactions
            model = ols(
                """qtot ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                          C(ssp_scenario):C(climate_model) + C(ssp_scenario):C(hydro_model) + 
                          C(climate_model):C(hydro_model) + C(ssp_scenario):year + 
                          C(climate_model):year + C(hydro_model):year""",
                data=basin_data,
            ).fit()

            # Get ANOVA table
            anova_table = anova_lm(model, typ=2)

            # Calculate total sum of squares
            total_ss = anova_table["sum_sq"].sum()

            # Calculate variance percentages for all terms (flexible approach)
            def safe_extract(term_name):
                """Safely extract variance and p-value for a term, return 0/NaN if not present."""
                if term_name in anova_table.index:
                    variance_pct = (
                        anova_table.loc[term_name, "sum_sq"] / total_ss
                    ) * 100
                    pvalue = anova_table.loc[term_name, "PR(>F)"]
                    return variance_pct, pvalue
                else:
                    return 0.0, np.nan

            # Main effects
            ssp_variance, ssp_pvalue = safe_extract("C(ssp_scenario)")
            model_variance, model_pvalue = safe_extract("C(climate_model)")
            hydro_variance, hydro_pvalue = safe_extract("C(hydro_model)")
            month_variance, month_pvalue = safe_extract("C(month)")
            year_variance, year_pvalue = safe_extract("year")

            # Two-way interactions
            ssp_model_interaction, ssp_model_pvalue = safe_extract(
                "C(ssp_scenario):C(climate_model)"
            )
            ssp_hydro_interaction, ssp_hydro_pvalue = safe_extract(
                "C(ssp_scenario):C(hydro_model)"
            )
            model_hydro_interaction, model_hydro_pvalue = safe_extract(
                "C(climate_model):C(hydro_model)"
            )
            ssp_year_interaction, ssp_year_pvalue = safe_extract("C(ssp_scenario):year")
            model_year_interaction, model_year_pvalue = safe_extract(
                "C(climate_model):year"
            )
            hydro_year_interaction, hydro_year_pvalue = safe_extract(
                "C(hydro_model):year"
            )

            residual_variance = (anova_table.loc["Residual", "sum_sq"] / total_ss) * 100

            # Get basin metadata
            basin_info = basin_data.iloc[0]

            results.append(
                {
                    "basin_id": basin_id,
                    "basin_name": basin_info["NAME"],
                    "region": basin_info["REGION"],
                    "area_km2": basin_info["area_km2"],
                    "n_observations": len(basin_data),
                    # Main effects
                    "ssp_variance_pct": ssp_variance,
                    "model_variance_pct": model_variance,
                    "hydro_variance_pct": hydro_variance,
                    "month_variance_pct": month_variance,
                    "year_variance_pct": year_variance,
                    # Two-way interactions
                    "ssp_model_interaction_pct": ssp_model_interaction,
                    "ssp_hydro_interaction_pct": ssp_hydro_interaction,
                    "model_hydro_interaction_pct": model_hydro_interaction,
                    "ssp_year_interaction_pct": ssp_year_interaction,
                    "model_year_interaction_pct": model_year_interaction,
                    "hydro_year_interaction_pct": hydro_year_interaction,
                    # Residual
                    "residual_variance_pct": residual_variance,
                    # P-values - Main effects
                    "ssp_pvalue": ssp_pvalue,
                    "model_pvalue": model_pvalue,
                    "hydro_pvalue": hydro_pvalue,
                    "month_pvalue": month_pvalue,
                    "year_pvalue": year_pvalue,
                    # P-values - Interactions
                    "ssp_model_pvalue": ssp_model_pvalue,
                    "ssp_hydro_pvalue": ssp_hydro_pvalue,
                    "model_hydro_pvalue": model_hydro_pvalue,
                    "ssp_year_pvalue": ssp_year_pvalue,
                    "model_year_pvalue": model_year_pvalue,
                    "hydro_year_pvalue": hydro_year_pvalue,
                    # Model fit
                    "r_squared": model.rsquared,
                    "mean_qtot": basin_data["qtot"].mean(),
                    "std_qtot": basin_data["qtot"].std(),
                }
            )

        except Exception as e:
            print(f"    Error processing basin {basin_id}: {e}")
            continue

    results_df = pd.DataFrame(results)
    print(f"Successfully analyzed {len(results_df)} basins")

    return results_df


def create_summary_statistics(results_df):
    """
    Create summary statistics across all basins.

    Args:
        results_df (pd.DataFrame): Basin-level ANOVA results

    Returns:
        pd.DataFrame: Summary statistics
    """
    print("Creating summary statistics...")

    # Overall statistics for all variance components
    variance_columns = [
        "ssp_variance_pct",
        "model_variance_pct",
        "month_variance_pct",
        "year_variance_pct",
        "ssp_model_interaction_pct",
        "ssp_year_interaction_pct",
        "model_year_interaction_pct",
        "residual_variance_pct",
    ]

    summary_stats = {"metric": ["Mean", "Median", "Std Dev", "Min", "Max"]}

    for col in variance_columns:
        summary_stats[col] = [
            results_df[col].mean(),
            results_df[col].median(),
            results_df[col].std(),
            results_df[col].min(),
            results_df[col].max(),
        ]

    summary_df = pd.DataFrame(summary_stats)

    # Significance counts for all factors
    alpha = 0.05
    sig_counts = {
        "factor": [
            "SSP Scenario",
            "Climate Model",
            "Month",
            "Year",
            "SSP×Model",
            "SSP×Year",
            "Model×Year",
        ],
        "significant_basins": [
            (results_df["ssp_pvalue"] < alpha).sum(),
            (results_df["model_pvalue"] < alpha).sum(),
            (results_df["month_pvalue"] < alpha).sum(),
            (results_df["year_pvalue"] < alpha).sum(),
            (results_df["ssp_model_pvalue"] < alpha).sum(),
            (results_df["ssp_year_pvalue"] < alpha).sum(),
            (results_df["model_year_pvalue"] < alpha).sum(),
        ],
        "total_basins": [len(results_df)] * 7,
        "percentage_significant": [
            ((results_df["ssp_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["model_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["month_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["year_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["ssp_model_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["ssp_year_pvalue"] < alpha).sum() / len(results_df)) * 100,
            ((results_df["model_year_pvalue"] < alpha).sum() / len(results_df)) * 100,
        ],
    }

    significance_df = pd.DataFrame(sig_counts)

    return summary_df, significance_df


def print_anova_results_summary(results_df, model_name="", title=None):
    """
    Print a comprehensive summary of ANOVA results.

    Args:
        results_df (pd.DataFrame): ANOVA results with variance and p-value columns
        model_name (str): Optional name of the temporal model (e.g., "5yr_rolling")
        title (str): Optional custom title for the summary
    """
    if title is None:
        title = (
            f"ANOVA RESULTS SUMMARY - {model_name.upper()}"
            if model_name
            else "ANOVA RESULTS SUMMARY"
        )

    print(f"\n{title}")
    print("=" * len(title))

    if len(results_df) == 0:
        print("No results to summarize.")
        return

    print(f"\nAnalyzed {len(results_df)} basins")
    print(f"Average R² = {results_df['r_squared'].mean():.3f}")

    # Identify available variance columns dynamically
    variance_cols = [
        col
        for col in results_df.columns
        if col.endswith("_pct") and not col.startswith("residual")
    ]
    pvalue_cols = [col for col in results_df.columns if col.endswith("_pvalue")]

    print("\nVariance Contributions (% across all basins):")

    # Group columns by category for better organization
    main_effects = []
    interactions = []

    for col in variance_cols:
        if "x_" in col or "interaction" in col:
            interactions.append(col)
        else:
            main_effects.append(col)

    if main_effects:
        print("MAIN EFFECTS:")
        for col in sorted(main_effects):
            # Create readable label from column name
            label = col.replace("_pct", "").replace("_", " ").title()
            if "ssp_scenario" in col:
                label = "SSP Scenario"
            elif "climate_model" in col:
                label = "Climate Model"
            elif "month" in col:
                label = "Month (seasonal)"
            elif "year" in col:
                label = "Year (temporal)"
            elif "decade" in col:
                label = "Decade"

            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            print(f"  {label:17s}: {mean_val:6.1f}% ± {std_val:.1f}%")

    if interactions:
        print("\nINTERACTIONS:")
        for col in sorted(interactions):
            # Create readable label from column name
            label = (
                col.replace("_pct", "").replace("_x_", " × ").replace("_", " ").title()
            )
            label = label.replace("Ssp", "SSP").replace("Climate Model", "Model")

            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            print(f"  {label:17s}: {mean_val:6.1f}% ± {std_val:.1f}%")

    # Residual variance
    residual_cols = [
        col
        for col in results_df.columns
        if col.startswith("residual") and col.endswith("_pct")
    ]
    if residual_cols:
        residual_col = residual_cols[0]
        mean_val = results_df[residual_col].mean()
        std_val = results_df[residual_col].std()
        print(f"\nRESIDUAL:         {mean_val:6.1f}% ± {std_val:.1f}%")

    # Significance summary
    if pvalue_cols:
        print(f"\nSignificant effects (p < 0.05):")
        for pval_col in sorted(pvalue_cols):
            # Create readable label from p-value column name
            label = (
                pval_col.replace("_pvalue", "")
                .replace("_x_", " × ")
                .replace("_", " ")
                .title()
            )
            label = label.replace("Ssp", "SSP").replace("Climate Model", "Model")
            if "ssp_scenario" in pval_col:
                label = "SSP Scenario"
            elif "climate_model" in pval_col:
                label = "Climate Model"
            elif "month" in pval_col:
                label = "Month (seasonal)"
            elif (
                "year" in pval_col and "ssp" not in pval_col and "model" not in pval_col
            ):
                label = "Year (temporal)"
            elif "decade" in pval_col:
                label = "Decade"

            sig_count = (results_df[pval_col] < 0.05).sum()
            sig_pct = (sig_count / len(results_df)) * 100
            print(f"{label:17s}: {sig_pct:5.1f}% of basins")


def main(drop_percentile=0, force_regen=False):
    """
    Main analysis function.

    Args:
        drop_percentile (float): Percentile threshold for dropping low-runoff basins (0-100).
                               E.g., 50 drops bottom 50% of basins by average qtot.
        force_regen (bool): Force regeneration of the prepared data file
    """

    print("=" * 60)
    print("HYDRO MODEL ANOVA ANALYSIS WITH TEMPORAL COMPARISON")
    if drop_percentile > 0:
        print(
            f"Basin filtering: Dropping bottom {drop_percentile}th percentile by runoff"
        )
    print("=" * 60)

    try:
        # Check if prepared data file exists
        prepared_data_file = OUTPUT_DIR / "combined_long_data_rl.csv"

        if prepared_data_file.exists() and not force_regen:
            print(f"Loading existing prepared data from {prepared_data_file}")
            print("(Use --regen flag to force regeneration)")
            long_data = pd.read_csv(prepared_data_file)

            # Convert date column back to datetime
            long_data["date"] = pd.to_datetime(long_data["date"])

            print(f"Loaded {len(long_data)} rows from prepared data file")
        else:
            if force_regen:
                print("Regenerating data (--regen flag specified)...")
            else:
                print("Prepared data file not found. Generating...")

            # Use the data_prep module to prepare data
            long_data = prepare_data(
                data_dir=DATA_DIR,
                output_file=prepared_data_file,
                variable="qtot",
                frequency="daily",  # Changed to daily as default
                hydro_models=None,  # Will auto-detect from available files
                climate_models=CLIMATE_MODELS,
                ssp_scenarios=SSP_SCENARIOS,
                time_period="future",
                drop_percentile=drop_percentile,
                outlier_threshold=1e15,
            )

        # Step 3: Perform temporal comparison analysis
        temporal_results = perform_anova_comparison(long_data)

        # Step 4: Perform standard ANOVA by basin (year-based)
        results_df = perform_anova_by_basin(long_data)

        # Step 5: Create summary statistics for year-based analysis
        summary_df, significance_df = create_summary_statistics(results_df)

        # Save results
        print("Saving results...")
        results_df.to_csv(OUTPUT_DIR / "basin_anova_results.csv", index=False)
        summary_df.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
        significance_df.to_csv(OUTPUT_DIR / "significance_summary.csv", index=False)

        # Save temporal comparison results
        for model_name, model_results in temporal_results.items():
            if len(model_results) > 0:
                model_results.to_csv(
                    OUTPUT_DIR / f"temporal_comparison_{model_name}.csv", index=False
                )

        # Data is already saved by prepare_data function when output_file is specified

        # Print main ANOVA results summary using reusable function
        print_anova_results_summary(results_df, title="MAIN ANOVA RESULTS (YEAR-BASED)")

        # Print detailed summaries for each temporal model
        for model_name, model_results in temporal_results.items():
            if len(model_results) > 0:
                print_anova_results_summary(model_results, model_name=model_name)

        # Quick comparison of SSP×time interactions across models
        print(f"\n{'=' * 60}")
        print("TEMPORAL MODEL COMPARISON - SSP×TIME INTERACTIONS")
        print("=" * 60)

        ssp_year_comparison = {}
        for model_name, model_results in temporal_results.items():
            if len(model_results) > 0:
                # Find SSP×year interaction columns
                ssp_year_cols = [
                    col
                    for col in model_results.columns
                    if "ssp" in col.lower()
                    and ("year" in col.lower() or "decade" in col.lower())
                    and "_pct" in col
                ]
                if ssp_year_cols:
                    col = ssp_year_cols[0]
                    mean_var = model_results[col].mean()
                    ssp_year_comparison[model_name] = mean_var
                    r2_mean = model_results["r_squared"].mean()
                    print(
                        f"{model_name:15s}: SSP×time = {mean_var:5.2f}%, R² = {r2_mean:.3f}"
                    )

        if ssp_year_comparison:
            best_model = max(
                ssp_year_comparison.keys(), key=lambda x: ssp_year_comparison[x]
            )
            print(
                f"\n** Best at capturing climate change: {best_model} ({ssp_year_comparison[best_model]:.2f}% SSP×time) **"
            )

        print("Results saved to anova_results/ directory")

    except Exception as e:
        print(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ANOVA Analysis for Hydro Model Output"
    )
    parser.add_argument(
        "drop_percentile",
        nargs="?",
        type=float,
        default=10,
        help="Percentile threshold for dropping low-runoff basins (0-100). Default: 10",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force regeneration of the prepared data file",
    )

    args = parser.parse_args()

    if not 0 <= args.drop_percentile <= 100:
        print("Error: drop_percentile must be between 0 and 100")
        sys.exit(1)

    print(f"Using threshold {args.drop_percentile}")
    main(drop_percentile=args.drop_percentile, force_regen=args.regen)
