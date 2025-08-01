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
from pathlib import Path
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/raghunathan/hydro_preprocess/anova_results/")
OUTPUT_DIR = Path("/home/raghunathan/hydro_preprocess/anova_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# All model and scenario definitions will be auto-discovered from data
CLIMATE_MODELS = None
SSP_SCENARIOS = None
HYDRO_MODELS = None

# Global flag to check if categories have been discovered
_categories_discovered = False


def auto_discover_categories(df):
    """
    Auto-discover unique values for categorical variables from the data.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        tuple: (hydro_models, climate_models, ssp_scenarios)
    """
    hydro_models = sorted(df["hydro_model"].unique())
    climate_models = sorted(df["climate_model"].unique())
    ssp_scenarios = sorted(df["ssp_scenario"].unique())

    logger.info("Auto-discovered categories:")
    logger.info(f"  Hydro models: {hydro_models}")
    logger.info(f"  Climate models: {climate_models}")
    logger.info(f"  SSP scenarios: {ssp_scenarios}")

    return hydro_models, climate_models, ssp_scenarios


def perform_temporal_anova(df, target_variable="qtot", model_name="base"):
    """
    Perform ANOVA analysis for a specific target variable.

    Args:
        df (pd.DataFrame): Long format dataframe
        target_variable (str): Target variable for ANOVA (qtot, qtot_mean_5yr, qtot_mean_30yr)
        model_name (str): Name for this model configuration

    Returns:
        pd.DataFrame: Results for all basins
    """
    logger.info(f"Performing ANOVA for {target_variable} ({model_name})...")
    match target_variable:
        case "qtot_mean":
            formula = f"""{target_variable} ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
            C(ssp_scenario):C(climate_model) + C(ssp_scenario):year + C(month):C(ssp_scenario)
            """
            logger.info(f"{target_variable} adding month X ssp interaction term")
        case _:
            # Standard formula template - same structure for all temporal representations
            formula = f"""{target_variable} ~ C(ssp_scenario) + C(climate_model) + C(hydro_model) + C(month) + year + 
                        C(ssp_scenario):C(climate_model) + C(ssp_scenario):year """

    results = []
    basin_ids = df["BASIN_ID"].unique()

    # Check if target variable exists and has data
    if target_variable not in df.columns:
        logger.error(f"Target variable '{target_variable}' not found in dataframe")
        return pd.DataFrame()

    # For rolling averages, filter out NaN values
    if target_variable != "qtot":
        df = df.dropna(subset=[target_variable])
        if len(df) == 0:
            logger.warning(f"No non-NaN data for {target_variable}")
            return pd.DataFrame()

    for i, basin_id in enumerate(basin_ids):
        if i % 50 == 0:
            logger.info(f"  Processing basin {i + 1}/{len(basin_ids)}: {basin_id}")

        # Filter data for this basin
        basin_data = df[df["BASIN_ID"] == basin_id].copy()

        # Skip if insufficient data
        min_obs = 150  # Need reasonable sample size
        if len(basin_data) < min_obs:
            logger.debug(
                f"Skipping basin {basin_id}: insufficient data ({len(basin_data)} < {min_obs})"
            )
            continue

        # Check for sufficient variation in factors
        n_ssp = basin_data["ssp_scenario"].nunique()
        n_climate = basin_data["climate_model"].nunique()
        n_hydro = basin_data["hydro_model"].nunique()

        if n_ssp < 2 or n_climate < 2 or n_hydro < 2:
            logger.debug(
                f"Skipping basin {basin_id}: insufficient factor levels (SSP:{n_ssp}, Climate:{n_climate}, Hydro:{n_hydro})"
            )
            continue

        try:
            # Fit the model
            model = ols(formula, data=basin_data).fit()
            anova_table = anova_lm(model, typ=2)

            # Extract variance components
            total_ss = anova_table["sum_sq"].sum()

            # Initialize results dictionary
            result = {
                "basin_id": basin_id,
                "basin_name": basin_data["NAME"].iloc[0],
                "region": basin_data["REGION"].iloc[0],
                "area_km2": basin_data["area_km2"].iloc[0]
                if "area_km2" in basin_data.columns
                else None,
                "n_observations": len(basin_data),
                "r_squared": model.rsquared,
                "temporal_model": model_name,
                "mean_value": basin_data[target_variable].mean(),
                "std_value": basin_data[target_variable].std(),
            }

            # Extract variance percentages for all terms
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
            logger.warning(f"Error processing basin {basin_id}: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    logger.info(f"Successfully analyzed {len(results_df)} basins for {model_name}")

    return results_df


def create_summary_statistics(results_df):
    """
    Create summary statistics across all basins.

    Args:
        results_df (pd.DataFrame): Basin-level ANOVA results

    Returns:
        pd.DataFrame: Summary statistics
    """
    logger.info("Creating summary statistics...")

    if len(results_df) == 0:
        logger.warning("No results to summarize")
        return pd.DataFrame(), pd.DataFrame()

    # Find all variance columns dynamically
    variance_columns = [
        col
        for col in results_df.columns
        if col.endswith("_pct") and not col.startswith("residual")
    ]
    variance_columns.append("residual_pct")  # Add residual if exists
    variance_columns = [col for col in variance_columns if col in results_df.columns]

    # Find all p-value columns dynamically
    pvalue_columns = [col for col in results_df.columns if col.endswith("_pvalue")]

    # Create summary statistics
    summary_stats = {"metric": ["Mean", "Median", "Std Dev", "Min", "Max"]}

    for col in variance_columns:
        if col in results_df.columns:
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
    sig_data = []

    for pval_col in pvalue_columns:
        if pval_col in results_df.columns:
            # Extract factor name from p-value column
            factor_name = (
                pval_col.replace("_pvalue", "")
                .replace("_x_", "×")
                .replace("_", " ")
                .title()
            )
            factor_name = factor_name.replace("Ssp", "SSP")

            sig_count = (results_df[pval_col] < alpha).sum()
            sig_pct = (sig_count / len(results_df)) * 100 if len(results_df) > 0 else 0

            sig_data.append(
                {
                    "factor": factor_name,
                    "significant_basins": sig_count,
                    "total_basins": len(results_df),
                    "percentage_significant": sig_pct,
                }
            )

    significance_df = pd.DataFrame(sig_data)

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

    logger.info(f"\n{title}")
    logger.info("=" * len(title))

    if len(results_df) == 0:
        logger.info("No results to summarize.")
        return

    logger.info(f"\nAnalyzed {len(results_df)} basins")
    logger.info(f"Average R² = {results_df['r_squared'].mean():.3f}")

    # Identify available variance columns dynamically
    variance_cols = [
        col
        for col in results_df.columns
        if col.endswith("_pct") and not col.startswith("residual")
    ]
    pvalue_cols = [col for col in results_df.columns if col.endswith("_pvalue")]

    logger.info("\nVariance Contributions (% across all basins):")

    # Group columns by category for better organization
    main_effects = []
    interactions = []

    for col in variance_cols:
        if "x_" in col or "interaction" in col:
            interactions.append(col)
        else:
            main_effects.append(col)

    if main_effects:
        logger.info("MAIN EFFECTS:")
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
            logger.info(f"  {label:17s}: {mean_val:6.1f}% ± {std_val:.1f}%")

    if interactions:
        logger.info("\nINTERACTIONS:")
        for col in sorted(interactions):
            # Create readable label from column name
            label = (
                col.replace("_pct", "").replace("_x_", " × ").replace("_", " ").title()
            )
            label = label.replace("Ssp", "SSP")

            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            logger.info(f"  {label:17s}: {mean_val:6.1f}% ± {std_val:.1f}%")

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
        logger.info(f"\nRESIDUAL:         {mean_val:6.1f}% ± {std_val:.1f}%")

    # Significance summary
    if pvalue_cols:
        logger.info("\nSignificant effects (p < 0.05):")
        for pval_col in sorted(pvalue_cols):
            # Create readable label from p-value column name
            label = (
                pval_col.replace("_pvalue", "")
                .replace("_x_", " × ")
                .replace("_", " ")
                .title()
            )
            label = label.replace("Ssp", "SSP")
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
            logger.info(f"{label:17s}: {sig_pct:5.1f}% of basins")


def main(input_file=None, target_variable=None, save_results=False):
    """
    Main analysis function.

    Args:
        input_file (str): Path to the prepared data CSV file
        target_variable (str): Specific variable to analyze (qtot, qtot_mean_5yr, qtot_mean_30yr)
                              If None, analyzes all available variables
    """
    logger.info("=" * 60)
    logger.info("HYDRO MODEL ANOVA ANALYSIS WITH TEMPORAL COMPARISON")
    logger.info("=" * 60)

    try:
        # Load prepared data file
        if input_file is None:
            input_file = OUTPUT_DIR / "qtot_monthly_rolling_averages.csv"

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading data from {input_path}...")
        long_data = pd.read_csv(input_path)

        # Convert date column to datetime
        if "date" in long_data.columns:
            long_data["date"] = pd.to_datetime(long_data["date"])

        logger.info(f"Loaded {len(long_data)} rows from {input_path.name}")

        # Auto-discover categories from the data
        global HYDRO_MODELS, CLIMATE_MODELS, SSP_SCENARIOS
        HYDRO_MODELS, CLIMATE_MODELS, SSP_SCENARIOS = auto_discover_categories(
            long_data
        )

        # Perform ANOVA for different temporal representations
        temporal_results = {}

        # Define available analyses
        available_analyses = {
            "qtot_mean": (
                "annual",
                lambda df: perform_temporal_anova(
                    df, target_variable="qtot_mean", model_name="annual"
                ),
            ),
            "qtot_mean_5yr": (
                "5yr_rolling",
                lambda df: perform_temporal_anova(
                    df.dropna(subset=["qtot_mean_5yr"]), "qtot_mean_5yr", "5yr_rolling"
                ),
            ),
            "qtot_mean_30yr": (
                "30yr_rolling",
                lambda df: perform_temporal_anova(
                    df.dropna(subset=["qtot_mean_30yr"]),
                    "qtot_mean_30yr",
                    "30yr_rolling",
                ),
            ),
        }

        # Run analysis based on target_variable parameter
        if target_variable:
            # Run only the specified variable
            if (
                target_variable in available_analyses
                and target_variable in long_data.columns
            ):
                model_name, analysis_func = available_analyses[target_variable]
                logger.info(f"Running analysis for {target_variable} only...")
                results = analysis_func(long_data)
                if len(results) > 0:
                    temporal_results[model_name] = results
                else:
                    logger.warning(f"No results generated for {target_variable}")
            else:
                if target_variable not in available_analyses:
                    logger.error(
                        f"Unknown target variable: {target_variable}. Available: {list(available_analyses.keys())}"
                    )
                else:
                    logger.error(
                        f"Target variable '{target_variable}' not found in data columns"
                    )
                return
        else:
            # Run all available analyses (original behavior)
            for var_name, (model_name, analysis_func) in available_analyses.items():
                if var_name in long_data.columns:
                    logger.info(f"Running analysis for {var_name}...")
                    results = analysis_func(long_data)
                    if len(results) > 0:
                        temporal_results[model_name] = results
        # Create summary statistics
        for model_name, results_df in temporal_results.items():
            if len(results_df) > 0:
                summary_df, significance_df = create_summary_statistics(results_df)
                if save_results:
                    # Save results
                    results_df.to_csv(
                        OUTPUT_DIR / f"basin_anova_results_{model_name}.csv",
                        index=False,
                    )
                    summary_df.to_csv(
                        OUTPUT_DIR / f"summary_statistics_{model_name}.csv", index=False
                    )
                    significance_df.to_csv(
                        OUTPUT_DIR / f"significance_summary_{model_name}.csv",
                        index=False,
                    )

            # Print summary
            print_anova_results_summary(results_df, model_name=model_name)

        # Compare temporal models
        if len(temporal_results) > 1:
            logger.info("\n" + "=" * 60)
            logger.info("TEMPORAL MODEL COMPARISON - SSP×TIME INTERACTIONS")
            logger.info("=" * 60)

            for model_name, results_df in temporal_results.items():
                if len(results_df) > 0:
                    # Find SSP×year interaction column
                    ssp_year_cols = [
                        col
                        for col in results_df.columns
                        if "ssp" in col.lower()
                        and "year" in col.lower()
                        and "_pct" in col
                    ]
                    if ssp_year_cols:
                        mean_var = results_df[ssp_year_cols[0]].mean()
                        r2_mean = results_df["r_squared"].mean()
                        logger.info(
                            f"{model_name:15s}: SSP×time = {mean_var:5.2f}%, R² = {r2_mean:.3f}"
                        )

        logger.info(f"\nResults saved to {OUTPUT_DIR}/")

    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ANOVA Analysis for Hydro Model Output"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the prepared data CSV file (default: anova_results/qtot_monthly_rolling_averages.csv)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=False,
        help="Store results from anova",
    )
    parser.add_argument(
        "--target-variable",
        type=str,
        default=None,
        choices=["qtot_mean", "qtot_mean_5yr", "qtot_mean_30yr"],
        help="Specific variable to analyze. If not specified, analyzes all available variables.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Update logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    main(
        input_file=args.input,
        target_variable=args.target_variable,
        save_results=args.save_results,
    )
