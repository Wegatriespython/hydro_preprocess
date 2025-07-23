#!/usr/bin/env python3
"""
Timeseries Visualization for Representative Basins
=================================================

Creates timeseries plots showing temporal patterns in qtot_30yr for representative basins.
Two plot types:
1. Grouped by SSP scenario (models as shaded regions)
2. Grouped by climate model (SSPs as shaded regions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/raghunathan/hydro_preprocess/anova_results")
PLOTS_DIR = Path("timeseries_plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Representative basins from file
REPRESENTATIVE_BASINS = [
    "Amazon",
    "Atlantic Ocean Seaboard",
    "Danube",
    "Ganges Bramaputra",
    "Indus",
    "Mekong",
    "Mississippi",
    "Nile",
    "Ob",
    "Persian Gulf Coast",
]

# Model and scenario definitions
CLIMATE_MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mpi-esm1-2-hr",
    "mri-esm2-0",
    "ukesm1-0-ll",
]
SSP_SCENARIOS = ["ssp126", "ssp370", "ssp585"]

# Color schemes
SSP_COLORS = {
    "ssp126": "#1f77b4",  # Blue - low emissions
    "ssp370": "#ff7f0e",  # Orange - medium emissions
    "ssp585": "#d62728",  # Red - high emissions
}

MODEL_COLORS = {
    "gfdl-esm4": "#1f77b4",
    "ipsl-cm6a-lr": "#ff7f0e",
    "mpi-esm1-2-hr": "#2ca02c",
    "mri-esm2-0": "#d62728",
    "ukesm1-0-ll": "#9467bd",
}


def load_timeseries_data():
    """Load the combined long-format data with all basins."""
    print("Loading timeseries data...")

    data_file = DATA_DIR / "combined_long_data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Loaded {len(df):,} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Available basins: {df['NAME'].nunique()}")

    return df


def find_representative_basins(df):
    """Find basin IDs for representative basin names."""
    print("Finding representative basins...")

    basin_matches = {}
    available_basins = df[["BASIN_ID", "NAME"]].drop_duplicates()

    for target_basin in REPRESENTATIVE_BASINS:
        # Try exact match first
        exact_match = available_basins[
            available_basins["NAME"].str.contains(target_basin, case=False, na=False)
        ]

        if len(exact_match) > 0:
            basin_id = exact_match.iloc[0]["BASIN_ID"]
            basin_name = exact_match.iloc[0]["NAME"]
            basin_matches[target_basin] = {"id": basin_id, "name": basin_name}
            print(f"  {target_basin} -> Basin {basin_id}: {basin_name}")
        else:
            print(f"  {target_basin} -> NOT FOUND")

    print(
        f"Successfully matched {len(basin_matches)}/{len(REPRESENTATIVE_BASINS)} basins"
    )
    return basin_matches


def create_annual_timeseries(df, basin_matches):
    """Create annual timeseries for easier visualization."""
    print("Creating annual timeseries...")

    # Filter for representative basins only
    basin_ids = [info["id"] for info in basin_matches.values()]
    df_filtered = df[df["BASIN_ID"].isin(basin_ids)].copy()

    # Add year column if not present and compute annual averages
    if "year" not in df_filtered.columns:
        df_filtered["year"] = df_filtered["date"].dt.year

    # Include hydro_model in grouping if available
    grouping_cols = ["BASIN_ID", "NAME", "climate_model", "ssp_scenario", "year"]
    if "hydro_model" in df_filtered.columns:
        grouping_cols.insert(-2, "hydro_model")  # Insert before year

    annual_data = (
        df_filtered.groupby(grouping_cols)
        .agg({"qtot_30yr": "mean", "REGION": "first"})
        .reset_index()
    )

    print(f"Created annual data: {len(annual_data)} basin-model-scenario-year combinations")
    
    # Show available hydro models if present
    if "hydro_model" in annual_data.columns:
        hydro_models = annual_data["hydro_model"].unique()
        print(f"Available hydro models: {hydro_models}")
    
    return annual_data


def plot_ssp_grouped_timeseries(annual_data, basin_matches, save_plots=True):
    """
    Create timeseries plots grouped by SSP scenario.
    Each SSP gets a shaded region representing model spread.
    """
    print("Creating SSP-grouped timeseries plots...")

    n_basins = len(basin_matches)
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= 10:  # Only plot first 10
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin
        basin_data = annual_data[annual_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Plot each SSP scenario with model spread
        for ssp in SSP_SCENARIOS:
            ssp_data = basin_data[basin_data["ssp_scenario"] == ssp]

            if len(ssp_data) == 0:
                continue

            # Calculate model spread (min/max envelope)
            yearly_stats = (
                ssp_data.groupby("year")["qtot_30yr"]
                .agg(["min", "max", "mean", "std"])
                .reset_index()
            )

            if len(yearly_stats) > 0:
                # Plot mean line
                ax.plot(
                    yearly_stats["year"],
                    yearly_stats["mean"],
                    color=SSP_COLORS[ssp],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{ssp.upper()}",
                )

                # Add shaded region for model spread
                ax.fill_between(
                    yearly_stats["year"],
                    yearly_stats["min"],
                    yearly_stats["max"],
                    color=SSP_COLORS[ssp],
                    alpha=0.2,
                )

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("qtot_30yr")
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True)

    # Remove empty subplots
    for j in range(len(basin_matches), 10):
        axes[j].remove()

    plt.suptitle(
        "Timeseries by SSP Scenario\n(Shaded regions show climate model spread)",
        fontsize=16,
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            PLOTS_DIR / "ssp_grouped_timeseries.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(PLOTS_DIR / "ssp_grouped_timeseries.pdf", bbox_inches="tight")

    plt.show()


def plot_model_grouped_timeseries(annual_data, basin_matches, save_plots=True):
    """
    Create timeseries plots grouped by climate model.
    Each model gets a shaded region representing SSP spread.
    """
    print("Creating model-grouped timeseries plots...")

    n_basins = len(basin_matches)
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= 10:  # Only plot first 10
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin
        basin_data = annual_data[annual_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Plot each climate model with SSP spread
        for model in CLIMATE_MODELS:
            model_data = basin_data[basin_data["climate_model"] == model]

            if len(model_data) == 0:
                continue

            # Calculate SSP spread (min/max envelope)
            yearly_stats = (
                model_data.groupby("year")["qtot_30yr"]
                .agg(["min", "max", "mean", "std"])
                .reset_index()
            )

            if len(yearly_stats) > 0:
                # Plot mean line
                ax.plot(
                    yearly_stats["year"],
                    yearly_stats["mean"],
                    color=MODEL_COLORS[model],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{model.upper()}",
                )

                # Add shaded region for SSP spread
                ax.fill_between(
                    yearly_stats["year"],
                    yearly_stats["min"],
                    yearly_stats["max"],
                    color=MODEL_COLORS[model],
                    alpha=0.2,
                )

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("qtot_30yr")
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)

    # Remove empty subplots
    for j in range(len(basin_matches), 10):
        axes[j].remove()

    plt.suptitle(
        "Timeseries by Climate Model\n(Shaded regions show SSP scenario spread)",
        fontsize=16,
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(
            PLOTS_DIR / "model_grouped_timeseries.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(PLOTS_DIR / "model_grouped_timeseries.pdf", bbox_inches="tight")

    plt.show()


def plot_hydro_grouped_timeseries(annual_data, basin_matches, save_plots=True):
    """
    Create timeseries plots grouped by hydro model.
    Each hydro model gets a shaded region representing climate model and SSP spread.
    Only creates plots if hydro_model column is available.
    """
    if "hydro_model" not in annual_data.columns:
        print("No hydro_model column found, skipping hydro-grouped plots")
        return

    print("Creating hydro-grouped timeseries plots...")
    
    # Define colors for hydro models
    hydro_models = annual_data["hydro_model"].unique()
    hydro_colors = {
        "CWatM": "#1f77b4",
        "H08": "#ff7f0e", 
        "MIROC-INTEG-LAND": "#2ca02c",
        "WaterGAP2-2e": "#d62728",
        "JULES-W2": "#9467bd",
    }
    
    # Use default colors for any models not in our predefined set
    import matplotlib.pyplot as plt
    default_colors = plt.cm.Set1(np.linspace(0, 1, len(hydro_models)))
    for i, model in enumerate(hydro_models):
        if model not in hydro_colors:
            hydro_colors[model] = default_colors[i]

    n_basins = len(basin_matches)
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= 10:  # Only plot first 10
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin
        basin_data = annual_data[annual_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Plot each hydro model with climate model and SSP spread
        for hydro_model in hydro_models:
            hydro_data = basin_data[basin_data["hydro_model"] == hydro_model]

            if len(hydro_data) == 0:
                continue

            # Calculate spread across climate models and SSPs (min/max envelope)
            yearly_stats = (
                hydro_data.groupby("year")["qtot_30yr"]
                .agg(["min", "max", "mean", "std"])
                .reset_index()
            )

            if len(yearly_stats) > 0:
                # Plot mean line
                ax.plot(
                    yearly_stats["year"],
                    yearly_stats["mean"],
                    color=hydro_colors[hydro_model],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{hydro_model}",
                )

                # Add shaded region for model/SSP spread
                ax.fill_between(
                    yearly_stats["year"],
                    yearly_stats["min"],
                    yearly_stats["max"],
                    color=hydro_colors[hydro_model],
                    alpha=0.2,
                )

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("qtot_30yr")
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)

    # Remove empty subplots
    for j in range(len(basin_matches), 10):
        axes[j].remove()

    plt.suptitle(
        "Timeseries by Hydro Model\n(Shaded regions show climate model and SSP spread)",
        fontsize=16,
    )
    plt.tight_layout()

    if save_plots:
        plt.savefig(PLOTS_DIR / "hydro_grouped_timeseries.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOTS_DIR / "hydro_grouped_timeseries.pdf", bbox_inches="tight")

    plt.show()


def create_summary_statistics_plots(annual_data, basin_matches):
    """Create summary plots showing variance patterns."""
    print("Creating summary statistics plots...")

    # Calculate variance decomposition for each basin
    variance_data = []

    for target_basin, basin_info in basin_matches.items():
        basin_id = basin_info["id"]
        basin_data = annual_data[annual_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            continue

        # Calculate total variance and components
        total_var = basin_data["qtot_30yr"].var()

        # SSP variance (between scenarios)
        ssp_means = basin_data.groupby("ssp_scenario")["qtot_30yr"].mean()
        ssp_var = ssp_means.var() if len(ssp_means) > 1 else 0

        # Climate model variance (between climate models)
        climate_model_means = basin_data.groupby("climate_model")["qtot_30yr"].mean()
        climate_model_var = climate_model_means.var() if len(climate_model_means) > 1 else 0

        # Hydro model variance (between hydro models) if available
        hydro_model_var = 0
        if "hydro_model" in basin_data.columns:
            hydro_model_means = basin_data.groupby("hydro_model")["qtot_30yr"].mean()
            hydro_model_var = hydro_model_means.var() if len(hydro_model_means) > 1 else 0

        # Year variance (temporal trend)
        year_means = basin_data.groupby("year")["qtot_30yr"].mean()
        year_var = year_means.var() if len(year_means) > 1 else 0

        variance_entry = {
            "basin": target_basin,
            "basin_name": basin_info["name"][:20],
            "total_variance": total_var,
            "ssp_variance": ssp_var,
            "climate_model_variance": climate_model_var,
            "year_variance": year_var,
            "ssp_pct": (ssp_var / total_var * 100) if total_var > 0 else 0,
            "climate_model_pct": (climate_model_var / total_var * 100) if total_var > 0 else 0,
            "year_pct": (year_var / total_var * 100) if total_var > 0 else 0,
        }
        
        # Add hydro model variance if available
        if "hydro_model" in basin_data.columns:
            variance_entry.update({
                "hydro_model_variance": hydro_model_var,
                "hydro_model_pct": (hydro_model_var / total_var * 100) if total_var > 0 else 0,
            })
        
        variance_data.append(variance_entry)

    if not variance_data:
        print("No variance data to plot")
        return

    variance_df = pd.DataFrame(variance_data)

    # Create variance decomposition plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Stacked bar plot of variance percentages
    basins = variance_df["basin"]
    ssp_pct = variance_df["ssp_pct"]
    climate_model_pct = variance_df["climate_model_pct"]
    year_pct = variance_df["year_pct"]
    
    # Check if hydro model data is available
    has_hydro_models = "hydro_model_pct" in variance_df.columns
    
    if has_hydro_models:
        hydro_model_pct = variance_df["hydro_model_pct"]
        residual_pct = 100 - (ssp_pct + climate_model_pct + hydro_model_pct + year_pct)
        
        # Stack bars with hydro model included
        ax1.bar(basins, ssp_pct, label="SSP Scenarios", color="skyblue", alpha=0.8)
        ax1.bar(basins, climate_model_pct, bottom=ssp_pct, label="Climate Models", color="lightcoral", alpha=0.8)
        ax1.bar(basins, hydro_model_pct, bottom=ssp_pct + climate_model_pct, label="Hydro Models", color="orange", alpha=0.8)
        ax1.bar(basins, year_pct, bottom=ssp_pct + climate_model_pct + hydro_model_pct, label="Temporal Trend", color="lightgreen", alpha=0.8)
        ax1.bar(basins, residual_pct, bottom=ssp_pct + climate_model_pct + hydro_model_pct + year_pct, label="Residual", color="lightgray", alpha=0.8)
    else:
        residual_pct = 100 - (ssp_pct + climate_model_pct + year_pct)
        
        # Stack bars without hydro model
        ax1.bar(basins, ssp_pct, label="SSP Scenarios", color="skyblue", alpha=0.8)
        ax1.bar(basins, climate_model_pct, bottom=ssp_pct, label="Climate Models", color="lightcoral", alpha=0.8)
        ax1.bar(basins, year_pct, bottom=ssp_pct + climate_model_pct, label="Temporal Trend", color="lightgreen", alpha=0.8)
        ax1.bar(basins, residual_pct, bottom=ssp_pct + climate_model_pct + year_pct, label="Residual", color="lightgray", alpha=0.8)

    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Variance Decomposition by Basin")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Total variance magnitudes
    ax2.bar(
        basins,
        np.log10(variance_df["total_variance"] + 1),
        color="steelblue",
        alpha=0.8,
    )
    ax2.set_ylabel("Log10(Total Variance + 1)")
    ax2.set_title("Total Variance Magnitude by Basin")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "variance_decomposition_by_basin.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return variance_df


def main():
    """Main visualization function."""

    print("=" * 60)
    print("TIMESERIES VISUALIZATION FOR REPRESENTATIVE BASINS")
    print("=" * 60)

    try:
        # Load data
        df = load_timeseries_data()

        # Find representative basins
        basin_matches = find_representative_basins(df)

        if not basin_matches:
            print("No representative basins found!")
            return

        # Create annual timeseries
        annual_data = create_annual_timeseries(df, basin_matches)

        # Create SSP-grouped plots
        plot_ssp_grouped_timeseries(annual_data, basin_matches)

        # Create climate model-grouped plots
        plot_model_grouped_timeseries(annual_data, basin_matches)
        
        # Create hydro model-grouped plots (if hydro models are available)
        plot_hydro_grouped_timeseries(annual_data, basin_matches)

        # Create summary statistics
        variance_df = create_summary_statistics_plots(annual_data, basin_matches)

        # Save variance summary
        if variance_df is not None:
            variance_df.to_csv(PLOTS_DIR / "basin_variance_summary.csv", index=False)

        print(f"\nVisualization complete!")
        print(f"Plots saved to: {PLOTS_DIR}/")
        print("Files created:")
        print("- ssp_grouped_timeseries.png/pdf")
        print("- model_grouped_timeseries.png/pdf")
        if "hydro_model" in annual_data.columns:
            print("- hydro_grouped_timeseries.png/pdf")
        print("- variance_decomposition_by_basin.png")
        print("- basin_variance_summary.csv")

    except Exception as e:
        print(f"Error in visualization: {e}")
        raise


if __name__ == "__main__":
    main()

