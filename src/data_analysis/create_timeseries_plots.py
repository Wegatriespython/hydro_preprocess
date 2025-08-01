#!/usr/bin/env python3
"""
Timeseries Visualization for Representative Basins
=================================================

Creates timeseries plots showing temporal patterns in var for representative basins.
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
import argparse

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/raghunathan/hydro_preprocess/anova_results/")
PLOTS_DIR = Path("/home/raghunathan/hydro_preprocess/anova_plots/timeseries_plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Representative basins from file
REPRESENTATIVE_BASINS = [
    "Amazon",
    "Congo",
    "Yangtze",
    "Ganges Bramaputra",
    "Indus",
    "Danube",
    "Mekong",
    "Mississippi",
    "Nile",
    "Persian Gulf Coast",
    "Australia Interior",
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
    "ssp370": "#2ca02c",  # Green - medium emissions
    "ssp585": "#d62728",  # Red - high emissions
}

CLIMATE_MODEL_COLORS = {
    "gfdl-esm4": "#1f77b4",
    "ipsl-cm6a-lr": "#ff7f0e",
    "mpi-esm1-2-hr": "#2ca02c",
    "mri-esm2-0": "#d62728",
    "ukesm1-0-ll": "#9467bd",
}
HYDRO_MODEL_COLORS = {
    "CWatM": "#1f77b4",
    "H08": "#ff7f0e",
    "MIROC-INTEG-LAND": "#2ca02c",
    "WaterGAP2-2e": "#d62728",
    "JULES-W2": "#9467bd",
}


def load_timeseries_data(var):
    """Load the combined long-format data with all basins."""
    print("Loading timeseries data...")

    data_file = DATA_DIR / f"qtot_monthly_rolling_averages.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"])

    # TEMPORARY: Exclude CWatM to test L-shaped jump hypothesis
    print("TEMPORARY: Excluding CWatM model to test transition")
    original_len = len(df)
    df = df[df["hydro_model"] != "CWatM"].copy()
    print(f"Excluded CWatM: {original_len:,} -> {len(df):,} data points")

    print(f"Loaded {len(df):,} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Available basins: {df['NAME'].nunique()}")

    return df


def find_representative_basins(df, var="qtot_mean_30yr"):
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


def create_annual_timeseries(df, basin_matches, var="qtot_mean_30yr"):
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
        .agg({f"{var}": "mean", "REGION": "first"})
        .reset_index()
    )

    print(
        f"Created annual data: {len(annual_data)} basin-model-scenario-year combinations"
    )

    # Show available hydro models if present
    if "hydro_model" in annual_data.columns:
        hydro_models = annual_data["hydro_model"].unique()
        print(f"Available hydro models: {hydro_models}")

    return annual_data


def create_monthly_timeseries(df, basin_matches, var="qtot_mean"):
    """Create monthly timeseries preserving month information."""
    print("Creating monthly timeseries...")

    # Filter for representative basins only
    basin_ids = [info["id"] for info in basin_matches.values()]
    df_filtered = df[df["BASIN_ID"].isin(basin_ids)].copy()

    # Ensure we have year and month columns
    if "year" not in df_filtered.columns:
        df_filtered["year"] = df_filtered["date"].dt.year
    if "month" not in df_filtered.columns:
        df_filtered["month"] = df_filtered["date"].dt.month

    # Add month name for better labeling
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    df_filtered["month_name"] = df_filtered["month"].map(month_names)

    print(
        f"Created monthly data: {len(df_filtered)} basin-model-scenario-month combinations"
    )

    return df_filtered


def create_regional_timeseries(df, var="qtot_mean_30yr", temporal_resolution="annual"):
    """
    Create regional aggregated timeseries at specified temporal resolution.

    Parameters:
    - df: Input dataframe with basin-level data
    - var: Variable to aggregate
    - temporal_resolution: Either "annual" or "monthly"
    """
    print(f"Creating regional {temporal_resolution} timeseries...")

    # Add year column if not present
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    # Base grouping columns
    grouping_cols = ["REGION", "climate_model", "ssp_scenario"]
    if "hydro_model" in df.columns:
        grouping_cols.append("hydro_model")

    # Add temporal columns based on resolution
    if temporal_resolution == "annual":
        grouping_cols.append("year")
    elif temporal_resolution == "monthly":
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        grouping_cols.extend(["year", "month"])
        # Preserve date column for monthly data
        date_col = ["date"]
    else:
        raise ValueError(f"Invalid temporal_resolution: {temporal_resolution}")

    # Aggregate by region (sum across basins within each region)
    agg_dict = {f"{var}": "sum"}
    if temporal_resolution == "monthly" and "date" in df.columns:
        agg_dict["date"] = "first"  # Keep the date for monthly data

    regional_data = df.groupby(grouping_cols).agg(agg_dict).reset_index()

    # Create fake BASIN_ID and NAME columns to work with existing plotting functions
    regional_data["BASIN_ID"] = regional_data["REGION"]
    regional_data["NAME"] = regional_data["REGION"]

    # Add month name for monthly data
    if temporal_resolution == "monthly":
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        regional_data["month_name"] = regional_data["month"].map(month_names)

    print(
        f"Created regional data: {len(regional_data)} region-model-scenario-{temporal_resolution} combinations"
    )
    print(f"Available regions: {regional_data['REGION'].unique()}")

    # Create region matches dictionary to work with existing plotting functions
    region_matches = {}
    for region in regional_data["REGION"].unique():
        region_matches[region] = {"id": region, "name": region}

    return regional_data, region_matches


def timeseries_plot(
    annual_data,
    basin_matches,
    group_by,
    color_map,
    var="qtot_mean_30yr",
    is_regional=False,
    plot_type="spread",
    title_suffix="",
    save_name=None,
):
    """
    Generic function to create timeseries plots with different grouping strategies.

    Parameters:
    - group_by: Column name to group by (e.g., 'ssp_scenario', 'climate_model', 'hydro_model')
    - color_map: Dictionary mapping group values to colors
    - plot_type: 'spread' for min/max envelope, 'lines' for individual lines
    - title_suffix: Additional text for the main title
    """
    print(f"Creating {group_by}-grouped timeseries plots...")

    # Filter data to start from 2030 & 2090
    annual_data = annual_data[
        (annual_data["year"] >= 2030) & (annual_data["year"] <= 2090)
    ].copy()
    print(
        f"Filtered data to start from 2030 and 2090. Data points remaining: {len(annual_data)}"
    )

    n_basins = len(basin_matches)
    if is_regional:
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        max_plots = 12
    else:
        fig, axes = plt.subplots(2, 5, figsize=(25, 12))
        max_plots = 10
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= max_plots:
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

        # Get unique groups
        groups = basin_data[group_by].unique()

        if plot_type == "spread":
            # For spread plots, we need a secondary grouping
            if group_by == "ssp_scenario":
                secondary_group = "climate_model"
            elif group_by == "climate_model":
                secondary_group = "ssp_scenario"
            else:
                secondary_group = None

            for group in groups:
                group_data = basin_data[basin_data[group_by] == group]

                if len(group_data) == 0:
                    continue

                # Calculate spread
                yearly_stats = (
                    group_data.groupby("year")[f"{var}"]
                    .agg(["min", "max", "mean", "std"])
                    .reset_index()
                )

                if len(yearly_stats) > 0:
                    # Plot mean line
                    ax.plot(
                        yearly_stats["year"],
                        yearly_stats["mean"],
                        color=color_map.get(group, "#000000"),
                        linewidth=2,
                        alpha=0.8,
                        label=f"{group.upper()}"
                        if hasattr(group, "upper")
                        else str(group),
                    )

                    # Add shaded region for spread
                    ax.fill_between(
                        yearly_stats["year"],
                        yearly_stats["min"],
                        yearly_stats["max"],
                        color=color_map.get(group, "#000000"),
                        alpha=0.2,
                    )

        elif plot_type == "lines":
            # For line plots (e.g., hydro models)
            for group in groups:
                group_data = basin_data[basin_data[group_by] == group]

                if len(group_data) == 0:
                    continue

                # Average across any other dimensions
                yearly_means = group_data.groupby("year")[f"{var}"].mean().reset_index()

                if len(yearly_means) > 0:
                    ax.plot(
                        yearly_means["year"],
                        yearly_means[f"{var}"],
                        color=color_map.get(group, "#000000"),
                        linewidth=2,
                        alpha=0.8,
                        label=group,
                    )

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var}")
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True)

    # Remove empty subplots
    for j in range(len(basin_matches), max_plots):
        if j < len(axes):
            axes[j].remove()

    plt.suptitle(title_suffix, fontsize=16)
    plt.tight_layout()

    if save_name:
        plt.savefig(PLOTS_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOTS_DIR / f"{save_name}.pdf", bbox_inches="tight")

    plt.show()


def plot_ssp_grouped_timeseries(
    annual_data, basin_matches, save_plots=True, var="qtot_mean_30yr", is_regional=False
):
    """
    Create timeseries plots grouped by SSP scenario.
    Each SSP gets a shaded region representing model spread.
    """
    timeseries_plot(
        annual_data,
        basin_matches,
        group_by="ssp_scenario",
        color_map=SSP_COLORS,
        var=var,
        is_regional=is_regional,
        plot_type="spread",
        title_suffix="Timeseries by SSP Scenario\n(Shaded regions show climate model spread)",
        save_name="ssp_grouped_timeseries" if save_plots else None,
    )


def plot_model_grouped_timeseries(
    annual_data, basin_matches, save_plots=True, var="qtot_mean_30yr", is_regional=False
):
    """
    Create timeseries plots grouped by climate model.
    Each model gets a shaded region representing SSP spread.
    """
    timeseries_plot(
        annual_data,
        basin_matches,
        group_by="climate_model",
        color_map=CLIMATE_MODELS,
        var=var,
        is_regional=is_regional,
        plot_type="spread",
        title_suffix="Timeseries by Climate Model\n(Shaded regions show SSP scenario spread)",
        save_name="model_grouped_timeseries" if save_plots else None,
    )


def plot_hydro_grouped_timeseries(
    annual_data, basin_matches, save_plots=True, var="qtot_mean_30yr", is_regional=False
):
    """
    Create timeseries plots grouped by hydro model.
    Each hydro model gets a shaded region representing climate model and SSP spread.
    Only creates plots if hydro_model column is available.
    """
    timeseries_plot(
        annual_data,
        basin_matches,
        group_by="hydro_model",
        color_map=HYDRO_MODEL_COLORS,
        var=var,
        is_regional=is_regional,
        plot_type="spread",
        title_suffix="Timeseries by Hydro Model\n(Shaded regions show climate model and SSP spread)",
        save_name="hydro_grouped_timeseries" if save_plots else None,
    )


def plot_monthly_climatology(
    monthly_data,
    basin_matches,
    group_by="ssp_scenario",
    color_map=None,
    var="qtot_mean",
    title_suffix="",
    save_name=None,
    is_regional=False,
):
    """
    Create monthly climatology plots showing average seasonal patterns.

    Parameters:
    - monthly_data: DataFrame with monthly timeseries data
    - basin_matches: Dictionary of basin information
    - group_by: Column to group by (e.g., 'ssp_scenario', 'climate_model')
    - color_map: Dictionary mapping group values to colors
    """
    print(f"Creating monthly climatology plots grouped by {group_by}...")

    # Set default color map if not provided
    if color_map is None:
        if group_by == "ssp_scenario":
            color_map = SSP_COLORS
        elif group_by == "climate_model":
            color_map = CLIMATE_MODEL_COLORS
        elif group_by == "hydro_model":
            color_map = HYDRO_MODEL_COLORS

    n_basins = len(basin_matches)
    if is_regional:
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        max_plots = 12
    else:
        fig, axes = plt.subplots(2, 5, figsize=(30, 12))
        max_plots = 10
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= max_plots:
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin
        basin_data = monthly_data[monthly_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Get unique groups
        groups = basin_data[group_by].unique()

        for group in groups:
            group_data = basin_data[basin_data[group_by] == group]

            if len(group_data) == 0:
                continue

            # Calculate monthly averages across all years
            monthly_avg = (
                group_data.groupby("month")
                .agg({var: ["mean", "std", "min", "max"]})
                .reset_index()
            )

            # Flatten column names
            monthly_avg.columns = [
                "month",
                f"{var}_mean",
                f"{var}_std",
                f"{var}_min",
                f"{var}_max",
            ]

            # Plot mean line
            ax.plot(
                monthly_avg["month"],
                monthly_avg[f"{var}_mean"],
                color=color_map.get(group, "#000000"),
                linewidth=2,
                alpha=0.8,
                label=f"{group.upper()}" if hasattr(group, "upper") else str(group),
                marker="o",
                markersize=4,
            )

            # Add shaded region for standard deviation
            ax.fill_between(
                monthly_avg["month"],
                monthly_avg[f"{var}_mean"] - monthly_avg[f"{var}_std"],
                monthly_avg[f"{var}_mean"] + monthly_avg[f"{var}_std"],
                color=color_map.get(group, "#000000"),
                alpha=0.2,
            )

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{var}")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.grid(True, alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True)

    # Remove empty subplots
    for j in range(len(basin_matches), max_plots):
        if j < len(axes):
            axes[j].remove()

    plt.suptitle(f"Monthly Climatology {title_suffix}", fontsize=16)
    plt.tight_layout()

    if save_name:
        plt.savefig(PLOTS_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOTS_DIR / f"{save_name}.pdf", bbox_inches="tight")

    plt.show()


def plot_seasonal_boxplots(
    monthly_data,
    basin_matches,
    group_by="ssp_scenario",
    var="qtot_mean",
    title_suffix="",
    save_name=None,
    is_regional=False,
):
    """
    Create seasonal boxplots showing distribution of values for each month.
    """
    print(f"Creating seasonal boxplots grouped by {group_by}...")

    n_basins = len(basin_matches)
    if is_regional:
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        max_plots = 12
    else:
        fig, axes = plt.subplots(2, 5, figsize=(30, 12))
        max_plots = 10
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= max_plots:
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin
        basin_data = monthly_data[monthly_data["BASIN_ID"] == basin_id]

        if len(basin_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Create boxplot data structure
        box_data = []
        positions = []
        colors = []
        labels = []

        groups = sorted(basin_data[group_by].unique())
        n_groups = len(groups)

        for month in range(1, 13):
            month_data = basin_data[basin_data["month"] == month]
            month_position = month - 0.5 + (0.8 / n_groups) * 0.5

            for g_idx, group in enumerate(groups):
                group_month_data = month_data[month_data[group_by] == group][var].values
                if len(group_month_data) > 0:
                    box_data.append(group_month_data)
                    positions.append(month_position + g_idx * (0.8 / n_groups))

                    if group_by == "ssp_scenario":
                        colors.append(SSP_COLORS.get(group, "#000000"))
                    elif group_by == "climate_model":
                        colors.append(CLIMATE_MODEL_COLORS.get(group, "#000000"))
                    elif group_by == "hydro_model":
                        colors.append(HYDRO_MODEL_COLORS.get(group, "#000000"))

                    if month == 1:  # Only add labels once
                        labels.append(group)

        # Create boxplots
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.8 / n_groups,
            patch_artist=True,
            showfliers=False,
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{var}")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend for first subplot
        if i == 0 and n_groups > 1:
            # Create custom legend
            from matplotlib.patches import Patch

            legend_elements = []
            for group in groups:
                if group_by == "ssp_scenario":
                    color = SSP_COLORS.get(group, "#000000")
                elif group_by == "climate_model":
                    color = CLIMATE_MODEL_COLORS.get(group, "#000000")
                elif group_by == "hydro_model":
                    color = HYDRO_MODEL_COLORS.get(group, "#000000")
                legend_elements.append(Patch(facecolor=color, alpha=0.7, label=group))
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Remove empty subplots
    for j in range(len(basin_matches), max_plots):
        if j < len(axes):
            axes[j].remove()

    plt.suptitle(f"Seasonal Distribution {title_suffix}", fontsize=16)
    plt.tight_layout()

    if save_name:
        plt.savefig(PLOTS_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOTS_DIR / f"{save_name}.pdf", bbox_inches="tight")

    plt.show()


def plot_monthly_heatmaps(
    monthly_data,
    basin_matches,
    scenario="ssp370",
    climate_model="gfdl-esm4",
    var="qtot_mean",
    title_suffix="",
    save_name=None,
    is_regional=False,
):
    """
    Create heatmaps showing monthly patterns over years.
    """
    print(f"Creating monthly heatmaps for {scenario}, {climate_model}...")

    n_basins = len(basin_matches)
    if is_regional:
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        max_plots = 12
    else:
        fig, axes = plt.subplots(2, 5, figsize=(30, 15))
        max_plots = 10
    axes = axes.flatten()

    for i, (target_basin, basin_info) in enumerate(basin_matches.items()):
        if i >= max_plots:
            break

        ax = axes[i]
        basin_id = basin_info["id"]
        basin_name = basin_info["name"]

        # Filter data for this basin, scenario, and model
        basin_data = monthly_data[
            (monthly_data["BASIN_ID"] == basin_id)
            & (monthly_data["ssp_scenario"] == scenario)
            & (monthly_data["climate_model"] == climate_model)
        ]

        if len(basin_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{target_basin}\n(No Data)")
            continue

        # Pivot to create year x month matrix
        pivot_data = basin_data.pivot_table(
            index="year", columns="month", values=var, aggfunc="mean"
        )

        # Create heatmap
        im = ax.imshow(
            pivot_data.values, aspect="auto", cmap="viridis", interpolation="nearest"
        )

        # Set ticks
        ax.set_xticks(range(12))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

        # Set y-axis to show every 5th year
        years = pivot_data.index
        year_ticks = range(0, len(years), 5)
        ax.set_yticks(year_ticks)
        ax.set_yticklabels([years[i] for i in year_ticks])

        ax.set_title(f"{target_basin}\n{basin_name[:30]}", fontsize=10)
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(var, fontsize=8)

    # Remove empty subplots
    for j in range(len(basin_matches), max_plots):
        if j < len(axes):
            axes[j].remove()

    plt.suptitle(
        f"Monthly Patterns Heatmap\n{scenario.upper()}, {climate_model} {title_suffix}",
        fontsize=16,
    )
    plt.tight_layout()

    if save_name:
        plt.savefig(PLOTS_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(PLOTS_DIR / f"{save_name}.pdf", bbox_inches="tight")

    plt.show()


def create_summary_statistics_plots(annual_data, basin_matches, var="qtot_mean_30yr"):
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
        total_var = basin_data[f"{var}"].var()

        # SSP variance (between scenarios)
        ssp_means = basin_data.groupby("ssp_scenario")[f"{var}"].mean()
        ssp_var = ssp_means.var() if len(ssp_means) > 1 else 0

        # Climate model variance (between climate models)
        climate_model_means = basin_data.groupby("climate_model")[f"{var}"].mean()
        climate_model_var = (
            climate_model_means.var() if len(climate_model_means) > 1 else 0
        )

        # Hydro model variance (between hydro models) if available
        hydro_model_var = 0
        if "hydro_model" in basin_data.columns:
            hydro_model_means = basin_data.groupby("hydro_model")[f"{var}"].mean()
            hydro_model_var = (
                hydro_model_means.var() if len(hydro_model_means) > 1 else 0
            )

        # Year variance (temporal trend)
        year_means = basin_data.groupby("year")[f"{var}"].mean()
        year_var = year_means.var() if len(year_means) > 1 else 0

        variance_entry = {
            "basin": target_basin,
            "basin_name": basin_info["name"][:20],
            "total_variance": total_var,
            "ssp_variance": ssp_var,
            "climate_model_variance": climate_model_var,
            "year_variance": year_var,
            "ssp_pct": (ssp_var / total_var * 100) if total_var > 0 else 0,
            "climate_model_pct": (climate_model_var / total_var * 100)
            if total_var > 0
            else 0,
            "year_pct": (year_var / total_var * 100) if total_var > 0 else 0,
        }

        # Add hydro model variance if available
        if "hydro_model" in basin_data.columns:
            variance_entry.update(
                {
                    "hydro_model_variance": hydro_model_var,
                    "hydro_model_pct": (hydro_model_var / total_var * 100)
                    if total_var > 0
                    else 0,
                }
            )

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
        ax1.bar(
            basins,
            climate_model_pct,
            bottom=ssp_pct,
            label="Climate Models",
            color="lightcoral",
            alpha=0.8,
        )
        ax1.bar(
            basins,
            hydro_model_pct,
            bottom=ssp_pct + climate_model_pct,
            label="Hydro Models",
            color="orange",
            alpha=0.8,
        )
        ax1.bar(
            basins,
            year_pct,
            bottom=ssp_pct + climate_model_pct + hydro_model_pct,
            label="Temporal Trend",
            color="lightgreen",
            alpha=0.8,
        )
        ax1.bar(
            basins,
            residual_pct,
            bottom=ssp_pct + climate_model_pct + hydro_model_pct + year_pct,
            label="Residual",
            color="lightgray",
            alpha=0.8,
        )
    else:
        residual_pct = 100 - (ssp_pct + climate_model_pct + year_pct)

        # Stack bars without hydro model
        ax1.bar(basins, ssp_pct, label="SSP Scenarios", color="skyblue", alpha=0.8)
        ax1.bar(
            basins,
            climate_model_pct,
            bottom=ssp_pct,
            label="Climate Models",
            color="lightcoral",
            alpha=0.8,
        )
        ax1.bar(
            basins,
            year_pct,
            bottom=ssp_pct + climate_model_pct,
            label="Temporal Trend",
            color="lightgreen",
            alpha=0.8,
        )
        ax1.bar(
            basins,
            residual_pct,
            bottom=ssp_pct + climate_model_pct + year_pct,
            label="Residual",
            color="lightgray",
            alpha=0.8,
        )

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
    parser = argparse.ArgumentParser(description="Generate timeseries for variables")
    parser.add_argument(
        "--clim_model_spread",
        action="store_true",
        default=False,
        help="Plot spread along climate models",
    )
    parser.add_argument(
        "--hydro_model_spread",
        action="store_true",
        default=False,
        help="Plot spread among hydrology models",
    )
    parser.add_argument(
        "--ssp_spread",
        action="store_true",
        default=False,
        help="Timeseries plot of spread in RCP forcing among models.",
    )
    parser.add_argument(
        "--var", type=str, default="qtot_mean_30yr", help="var to plot for timeseries"
    )
    parser.add_argument(
        "--regional",
        action="store_true",
        default=False,
        help="Create regional aggregated plots instead of basin plots",
    )
    # Monthly visualization arguments
    parser.add_argument(
        "--monthly",
        action="store_true",
        default=False,
        help="Create monthly climatology plots showing seasonal patterns",
    )
    parser.add_argument(
        "--seasonal",
        action="store_true",
        default=False,
        help="Create seasonal boxplots showing monthly distributions",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="Create monthly heatmaps showing patterns over years",
    )
    parser.add_argument(
        "--monthly_var",
        type=str,
        default="qtot_mean",
        help="Variable to use for monthly plots (default: qtot_mean)",
    )
    args = parser.parse_args()
    print("=" * 60)
    if args.regional:
        print("TIMESERIES VISUALIZATION FOR REGIONS")
    else:
        print("TIMESERIES VISUALIZATION FOR REPRESENTATIVE BASINS")
    print("=" * 60)
    var = args.var
    try:
        # Load data
        df = load_timeseries_data(var)

        if args.regional:
            # Create regional aggregated data
            annual_data, region_matches = create_regional_timeseries(
                df, var, temporal_resolution="annual"
            )
            matches_to_use = region_matches
            print("Using regional aggregation mode")
        else:
            # Find representative basins
            basin_matches = find_representative_basins(df, var)

            if not basin_matches:
                print("No representative basins found!")
                return

            # Create annual timeseries
            annual_data = create_annual_timeseries(df, basin_matches, var)
            matches_to_use = basin_matches
            print("Using basin-level mode")

        # Create SSP-grouped plots
        if args.ssp_spread:
            timeseries_plot(
                annual_data,
                matches_to_use,
                group_by="ssp_scenario",
                color_map=SSP_COLORS,
                var=var,
                is_regional=args.regional,
                plot_type="spread",
                title_suffix="Timeseries by SSP Scenario\n(Shaded regions show Climate and Hydro model spread)",
                save_name="ssp_grouped_timeseries",
            )

        # Create climate model-grouped plots
        if args.clim_model_spread:
            timeseries_plot(
                annual_data,
                matches_to_use,
                group_by="climate_model",
                color_map=CLIMATE_MODEL_COLORS,
                var=var,
                is_regional=args.regional,
                plot_type="spread",
                title_suffix="Timeseries by Climate Model\n(Shaded regions show Hydro Model and SSP spread)",
                save_name="model_grouped_timeseries",
            )

        # Create hydro model-grouped plots (if hydro models are available)
        if args.hydro_model_spread:
            timeseries_plot(
                annual_data,
                matches_to_use,
                group_by="hydro_model",
                color_map=HYDRO_MODEL_COLORS,
                var=var,
                is_regional=args.regional,
                plot_type="spread",
                title_suffix="Timeseries by Hydro Model\n(Shaded regions show Climate model and SSP spread)",
                save_name="hydro_grouped_timeseries",
            )

        # Create monthly visualizations if requested
        if args.monthly or args.seasonal or args.heatmap:
            print("\n" + "=" * 60)
            print("CREATING MONTHLY VISUALIZATIONS")
            print("=" * 60)

            # Create monthly timeseries data
            if args.regional:
                monthly_data, _ = create_regional_timeseries(
                    df, args.monthly_var, temporal_resolution="monthly"
                )
            else:
                monthly_data = create_monthly_timeseries(
                    df, matches_to_use, var=args.monthly_var
                )

            # Monthly climatology plots
            if args.monthly:
                # Group by SSP scenario
                plot_monthly_climatology(
                    monthly_data,
                    matches_to_use,
                    group_by="ssp_scenario",
                    var=args.monthly_var,
                    title_suffix="by SSP Scenario",
                    save_name="monthly_climatology_ssp",
                    is_regional=args.regional,
                )

                # Group by climate model
                plot_monthly_climatology(
                    monthly_data,
                    matches_to_use,
                    group_by="climate_model",
                    var=args.monthly_var,
                    title_suffix="by Climate Model",
                    save_name="monthly_climatology_model",
                    is_regional=args.regional,
                )

                # Group by hydro model if available
                if "hydro_model" in monthly_data.columns:
                    plot_monthly_climatology(
                        monthly_data,
                        matches_to_use,
                        group_by="hydro_model",
                        var=args.monthly_var,
                        title_suffix="by Hydro Model",
                        save_name="monthly_climatology_hydro",
                        is_regional=args.regional,
                    )

            # Seasonal boxplots
            if args.seasonal:
                plot_seasonal_boxplots(
                    monthly_data,
                    matches_to_use,
                    group_by="ssp_scenario",
                    var=args.monthly_var,
                    title_suffix="by SSP Scenario",
                    save_name="seasonal_boxplots_ssp",
                    is_regional=args.regional,
                )

                plot_seasonal_boxplots(
                    monthly_data,
                    matches_to_use,
                    group_by="climate_model",
                    var=args.monthly_var,
                    title_suffix="by Climate Model",
                    save_name="seasonal_boxplots_model",
                    is_regional=args.regional,
                )

            # Monthly heatmaps
            if args.heatmap:
                # Create heatmaps for different scenario/model combinations
                scenarios = ["ssp126", "ssp370", "ssp585"]
                models = ["gfdl-esm4", "mpi-esm1-2-hr"]

                for scenario in scenarios:
                    for model in models:
                        plot_monthly_heatmaps(
                            monthly_data,
                            matches_to_use,
                            scenario=scenario,
                            climate_model=model,
                            var=args.monthly_var,
                            save_name=f"monthly_heatmap_{scenario}_{model}",
                            is_regional=args.regional,
                        )

        # Create summary statistics
        # variance_df = create_summary_statistics_plots(annual_data, basin_matches, var)

        # # Save variance summary
        # if variance_df is not None:
        #     variance_df.to_csv(PLOTS_DIR / "basin_variance_summary.csv", index=False)

    except Exception as e:
        print(f"Error in visualization: {e}")
        raise


if __name__ == "__main__":
    main()
