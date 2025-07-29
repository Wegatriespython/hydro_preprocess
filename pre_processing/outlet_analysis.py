#!/usr/bin/env python3
"""
Generic Basin Outlet Analysis
Configurable tool for analyzing basin outlets and flow accumulation.
Supports multiple basins and configurable output generation.
"""

import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import argparse
import os
from pathlib import Path

warnings.filterwarnings("ignore")


# Basin name to ID mapping for representative basins
BASIN_NAME_TO_ID = {
    "Amazon": 9.0,
    "Congo": 8.0,
    "Yangtze": 159.0,
    "Ganges Bramaputra": 98.0,
    "Indus": 99.0,
    "Danube": 111.0,
    "Mekong": 149.0,
    "Mississippi": 127.0,
    "Nile": 41.0,
    "Persian Gulf Coast": 95.0,
    "Australia Interior": 18.0,
}


def load_representative_basins(basin_file):
    """Load list of representative basins from file"""

    with open(basin_file, "r") as f:
        content = f.read()
        # Extract basin names from the Python list format
        import ast

        start = content.find("[")
        end = content.find("]") + 1
        list_str = content[start:end]
        basins = ast.literal_eval(list_str)
        return basins


def load_basin_data(
    basin_name,
    mapping_file="/home/raghunathan/hydro_preprocess/pre_processing/mapp_basin_shape.csv",
    shapefile_path="/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_R12.shp",
):
    """Load basin data for specified basin

    Args:
        basin_name: Name of the basin to load
        mapping_file: Path to the basin mapping CSV file
        shapefile_path: Path to the basin shapefile

    Returns:
        tuple: (basin_data_df, basin_info_gdf_row)
    """
    if basin_name not in BASIN_NAME_TO_ID:
        raise ValueError(
            f"Basin '{basin_name}' not found. Available basins: {list(BASIN_NAME_TO_ID.keys())}"
        )

    basin_id = BASIN_NAME_TO_ID[basin_name]
    print(f"Loading {basin_name} basin data (ID: {basin_id})...")

    # Load mapping data
    df = pd.read_csv(mapping_file)
    local_basins = gpd.read_file(shapefile_path)

    # Get basin data
    basin_data = df[df["local_basin_id"] == basin_id].copy()

    if len(basin_data) == 0:
        raise ValueError(f"No data found for basin '{basin_name}' (ID: {basin_id})")

    basin_info = local_basins[local_basins["BASIN_ID"] == basin_id].iloc[0]

    print(f"{basin_name} basin: {len(basin_data)} cells")
    print(f"Lat range: {basin_data['lat'].min():.1f} to {basin_data['lat'].max():.1f}")
    print(f"Lon range: {basin_data['lon'].min():.1f} to {basin_data['lon'].max():.1f}")

    return basin_data, basin_info


def build_flow_network(basin_data, basin_name):
    """Build flow network from basin cells"""
    print(f"Building flow network for {basin_name}...")

    # Load flow direction data
    ds = xr.open_dataset(
        "/home/raghunathan/ISIMIP/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_flowdir_cru.nc"
    )

    # Direction offsets (D8)
    direction_offsets = {
        1: (0, 1),  # East
        2: (1, 1),  # Southeast
        3: (1, 0),  # South
        4: (1, -1),  # Southwest
        5: (0, -1),  # West
        6: (-1, -1),  # Northwest
        7: (-1, 0),  # North
        8: (-1, 1),  # Northeast
    }

    # Create network
    network = {}
    coord_to_cell = {}
    outlets = []

    for idx, row in basin_data.iterrows():
        i, j = int(row["grid_i"]), int(row["grid_j"])
        flow_dir = ds.flowdirection.values[i, j]

        cell_id = f"{i}_{j}"
        coord_to_cell[(i, j)] = cell_id

        network[cell_id] = {
            "lat": row["lat"],
            "lon": row["lon"],
            "isimip_basin_id": row["isimip_basin_id"],
            "flow_direction": flow_dir,
            "downstream": None,
            "upstream": [],
            "flow_accumulation": 1,  # Each cell contributes 1 unit
        }

        # Check if this is an outlet (flows to ocean or undefined)
        if flow_dir == -1:
            outlets.append(cell_id)
            network[cell_id]["is_outlet"] = True
        else:
            network[cell_id]["is_outlet"] = False

    # Build connections
    boundary_outlets = []

    for cell_id, cell_data in network.items():
        flow_dir = cell_data["flow_direction"]

        if flow_dir in direction_offsets:
            # Calculate downstream coordinates
            i, j = [int(x) for x in cell_id.split("_")]
            di, dj = direction_offsets[flow_dir]
            downstream_i, downstream_j = i + di, j + dj

            # Check if downstream cell exists in our basin
            if (downstream_i, downstream_j) in coord_to_cell:
                downstream_id = coord_to_cell[(downstream_i, downstream_j)]
                network[cell_id]["downstream"] = downstream_id
                network[downstream_id]["upstream"].append(cell_id)
            else:
                # Flow exits basin - this is a boundary outlet
                boundary_outlets.append(cell_id)
                network[cell_id]["is_outlet"] = True
                outlets.append(cell_id)

    ds.close()

    print(f"Network built: {len(network)} cells")
    print(
        f"Direct outlets (ocean): {len([o for o in outlets if network[o]['flow_direction'] == -1])}"
    )
    print(f"Boundary outlets: {len(boundary_outlets)}")
    print(f"Total outlets: {len(outlets)}")

    return network, outlets


def calculate_flow_accumulation(network, outlets):
    """Calculate flow accumulation using topological sorting

    This algorithm processes cells from sources (headwaters) to sinks (outlets)
    ensuring all upstream contributions are calculated before downstream cells.
    Uses Kahn's algorithm for topological sorting to handle complex networks.
    """
    print("Calculating flow accumulation...")

    # Initialize all cells with flow accumulation = 1 (their own contribution)
    for cell_id in network:
        network[cell_id]["flow_accumulation"] = 1

    # Find all source cells (no upstream connections)
    sources = [
        cell_id for cell_id, data in network.items() if len(data["upstream"]) == 0
    ]
    print(f"Found {len(sources)} source cells")

    # Implement Kahn's algorithm for topological sorting
    # Count incoming edges (upstream connections) for each cell
    in_degree = {}
    for cell_id in network:
        in_degree[cell_id] = len(network[cell_id]["upstream"])

    # Initialize queue with source cells (in_degree = 0)
    from collections import deque

    queue = deque(sources)
    processed_count = 0

    # Process cells in topological order
    while queue:
        current_cell = queue.popleft()
        processed_count += 1

        # Get downstream cell
        downstream_cell = network[current_cell]["downstream"]

        if downstream_cell is not None:
            # Add current cell's flow accumulation to downstream cell
            network[downstream_cell]["flow_accumulation"] += network[current_cell][
                "flow_accumulation"
            ]

            # Decrease in-degree of downstream cell
            in_degree[downstream_cell] -= 1

            # If downstream cell has no more upstream dependencies, add to queue
            if in_degree[downstream_cell] == 0:
                queue.append(downstream_cell)

    # Check if we processed all cells (detect cycles)
    if processed_count != len(network):
        print(f"Warning: Only processed {processed_count}/{len(network)} cells")
        print("This may indicate cycles in the flow network")

        # Handle any remaining unprocessed cells with cycles
        # Use DFS with cycle detection for remaining cells
        remaining_cells = [cell_id for cell_id in network if in_degree[cell_id] > 0]
        if remaining_cells:
            print(f"Processing {len(remaining_cells)} cells with potential cycles...")
            _handle_cycles(network, remaining_cells)

    print(f"Flow accumulation calculated for all {len(network)} cells")

    # Calculate outlet accumulations for return value
    outlet_accumulations = {}
    for outlet_id in outlets:
        accumulation = network[outlet_id]["flow_accumulation"]
        outlet_accumulations[outlet_id] = accumulation

        cell_data = network[outlet_id]
        outlet_type = "Ocean" if cell_data["flow_direction"] == -1 else "Boundary"
        print(
            f"Outlet at {cell_data['lat']:.1f}°, {cell_data['lon']:.1f}°: {accumulation} cells ({outlet_type})"
        )

    return outlet_accumulations


def _handle_cycles(network, cells_with_cycles):
    """Handle cells that may be part of cycles using DFS with cycle detection

    This is a fallback for handling any remaining cells that weren't processed
    by the main topological sort, likely due to cycles in the flow network.
    """
    visited = set()
    recursion_stack = set()

    def dfs_accumulate(cell_id):
        if cell_id in recursion_stack:
            # Cycle detected - assign minimal accumulation to break cycle
            print(f"Cycle detected involving cell {cell_id}")
            return network[cell_id]["flow_accumulation"]

        if cell_id in visited:
            return network[cell_id]["flow_accumulation"]

        visited.add(cell_id)
        recursion_stack.add(cell_id)

        # Calculate accumulation from upstream cells
        upstream_total = 0
        for upstream_id in network[cell_id]["upstream"]:
            upstream_total += dfs_accumulate(upstream_id)

        # Set flow accumulation (own contribution + upstream)
        network[cell_id]["flow_accumulation"] = 1 + upstream_total

        recursion_stack.remove(cell_id)
        return network[cell_id]["flow_accumulation"]

    # Process all cells with potential cycles
    for cell_id in cells_with_cycles:
        if cell_id not in visited:
            dfs_accumulate(cell_id)


def find_main_outlet(network, outlet_accumulations, basin_name):
    """Find the main outlet with highest flow accumulation"""
    print(f"\\nFinding main outlet for {basin_name}...")

    # Sort outlets by accumulation
    sorted_outlets = sorted(
        outlet_accumulations.items(), key=lambda x: x[1], reverse=True
    )

    print("Top 10 outlets by flow accumulation:")
    for i, (outlet_id, accumulation) in enumerate(sorted_outlets[:10]):
        cell_data = network[outlet_id]
        outlet_type = "Ocean" if cell_data["flow_direction"] == -1 else "Boundary"
        print(
            f"  {i + 1}. {cell_data['lat']:.2f}°, {cell_data['lon']:.2f}°: {accumulation} cells ({outlet_type})"
        )

    # Main outlet
    main_outlet_id, main_accumulation = sorted_outlets[0]
    main_cell = network[main_outlet_id]

    print(f"\\nMAIN {basin_name.upper()} OUTLET:")
    print(f"   Location: {main_cell['lat']:.3f}°N, {main_cell['lon']:.3f}°W")
    print(
        f"   Flow accumulation: {main_accumulation} cells ({main_accumulation / len(network) * 100:.1f}% of basin)"
    )
    print(
        f"   Type: {'Ocean outlet' if main_cell['flow_direction'] == -1 else 'Boundary exit'}"
    )
    print(f"   ISIMIP Basin ID: {main_cell['isimip_basin_id']}")

    return main_outlet_id, main_cell, sorted_outlets


def visualize_flow_accumulation(
    network, main_outlet_id, basin_info, basin_name, output_dir="."
):
    """Visualize the flow accumulation"""
    print(f"Creating flow accumulation visualization for {basin_name}...")

    # Extract data for plotting
    lats = [data["lat"] for data in network.values()]
    lons = [data["lon"] for data in network.values()]
    accumulations = [data["flow_accumulation"] for data in network.values()]

    # Main outlet location
    main_cell = network[main_outlet_id]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Flow accumulation
    scatter1 = ax1.scatter(
        lons,
        lats,
        c=accumulations,
        cmap="Blues",
        s=np.log10(np.array(accumulations) + 1) * 20,
        alpha=0.7,
    )
    ax1.scatter(
        main_cell["lon"],
        main_cell["lat"],
        c="red",
        s=200,
        marker="*",
        label=f"Main outlet ({main_cell['flow_accumulation']} cells)",
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(f"{basin_name} Basin - Flow Accumulation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="Flow Accumulation (cells)")

    # Plot 2: Log scale accumulation for better visibility
    log_accumulations = np.log10(np.array(accumulations))
    scatter2 = ax2.scatter(
        lons, lats, c=log_accumulations, cmap="viridis", s=15, alpha=0.8
    )
    ax2.scatter(
        main_cell["lon"],
        main_cell["lat"],
        c="red",
        s=200,
        marker="*",
        label=f"Main outlet",
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title(f"{basin_name} Basin - Log Flow Accumulation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Log10(Flow Accumulation)")

    plt.tight_layout()
    output_file = os.path.join(
        output_dir, f"{basin_name.lower().replace(' ', '_')}_flow_accumulation.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")

    return fig


def save_results(network, sorted_outlets, basin_name, output_dir=".", save_csv=False):
    """Save results to files"""
    if not save_csv:
        print("Skipping CSV output (not requested)")
        return

    print(f"Saving results for {basin_name}...")

    # Save all outlets
    outlet_data = []
    for outlet_id, accumulation in sorted_outlets:
        cell_data = network[outlet_id]
        outlet_data.append(
            {
                "lat": cell_data["lat"],
                "lon": cell_data["lon"],
                "flow_accumulation": accumulation,
                "outlet_type": "Ocean"
                if cell_data["flow_direction"] == -1
                else "Boundary",
                "isimip_basin_id": cell_data["isimip_basin_id"],
            }
        )

    outlet_df = pd.DataFrame(outlet_data)
    outlet_file = os.path.join(
        output_dir, f"{basin_name.lower().replace(' ', '_')}_outlets_ranked.csv"
    )
    outlet_df.to_csv(outlet_file, index=False)

    # Save full network data
    network_data = []
    for cell_id, data in network.items():
        i, j = [int(x) for x in cell_id.split("_")]
        network_data.append(
            {
                "grid_i": i,
                "grid_j": j,
                "lat": data["lat"],
                "lon": data["lon"],
                "flow_accumulation": data["flow_accumulation"],
                "flow_direction": data["flow_direction"],
                "is_outlet": data["is_outlet"],
                "isimip_basin_id": data["isimip_basin_id"],
            }
        )

    network_df = pd.DataFrame(network_data)
    network_file = os.path.join(
        output_dir, f"{basin_name.lower().replace(' ', '_')}_flow_network.csv"
    )
    network_df.to_csv(network_file, index=False)

    print("Results saved:")
    print(f"  - {outlet_file}: All outlets ranked by flow")
    print(f"  - {network_file}: Complete flow network data")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generic Basin Outlet Analysis Tool")

    parser.add_argument(
        "--basin",
        "-b",
        type=str,
        help="Name of basin to analyze (e.g., 'Amazon', 'Congo'). Use --list-basins to see available options.",
    )

    parser.add_argument(
        "--list-basins",
        "-l",
        action="store_true",
        help="List all available representative basins",
    )

    parser.add_argument(
        "--all-basins",
        "-a",
        action="store_true",
        help="Process all representative basins",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)",
    )

    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save CSV results (outlets and network data)",
    )

    parser.add_argument(
        "--mapping-file",
        "-m",
        type=str,
        default="mapp_basin_shape.csv",
        help="Path to basin mapping CSV file (default: mapp_basin_shape.csv)",
    )

    parser.add_argument(
        "--shapefile",
        "-s",
        type=str,
        default="/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_R12.shp",
        help="Path to basin shapefile",
    )

    return parser.parse_args()


def analyze_basin(basin_name, args):
    """Analyze a single basin"""
    print(f"\\n{'=' * 60}")
    print(f"Analyzing {basin_name} Basin")
    print(f"{'=' * 60}")

    try:
        # Load data
        basin_data, basin_info = load_basin_data(
            basin_name, mapping_file=args.mapping_file, shapefile_path=args.shapefile
        )

        # Build network
        network, outlets = build_flow_network(basin_data, basin_name)

        # Calculate flow accumulation
        outlet_accumulations = calculate_flow_accumulation(network, outlets)

        # Find main outlet
        main_outlet_id, main_cell, sorted_outlets = find_main_outlet(
            network, outlet_accumulations, basin_name
        )

        # Generate visualization (PNG output - always generated)
        fig = visualize_flow_accumulation(
            network, main_outlet_id, basin_info, basin_name, args.output_dir
        )

        # Save CSV results if requested
        save_results(
            network, sorted_outlets, basin_name, args.output_dir, args.save_csv
        )

        print(f"\\n{basin_name} analysis complete!")
        return main_cell, network

    except Exception as e:
        print(f"Error analyzing {basin_name}: {str(e)}")
        return None, None


def main():
    """Main analysis function"""
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.list_basins:
        print("Available representative basins:")
        representative_basins = load_representative_basins(
            "/home/raghunathan/hydro_preprocess/represent_basin.txt"
        )
        for i, basin in enumerate(representative_basins, 1):
            basin_id = BASIN_NAME_TO_ID.get(basin, "Unknown")
            print(f"  {i:2d}. {basin} (ID: {basin_id})")
        return

    if args.all_basins:
        print("Processing all representative basins...")
        representative_basins = load_representative_basins(
            "/home/raghunathan/hydro_preprocess/represent_basin.txt"
        )

        results = {}

        for basin_name in representative_basins:
            if basin_name in BASIN_NAME_TO_ID:
                main_cell, network = analyze_basin(basin_name, args)
                results[basin_name] = (main_cell, network)
            else:
                print(f"Warning: Basin '{basin_name}' not found in mapping")

        print(f"\\n\\nProcessed {len(results)} basins successfully")
        return results

    elif args.basin:
        # Analyze single basin
        return analyze_basin(args.basin, args)

    else:
        print("Error: Must specify --basin, --all-basins, or --list-basins")
        print("Use --help for usage information")
        return None


if __name__ == "__main__":
    main()
