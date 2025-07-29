#!/usr/bin/env python3
"""
Basin Mapping Analysis Script
Creates mapping between ISIMIP basins and local shapefile basins,
and generates layered visualization.
"""

import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load ISIMIP and local basin data"""
    print("Loading datasets...")

    # Load local shapefile basins
    local_basins = gpd.read_file("basins_delineated/basins_by_region_R12.shp")
    print(f"Loaded {len(local_basins)} local basins")

    # Load ISIMIP basin data
    ds = xr.open_dataset(
        "ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_basins_cru.nc"
    )
    print(f"Loaded ISIMIP data: {ds.sizes['lat']} x {ds.sizes['lon']} grid")

    return local_basins, ds


def valid_isimip_basins(ds):
    """Clean na basins"""

    # Get all valid basin data
    basin_data = ds.basinnumber.values
    valid_mask = ~np.isnan(basin_data)
    valid_basins = basin_data[valid_mask]

    # Count cells per basin
    basin_counts = Counter(valid_basins)
    basin_sizes = np.array(list(basin_counts.values()))
    return valid_basins, basin_counts


def create_basin_mapping(ds, local_basins, keep_basins):
    """Create mapping between ISIMIP basins and local basins"""
    print("Creating spatial mapping between basin datasets...")

    # Create mapping dictionary: local_basin_id -> list of ISIMIP basin info
    mapping = defaultdict(list)

    # Get coordinates for each ISIMIP grid cell
    lats, lons = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    basin_data = ds.basinnumber.values

    # Process each grid cell
    total_cells = np.sum(~np.isnan(basin_data))
    processed = 0

    for i in range(basin_data.shape[0]):
        for j in range(basin_data.shape[1]):
            basin_id = basin_data[i, j]

            if np.isnan(basin_id) or basin_id not in keep_basins:
                continue

            processed += 1
            if processed % 5000 == 0:
                print(
                    f"Processed {processed}/{total_cells} cells ({processed / total_cells * 100:.1f}%)"
                )

            # Create point for this grid cell
            lat, lon = lats[i, j], lons[i, j]
            point = Point(lon, lat)

            # Find which local basin(s) contain this point
            for idx, local_basin in local_basins.iterrows():
                if local_basin.geometry.contains(point):
                    mapping[local_basin["BASIN_ID"]].append(
                        {
                            "isimip_basin_id": int(basin_id),
                            "lat": lat,
                            "lon": lon,
                            "grid_i": i,
                            "grid_j": j,
                        }
                    )
                    break  # Assume each point belongs to only one local basin

    print(f"Completed mapping for {processed} ISIMIP basin cells")
    return dict(mapping)


def save_mapping_dataset(mapping, filename="mapp_basin_shape.csv"):
    """Save the basin mapping as a structured dataset"""
    print(f"Saving mapping dataset to {filename}...")

    # Convert to DataFrame
    rows = []
    for local_basin_id, isimip_basins in mapping.items():
        for basin_info in isimip_basins:
            rows.append(
                {
                    "local_basin_id": local_basin_id,
                    "isimip_basin_id": basin_info["isimip_basin_id"],
                    "lat": basin_info["lat"],
                    "lon": basin_info["lon"],
                    "grid_i": basin_info["grid_i"],
                    "grid_j": basin_info["grid_j"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

    # Print summary statistics
    print(f"Saved {len(df)} ISIMIP basin-cell mappings")
    print(f"Covers {df['local_basin_id'].nunique()} local basins")
    print(f"Contains {df['isimip_basin_id'].nunique()} unique ISIMIP basins")

    return df


def print_summary_statistics(mapping, local_basins):
    """Print summary statistics of the mapping"""
    print("\n=== BASIN MAPPING SUMMARY ===")

    total_local = len(local_basins)
    mapped_local = len(mapping)

    print(f"Local basins: {total_local} total, {mapped_local} have ISIMIP children")
    print(
        f"Coverage: {mapped_local / total_local * 100:.1f}% of local basins have child basins"
    )

    # Child basin statistics
    child_counts = [len(children) for children in mapping.values()]
    if child_counts:
        print("\nChild basin statistics:")
        print(f"  Total ISIMIP cells mapped: {sum(child_counts)}")
        print(f"  Average children per local basin: {np.mean(child_counts):.1f}")
        print(f"  Median children per local basin: {np.median(child_counts):.1f}")
        print(f"  Max children in one local basin: {max(child_counts)}")

        # Show top local basins by child count
        basin_child_counts = [(bid, len(children)) for bid, children in mapping.items()]
        basin_child_counts.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 5 local basins by child count:")
        for bid, count in basin_child_counts[:5]:
            local_name = local_basins[local_basins["BASIN_ID"] == bid]["NAME"].iloc[0]
            print(f"  {local_name} (ID {bid}): {count} children")


def main():
    """Main analysis pipeline"""
    print("Starting Basin Mapping Analysis")
    print("=" * 50)

    # Load data
    local_basins, ds = load_data()

    # Filter ISIMIP basins
    keep_basins, _ = valid_isimip_basins(ds)

    # Create basin mapping
    mapping = create_basin_mapping(ds, local_basins, keep_basins)

    # Save mapping dataset
    map_file_name = "mapp_basin_shape.csv"
    mapping_df = save_mapping_dataset(mapping, map_file_name)

    # Print summary
    print_summary_statistics(mapping, local_basins)

    # Cleanup
    ds.close()

    print("\nAnalysis complete!")
    print(f"- Mapping saved to: {map_file_name}")

    return mapping_df


if __name__ == "__main__":
    mapping_df = main()

