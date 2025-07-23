#!/usr/bin/env python3
"""
Simplified test to check if discharge outlet pixels (maximum locations) move over time.
Uses common Python GIS packages to verify the outlet-pixel approach.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json


def rasterize_basin(basin_geom, transform, shape):
    """Convert basin geometry to raster mask."""
    mask = features.rasterize(
        [(basin_geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8"
    )
    return mask.astype(bool)


def track_outlet_movement(
    nc_file: str, shapefile: str, maximizer_func, method_name: str,
    n_basins: int = 20, n_timesteps: int = 50
) -> Dict:
    """
    Track movement of outlet pixels (discharge maximum locations) over time.

    Returns:
        Dictionary with analysis results
    """
    print(
        f"\nTracking outlet pixel movement for {n_basins} basins over {n_timesteps} timesteps using {method_name}..."
    )

    # Load data
    print("Loading data...")
    gdf = gpd.read_file(shapefile)
    gdf = gdf.to_crs("EPSG:4326")
    ds = xr.open_dataset(nc_file)

    # Limit analysis
    n_basins = min(n_basins, len(gdf))
    n_timesteps = min(n_timesteps, len(ds.time))

    # Get grid info
    lons = ds.lon.values
    lats = ds.lat.values
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])

    # Create transform for rasterization
    transform = from_bounds(
        lons.min() - lon_res / 2,
        lats.min() - lat_res / 2,
        lons.max() + lon_res / 2,
        lats.max() + lat_res / 2,
        len(lons),
        len(lats),
    )

    # Results storage
    results = {
        "basin_analysis": {},
        "summary": {
            "method": method_name,
            "total_basins": n_basins,
            "fixed_outlet_basins": 0,
            "moving_outlet_basins": 0,
            "empty_basins": 0,
        },
    }

    # Analyze each basin
    for basin_idx in range(n_basins):
        # Use actual BASIN_ID from shapefile, not index-based naming
        basin_id = gdf.iloc[basin_idx]["BASIN_ID"] if "BASIN_ID" in gdf.columns else f"basin_{basin_idx}"
        print(f"\rAnalyzing basin {basin_idx + 1}/{n_basins} (ID: {basin_id})", end="")

        # Get basin geometry
        basin_geom = gdf.iloc[basin_idx].geometry

        # Track outlet locations
        outlet_locations = []
        outlet_values = []

        # Sample timesteps (every 5th to speed up)
        for t in range(0, n_timesteps, 5):
            # Get discharge data
            dis_data = ds.dis.isel(time=t).values

            # Use provided maximizer function
            max_loc, max_val = maximizer_func(dis_data, basin_geom, gdf.iloc[basin_idx], 
                                              lons, lats, transform, method_name)
            if max_loc is not None:
                outlet_locations.append(max_loc)
                outlet_values.append(float(max_val))

        # Analyze this basin
        if len(outlet_locations) == 0:
            results["basin_analysis"][str(float(basin_id))] = {
                "status": "empty",
                "n_unique_locations": 0,
            }
            results["summary"]["empty_basins"] += 1
        else:
            unique_locations = list(set(outlet_locations))
            n_unique = len(unique_locations)

            results["basin_analysis"][str(float(basin_id))] = {
                "status": "fixed" if n_unique == 1 else "moving",
                "n_unique_locations": n_unique,
                "unique_locations": [
                    (int(loc[0]), int(loc[1])) for loc in unique_locations
                ],
                "max_value_range": [
                    float(min(outlet_values)),
                    float(max(outlet_values)),
                ],
                "n_observations": len(outlet_locations),
            }

            if n_unique == 1:
                results["summary"]["fixed_outlet_basins"] += 1
            else:
                results["summary"]["moving_outlet_basins"] += 1

    print("\n")  # New line after progress

    # Calculate percentages
    results["summary"]["fixed_outlet_percentage"] = (
        results["summary"]["fixed_outlet_basins"] / n_basins * 100
    )
    results["summary"]["moving_outlet_percentage"] = (
        results["summary"]["moving_outlet_basins"] / n_basins * 100
    )

    ds.close()
    return results


def print_analysis_summary(results: Dict):
    """Print a summary of the outlet movement analysis."""
    summary = results["summary"]

    print("\n=== OUTLET PIXEL MOVEMENT ANALYSIS ===")
    print(f"Total basins analyzed: {summary['total_basins']}")
    print(
        f"Basins with FIXED outlet: {summary['fixed_outlet_basins']} ({summary['fixed_outlet_percentage']:.1f}%)"
    )
    print(
        f"Basins with MOVING outlet: {summary['moving_outlet_basins']} ({summary['moving_outlet_percentage']:.1f}%)"
    )
    print(f"Empty basins: {summary['empty_basins']}")

    # Show examples of moving outlets
    print("\n=== Examples of basins with moving outlets ===")
    count = 0
    for basin_id, info in results["basin_analysis"].items():
        if info["status"] == "moving" and count < 5:
            print(f"\nBasin {basin_id}:")
            print(f"  Number of unique outlet locations: {info['n_unique_locations']}")
            print(f"  Locations: {info['unique_locations'][:3]}...")  # Show first 3
            print(
                f"  Discharge range: {info['max_value_range'][0]:.2f} - {info['max_value_range'][1]:.2f} m³/s"
            )
            count += 1


def maximizer_rasterio(dis_data, basin_geom, basin_row, lons, lats, transform, method_name):
    """Maximizer using rasterio rasterization."""
    from shapely.geometry import Point
    # Rasterize basin
    basin_mask = rasterize_basin(basin_geom, transform, (len(lats), len(lons)))
    
    # Apply basin mask
    masked_dis = np.where(basin_mask, dis_data, np.nan)
    
    # Find maximum
    if not np.all(np.isnan(masked_dis)):
        max_idx = np.nanargmax(masked_dis)
        max_loc = np.unravel_index(max_idx, masked_dis.shape)
        max_val = masked_dis[max_loc]
        
        # Assertion: Check if the outlet is inside the basin
        lat, lon = lats[max_loc[0]], lons[max_loc[1]]
        assert basin_geom.contains(Point(lon, lat)), f"[{method_name}] Outlet at {max_loc} (lon: {lon}, lat: {lat}) is outside the basin"

        return max_loc, max_val
    return None, None


def maximizer_rasterstats(dis_data, basin_geom, basin_row, lons, lats, transform, method_name):
    """Maximizer using rasterstats."""
    from rasterstats import zonal_stats
    from shapely.geometry import Point
    
    # Calculate zonal statistics
    stats = zonal_stats(basin_geom, dis_data, 
                       affine=transform, 
                       stats=['max'],
                       all_touched=True)
    
    if stats and stats[0]['max'] is not None:
        max_val = stats[0]['max']
        
        # Find location of maximum
        basin_mask = rasterize_basin(basin_geom, transform, (len(lats), len(lons)))
        masked_dis = np.where(basin_mask, dis_data, np.nan)
        
        # Get all indices matching the max value
        max_indices = np.argwhere(masked_dis == max_val)
        
        # Select the first valid location
        for idx in max_indices:
            max_loc = tuple(idx)
            lat, lon = lats[max_loc[0]], lons[max_loc[1]]
            point = Point(lon, lat)
            if basin_geom.contains(point):
                # Assertion: Check if the outlet is inside the basin
                assert basin_geom.contains(point), f"[{method_name}] Outlet at {max_loc} (lon: {lon}, lat: {lat}) is outside the basin"
                return max_loc, max_val
        
        # Fallback if no point is strictly inside (e.g., due to boundary issues)
        # but this should ideally not be reached if logic is correct
        if len(max_indices) > 0:
            max_loc = tuple(max_indices[0])
            return max_loc, max_val

    return None, None


def maximizer_numpy(dis_data, basin_geom, basin_row, lons, lats, transform, method_name):
    """Maximizer using numpy with manual basin mask."""
    # Create basin mask using shapely contains
    from shapely.geometry import Point
    
    basin_mask = np.zeros(dis_data.shape, dtype=bool)
    for i in range(len(lats)):
        for j in range(len(lons)):
            point = Point(lons[j], lats[i])
            if basin_geom.contains(point):
                basin_mask[i, j] = True
    
    # Apply mask and find maximum
    masked_dis = np.where(basin_mask, dis_data, np.nan)
    
    if not np.all(np.isnan(masked_dis)):
        max_idx = np.nanargmax(masked_dis)
        max_loc = np.unravel_index(max_idx, masked_dis.shape)
        max_val = masked_dis[max_loc]

        # Assertion: Check if the outlet is inside the basin
        lat, lon = lats[max_loc[0]], lons[max_loc[1]]
        assert basin_geom.contains(Point(lon, lat)), f"[{method_name}] Outlet at {max_loc} (lon: {lon}, lat: {lat}) is outside the basin"

        return max_loc, max_val
    return None, None


def test_multiple_packages(config):
    """Test multiple package implementations for finding outlet pixels."""
    print("\n=== Testing multiple methods for outlet pixel identification ===")
    
    methods = [
        ("rasterio", maximizer_rasterio),
        ("rasterstats", maximizer_rasterstats),
        ("numpy", maximizer_numpy)
    ]
    
    all_results = {}
    
    for method_name, maximizer_func in methods:
        print(f"\nTesting {method_name}...")
        try:
            results = track_outlet_movement(
                config["nc_file"],
                config["shapefile"],
                maximizer_func,
                method_name,
                config["n_basins"],
                config["n_timesteps"]
            )
            all_results[method_name] = results
            print_analysis_summary(results)
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            all_results[method_name] = {"error": str(e)}
    
    # Compare results across methods
    compare_method_results(all_results)
    
    return all_results


def compare_method_results(all_results):
    """Compare outlet pixel results across different methods."""
    print("\n\n=== COMPARISON ACROSS METHODS ===")
    
    # Extract successful methods
    successful_methods = {k: v for k, v in all_results.items() if "error" not in v}
    
    if len(successful_methods) < 2:
        print("Not enough successful methods to compare")
        return
    
    # Compare summary statistics
    print("\nSummary comparison:")
    print(f"{'Method':<15} {'Fixed':<10} {'Moving':<10} {'Empty':<10} {'Fixed %':<10}")
    print("-" * 55)
    
    for method, results in successful_methods.items():
        summary = results['summary']
        print(f"{method:<15} {summary['fixed_outlet_basins']:<10} "
              f"{summary['moving_outlet_basins']:<10} {summary['empty_basins']:<10} "
              f"{summary['fixed_outlet_percentage']:<10.1f}")
    
    # Compare basin-by-basin results
    method_names = list(successful_methods.keys())
    if len(method_names) >= 2:
        print(f"\nDetailed comparison between {method_names[0]} and {method_names[1]}:")
        
        method1_basins = successful_methods[method_names[0]]['basin_analysis']
        method2_basins = successful_methods[method_names[1]]['basin_analysis']
        
        agreement_count = 0
        disagreement_count = 0
        
        for basin_id in method1_basins:
            if basin_id in method2_basins:
                loc1 = method1_basins[basin_id].get('unique_locations', [])
                loc2 = method2_basins[basin_id].get('unique_locations', [])
                
                if loc1 and loc2:
                    # Convert to sets for comparison
                    loc1_set = set(tuple(l) for l in loc1)
                    loc2_set = set(tuple(l) for l in loc2)
                    
                    if loc1_set == loc2_set:
                        agreement_count += 1
                    else:
                        disagreement_count += 1
                        if disagreement_count <= 5:  # Show first 5 disagreements
                            print(f"\nBasin {basin_id} disagreement:")
                            print(f"  {method_names[0]}: {loc1_set}")
                            print(f"  {method_names[1]}: {loc2_set}")
        
        print(f"\nAgreement on outlet locations: {agreement_count} basins")
        print(f"Disagreement on outlet locations: {disagreement_count} basins")
        
        if agreement_count + disagreement_count > 0:
            agreement_percentage = agreement_count / (agreement_count + disagreement_count) * 100
            print(f"Agreement percentage: {agreement_percentage:.1f}%")


def main():
    """Main test function."""
    print("=== DISCHARGE OUTLET PIXEL CONSTANCY TEST ===")
    print("Testing if discharge maximum locations (outlets) remain fixed over time")

    # Configuration - adjust paths as needed
    config = {
        "nc_file": "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_dis_global_monthly_2015_2100.nc",
        "shapefile": "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp",
        "n_basins": 10,  # Reduced for faster testing with multiple methods
        "n_timesteps": 30,  # Reduced for faster testing
    }

    # Check if files exist
    if not Path(config["nc_file"]).exists():
        print(f"\nError: NetCDF file not found: {config['nc_file']}")
        return

    if not Path(config["shapefile"]).exists():
        print(f"\nError: Shapefile not found: {config['shapefile']}")
        return

    # Test multiple packages and compare results
    all_results = test_multiple_packages(config)

    # Save combined results
    output_file = "outlet_movement_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nDetailed comparison results saved to {output_file}")


if __name__ == "__main__":
    main()

