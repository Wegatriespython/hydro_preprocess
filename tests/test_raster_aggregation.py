#!/usr/bin/env python3
"""
Pytest test suite for comparing Julia Rasters.jl vs Python rasterio aggregation methods.
"""

import subprocess
import pandas as pd
import numpy as np
import os
import sys
import pytest

class TestRasterAggregation:
    """Test class for raster aggregation method comparison."""
    
    @pytest.fixture(scope="class")
    def test_outputs(self):
        """Run Julia and Python tests and return output filenames."""
        # Change to tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_dir)
        
        # Output files
        julia_output = "julia_test_output.csv"
        python_output = "python_test_output.csv"
        
        # Clean up previous outputs
        for f in [julia_output, python_output]:
            if os.path.exists(f):
                os.remove(f)
        
        # Run Julia test
        project_path = os.path.join(os.path.dirname(test_dir), "pre_processing")
        result = subprocess.run(["julia", f"--project={project_path}", "test_julia_raster.jl"], 
                              capture_output=True, text=True, check=True)
        assert os.path.exists(julia_output), "Julia test failed to produce output"
        
        # Run Python test
        result = subprocess.run(["uv", "run", "python", "test_python_raster.py"], 
                              capture_output=True, text=True, check=True)
        assert os.path.exists(python_output), "Python test failed to produce output"
        
        yield julia_output, python_output
        
        # Clean up
        for f in [julia_output, python_output]:
            if os.path.exists(f):
                os.remove(f)
    
    @pytest.fixture(scope="class")
    def comparison_data(self, test_outputs):
        """Load and prepare comparison data."""
        julia_output, python_output = test_outputs
        
        # Load results
        julia_df = pd.read_csv(julia_output)
        python_df = pd.read_csv(python_output)
        
        # Extract numeric columns
        julia_cols = [col for col in julia_df.columns if col.startswith('timestep_')]
        python_cols = [col for col in python_df.columns if col.startswith('timestep_')]
        
        julia_data = julia_df[julia_cols].values
        python_data = python_df[python_cols].values
        
        return julia_data, python_data, julia_df, python_df
    
    def test_output_shapes_match(self, comparison_data):
        """Test that Julia and Python outputs have matching shapes."""
        julia_data, python_data, julia_df, python_df = comparison_data
        
        assert julia_data.shape == python_data.shape, \
            f"Shape mismatch: Julia {julia_data.shape} vs Python {python_data.shape}"
        
        assert len(julia_df) == len(python_df), \
            f"Basin count mismatch: Julia {len(julia_df)} vs Python {len(python_df)}"
    
    def test_overall_correlation(self, comparison_data):
        """Test overall correlation between Julia and Python results."""
        julia_data, python_data, _, _ = comparison_data
        
        correlation = np.corrcoef(julia_data.flatten(), python_data.flatten())[0, 1]
        
        assert correlation >= 0.99, \
            f"Overall correlation {correlation:.6f} is below acceptable threshold of 0.99"
    
    def test_basin_correlations(self, comparison_data):
        """Test per-basin correlations between Julia and Python results."""
        julia_data, python_data, _, _ = comparison_data
        
        basin_corrs = []
        for i in range(len(julia_data)):
            if np.std(julia_data[i]) > 1e-10 and np.std(python_data[i]) > 1e-10:
                corr = np.corrcoef(julia_data[i], python_data[i])[0, 1]
                basin_corrs.append(corr)
        
        mean_basin_corr = np.mean(basin_corrs)
        min_basin_corr = np.min(basin_corrs)
        
        assert mean_basin_corr >= 0.98, \
            f"Mean basin correlation {mean_basin_corr:.6f} is below acceptable threshold of 0.98"
        
        assert min_basin_corr >= 0.95, \
            f"Minimum basin correlation {min_basin_corr:.6f} is below acceptable threshold of 0.95"
    
    def test_relative_differences(self, comparison_data):
        """Test relative differences between Julia and Python results."""
        julia_data, python_data, _, _ = comparison_data
        
        # Avoid division by zero
        mask = np.abs(julia_data) > 1e-10
        rel_diff = np.mean(np.abs(julia_data[mask] - python_data[mask]) / np.abs(julia_data[mask]))
        max_rel_diff = np.max(np.abs(julia_data[mask] - python_data[mask]) / np.abs(julia_data[mask]))
        
        assert rel_diff <= 0.1, \
            f"Mean relative difference {rel_diff*100:.2f}% exceeds acceptable threshold of 10%"
        
        assert max_rel_diff <= 0.5, \
            f"Maximum relative difference {max_rel_diff*100:.2f}% exceeds acceptable threshold of 50%"
    
    def test_no_extreme_outliers(self, comparison_data):
        """Test that there are no extreme outliers in the differences."""
        julia_data, python_data, _, _ = comparison_data
        
        abs_diff = np.abs(julia_data - python_data)
        julia_scale = np.median(np.abs(julia_data))
        
        # Check that no absolute difference is more than 100x the median scale
        max_acceptable_diff = 100 * julia_scale
        extreme_outliers = np.sum(abs_diff > max_acceptable_diff)
        
        assert extreme_outliers == 0, \
            f"Found {extreme_outliers} extreme outliers with differences > {max_acceptable_diff:.2e}"
    
    def test_data_ranges_similar(self, comparison_data):
        """Test that Julia and Python data ranges are similar."""
        julia_data, python_data, _, _ = comparison_data
        
        julia_range = np.max(julia_data) - np.min(julia_data)
        python_range = np.max(python_data) - np.min(python_data)
        
        range_ratio = python_range / julia_range
        
        assert 0.5 <= range_ratio <= 2.0, \
            f"Data range ratio {range_ratio:.2f} indicates significant scaling difference"
    
    def test_basin_metadata_consistency(self, comparison_data):
        """Test that basin metadata is consistent between outputs."""
        _, _, julia_df, python_df = comparison_data
        
        # Check basin IDs match
        assert julia_df['BASIN_ID'].tolist() == python_df['BASIN_ID'].tolist(), \
            "Basin IDs do not match between Julia and Python outputs"
        
        # Check basin names match
        assert julia_df['BCU_name'].tolist() == python_df['BCU_name'].tolist(), \
            "Basin names do not match between Julia and Python outputs"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])