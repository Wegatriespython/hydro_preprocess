"""
Pytest configuration for raster aggregation tests.
"""

import pytest
import os
import sys

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

def pytest_collection_modifyitems(config, items):
    """Add slow marker to tests that run Julia/Python subprocesses."""
    for item in items:
        if "test_outputs" in item.fixturenames:
            item.add_marker(pytest.mark.slow)

@pytest.fixture(scope="session")
def test_data_available():
    """Check if required test data files are available."""
    required_files = [
        "/mnt/p/watxene/ISIMIP/ISIMIP3b/OutputData/CWatM/ssp126/GFDL-ESM4/cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc",
        "/home/raghunathan/ISIMIP/basins_delineated/basins_by_region_simpl_R12.shp"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        pytest.skip(f"Required test data files not found: {missing_files}")
    
    return True