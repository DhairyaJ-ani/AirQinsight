"""Unit and integration tests for multi-pollutant sample GeoTIFF datasets.

Verifies:
1. 305 authentic GeoTIFF rasters exist (61 days x 5 pollutants).
2. Rasterio grid dimensions, bounds, CRS, nodata values match the Mumbai reference grid.
3. Manifest integrity: counts, columns, dates, validity, and metadata attributes.
4. Downscale router endpoints serve dates and GeoTIFFs for all 5 new pollutants.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
import rasterio

from app.dependencies import get_current_user
from app.main import app

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
SAMPLES_DIR = DATA_DIR / "pollutant_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.csv"

EXPECTED_POLLUTANTS = ["PM25", "PM10", "SO2", "O3", "CO"]
REFERENCE_BOUNDS = (72.77, 18.865, 73.12, 19.32)


def test_manifest_structure_and_completeness():
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        reader = list(csv.DictReader(fh))

    assert len(reader) == 305, f"Expected 305 rows (61 days * 5 pollutants), found {len(reader)}"

    pollutant_counts = {}
    for row in reader:
        p = row["pollutant"]
        pollutant_counts[p] = pollutant_counts.get(p, 0) + 1
        filepath_str = row["filepath"]
        full_path = BACKEND_DIR / filepath_str.removeprefix("backend/")
        assert full_path.exists(), f"File {filepath_str} listed in manifest does not exist at {full_path}"
        assert row["crs"] == "EPSG:4326"
        assert float(row["valid_pixel_fraction"]) > 0.0

    for p in EXPECTED_POLLUTANTS:
        assert pollutant_counts.get(p) == 61, f"Pollutant {p} expected 61 files, got {pollutant_counts.get(p)}"


@pytest.mark.parametrize("pollutant", EXPECTED_POLLUTANTS)
def test_pollutant_geotiff_properties(pollutant: str):
    subfolder = SAMPLES_DIR / pollutant
    assert subfolder.exists()
    tifs = sorted(subfolder.glob("*.tif"))
    assert len(tifs) == 61

    # Check first, middle, and last file in each pollutant folder
    check_files = [tifs[0], tifs[30], tifs[-1]]
    for f in check_files:
        with rasterio.open(f) as src:
            assert src.crs.to_string() == "EPSG:4326"
            assert src.width == 10
            assert src.height == 13
            assert src.count == 1
            assert np.isnan(src.nodata)

            data = src.read(1)
            assert data.shape == (13, 10)
            valid = data[np.isfinite(data)]
            assert valid.size > 0
            assert np.all(valid >= 0.0)

            # Bounds within 0.001 deg of reference
            w, s, e, n = src.bounds
            assert abs(w - REFERENCE_BOUNDS[0]) < 0.001
            assert abs(s - REFERENCE_BOUNDS[1]) < 0.001
            assert abs(e - REFERENCE_BOUNDS[2]) < 0.001
            assert abs(n - REFERENCE_BOUNDS[3]) < 0.001


@pytest.mark.asyncio
@pytest.mark.parametrize("pollutant", ["pm25", "pm10", "so2", "o3", "co"])
async def test_downscale_router_serves_dates_for_pollutant(auth_client, pollutant: str):
    response = await auth_client.get(f"/api/v1/downscale/dates?pollutant={pollutant}")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert len(data["dates"]) == 61
    assert data["dates"][0].startswith("2025-11-01")
    assert data["dates"][-1].startswith("2025-12-31")


@pytest.mark.asyncio
@pytest.mark.parametrize("pollutant", ["pm25", "pm10", "so2", "o3", "co"])
async def test_downscale_router_serves_geotiff_for_pollutant(auth_client, pollutant: str):
    response = await auth_client.get(f"/api/v1/downscale/geotiff?timestamp=2025-11-15T00:00:00Z&pollutant={pollutant}")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/tiff", "application/octet-stream"]
    assert len(response.content) > 100
