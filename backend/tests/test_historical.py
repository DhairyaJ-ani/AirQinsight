"""Tests for AirQ Insight historical dataset endpoints and multi-pollutant registry.

Covers:
1. Historical dataset listing and metadata validation.
2. Pollutant availability matrix (demonstrating NO₂ available, SO₂/PM/O₃ unavailable for raster with reasoning).
3. Manifest endpoint for both 61-day dataset and 8-sample dataset.
4. Coverage endpoint reporting total days, days with data, and missing fractions.
5. Grid endpoint: row-major JSON array with NaN as null, correct units (µmol/m²), and measurement type.
6. GeoTIFF byte streaming.
7. Pollutant registry scientific integrity (units, standards, capabilities).
8. Specialist agent routing & multi-pollutant threshold adherence.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml_engine.pollutant_registry import (
    POLLUTANT_REGISTRY,
    get_registry_entry,
    get_standard,
    list_registry_entries,
)
from app.agent.state import AgentState

client = TestClient(app)


def test_historical_datasets_list():
    resp = client.get("/api/v1/historical/datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    assert isinstance(datasets, list)
    assert len(datasets) >= 1

    ids = [d["id"] for d in datasets]
    assert "no2_61day_2025" in ids
    
    # Verify metadata fields on no2_61day_2025
    no2_ds = next(d for d in datasets if d["id"] == "no2_61day_2025")
    assert no2_ds["pollutant"] == "no2"
    assert no2_ds["measurement_type"] == "satellite_column"
    assert no2_ds["units"] == "µmol/m²"
    assert no2_ds["crs"] == "EPSG:4326"
    assert no2_ds["native_resolution_deg"] == 0.035


def test_historical_pollutants_availability():
    resp = client.get("/api/v1/historical/pollutants/availability")
    assert resp.status_code == 200
    avail = resp.json()
    assert "no2" in avail
    assert avail["no2"]["has_raster_data_nov_dec_2025"] is True
    assert "no2_61day_2025" in avail["no2"]["dataset_ids"]

    # Now all target pollutants (PM2.5, PM10, SO2, CO, O3) have real 61-day GeoTIFF datasets
    for p in ["pm25", "pm10", "so2", "co", "o3"]:
        assert p in avail
        assert avail[p]["has_raster_data_nov_dec_2025"] is True
        assert len(avail[p]["dataset_ids"]) >= 1



def test_historical_manifest():
    resp = client.get("/api/v1/historical/no2_61day_2025/manifest")
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["id"] == "no2_61day_2025"
    assert manifest["pollutant"] == "no2"
    assert manifest["date_count"] == 61
    assert "2025-11-05" in manifest["dates_with_data"]
    assert len(manifest["per_date_coverage"]) == 61


def test_historical_coverage():
    resp = client.get("/api/v1/historical/no2_61day_2025/no2/coverage")
    assert resp.status_code == 200
    cov = resp.json()
    assert cov["dataset_id"] == "no2_61day_2025"
    assert cov["pollutant"] == "no2"
    assert cov["total_days_in_window"] == 61
    assert cov["days_with_data"] == 61
    assert len(cov["dates"]) == 61

    # Requesting non-existent pollutant on no2 dataset raises 404
    resp_bad = client.get("/api/v1/historical/no2_61day_2025/pm25/coverage")
    assert resp_bad.status_code == 404


def test_historical_grid_endpoint():
    resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/grid")
    assert resp.status_code == 200
    grid = resp.json()
    assert grid["dataset_id"] == "no2_61day_2025"
    assert grid["pollutant"] == "no2"
    assert grid["measurement_type"] == "satellite_column"
    assert grid["units"] == "µmol/m²"
    assert grid["date"] == "2025-11-05"
    assert grid["width"] == 10
    assert grid["height"] == 13
    assert len(grid["values"]) == 13
    assert len(grid["values"][0]) == 10
    assert grid["valid_pixels"] > 0

    # Ensure NaN pixels are serialized as None/null, not 0
    # Nov 1, 2, 3 have cloud-masked NaN values
    resp_nov1 = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-01/grid")
    assert resp_nov1.status_code == 200
    grid_nov1 = resp_nov1.json()
    # Should have some null values
    flattened = [val for row in grid_nov1["values"] for val in row]
    assert any(v is None for v in flattened)


def test_historical_geotiff_stream():
    resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/geotiff")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/tiff"
    assert len(resp.content) > 0


def test_pollutant_registry_metadata():
    entries = list_registry_entries()
    assert len(entries) >= 6
    no2 = get_registry_entry("no2")
    assert no2.formula == "NO2"
    assert no2.model_capability == "production"

    pm25 = get_registry_entry("pm25")
    assert pm25.model_capability == "production"
    assert pm25.canonical_units == "µg/m³"

    # Verify regulatory standard lookup
    std_no2 = get_standard("no2", "24h")
    assert std_no2 is not None
    assert std_no2.concentration_limit == 80.0
    assert std_no2.unit == "µg/m³"

    std_pm25 = get_standard("pm25", "24h")
    assert std_pm25 is not None
    assert std_pm25.concentration_limit == 60.0
    assert std_pm25.unit == "µg/m³"


@pytest.mark.asyncio
async def test_specialist_agent_compliance_guard():
    from app.agent.specialists import compliance_agent

    # When satellite column data is supplied, compliance agent must set measurement_type_warning
    # and NOT blindly report a violation of surface concentration standard
    state: AgentState = {
        "pollutant": "no2",
        "measurement_type": "satellite_column",
        "pollutant_units": "µmol/m²",
        "data_source": "Sentinel-5P TROPOMI",
        "user_query": "Check compliance for Mumbai",
        "analysis": {
            "mean_no2": 120.0,
            "max_no2": 250.0,
            "hotspot_count": 3,
            "evidence": [],
        },
    }
    result = await compliance_agent(state)
    compliance = result.get("compliance_result")
    assert compliance is not None
    assert compliance["measurement_type_warning"] is not None
    assert "column" in compliance["measurement_type_warning"].lower()


@pytest.mark.asyncio
async def test_specialist_agent_xai_pollutant_propagation():
    from app.agent.specialists import xai_agent

    state: AgentState = {
        "pollutant": "pm25",
        "measurement_type": "surface_concentration",
        "pollutant_units": "µg/m³",
        "data_source": "CPCB Ground Monitoring Network",
        "user_query": "Explain PM2.5 driver in Mumbai",
    }
    result = await xai_agent(state)
    xai = result.get("xai_result")
    assert xai is not None
    assert xai["pollutant"] == "pm25"
    assert "model_used" in xai
    assert any("pm25" in ev.lower() for ev in xai["evidence"])


