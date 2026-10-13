from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_current_user
from app.services.upload_data import available_dates, data_dir, point_forecast

FAKE_USER = {
    "id": "test-user-multipol",
    "email": "test@multipol.test",
    "full_name": "Multi-Pollutant Test User",
    "role": "NORMAL_USER",
    "is_blocked": False,
    "created_at": "2024-01-01T00:00:00+00:00",
}

@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user] = lambda: FAKE_USER
    yield
    app.dependency_overrides[get_current_user] = lambda: FAKE_USER


client = TestClient(app)

POLLUTANTS_WITH_RASTERS = ["no2", "pm25", "pm10", "so2", "o3", "co"]
CAMS_POLLUTANTS = ["pm25", "pm10", "so2", "o3", "co"]


# ── 1. Home pollutant selection — downscale/dates per pollutant ───────────────

class TestDownscaleDates:
    """GET /api/v1/downscale/dates?pollutant=X returns dates for that pollutant's raster folder."""

    def test_no2_dates_returned(self):
        resp = client.get("/api/v1/downscale/dates?pollutant=no2")
        assert resp.status_code == 200
        data = resp.json()
        assert "dates" in data
        assert data["pollutant"] == "no2"
        assert len(data["dates"]) >= 61  # 61-day NO2 dataset

    @pytest.mark.parametrize("pol", CAMS_POLLUTANTS)
    def test_cams_pollutant_dates_returned(self, pol):
        """Each CAMS pollutant should return at least 1 date from its own folder."""
        resp = client.get(f"/api/v1/downscale/dates?pollutant={pol}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pollutant"] == pol
        assert len(data["dates"]) >= 1, (
            f"Pollutant {pol} has no dates from its raster folder — check pollutant_samples/{pol.upper()}/"
        )


# ── 2. Plume Flow pollutant selection — historical geotiff per pollutant ──────

class TestHistoricalGeoTiffPerPollutant:
    """The historical geotiff endpoint must serve each pollutant's own raster bytes."""

    def test_no2_geotiff_served(self):
        resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/geotiff")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/tiff"
        assert len(resp.content) > 0

    @pytest.mark.parametrize("pol,ds_id", [
        ("pm25", "pm25_cams_61day_2025"),
        ("pm10", "pm10_cams_61day_2025"),
        ("so2", "so2_cams_61day_2025"),
        ("o3", "o3_cams_61day_2025"),
        ("co", "co_cams_61day_2025"),
    ])
    def test_cams_pollutant_geotiff_served(self, pol, ds_id):
        """Each CAMS pollutant's own historical geotiff is non-empty."""
        resp = client.get(f"/api/v1/historical/{ds_id}/{pol}/2025-11-05/geotiff")
        assert resp.status_code == 200, (
            f"{pol}: expected 200 from historical/{ds_id}/{pol}/2025-11-05/geotiff, "
            f"got {resp.status_code}: {resp.text[:200]}"
        )
        assert resp.headers["content-type"] == "image/tiff"
        assert len(resp.content) > 0

    def test_no2_geotiff_not_returned_for_o3_request(self):
        """Requesting O3 from its own dataset must NOT silently return a NO2 raster.
        We check that the grid JSON has measurement_type=surface_concentration, not satellite_column.
        """
        resp = client.get("/api/v1/historical/o3_cams_61day_2025/o3/2025-11-05/grid")
        assert resp.status_code == 200
        grid = resp.json()
        assert grid["pollutant"] == "o3"
        assert grid["measurement_type"] == "surface_concentration"
        assert grid["units"] == "µg/m³"

        # Also verify NO2 dataset uses satellite_column
        resp_no2 = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/grid")
        assert resp_no2.json()["measurement_type"] == "satellite_column"
        assert resp_no2.json()["units"] == "µmol/m²"


# ── 3. Correct raster retrieval and rendering ─────────────────────────────────

class TestRasterMetadata:
    """Grid endpoint returns correct units and measurement type per pollutant."""

    @pytest.mark.parametrize("pol,ds_id,expected_mtype,expected_units", [
        ("no2", "no2_61day_2025", "satellite_column", "µmol/m²"),
        ("pm25", "pm25_cams_61day_2025", "surface_concentration", "µg/m³"),
        ("pm10", "pm10_cams_61day_2025", "surface_concentration", "µg/m³"),
        ("so2", "so2_cams_61day_2025", "surface_concentration", "µg/m³"),
        ("o3", "o3_cams_61day_2025", "surface_concentration", "µg/m³"),
        ("co", "co_cams_61day_2025", "surface_concentration", "µg/m³"),
    ])
    def test_grid_returns_correct_measurement_metadata(self, pol, ds_id, expected_mtype, expected_units):
        resp = client.get(f"/api/v1/historical/{ds_id}/{pol}/2025-11-05/grid")
        assert resp.status_code == 200
        grid = resp.json()
        assert grid["measurement_type"] == expected_mtype, (
            f"{pol}: expected measurement_type={expected_mtype}, got {grid.get('measurement_type')}"
        )
        assert grid["units"] == expected_units, (
            f"{pol}: expected units={expected_units}, got {grid.get('units')}"
        )


# ── 4. Uploaded dataset discovery ─────────────────────────────────────────────

class TestUploadDataDiscovery:
    """upload_data.available_dates() returns correct dates for each pollutant folder."""

    def test_no2_available_dates(self):
        dates = available_dates("no2")
        assert len(dates) >= 61
        assert "2025-11-01" in dates
        assert "2025-12-31" in dates

    @pytest.mark.parametrize("pol", CAMS_POLLUTANTS)
    def test_cams_available_dates_nonempty(self, pol):
        dates = available_dates(pol)
        assert len(dates) >= 1, (
            f"available_dates('{pol}') returned 0 dates — "
            f"check backend/data/pollutant_samples/{pol.upper()}/"
        )

    def test_no2_data_dir_is_test_data(self):
        d = data_dir("no2")
        assert d.exists()
        assert "test_data" in str(d)

    @pytest.mark.parametrize("pol", CAMS_POLLUTANTS)
    def test_cams_data_dir_is_pollutant_samples(self, pol):
        d = data_dir(pol)
        assert d.exists(), f"data_dir('{pol}') does not exist: {d}"
        assert "pollutant_samples" in str(d)


# ── 5. Pinpoint Intelligence — point_forecast returns correct pollutant metadata

class TestPointForecast:
    """upload_data.point_forecast returns correct pollutant, units, and measurement_type."""

    @pytest.mark.parametrize("pol,expected_mtype,expected_units", [
        ("no2", "satellite_column", "µmol/m²"),
        ("pm25", "surface_concentration", "µg/m³"),
        ("o3", "surface_concentration", "µg/m³"),
    ])
    def test_point_forecast_correct_metadata(self, pol, expected_mtype, expected_units):
        result = point_forecast(lat=19.05, lon=72.85, date="2025-11-05", pollutant=pol)
        assert result["pollutant"] == pol
        assert result["measurement_type"] == expected_mtype
        assert result["units"] == expected_units
        # value must be numeric and non-negative
        assert isinstance(result["value"], (int, float))
        assert result["value"] >= 0

    def test_point_forecast_no_stale_no2_when_o3_selected(self):
        """point_forecast for o3 must NOT carry NO2 units or satellite_column type."""
        result = point_forecast(lat=19.05, lon=72.85, date="2025-11-05", pollutant="o3")
        assert result["measurement_type"] != "satellite_column", (
            "O3 point_forecast returned satellite_column — stale NO2 type leaked in"
        )
        assert result["units"] != "µmol/m²", (
            "O3 point_forecast returned µmol/m² — stale NO2 units leaked in"
        )

    def test_point_forecast_horizons_present(self):
        result = point_forecast(lat=19.05, lon=72.85, date="2025-11-05", pollutant="pm25")
        assert "horizons" in result
        assert len(result["horizons"]) >= 1
        for h in result["horizons"]:
            assert "hours" in h
            assert "value" in h
            assert h["value"] >= 0


# ── 6. NoData handling ────────────────────────────────────────────────────────

class TestNoDataHandling:
    """NaN/cloud-masked pixels must be serialized as null in JSON, not as 0."""

    def test_no2_nodata_is_null_not_zero(self):
        # 2025-11-01 has cloud-masked pixels (small file, mostly NaN)
        resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-01/grid")
        assert resp.status_code == 200
        grid = resp.json()
        flat = [v for row in grid["values"] for v in row]
        assert any(v is None for v in flat), (
            "Expected some null values for cloud-masked pixels on 2025-11-01"
        )
        # Ensure valid_pixels count excludes null cells
        assert grid["valid_pixels"] < grid["width"] * grid["height"]

    def test_coverage_missing_fraction_reported(self):
        resp = client.get("/api/v1/historical/no2_61day_2025/no2/coverage")
        assert resp.status_code == 200
        cov = resp.json()
        # Check that per-date coverage entries include cloud fraction or valid fraction
        dates = cov["dates"]
        assert len(dates) > 0
        first = dates[0]
        assert "available" in first


# ── 7. Historical vs. forecast separation ────────────────────────────────────

class TestHistoricalVsForecastSeparation:
    """Historical endpoint serves historical rasters; forecasting is separate."""

    def test_historical_returns_historical_not_forecast(self):
        resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/grid")
        assert resp.status_code == 200
        grid = resp.json()
        assert grid["date"] == "2025-11-05"
        # Historical data should NOT have forecast_horizon field
        assert "forecast_horizon" not in grid

    def test_downscale_dates_respects_no2_pollutant(self):
        resp = client.get("/api/v1/downscale/dates?pollutant=no2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pollutant"] == "no2"


# ── 8. Existing NO2 workflow preserved ────────────────────────────────────────

class TestExistingNo2WorkflowPreserved:
    """Existing NO2 endpoints continue to work correctly after multi-pollutant additions."""

    def test_no2_61day_manifest_still_61(self):
        resp = client.get("/api/v1/historical/no2_61day_2025/manifest")
        assert resp.status_code == 200
        manifest = resp.json()
        assert manifest["date_count"] == 61

    def test_no2_sentinel_test_samples_still_separate(self):
        """The 8-image test sample dataset must remain distinct from the 61-day main dataset."""
        resp = client.get("/api/v1/historical/no2_samples_8day/manifest")
        assert resp.status_code == 200
        manifest = resp.json()
        assert manifest["id"] == "no2_samples_8day"
        # Must NOT report 61 dates
        assert manifest["date_count"] < 61

    def test_no2_geotiff_still_available(self):
        resp = client.get("/api/v1/historical/no2_61day_2025/no2/2025-11-05/geotiff")
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_pollutant_availability_no2_rasters_still_true(self):
        resp = client.get("/api/v1/historical/pollutants/availability")
        assert resp.status_code == 200
        avail = resp.json()
        assert avail["no2"]["has_raster_data_nov_dec_2025"] is True


# ── 9. Multi-pollutant coverage endpoint ──────────────────────────────────────

class TestCamsCoverageEndpoints:
    """CAMS dataset coverage endpoints return correct pollutant metadata."""

    @pytest.mark.parametrize("pol,ds_id", [
        ("pm25", "pm25_cams_61day_2025"),
        ("o3", "o3_cams_61day_2025"),
    ])
    def test_cams_coverage_returns_correct_pollutant(self, pol, ds_id):
        resp = client.get(f"/api/v1/historical/{ds_id}/{pol}/coverage")
        assert resp.status_code == 200
        cov = resp.json()
        assert cov["pollutant"] == pol
        assert cov["dataset_id"] == ds_id
        assert cov["days_with_data"] >= 1


# ── 10. Source attribution abstention for O3 ──────────────────────────────────

class TestSourceAttributionAbstention:
    """The system must not present quantitative source attribution for O3.

    O3 is a secondary photochemical pollutant; proximity to a facility does not
    establish emission causality. The implementation in constants.ts flags
    isQuantitativeAttributionSupported=False for non-NO2. The backend point_forecast
    does not return attribution_percentage for O3.
    """

    def test_point_forecast_o3_has_no_attribution_percentage(self):
        result = point_forecast(lat=19.05, lon=72.85, date="2025-11-05", pollutant="o3")
        # The backend point_forecast does not return source attribution at all —
        # that is a frontend responsibility. Verify the endpoint result is clean.
        assert result["pollutant"] == "o3"
        # Should NOT have any NO2-specific attribution fields
        assert "attribution_percentage" not in result
        assert result.get("dataset_id") == "o3_cams_61day_2025"

    @pytest.mark.asyncio
    async def test_analysis_agent_o3_no_quantitative_attribution(self):
        """Analysis agent for O3 must not claim quantitative source percentages."""
        from app.agent.specialists import analysis_agent

        state = {
            "pollutant": "o3",
            "measurement_type": "surface_concentration",
            "pollutant_units": "µg/m³",
            "data_source": "CAMS surface O3",
            "user_query": "Analyse O3 in Mumbai",
            "tool_results": {
                "analysis": {
                    "status": "success",
                    "data": {
                        "current": {"mean": 95.0, "pct_vs_naaqs": -5.0},
                        "hotspots": [],
                        "anomalies": [],
                        "population": {},
                        "forecast": {},
                    },
                }
            },
        }
        result = await analysis_agent(state)
        ar = result.get("analysis_result", {})
        summary = ar.get("summary", "")
        # Must not claim percentage attribution for O3
        assert "%" not in summary or "above" in summary or "below" in summary, (
            "Analysis summary for O3 should not contain unexplained percentage attributions"
        )
        # Evidence must record that pollutant=o3
        evidence = ar.get("evidence", [])
        assert any("o3" in ev.lower() for ev in evidence)
