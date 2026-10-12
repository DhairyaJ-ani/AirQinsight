import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ml_engine.service import RUNS_ROOT


_RUN = RUNS_ROOT / "abc123"
FAKE_MAP = {
    "bbox": [72.8, 18.9, 73.1, 19.2],
    "date": "2024-01-15",
    "resolution_m": 263,
    "units": "ug m-3",
    "surface_tif": str(_RUN / "daily" / "2024-01-15" / "no2_surface_fine.tif"),
    "raw_tif": str(_RUN / "daily" / "2024-01-15" / "no2_raw_coarse.tif"),
    "gapfilled_tif": str(_RUN / "daily" / "2024-01-15" / "no2_gapfilled_coarse.tif"),
    "hazard_geojson": str(_RUN / "daily" / "2024-01-15" / "no2_hazard_bands.geojson"),
    "surface_netcdf": str(_RUN / "no2_surface_fine.nc"),
    "metrics": {"gapfill_holdout_r2": 0.93},
}


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/map — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_map_success(auth_client):
    """Authenticated request returns the ML engine's map for the requested day (engine mocked)."""
    mock_supabase = MagicMock()

    with patch("app.services.activity_service.log_activity"):
        with patch("app.api.v1.downscale.router.get_supabase", return_value=mock_supabase):
            with patch("app.services.downscale_service.generate_map", return_value=FAKE_MAP) as gen:
                response = await auth_client.get(
                    "/api/v1/downscale/map",
                    params={"bbox": "72.8,18.9,73.1,19.2", "timestamp": "2024-01-15T10:00:00Z"},
                )

    assert response.status_code == 200
    assert gen.call_args.kwargs["date"] == "2024-01-15"
    data = response.json()
    assert data["resolution"] == "263m"
    assert data["format"] == "GeoTIFF"
    assert data["bbox"] == [72.8, 18.9, 73.1, 19.2]
    assert data["grid_url"] == "/files/abc123/daily/2024-01-15/no2_surface_fine.tif"
    assert data["hazard_geojson_url"].endswith("no2_hazard_bands.geojson")
    assert data["units"] == "ug m-3"


@pytest.mark.asyncio
async def test_get_map_invalid_bbox(auth_client):
    """A malformed bbox is rejected with 422 before the ML engine runs."""
    with patch("app.services.activity_service.log_activity"):
        with patch("app.api.v1.downscale.router.get_supabase", return_value=MagicMock()):
            response = await auth_client.get(
                "/api/v1/downscale/map",
                params={"bbox": "73.1,18.9,72.8", "timestamp": "2024-01-15T10:00:00Z"},
            )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_map_engine_failure(auth_client):
    """An ML engine failure (e.g. Earth Engine outage) surfaces as 503."""
    with patch("app.services.activity_service.log_activity"):
        with patch("app.api.v1.downscale.router.get_supabase", return_value=MagicMock()):
            with patch("app.services.downscale_service.generate_map", side_effect=RuntimeError("quota")):
                response = await auth_client.get(
                    "/api/v1/downscale/map",
                    params={"bbox": "72.8,18.9,73.1,19.2", "timestamp": "2024-01-15T10:00:00Z"},
                )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/map — unauthenticated → 401
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_map_unauthenticated(client):
    """Request without Authorization header returns 401."""
    response = await client.get(
        "/api/v1/downscale/map",
        params={"bbox": "72.8,18.9,73.1,19.2", "timestamp": "2024-01-15T10:00:00Z"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/map — missing query params → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_map_missing_params(auth_client):
    """Request missing the required bbox parameter returns 422."""
    response = await auth_client.get(
        "/api/v1/downscale/map",
        params={"timestamp": "2024-01-15T10:00:00Z"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_map_missing_timestamp(auth_client):
    """Request missing the required timestamp parameter returns 422."""
    response = await auth_client.get(
        "/api/v1/downscale/map",
        params={"bbox": "72.8,18.9,73.1,19.2"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/dates — available dates from test_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_available_dates(auth_client):
    """GET /api/v1/downscale/dates returns list of available test data dates."""
    response = await auth_client.get("/api/v1/downscale/dates")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert "count" in data
    assert len(data["dates"]) > 0
    assert "2025-11-05" in data["dates"]


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/geotiff — test data GeoTIFF raster stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_geotiff_stream(auth_client):
    """GET /api/v1/downscale/geotiff returns valid image/tiff."""
    response = await auth_client.get(
        "/api/v1/downscale/geotiff",
        params={"timestamp": "2025-11-05T12:00:00Z"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# Ensemble downscaler pipeline wiring
# ---------------------------------------------------------------------------


def test_ensemble_downscaler_use_ensemble_flag():
    """NO2Pipeline.run() accepts use_ensemble=True without signature error."""
    import inspect
    from ml_engine.pipeline import NO2Pipeline

    sig = inspect.signature(NO2Pipeline.run)
    assert "use_ensemble" in sig.parameters, "use_ensemble param missing from NO2Pipeline.run()"
    param = sig.parameters["use_ensemble"]
    assert param.default is False, "use_ensemble should default to False"


# ---------------------------------------------------------------------------
# Ensemble downscaler XAI — explain_ensemble_downscaler
# ---------------------------------------------------------------------------


def test_explain_ensemble_downscaler_output_structure():
    """explain_ensemble_downscaler returns a valid XAIExplanation with combined confidence."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from ml_engine.downscaling.ensemble import EnsembleDownscaler
    from ml_engine.xai import explain_ensemble_downscaler, XAIExplanation

    N, D = 80, 8
    rng = np.random.RandomState(0)
    X = rng.randn(N, D).astype(np.float32)
    y = 40.0 + 5.0 * X[:, 0] - 3.0 * X[:, 2] + rng.randn(N) * 2

    ens = EnsembleDownscaler()
    ens.xgb_model = xgb.XGBRegressor(n_estimators=8, max_depth=3, random_state=0)
    ens.xgb_model.fit(X, y)
    ens.rf_model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0, n_jobs=1)
    ens.rf_model.fit(X, y)
    ens.lgb_model = lgb.LGBMRegressor(n_estimators=8, max_depth=3, random_state=0, n_jobs=1, verbosity=-1)
    ens.lgb_model.fit(X, y)
    ens.features = True  # signal that models are ready (FeatureBuilder sentinel)

    feature_names = [f"f{i}" for i in range(D)]
    x_row = X[0]

    explanation = explain_ensemble_downscaler(ens, x_row, feature_names=feature_names)

    assert isinstance(explanation, XAIExplanation)
    assert 0.0 <= explanation.confidence <= 1.0, "confidence must be in [0, 1]"
    assert explanation.predicted_value > 0
    assert len(explanation.top_contributors) > 0
    # Executive summary must mention ensemble disagreement
    assert "Ensemble inter-model disagreement" in explanation.executive_summary


def test_explain_ensemble_downscaler_raises_when_unfitted():
    """explain_ensemble_downscaler raises RuntimeError if the ensemble is not trained."""
    from ml_engine.downscaling.ensemble import EnsembleDownscaler
    from ml_engine.xai import explain_ensemble_downscaler

    ens = EnsembleDownscaler()  # not fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        explain_ensemble_downscaler(ens, np.zeros(8))


# ---------------------------------------------------------------------------
# GET /api/v1/downscale/map — ensemble=true query param
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_map_ensemble_flag_threaded(auth_client):
    """?ensemble=true is accepted and threaded through to generate_map."""
    mock_supabase = MagicMock()
    FAKE_MAP_ENS = {**FAKE_MAP, "ensemble": True, "ensemble_metrics": {"weights": {"xgboost": 0.5}}}

    with patch("app.services.activity_service.log_activity"):
        with patch("app.api.v1.downscale.router.get_supabase", return_value=mock_supabase):
            with patch(
                "app.services.downscale_service.generate_map",
                return_value=FAKE_MAP_ENS,
            ) as gen:
                response = await auth_client.get(
                    "/api/v1/downscale/map",
                    params={
                        "bbox": "72.8,18.9,73.1,19.2",
                        "timestamp": "2024-01-15T10:00:00Z",
                        "ensemble": "true",
                    },
                )

    assert response.status_code == 200
    # Verify the ensemble flag was forwarded to generate_map
    assert gen.call_args.kwargs.get("use_ensemble") is True
    data = response.json()
    assert data["ensemble_active"] is True
