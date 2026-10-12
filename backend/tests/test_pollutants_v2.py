"""Test suite for AirQ Insight v2.0 multi-pollutant additions.

Covers:
1. Pollutant registry configuration
2. Compliance engine calculations
3. NAMP yearly multi-pollutant dataset loader
4. PM2.5 LightGBM predictor inference
5. Multi-pollutant FastAPI endpoints (including Mumbai filters)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml_engine.pollutant_config import POLLUTANTS, get_pollutant, list_pollutants
from ml_engine.compliance import assess_point, assess_dataframe, ComplianceRecord
from app.services.pollutant_service import (
    get_yearly_data,
    get_mumbai_data,
    get_city_trends,
    predict_pm25,
)

client = TestClient(app)


def test_pollutant_registry():
    items = list_pollutants()
    keys = [p["key"] for p in items]
    assert "no2" in keys
    assert "pm25" in keys
    assert "pm10" in keys
    assert "so2" in keys
    assert "co" in keys
    assert "o3" in keys

    pm25 = get_pollutant("pm25")
    assert pm25.naaqs_annual == 40.0
    assert pm25.naaqs_24h == 60.0
    assert pm25.openaq_parameter_id == 2


def test_compliance_engine():
    # Compliant check
    res_clean = assess_point("pm25", 25.0)
    assert res_clean.naaqs_status == "compliant"
    assert res_clean.naaqs_exceedance_fraction < 0

    # Exceeded check
    res_dirty = assess_point("pm25", 65.0)
    assert res_dirty.naaqs_status == "exceeded"
    assert res_dirty.naaqs_exceedance_fraction > 0
    assert res_dirty.naaqs_threshold == 40.0


def test_pollutant_service_namp_data():
    all_data = get_yearly_data()
    assert len(all_data) > 0

    mumbai_data = get_mumbai_data()
    assert len(mumbai_data) >= 3
    years = [r["year"] for r in mumbai_data]
    assert 2021 in years
    assert 2022 in years
    assert 2023 in years


def test_pm25_model_prediction():
    pred = predict_pm25(city="mumbai", year=2024, so2=18.0, no2=29.0, pm10=108.0)
    assert "pm25_ugm3" in pred
    assert pred["pm25_ugm3"] > 0
    assert pred["city"].lower() == "mumbai"
    assert pred["year"] == 2024
    assert pred["naaqs_status"] in ["compliant", "exceeded"]


def test_api_list_pollutants():
    res = client.get("/api/v1/pollutants/")
    assert res.status_code == 200
    data = res.json()
    keys = [p["key"] for p in data]
    assert "pm25" in keys
    assert "no2" in keys


def test_api_mumbai_endpoint():
    res = client.get("/api/v1/pollutants/mumbai")
    assert res.status_code == 200
    data = res.json()
    assert data["city"] == "Mumbai"
    assert "data" in data
    assert len(data["data"]) >= 3


def test_api_predict_pm25():
    res = client.get(
        "/api/v1/pollutants/mumbai/predict-pm25",
        params={"year": 2025, "so2": 15.0, "no2": 25.0, "pm10": 95.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["city"].lower() == "mumbai"
    assert data["year"] == 2025
    assert "pm25_ugm3" in data
    assert data["pm25_ugm3"] > 0


def test_api_model_registry():
    res = client.get("/api/v1/pollutants/models")
    assert res.status_code == 200
    models = res.json()
    keys = [m["key"] for m in models]
    assert "pm25" in keys
    assert "pm10" in keys
    assert "no2" in keys
    pm25_info = next(m for m in models if m["key"] == "pm25")
    assert pm25_info["has_lightgbm"] is True


def test_three_component_ensemble_unit():
    from ml_engine.ensemble import ThreeComponentEnsemble
    ens = ThreeComponentEnsemble("pm25")
    pred = ens.combine_point(ai_val=45.0, lgbm_val=40.0, physics_val=38.0)
    assert pred.combined_concentration > 0
    assert pred.status == "production"
    assert "ai" in pred.active_components
    assert "lgbm" in pred.active_components
    assert "physics" in pred.active_components

    # Test adaptive fallback when one component is missing
    fallback_pred = ens.combine_point(ai_val=None, lgbm_val=42.0, physics_val=39.0)
    assert fallback_pred.fallback_applied is True
    assert fallback_pred.combined_concentration > 0


def test_api_ensemble_endpoint():
    res = client.get(
        "/api/v1/pollutants/mumbai/ensemble/pm25",
        params={"year": 2024, "so2": 15.0, "no2": 25.0, "pm10": 90.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["city"] == "mumbai"
    assert "combined_concentration" in data
    assert data["combined_concentration"] > 0
    assert "weights" in data
    assert "components" in data


def test_api_xai_explain_endpoint():
    res = client.get(
        "/api/v1/pollutants/mumbai/explain/pm25",
        params={"year": 2024, "so2": 15.0, "no2": 25.0, "pm10": 90.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["city"] == "mumbai"
    assert "combined_prediction" in data
    assert "model_agreement" in data
    assert "lightgbm_explanation" in data
    assert "physics_explanation" in data
    assert "top_features" in data["lightgbm_explanation"]


def test_pm10_and_so2_artifact_loading_and_prediction():
    from ml_engine.registry import get_model_registry
    registry = get_model_registry()

    # PM10
    booster_pm10, meta_pm10, _ = registry.load_lightgbm("pm10")
    assert booster_pm10 is not None
    assert meta_pm10["target"] == "pm10"
    assert meta_pm10["target_unit"] == "µg/m³"
    pred_pm10 = registry.predict_lightgbm("pm10", {"year": 2024, "so2": 15.0, "no2": 25.0, "lat": 19.07, "lon": 72.87})
    assert pred_pm10 > 0
    assert pred_pm10 < 1000.0

    # SO2
    booster_so2, meta_so2, _ = registry.load_lightgbm("so2")
    assert booster_so2 is not None
    assert meta_so2["target"] == "so2"
    assert meta_so2["target_unit"] == "µg/m³"
    pred_so2 = registry.predict_lightgbm("so2", {"year": 2024, "no2": 25.0, "pm10": 90.0, "lat": 19.07, "lon": 72.87})
    assert pred_so2 > 0
    assert pred_so2 < 1000.0


def test_unsupported_pollutant_handling():
    from ml_engine.training.train_pollutant import prepare_pollutant_dataset
    with pytest.raises(ValueError, match="insufficient_data"):
        prepare_pollutant_dataset("data/namp_openaq_all_pollutants.csv", "co")

    with pytest.raises(ValueError, match="Unsupported pollutant"):
        prepare_pollutant_dataset("data/namp_openaq_all_pollutants.csv", "unknown_gas")

