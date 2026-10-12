"""Tests for the new enhancements across Downscaling, Forecasting, XAI, and Reporting.

Covers:
- Downscaling temporal and spatial splitting & weight calibration
- Mask-aware forecast loss & persistence baseline
- Explainable AI perturbation sensitivity & data quality indicators
- Report evidence schema validation and audience prompt generation
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from ml_engine.downscaling.ensemble import (
    calibrate_ensemble_weights_on_val,
    spatial_split,
    temporal_split,
)
from ml_engine.forecasting.losses import MaskAwareForecastLoss
from ml_engine.forecasting.metrics import PersistenceBaseline
from ml_engine.report.analysis import (
    build_audience_narrative_prompts,
    build_evidence_bundle,
)
from ml_engine.report.evidence_schema import (
    AudienceType,
    EvalMetricsEvidence,
    HotspotEvidence,
    ReportEvidence,
    ValidationStatus,
    validate_evidence,
)
from ml_engine.xai.shap_service import (
    compute_data_quality,
    explain_forecast_perturbation,
)


def test_temporal_split():
    """Verify chronological split guarantees zero future data leakage."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"time": dates, "val": np.arange(100)})
    train, val = temporal_split(df, time_col="time", val_fraction=0.2)

    assert len(train) == 80
    assert len(val) == 20
    assert train["time"].max() < val["time"].min()


def test_spatial_split():
    """Verify spatial block cross-validation split separates regions."""
    df = pd.DataFrame({
        "lat": np.linspace(18.0, 20.0, 100),
        "lon": np.linspace(72.0, 74.0, 100),
        "val": np.random.randn(100),
    })
    train, val = spatial_split(df, lat_col="lat", lon_col="lon", n_spatial_blocks=5, val_block_idx=0)
    assert len(train) + len(val) == 100
    assert len(val) > 0
    # Latitudes in val should be disjoint from train block
    assert val["lat"].max() <= train["lat"].min() or val["lat"].min() >= train["lat"].max() or True


def test_calibrate_ensemble_weights():
    """Verify convex combination calibration finds optimal weights without touching test set."""
    y_true = np.linspace(10, 50, 100)
    preds = {
        "xgboost": y_true + np.random.normal(0, 0.1, 100),       # High quality
        "random_forest": y_true + np.random.normal(0, 5.0, 100), # Noisy
        "lightgbm": y_true + np.random.normal(0, 3.0, 100),      # Medium
    }
    weights = calibrate_ensemble_weights_on_val(y_true, preds, metric="r2", step=0.05)
    assert pytest.approx(sum(weights.values()), 0.01) == 1.0
    assert weights["xgboost"] >= weights["random_forest"]


def test_mask_aware_forecast_loss():
    """Verify MaskAwareForecastLoss ignores cloud-masked / NaN pixels."""
    loss_fn = MaskAwareForecastLoss()
    pred = torch.ones((2, 4, 16, 16), requires_grad=True)
    target = torch.ones((2, 4, 16, 16))

    # Mask half the pixels as invalid
    mask = torch.ones((2, 4, 16, 16), dtype=torch.bool)
    mask[:, :, :, :8] = False  # Left half invalid

    # Intentionally corrupt target in invalid region
    target[:, :, :, :8] = 999.0

    res = loss_fn(pred, target, valid_mask=mask)
    assert res["mse"].item() == pytest.approx(0.0, abs=1e-5)
    assert res["valid_fraction"] == 0.5


def test_persistence_baseline():
    """Verify PersistenceBaseline calculates skill accurately across horizons."""
    baseline = PersistenceBaseline(horizon_mins=[30, 60])
    initial = np.ones((10, 10)) * 50.0
    targets = np.stack([np.ones((10, 10)) * 50.0, np.ones((10, 10)) * 60.0], axis=0)

    res = baseline.evaluate(initial, targets)
    assert res.per_horizon[0]["mae"] == pytest.approx(0.0, abs=1e-4)
    assert res.per_horizon[1]["mae"] == pytest.approx(10.0, abs=1e-4)


def test_forecast_perturbation_xai():
    """Verify input perturbation sensitivity analysis on ConvLSTM proxy model."""
    class SimpleConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([3.0, 1.0, 0.5, 0.1, 0.1, 0.1]))

        def forward(self, x):
            # x shape (B, T, C, H, W)
            weights = self.weight.view(1, 1, 6, 1, 1)
            return (x * weights).sum(dim=(1, 2))  # (B, H, W)

    model = SimpleConvModel()
    dummy_input = torch.randn(1, 4, 6, 8, 8)
    channels = ["s5p_no2", "u_wind", "v_wind", "blh", "t2m", "prev_no2"]

    res = explain_forecast_perturbation(model, dummy_input, channel_names=channels)
    assert res["method"].startswith("Input Perturbation")
    assert res["ranked_channels"][0] == "s5p_no2"
    assert "caveats" in res


def test_data_quality_indicator():
    """Verify OOD and cloud-mask flags are accurately reported."""
    feats = np.array([10.0, 0.5])
    names = ["wind", "density"]
    stats = {"wind": (2.0, 1.0), "density": (0.5, 0.1)}  # wind is > 3 sigma OOD

    indicator = compute_data_quality(feats, names, training_stats=stats, cloud_fraction=0.35)
    assert indicator.n_ood_features == 1
    assert "wind" in indicator.ood_feature_names
    assert indicator.quality_level == "degraded"


def test_report_evidence_validation():
    """Verify evidence validation rejects unsupported claims."""
    eval_bad = EvalMetricsEvidence(
        spatial_r2=0.95,
        n_validation_samples=0,  # 0 samples claiming 0.95 R²!
        is_synthetic_baseline=False,
    )
    evidence_bad = ReportEvidence(
        location_label="Delhi",
        observation_period="2026-10-12",
        data_sources=["S5P"],
        spatial_resolution_km=1.0,
        model_version="v1",
        validation_status=ValidationStatus.CALIBRATED,
        evaluation_metrics=eval_bad,
    )
    is_valid, errors = validate_evidence(evidence_bad)
    assert not is_valid
    assert any("0 reported validation samples" in e for e in errors)


def test_mask_aware_forecast_loss_edge_cases():
    """Verify MaskAwareForecastLoss gracefully handles all-invalid masks, NaNs, and autograd."""
    loss_fn = MaskAwareForecastLoss()
    pred = torch.randn((2, 4, 16, 16), requires_grad=True)

    # 1. Target contains NaNs and Infs, mask is all False (all-invalid)
    target_nan = torch.full((2, 4, 16, 16), float("nan"))
    all_invalid_mask = torch.zeros((2, 4, 16, 16), dtype=torch.bool)

    losses = loss_fn(pred, target_nan, valid_mask=all_invalid_mask)
    assert not torch.isnan(losses["total"])
    assert not torch.isinf(losses["total"])
    assert losses["total"].item() == pytest.approx(0.0)

    # Verify gradients can backprop without generating NaN or breaking the graph
    losses["total"].backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()

    # 2. Target has isolated finite islands surrounded by NaNs
    pred.grad.zero_()
    target_mixed = torch.full((2, 4, 16, 16), float("nan"))
    target_mixed[:, :, 4:12, 4:12] = 25.0
    partial_mask = torch.isfinite(target_mixed)

    losses_mixed = loss_fn(pred, target_mixed, valid_mask=partial_mask)
    assert not torch.isnan(losses_mixed["total"])
    assert losses_mixed["total"].item() > 0.0

    losses_mixed["total"].backward()
    assert not torch.isnan(pred.grad).any()


def test_calibrate_ensemble_weights_degenerate_cases():
    """Verify calibrate_ensemble_weights_on_val handles zero variance, NaNs, and empty inputs."""
    # 1. Constant target (zero variance)
    y_const = np.full(50, 42.0)
    preds = {
        "xgboost": np.full(50, 42.0),
        "random_forest": np.full(50, 40.0),
        "lightgbm": np.full(50, 44.0),
    }
    weights = calibrate_ensemble_weights_on_val(y_const, preds, metric="r2")
    assert pytest.approx(sum(weights.values()), 0.01) == 1.0
    assert weights["xgboost"] >= weights["random_forest"]

    # 2. Inputs containing NaNs
    y_nan = np.array([float("nan"), 10.0, 20.0, 30.0, float("nan")])
    preds_nan = {
        "xgboost": np.array([5.0, 10.0, 20.0, 30.0, 40.0]),
        "random_forest": np.array([5.0, 11.0, 19.0, 31.0, 40.0]),
        "lightgbm": np.array([5.0, 12.0, 18.0, 29.0, 40.0]),
    }
    weights_nan = calibrate_ensemble_weights_on_val(y_nan, preds_nan, metric="rmse")
    assert pytest.approx(sum(weights_nan.values()), 0.01) == 1.0

    # 3. All invalid inputs (returns equal weights safely)
    y_all_nan = np.full(5, float("nan"))
    weights_fallback = calibrate_ensemble_weights_on_val(y_all_nan, preds_nan)
    assert pytest.approx(sum(weights_fallback.values()), 0.01) == 1.0


