"""Multi-Pollutant Explainable AI (XAI) Suite — AirQ Insight v2.0.

Provides comprehensive, multi-layer explanations:
1. Individual Model Explanations:
   - TreeSHAP explanations for LightGBM regression models (feature contributions,
     positive/negative drivers, base values).
   - Physics parameter sensitivity breakdown (diffusion, decay, advection impact).
2. Ensemble Breakdown:
   - Component contribution breakdown (AI, LightGBM, Physics).
   - Effective normalized weights and fallback provenance.
   - Uncertainty & agreement scoring (variance across component estimates).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import shap

from ml_engine.registry import get_model_registry
from ml_engine.pollutant_config import get_pollutant

log = logging.getLogger(__name__)


def explain_lightgbm_prediction(
    pollutant: str,
    features_dict: dict[str, Any],
) -> dict[str, Any]:
    """Compute local TreeSHAP attribution for a LightGBM prediction."""
    target = pollutant.lower()
    registry = get_model_registry()
    booster, meta, state_to_idx = registry.load_lightgbm(target)

    feature_names = meta.get("features", [])
    row = dict(features_dict)
    if "state" in row and "state_enc" not in row:
        row["state_enc"] = state_to_idx.get(row["state"], 0)

    X_row = np.array([[float(row.get(f, 0.0)) for f in feature_names]], dtype=np.float64)

    # Compute TreeSHAP values
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_row)[0]
    expected_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
    pred_val = float(booster.predict(X_row)[0])

    contributions = []
    total_abs_shap = float(np.sum(np.abs(shap_vals))) + 1e-6
    for name, val, sv in zip(feature_names, X_row[0], shap_vals):
        pct = round(abs(float(sv)) / total_abs_shap * 100.0, 1)
        contributions.append({
            "feature": name,
            "feature_value": round(float(val), 2),
            "shap_value": round(float(sv), 3),
            "percentage_impact": pct,
            "direction": "increasing" if sv > 0 else "decreasing",
        })

    # Sort by absolute impact
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    positive = [c for c in contributions if c["shap_value"] > 0]
    negative = [c for c in contributions if c["shap_value"] < 0]

    top_drivers_text = ", ".join(
        [f"{c['feature']} ({c['percentage_impact']}%)" for c in contributions[:3]]
    )
    summary_text = (
        f"Predicted {target.upper()} concentration is driven predominantly by {top_drivers_text}. "
        f"Baseline expectation was {expected_val:.1f} µg/m³; net feature adjustments shifted prediction to {pred_val:.1f} µg/m³."
    )

    return {
        "pollutant": target,
        "model_type": "LightGBM",
        "predicted_value": round(pred_val, 2),
        "base_value": round(expected_val, 2),
        "summary": summary_text,
        "top_features": contributions,
        "positive_contributors": positive,
        "negative_contributors": negative,
    }


def explain_physics_sensitivity(
    pollutant: str,
    wind_speed_ms: float = 2.5,
    mixing_height_m: float = 800.0,
) -> dict[str, Any]:
    """Provide physics-based sensitivity explanation for the pollutant's dispersion."""
    target = pollutant.lower()
    cfg = get_pollutant(target)

    # Physical parameters
    diffusivity = cfg.diffusivity_m2_s
    lifetime_h = cfg.chemical_lifetime_h

    # Sensitivity indicators
    ventilation_factor = wind_speed_ms * (mixing_height_m / 1000.0)
    dispersion_speed = "fast" if ventilation_factor > 3.0 else ("moderate" if ventilation_factor > 1.5 else "poor / stagnant")

    decay_note = f"Chemical half-life ~{lifetime_h}h" if lifetime_h else "Non-reactive / conservative transport"

    narrative = (
        f"Physics solver models horizontal advection via wind ({wind_speed_ms:.1f} m/s) and atmospheric diffusion "
        f"(eddy coefficient K={diffusivity} m²/s). Ventilation index is {ventilation_factor:.2f} ({dispersion_speed} dispersion). {decay_note}."
    )

    return {
        "pollutant": target,
        "diffusivity_m2_s": diffusivity,
        "chemical_lifetime_h": lifetime_h,
        "ventilation_factor": round(ventilation_factor, 2),
        "dispersion_condition": dispersion_speed,
        "physics_narrative": narrative,
    }


def explain_ensemble_prediction(
    pollutant: str,
    features_dict: dict[str, Any],
    physics_estimate: Optional[float] = None,
    ai_estimate: Optional[float] = None,
) -> dict[str, Any]:
    """Comprehensive explanation of the three-component ensemble prediction."""
    target = pollutant.lower()
    registry = get_model_registry()

    # Run ensemble
    ens_res = registry.predict_ensemble(
        target,
        features_dict=features_dict,
        physics_estimate=physics_estimate,
        ai_estimate=ai_estimate,
    )

    # TreeSHAP for LightGBM component
    lgbm_xai = {}
    try:
        lgbm_xai = explain_lightgbm_prediction(target, features_dict)
    except Exception as exc:
        log.warning("LightGBM XAI failed for %s: %s", target, exc)

    # Physics sensitivity
    physics_xai = explain_physics_sensitivity(target)

    # Uncertainty / Agreement assessment
    valid_vals = [v for v in ens_res.components.values() if v is not None]
    spread = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
    cv_agreement = (spread / (ens_res.combined_concentration + 1e-6)) if ens_res.combined_concentration > 0 else 0.0

    confidence = "high" if cv_agreement < 0.15 else ("moderate" if cv_agreement < 0.35 else "low / divergent")

    return {
        "pollutant": target,
        "combined_prediction": ens_res.to_dict(),
        "model_agreement": {
            "component_spread_ugm3": round(spread, 2),
            "coefficient_of_variation": round(cv_agreement, 3),
            "confidence_assessment": confidence,
        },
        "lightgbm_explanation": lgbm_xai,
        "physics_explanation": physics_xai,
    }
