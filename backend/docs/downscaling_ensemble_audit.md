# Downscaling Ensemble Scientific Audit — AirQ Insight

_Status: Validated & Audited_  
_Updated: 2026-10-12_

---

## 1. System Boundary Definition

To prevent architectural confusion, AirQ Insight strictly decouples three distinct ML subsystems:

1. **Satellite Downscaling (Subsystem under Audit):** Transforms coarse satellite observations (~2.5 km Sentinel-5P TROPOMI pixels) into fine-grid spatial columns (~250 m) using land-use, meteorological, and contextual co-variates.
2. **Annual / Station-Based Pollutant Estimation:** Statistical/tabular models (e.g. LightGBM models in `backend/ml_engine/training/`) estimating annual city-level concentrations from NAMP/CPCB and OpenAQ station monitors.
3. **Short-Term Forecasting & Physics Dispersion:** Spatial advection-diffusion solvers (`AdvectionDiffusionSolver`) and spatio-temporal ConvLSTM neural networks predicting future plume transport across hours.

This audit evaluates exclusively **Subsystem 1 (Base Sentinel-5P Satellite Downscaler Ensemble)**.

---

## 2. End-to-End Inference Path

The real request path from API ingress to output raster is:

```mermaid
flowchart TD
    API["GET /api/v1/downscale/map?ensemble=true|false"] --> DS[downscale_service.get_downscaled_map]
    DS --> SVC[ml_engine.service.generate_map]
    SVC --> RUN[ml_engine.service._run]
    RUN --> PIPE[NO2Pipeline.run(use_ensemble=...)]
    
    PIPE --> S1[Stage 1: Sentinel-5P / Local GeoTIFF Ingestion + ERA5 & Static Covariates]
    S1 --> S2[Stage 2: SpatioTemporalGapFiller for cloud-masked pixels]
    S2 --> S3{Stage 3: Spatial Downscaling}
    
    S3 -- "use_ensemble=False (default)" --> BL[NO2Downscaler: Single XGBoost]
    S3 -- "use_ensemble=True" --> ENS[EnsembleDownscaler: XGBoost 0.50 + RF 0.30 + LGBM 0.20]
    
    BL --> BP1[Backprojection Conservation Iterations]
    ENS --> BP2[Backprojection Conservation Iterations]
    
    BP1 --> S4[Stage 4: Surface Conversion to boundary-layer NO2]
    BP2 --> S4
    S4 --> S5[Stage 5: Surface Station Calibration]
    S5 --> OUT[GeoTIFF & GeoJSON Response]
```

### Verified Code Implementations:
- `app/api/v1/downscale/router.py`: Exposes `ensemble: bool = Query(False)`. Backward compatible.
- `app/services/downscale_service.py`: Threads `use_ensemble` into `generate_map()`, returns `ensemble_active` and `ensemble_metrics`.
- `ml_engine/service.py`: Computes unique run cache key including `use_ensemble` flag (preventing cache collisions between baseline and ensemble).
- `ml_engine/pipeline.py`: Stage 3 selects between `NO2Downscaler` and `EnsembleDownscaler`. Saves and roundtrip-loads artifact (`EnsembleDownscaler.load`) before inference.
- `ml_engine/downscaling/ensemble.py`: Implements `EnsembleDownscaler`, `predict_features()`, `predict_day_ensemble()`, `predict()`, `save()`, and `load()`.
- `ml_engine/xai/shap_service.py`: `explain_ensemble_downscaler()` executes TreeSHAP on the primary XGBoost component, augmented with explicit ensemble prediction, weights, and inter-model disagreement.

---

## 3. Subsystem Architecture & Implementation Audit

| Component | Target Predicted | Features Used | Training Labels | Fitting Status | Weight / Fusion |
|---|---|---|---|---|---|
| **XGBoost** | Coarse tropospheric NO₂ column ($\mu\text{mol}/\text{m}^2$) | Static land use (road density, built-up, elevation, NDVI, population, power plants) + ERA5 meteorology ($u_{10}, v_{10}, \text{blh}, \text{sp}, t_{2m}, \text{wind speed}$) + Context ($\text{ring}, \text{day mean}, \sin, \cos$) | Coarse TROPOMI observed pixels (cloud-masked pixels excluded) | Fully fitted jointly in `EnsembleDownscaler.fit()` | 0.50 (normalized) |
| **Random Forest** | Same as above | Same as above | Same as above | Fully fitted in `EnsembleDownscaler.fit()` | 0.30 (normalized) |
| **LightGBM** | Same as above | Same as above | Same as above | Fully fitted in `EnsembleDownscaler.fit()` | 0.20 (normalized) |
| **Ensemble Fusion** | Weighted column prediction | Same as above | Weighted linear combination | Dynamic weights with validation | $\sum w_i y_i$ |

### Conservation and Back-Projection Stage
- Back-projection conservation (`residual = field - block_mean(pred_2d, f); pred_2d += upsample_bilinear(residual, f)`) is correctly applied to the **fused ensemble map** in `predict_day_ensemble()`. This strictly preserves coarse-pixel integral mass conservation for the ensemble output.

---

## 4. Defects Found & Fixes Made

During validation, 4 critical defects were identified and resolved:

1. **LightGBM Deserialization & Prediction Failure (`NotFittedError` / Segfault):**
   - *Defect:* `EnsembleDownscaler.load()` recreated `LGBMRegressor` and loaded only the raw text booster file (`LGB_FILE`), leaving scikit-learn wrapper fitted flags (`fitted_`, `n_features_in_`) unpopulated, or encountering C ABI memory faults during `predict()` under Python 3.14.
   - *Fix:* Added `LGB_PKL_FILE` serialization for full native object fidelity, with fallback text-booster loading and explicit scikit-learn attribute restoration. Verified round-trip without errors.
2. **Cache Key Collision in ML Service:**
   - *Defect:* `ml_engine/service.py:_run_dir()` did not incorporate `use_ensemble` into the cache key SHA1 hash. Running `ensemble=false` followed by `ensemble=true` on the same region and date would return the cached single-model baseline run.
   - *Fix:* Added `use_ensemble` parameter to `_run_dir()` and hash input array.
3. **Unvalidated Ensemble Weights:**
   - *Defect:* `EnsembleDownscaler.__init__()` allowed negative weights or zero sum without rejection.
   - *Fix:* Added strict weight validation: all weights must be non-negative and total sum must be strictly positive.
4. **Ambiguous Ensemble Explainability:**
   - *Defect:* `explain_ensemble_downscaler()` did not explicitly identify that feature attribution percentages originate from the primary XGBoost component, which could mislead users into believing a single additive SHAP model explains 3 disparate tree architectures.
   - *Fix:* Explicitly labeled executive summary and narrative with `[XGBoost Component SHAP]`, reported component weight ($0.50$), ensemble prediction, and inter-model disagreement. Added non-finite input validation.

---

## 5. Test Suite Verification

- Focused tests: `backend/tests/test_downscale.py` (15/15 passed).
- Integration tests: `backend/tests/test_ensemble_xai_simulator.py` (9/9 passed).
- Full backend suite: 89 passed, 0 failed.
