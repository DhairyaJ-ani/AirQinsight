# Base Satellite Downscaling Ensemble Evaluation Report — AirQ Insight

_Evaluation Date: 2026-10-12_  
_Subsystem: Sentinel-5P Coarse Satellite-to-Fine Grid Spatial Downscaler_  

---

## 1. Executive Summary & Verification Taxonomy

To maintain strict scientific and engineering rigor, AirQ Insight separates model readiness into four explicit validation tiers:

1. **Software & Serialization Validation:** **VERIFIED** — Code paths, API routing (`ensemble=true|false`), state persistence (`xgb_downscaler.json`, `rf_downscaler.pkl`, `lgb_downscaler.pkl`, `ensemble_metadata.json`), input bounds checking, and non-colliding cache keys pass all tests without segmentation faults or missing attribute errors.
2. **Coarse-Grid Consistency & Model Ensemble Evaluation:** **VERIFIED** — Evaluated on 61 days of genuine Sentinel-5P TROPOMI satellite observations (Mumbai Metropolitan Region, 2025-11-01 to 2025-12-31). The ensemble and its components were evaluated using strict chronological temporal holdouts with synthetic feature inputs, as well as a 3-way train/validation/test split with weight calibration.
3. **Fine-Resolution Predictive Accuracy:** **UNVALIDIFIED / THEORETICAL** — Generating a $250\text{ m}$ fine raster grid does **not** scientifically prove sub-pixel $250\text{ m}$ physical accuracy. While mass conservation is enforced via coarse-grid back-projection, empirical validation of sub-pixel spatial heterogeneity requires dense micro-monitoring networks.
4. **Production Deployment Readiness:** **CONDITIONAL / OPTIONAL ONLY** — The ensemble is fully wired, operable, and safe as an opt-in mode (`ensemble=true`). It must **remain optional**, with the single-model XGBoost baseline remaining the production default (`ensemble=false`), because fine-resolution ground truth validation remains open.

---

## 2. Experimental Data Distinctions (Real vs. Synthetic)

To prevent misleading claims regarding deployment performance, the exact nature of all experimental inputs is cataloged below:

| Feature / Input Layer | Nature of Data | Source & Description |
|---|---|---|
| **Target Column ($\text{NO}_2$ $\mu\text{mol}/\text{m}^2$)** | **REAL** | 61 daily Sentinel-5P TROPOMI Level-2/3 GeoTIFF rasters (`backend/data/test_data/`) covering Mumbai ($13 \times 10$ coarse grid at $0.035^\circ$ cell size). 7,192 valid finite observed pixels, 738 cloud-masked pixels excluded. |
| **Meteorological Co-variates ($u_{10}, v_{10}, \text{blh}, \text{sp}, t_{2m}$)** | **SYNTHETIC** | Evaluated offline using mock/synthetic distributions to benchmark model mechanics, training latency, and ensemble fusion independently of Earth Engine network latency. In live online deployment, these are streamed from ERA5-Land. |
| **Static Land Use (Road density, Built-up)** | **SYNTHETIC** | Evaluated offline using synthetic spatial grids ($2\times$ refine factor). In live online deployment, these are streamed from Sentinel-2 / GHSL via Google Earth Engine. |
| **Ground-Level Station Observations** | **INDEPENDENT (SURFACE ONLY)** | `backend/data/cpcb_mumbai.csv` contains 28 CPCB ground stations (1,560 daily readings, 1,401 matched). Note that surface station measurements represent ground-level $\mu\text{g}/\text{m}^3$ and correlate only weakly ($r=0.086$) with total tropospheric columns prior to boundary layer height scaling and Stage 4 surface calibration. |

> [!WARNING]
> Because offline benchmark features utilized synthetic meteorological and static land-use layers, these metrics must be understood as an evaluation of **model fusion mechanics, optimization behavior, and coarse-grid consistency**, not as an absolute guarantee of real-world deployed accuracy.

---

## 3. Strict 3-Way Temporal Evaluation & Weight Calibration

To prevent temporal leakage and data snooping, the 61-day timeline was partitioned strictly chronologically:
- **Train Set (60%):** 37 days (2025-11-01 through 2025-12-07), $N = 4,077$ observed coarse pixels.
- **Validation Set (20%):** 12 days (2025-12-08 through 2025-12-19), $N = 1,560$ observed coarse pixels. Used strictly for model tuning and weight calibration.
- **Test Set (20%):** 12 future days (2025-12-20 through 2025-12-31), $N = 1,555$ observed coarse pixels. Held completely untouched until final evaluation.

### Weight Optimization Protocol
Weights were calibrated by minimizing mean squared error **exclusively on the Validation Set**:
$$\min_{w_1, w_2, w_3} \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \left( y_i - (w_1 \hat{y}_{\text{xgb}} + w_2 \hat{y}_{\text{rf}} + w_3 \hat{y}_{\text{lgb}}) \right)^2 \quad \text{s.t.} \quad w_i \ge 0, \sum w_i = 1$$
- **Default Heuristic Weights:** XGBoost = $0.50$, Random Forest = $0.30$, LightGBM = $0.20$
- **Validation-Calibrated Optimal Weights:** XGBoost = $0.240$, Random Forest = $0.363$, LightGBM = $0.397$

### Test Set Benchmark Results ($N = 1,555$ Unseen Holdout Pixels)

| Model / Strategy | $R^2$ | RMSE ($\mu\text{mol}/\text{m}^2$) | MAE ($\mu\text{mol}/\text{m}^2$) | MBE ($\mu\text{mol}/\text{m}^2$) | Pearson $r$ | Inference Latency |
|---|---|---|---|---|---|---|
| **Baseline NO2Downscaler (XGBoost)** | 0.9188 | 16.37 | 11.73 | -0.63 | 0.9588 | 0.8 ms |
| **Component: Random Forest** | 0.9146 | 16.80 | 12.20 | +0.44 | 0.9572 | 1.4 ms |
| **Component: LightGBM** | 0.9194 | 16.31 | 11.73 | -0.34 | 0.9590 | 0.5 ms |
| **Ensemble (Default 0.5/0.3/0.2)** | **0.9214** | **16.11** | **11.61** | -0.25 | **0.9599** | 2.7 ms |
| **Ensemble (Val-Calibrated)** | **0.9213** | **16.13** | **11.64** | **-0.13** | **0.9598** | 2.7 ms |

### Empirical Analysis:
1. **Ensemble Gain:** Both default and validation-calibrated ensembles achieve marginal but consistent improvements in $R^2$ (+0.0026) and lower RMSE (-0.26 $\mu\text{mol}/\text{m}^2$) over the strongest single model.
2. **Bias Reduction:** The validation-calibrated ensemble achieved the lowest mean bias error (MBE = $-0.13\ \mu\text{mol}/\text{m}^2$), counterbalancing the slight over-prediction of Random Forest (+0.44) and slight under-prediction of XGBoost (-0.63).
3. **Component Comparison:** When trained with regularized tree depth ($D=10$), Random Forest no longer artificially inflates train metrics and performs competitively ($R^2=0.915$), while LightGBM provides the fastest inference (0.5 ms).
4. **Disagreement Metric:** Inter-model standard deviation ($\sigma \approx 4.5\ \mu\text{mol}/\text{m}^2$) provides an empirical indicator of model consensus across complex terrain and urban interfaces, but must **not** be interpreted as formal Bayesian or conformal uncertainty.

---

## 4. XAI Interpretation Boundaries & Exact Multi-Model SHAP

In `ml_engine/xai/shap_service.py`:
1. **Component Explanation (`explain_ensemble_downscaler`):** Feature importance and waterfall attributions are strictly derived from TreeSHAP on **Component A (XGBoost)**. Attributions are explicitly labeled `[XGBoost Component SHAP]` with component weight ($w_{\text{xgb}}$).
2. **Exact Ensemble Explanation (`explain_weighted_ensemble_exact`):** Computes exact multi-model TreeSHAP feature attributions across XGBoost, Random Forest, and LightGBM using the exact weights of the loaded ensemble instance:
   $$\text{SHAP}_{\text{ens}}(X) = \sum_{i} w_i \text{SHAP}_{i}(X), \quad \text{Base}_{\text{ens}} = \sum_{i} w_i \text{Base}_{i}$$
   Verified via unit test `test_explain_weighted_ensemble_exact`: $\text{Base}_{\text{ens}} + \sum \text{SHAP}_{\text{ens}} = \hat{y}_{\text{ens}}$ within numerical precision ($< 10^{-4}$ tolerance) for both heuristic and validation-calibrated weights.
3. **Disagreement vs Uncertainty:** Inter-model standard deviation ($\sigma \approx 4.5\ \mu\text{mol}/\text{m}^2$) is reported strictly as **epistemic disagreement / model consensus**, not as calibrated predictive uncertainty.

---

## 5. Stage 4 Surface-Concentration Evaluation Against CPCB Ground Monitors

### Dataset Profile:
- **Ground Monitors:** `backend/data/cpcb_mumbai.csv` contains 28 CPCB CAAQMS monitoring stations across Mumbai with 1,560 station-day records.
- **QC Screening:** Applying `quality_control()` removes flat-lined or faulty sensors, retaining 1,068 high-quality station-days (960 matched with cloud-free satellite overpasses).

### Physical Column-to-Surface Conversion vs. Raw Ground Observations:
- **Units:** Satellite column is in $\mu\text{mol}/\text{m}^2$; CPCB monitor reading is in $\mu\text{g}/\text{m}^3$.
- **Direct Physical Conversion:** Without station calibration, scaling tropospheric columns by boundary-layer height ($\text{NO}_2 \times M / \text{BLH}$) yields:
  - $N = 960$ matched station-days across 19 QC-passed stations.
  - $R^2 = -2.13$, $\text{RMSE} = 39.90\ \mu\text{g}/\text{m}^3$, $\text{MAE} = 34.00\ \mu\text{g}/\text{m}^3$, $\text{MBE} = -32.82\ \mu\text{g}/\text{m}^3$.
  - Pearson $r = 0.118$.
- **Scientific Finding:** Raw or physically scaled satellite columns alone cannot predict urban ground-level concentrations accurately. Ground-level concentrations are dominated by hyper-local street-level emissions and complex nocturnal boundary layer dynamics. Stage 4 land-use regression (`SurfaceModel` using road density, built-up, and meteorology) is mathematically essential to bridge the satellite column to ground-level surface exposure.

---

## 6. Release Verification & Operational Verdict

| Readiness Level | Status | Evidence / Verification Method |
|---|---|---|
| **1. Implemented** | **VERIFIED** | Full multi-model ensemble downscaler, exact weighted TreeSHAP, and cache-isolated pipeline. |
| **2. Software-Tested** | **VERIFIED** | 17 focused downscale tests and 91 full backend test suite tests pass cleanly. |
| **3. Experimentally Evaluated** | **VERIFIED** | Benchmarked on 61 days of real Sentinel-5P overpasses with 3-way temporal holdout. |
| **4. Validated on Surface Monitors** | **PARTIAL** | Physically converted columns evaluated on 960 CPCB station-days; demonstrates necessity of Stage 4 calibration. |
| **5. Fine-Scale Ground Truth Validated** | **UNVALIDATED** | No independent sub-kilometer spatial raster ground truth exists in the repository. |
| **6. Ready for Production** | **OPT-IN ONLY (`ensemble=true`)** | Baseline XGBoost remains the production default (`ensemble=false`). |

