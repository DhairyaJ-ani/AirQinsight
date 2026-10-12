# AirQ Insight v2.0: Multi-Pollutant Training and Audit Report

## 1. Executive Summary & Audit Context

AirQ Insight is an environmental intelligence and community protection platform transitioning from an initial NO₂-focused downscaling and advection-diffusion forecasting system to a full multi-pollutant system covering **PM₂.₅, PM₁₀, NO₂, SO₂, CO, and O₃**.

This audit rigorously inspects:
1. The CPCB/NAMP national ambient air quality dataset (`backend/data/namp_all_pollutants_clean.csv`).
2. The initial PM₂.₅ LightGBM model (`backend/models/pm25_lightgbm/`).
3. External ground monitoring data (OpenAQ v3 API).
4. Spatial-temporal forecasting models (`ConvLSTM` + advection-diffusion physics).
5. Explainable AI (`SHAPService`) and model serving registry.

---

## 2. Dataset Audit & Quality Analysis (`namp_all_pollutants_clean.csv`)

### 2.1 Dataset Inventory & Provenance
* **File Source**: Extracted from CPCB yearly PDF reports (2015–2023) in `backend/data/all_pollutants/`.
* **Total Rows**: 3,752 city-year observation rows.
* **Columns**: `year, state, city, so2, no2, pm10, pm25, lat, lon`.
* **Missingness per column**:
  - `year`, `state`, `city`: 0 missing (100% complete).
  - `so2`: 33 missing (99.1% coverage).
  - `no2`: 28 missing (99.3% coverage).
  - `pm10`: 28 missing (99.3% coverage).
  - `pm25`: 860 missing (77.1% coverage).
  - `lat`, `lon`: 3,487 missing (92.9% missing; only 265 rows mapped to coordinates in default dictionary).

### 2.2 Critical Data Observations & Vulnerabilities Discovered
1. **Multi-Pollutant Table Shift Across Years**:
   - In 2021, 2022, and 2023, CPCB published integrated tables with all 4 pollutants (`SO2, NO2, PM10, PM2.5`) side-by-side.
   - In 2015–2020, CPCB PDFs presented separate sub-sections for each pollutant (e.g., SO₂ section, NO₂ section, PM₁₀ section, PM₂.₅ section). The initial PDF parser regex assumed 4 columns side-by-side, causing columns from single-pollutant tables in 2015–2020 to be misaligned (e.g., days/min/max/average captured instead of SO2/NO2/PM10/PM25).
   - In 2021–2023, the data format is verified and valid (`SO2, NO2, PM10, PM2.5` annual averages in µg/m³).
2. **Missing Lat/Lon Imputation Leakage Risk**:
   - The initial training script imputed missing coordinates using the `state` median. However, because states had malformed names in earlier years, median imputation created arbitrary coordinates.
   - Geographic coordinates (`lat, lon`) when heavily imputed can act as noisy proxies or leak target geography if not partitioned by state.
3. **Target Definition & Resolution**:
   - Target variable: Annual-mean surface concentration (µg/m³).
   - Temporal resolution: Annual statistics (not 30-minute high-frequency forecasts).
   - Spatial resolution: City-level aggregate monitoring station averages.

---

## 3. Verification of Existing PM₂.₅ Model Artifacts

### 3.1 Saved Model Inspection
* **Artifact Path**: `backend/models/pm25_lightgbm/pm25_model.txt`
* **Metadata**: `backend/models/pm25_lightgbm/pm25_metadata.json`
* **State Encoder**: `backend/models/pm25_lightgbm/pm25_state_enc.json`
* **Features**: `['year', 'so2', 'no2', 'pm10', 'lat', 'lon', 'state_enc']`
* **Reported Metrics**:
  - Independent Test (2022–2023 holdout): $R^2 = 0.750$, $\text{RMSE} = 10.884\ \mu\text{g/m}^3$, $\text{MAE} = 7.907\ \mu\text{g/m}^3$.
  - GroupKFold CV (state groups): $R^2 = 0.847$, $\text{RMSE} = 9.561\ \mu\text{g/m}^3$.
* **Artifact Loadability**: Successfully verified with `lightgbm.Booster(model_file=...)`.

### 3.2 Evaluation Methodology Assessment
* **Temporal Leakage**: None in the holdout split; test set strictly held out the latest years (2022–2023).
* **Spatial Leakage**: Managed via `GroupKFold(state_enc)`, ensuring entire states were placed together in validation folds.
* **Pre-processing Leakage**: State encoding and coordinate imputation must be fitted strictly on the training partition and transformed on test sets to prevent preprocessing leakage.

---

## 4. OpenAQ Ground Station Capabilities (Mumbai Audit)

* Tested live query using OpenAQ v3 API with key in `.env`.
* **Mumbai Bounding Box**: `[72.77°E, 18.88°N, 73.12°E, 19.32°N]`.
* Discovered **60 active sensor monitors** for PM₂.₅ across Mumbai metropolitan stations.
* Validated that live monitoring feeds match OpenAQ v3 parameter IDs (`pm25: 2, pm10: 1, no2: 5, so2: 9, co: 4, o3: 7`).

---

## 5. Three-Component Ensemble Architecture Design

To achieve an explainable multi-pollutant environmental intelligence platform:

1. **Component A: Spatiotemporal AI (Neural / ConvLSTM)**
   - Reuses existing ConvLSTM architecture (`models/forecasting/checkpoint_best.pt`).
   - Dedicated to high-frequency spatial forecasting (30, 60, 90, 120 min) and spatial downscaling.
2. **Component B: LightGBM Regressor (Tabular)**
   - Fast, CPU-friendly multi-pollutant predictor trained per species (`PM2.5, PM10, NO2, SO2`).
   - Uses co-pollutants, meteorological and geographic covariates.
3. **Component C: Physics-Informed Advection-Diffusion Solver**
   - Reuses `ImprovedPhysicsSolver` with pollutant-specific decay rate $\tau$ and dispersion coefficient $K$.
   - Advection governed by ERA5 wind fields, enforcing mass conservation.

### Ensemble Fusion:
$$\hat{C}_{\text{ensemble}} = w_{\text{ai}} \hat{C}_{\text{ai}} + w_{\text{lgbm}} \hat{C}_{\text{lgbm}} + w_{\text{physics}} \hat{C}_{\text{physics}}$$
with $\sum w_i = 1$, $w_i \ge 0$, adaptive fallback if any component is unavailable.

---

## 6. Recommended Implementation Sequence

1. **Phase 2 & 3**: Refine data preparation pipeline with robust validation, clean splits, and reproducible CLI for PM₂.₅.
2. **Phase 4**: Implement generalized multi-pollutant trainer (`train_pollutant.py`) supporting PM₂.₅, PM₁₀, SO₂, and NO₂ with strict data sufficiency checks (`insufficient_data` flag for CO/O₃).
3. **Phase 5**: Build the unified three-component ensemble module with adaptive weights and benchmark validation.
4. **Phase 6 & 7**: Implement a lazy-loading Multi-Pollutant Model Registry and upgrade Explainable AI (TreeSHAP for LightGBM + physics sensitivity + ensemble contribution breakdown).
5. **Phase 8 & 9**: Integrate with `/api/v1/pollutants/` preserving backward-compatible schemas and add test coverage.
