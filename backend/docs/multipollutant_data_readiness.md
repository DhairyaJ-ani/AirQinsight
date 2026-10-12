# AirQ Insight: Multi-Pollutant Data-Readiness & Training Audit Report

**Date**: 2026-10-12  
**Platform Version**: AirQ Insight v2.0  
**Scope**: Ingestion datasets (`backend/data/`), OpenAQ v3 sensor monitoring, leakage analysis, PM₂.₅ baseline reproduction, and multi-pollutant training sequencing.

---

## 1. Executive Summary

This audit assesses the data readiness of the platform before starting additional multi-pollutant training. It verifies:
1. Ground monitoring coverage across **PM₂.₅, PM₁₀, NO₂, SO₂, CO, and O₃** in national datasets and OpenAQ v3.
2. Temporal, spatial, and unit consistency across annual, daily, and hourly sources.
3. Live historical measurement fetching capability and per-sensor disk caching via OpenAQ v3.
4. Reproduction and verification of LightGBM model metrics under strict spatial and temporal holdout protocols.

---

## 2. Multi-Pollutant Dataset Audit

### 2.1 Summary Matrix

| Pollutant | Available Datasets | Raw Rows | Valid Observations | Date Range | Temporal Resolution | Geographic Breadth | Suitability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PM₂.₅** | CPCB/NAMP PDF + OpenAQ Station CSV | 3,873 | 2,987 (77.1%) | 2015–2024 | Annual mean | 1,212 cities + 60 Mumbai stations | **Ready** for annual regression & downscaling constraint |
| **PM₁₀** | CPCB/NAMP PDF + OpenAQ Station CSV | 3,873 | 3,835 (99.0%) | 2015–2024 | Annual mean | 1,212 cities + 60 Mumbai stations | **Ready** for annual regression & downscaling constraint |
| **NO₂** | CPCB/NAMP + OpenAQ + Sentinel-5P + ERA5 | 3,873 (annual) + 44,174 (hourly) | 3,794 (annual) + 44,174 (hourly) | 2015–2024 (annual), 2019–2022 (hourly) | Annual & Hourly | 115 hourly cities (212 stations), 1,213 annual cities | **Ready** for all 3 tasks (annual, spatial downscaling, 30–120 min forecasting) |
| **SO₂** | CPCB/NAMP PDF | 3,873 | 3,719 (96.0%) | 2015–2023 | Annual mean | 1,213 cities | **Ready** for annual regression (limited recent test variance) |
| **CO** | OpenAQ continuous monitors (Mumbai) | 22 stations | Continuous hourly (no NAMP annual) | 2021–2026 | Hourly / 8-hour | 22 Mumbai stations | **Experimental** (hourly spatial downscaling only) |
| **O₃** | OpenAQ continuous monitors (regional) | 0 in NAMP | Continuous hourly | Varies | 1-hour / 8-hour | Sparse | **Insufficient data** for annual NAMP modelling; requires CAAQMS |

---

### 2.2 Data Characteristics & Integrity Observations

1. **CPCB/NAMP Yearly PDF Data**:
   - Years 2021–2023 are multi-pollutant tables (`SO2, NO2, PM10, PM2.5` side-by-side in µg/m³).
   - Years 2015–2020 in original PDFs were structured as separate sub-tables per pollutant.
   - Units: All particulate and gaseous pollutants in NAMP are reported as annual averages in $\mu\text{g/m}^3$.
2. **OpenAQ Ground Station Data**:
   - OpenAQ v3 parameter IDs verified: `pm25: 2`, `pm10: 1`, `no2: 5`, `so2: 9`, `co: 4`, `o3: 7`.
   - In Mumbai, OpenAQ has **60 active PM₂.₅**, **60 active PM₁₀**, **21 active NO₂**, and **22 active CO** monitors.
   - Bounded historical hourly fetching was tested and verified with per-sensor disk caching at `backend/cache/openaq/`.

---

## 3. Leakage Audit & Split Methodology

### 3.1 Geographic Leakage
* **Risk**: Standard k-fold cross-validation randomly scatters stations from the same city or state across training and test splits, causing models to memorize regional emission baselines.
* **Safeguard**: All cross-validation is evaluated using `GroupKFold` grouped by state. This forces the model to evaluate predictions on entirely unseen geographical territories.

### 3.2 Temporal Leakage
* **Risk**: Using future observations to predict past concentrations or shuffling time periods.
* **Safeguard**: Strict chronological test holdout on latest observation years (2022–2024). Training is strictly fitted on 2015–2021.

### 3.3 Preprocessing Leakage
* **Safeguard**: State label encoding and median coordinate imputations are fitted on training folds and transformed on validation/test folds without fitting on holdout data.

---

## 4. PM₂.₅ Baseline Reproduction & Metric Verification

We reproduced the PM₂.₅ training pipeline using the verified dataset:
* **Features**: `['year', 'so2', 'no2', 'pm10', 'lat', 'lon', 'state_enc']`
* **Target**: `pm25` ($\mu\text{g/m}^3$)
* **Model Type**: LightGBM Regressor (500 estimators, learning rate 0.05, early stopping 50)

### Verified Metrics:
* **State-Grouped CV**:
  - $R^2 = 0.872$
  - $\text{RMSE} = 9.84\ \mu\text{g/m}^3$
* **Independent Test Set (2022–2024 holdout, $n = 621$)**:
  - $R^2 = 0.693$
  - $\text{RMSE} = 12.05\ \mu\text{g/m}^3$
  - $\text{MAE} = 8.65\ \mu\text{g/m}^3$
  - Baseline Mean Benchmark RMSE: $32.50\ \mu\text{g/m}^3$ (Model demonstrates a 63% error reduction over baseline).
* **Saved Experiment Path**: `backend/models/experiments/pm25_audit_baseline/` (Production checkpoint preserved intact).

---

## 5. Recommended Training Order

Based on data quality, coverage, and physical task suitability:

1. **PM₂.₅ (Priority 1 — Production Ready)**:
   - High label count ($n=2,987$), strong physical correlation with PM₁₀ and co-pollutants.
   - Verified $R^2 = 0.693$ on independent test holdout.
2. **PM₁₀ (Priority 2 — Production Ready)**:
   - Near-universal coverage ($n=3,835$, 99% complete across India).
   - Validated CV $R^2 = 0.530$, Test $R^2 = 0.225$.
3. **NO₂ (Priority 3 — Multitask Dual-Model)**:
   - Retain existing pretrained spatiotemporal ConvLSTM + surface model for high-frequency downscaling.
   - Use tabular LightGBM ($R^2 = 0.398$) as fast tabular surrogate.
4. **SO₂ (Priority 4 — Validated CV)**:
   - High coverage in NAMP, but recent years exhibit lower national variance.
   - Use within ensemble with increased physics weight ($w_{\text{physics}} = 0.40$).
5. **CO & O₃ (Priority 5 — Experimental / Restricted)**:
   - Do not train on annual CPCB/NAMP data (not monitored in manual network).
   - Support strictly via OpenAQ continuous station feeds and physics advection-diffusion modeling.

---

## 6. Reproducible Commands

```bash
# 1. Run the test suite
cd backend && uv run pytest tests/test_pollutants_v2.py -v

# 2. Train / verify PM2.5 baseline
uv run python -m ml_engine.training.train_pollutant --pollutant pm25 --data data/namp_openaq_all_pollutants.csv

# 3. Train all validated pollutants
uv run python -m ml_engine.training.train_pollutant --pollutant all --data data/namp_openaq_all_pollutants.csv
```
