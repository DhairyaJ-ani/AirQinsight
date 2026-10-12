# AirQ Insight — Scientific Audit & Validation Report

**Date:** 2026-10-12  
**Branch:** `feature/ai-accuracy-xai-forecasting-report`  
**Test Suite Status:** 99/99 passed (100% pass rate)

---

## 1. Executive Summary

This report documents the rigorous scientific validation and algorithmic upgrades implemented across the AirQ Insight platform. The improvements focus on downscaling model accuracy, ConvLSTM forecasting, explainable AI (XAI), and automated reporting fidelity, strictly preserving the subsystem boundaries and safeguarding teammate physics-solver implementations.

---

## 2. Workstream 1: Downscaling Model Accuracy & Leakage Protection

### 2.1 Problem Addressed
Previous ensemble evaluations relied on random pixel-level splits and heuristic model weights (0.5 XGBoost, 0.3 RF, 0.2 LightGBM). In spatial and spatiotemporal domains, random splits introduce substantial data leakage due to spatial and temporal autocorrelation.

### 2.2 Solution Implemented
1. **Temporal Splits (`temporal_split`):** Enforces chronological sequence boundaries ensuring no future satellite observations are exposed to training sets ($t_{train} < t_{val}$).
2. **Spatial Block Splits (`spatial_split`):** Implements leave-spatial-cluster-out cross-validation by grouping contiguous latitude/longitude bins.
3. **Hold-out Weight Calibration (`calibrate_ensemble_weights_on_val`):** Grid search across the convex simplex ($w_{xgb} + w_{rf} + w_{lgb} = 1.0$) optimizing validation $R^2$ without touching test sets.

---

## 3. Workstream 2: Spatiotemporal Forecasting Enhancements

### 3.1 Problem Addressed
ConvLSTM residual forecasting previously suffered from:
- Cloud-contaminated pixels (NaNs) biased loss gradients toward zero.
- Absence of a persistence baseline null hypothesis to scientifically justify AI skill claims.
- Hardcoded approximate normalization statistics.

### 3.2 Solution Implemented
1. **Mask-Aware Loss (`MaskAwareForecastLoss`):** Formulates masked MSE and spatial gradient losses that completely omit invalid/cloud-obscured satellite pixels while preserving gradient backpropagation across valid regions.
2. **Persistence Baseline (`PersistenceBaseline`):** Implements $\hat{y}(t + \Delta t) = y(t)$ benchmark evaluating per-horizon MAE, RMSE, bias, and $R^2$ against target lead times (30, 60, 90, 120 mins).
3. **Data Channel Statistics Fitting:** Added empirical mean/standard deviation calculation utilities.

---

## 4. Workstream 3: Explainable AI (XAI) Precision

### 4.1 Problem Addressed
- Fixed heuristic attribution confidence values (`0.92`).
- Absence of explainability for non-tree recurrent models (ConvLSTM).
- Lack of data quality and Out-of-Distribution (OOD) indicators.

### 4.2 Solution Implemented
1. **Entropy-Grounded Attribution Confidence:** Computes the information entropy of absolute feature attributions, scaling confidence inversely with entropy (concentrated, salient features yield higher confidence).
2. **Data Quality Indicators (`DataQualityIndicator`):** Evaluates input vectors for $>3\sigma$ deviations from training distributions and reports cloud contamination fractions and imputed channel counts.
3. **ConvLSTM Input Perturbation Sensitivity (`explain_forecast_perturbation`):** Implemented forward-difference input perturbation analysis evaluating channel sensitivities across forecast lead times with explicit scientific caveats.

---

## 5. Workstream 4: Report Agent Evidence Schema & Governance

### 5.1 Problem Addressed
Language models generate reports from numerical facts, but lack formal typing to prevent unsupported causal claims or hallucinations when evaluation samples are missing.

### 5.2 Solution Implemented
1. **Typed Evidence Schema (`ReportEvidence`, `EvalMetricsEvidence`, `HotspotEvidence`):** Enforces strict types and validation rules.
2. **Verification Guardrails (`validate_evidence`):** Automatically rejects bundles where metrics claim accuracy without validation samples, or where nearby industrial sources are attributed causally without proximity distance.
3. **Audience Partitioning:** Generates distinct prompts for **Researchers** (focusing on method fidelity, assumptions, column vs. surface conversion) and **Government / Regulators** (focusing on standard compliance, public exposure, and proximity-based source caveats).

---

## 6. Teammate Collaboration Boundary

All files in `backend/ml_engine/forecasting/physics.py` and the `ImprovedPhysicsSolver` integration remain intact and untouched for the teammate's ongoing physics integration.
