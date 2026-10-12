# AirQ Insight — Physics Model Plan

**Branch**: `feat/physics-model`
**Scope**: SRS §7.4 (physics-based ensemble member) and §8 (30–120 min forecasting), built on top of the existing NO₂ downscaling model.

## 1. What the physics model is

It is not trained from data. It is the transport equation

    ∂C/∂t + u·∇C = K∇²C − k·C + S,    k = 1/τ_chem + v_d/H

solved on the model's fine grid. Data only supplies its **inputs** (starting map, wind, boundary-layer height, terrain, sources) and its **validation**.

| Term | Meaning | Source |
|---|---|---|
| C(t=0) | current concentration map | main NO₂ downscaler (other pollutants: see §5) |
| u, v | wind | ERA5 / ERA5-Land (`ingestion/gee.py`) |
| H | mixing-layer depth for dry deposition | `fetch_blh` in `ingestion/gee.py` (GEOS-CF ZPBL, bulk-parametrisation fallback) |
| K, τ_chem, v_d | per-pollutant constants | `pollutant_config.py` |
| S | persistent emissions | existing static layers (power, industry, traffic masks) |

## 2. Verified state of the repo (2026-10-12)

1. **Two solvers exist.** `ml_engine/dispersion.py` (legacy, NO₂-only, constant wind, horizons 1/3/6 h) is what `pipeline.py` and `service.py` call today. `ml_engine/forecasting/physics.py` (`ImprovedPhysicsSolver`: hourly wind, 30-min frames, terrain, source persistence) is what `forecasting/inference.py` calls. They are not unified.
2. **Done on this branch**: `ImprovedPhysicsSolver` is pollutant-aware (`PhysicsConfig.for_pollutant`), has a dry-deposition sink, a `mass_policy`, an `advection_order`, and accepts boundary-layer height per call. Default NO₂ output is bit-identical to `main`.
3. **Hourly ground truth is now fetched on demand** (OpenAQ v3, cached under `backend/cache/hindcast/<city>/`, gitignored). OpenAQ's Indian NO₂ series end in 2022-10, so hindcasts use 2021–22. Mumbai has only ~5 stations with continuous hourly data; Delhi has 44 reporting.
4. **The NAMP annual CSV is corrupted** (scrambled 2015–2020 columns, broken state column, 2011–2014 duplicated). It is not usable for this model and should be re-parsed before any PM/SO₂ baseline is trusted.

## 3. Solver accuracy benchmark (no external data)

Gaussian puff, 250 m cells, K = 50 m²/s, 1–2 h. Analytic added variance per axis is 2Kt.

| Case | Added variance vs analytic |
|---|---|
| No wind | exact (1.00×) |
| Wind 1–6 m/s, bilinear advection (original) | 1.33×–1.50× along the wind (numerical diffusion ≈ 17–25 m²/s); cross-wind exact |
| Wind 1–6 m/s, cubic advection | 1.00× |

Centroid position and total mass are correct in every case. A hard-edged hotspot showed no ringing or mass gain with cubic. Cubic is therefore used for all non-NO₂ pollutants. NO₂ stays bilinear by default so existing results do not shift; switching it is a one-line config change once validated.

## 4. Phases and status

**Phase A — Solver correctness: done**
- Pollutant-specific loss terms, dry deposition with per-call boundary-layer height, mass policy, cubic advection option. ✔
- Analytic benchmarks as regression tests. ✔
- Not done: an explicit emission term for conservative pollutants (PM, CO; needs inventories); time-varying boundary-layer height inside a run.

**Phase B — Attach to the main model: done as opt-in, deliberately not the default**
- `service.forecast_point(..., solver="legacy" | "improved")`. Legacy stays the default. The response gains a `solver` key (backward compatible). Tests: `tests/test_forecast_point_solver.py`. ✔
- The spatial forecast (`trends_service._run_spatial_forecast`) already uses `ForecastEngine` / `ImprovedPhysicsSolver` with NO₂ defaults. ✔ (unchanged)
- Not wired, on purpose: the annual `/pollutants/{city}/ensemble/{pollutant}` endpoint takes a caller-supplied `physics_estimate`. It blends annual tabular values, so a 1–2 h transport result does not belong in it, and its LightGBM members were trained on the corrupted NAMP CSV.
- Not done: What-If Simulator scenario controls mapped onto solver inputs.

**Phase C — Validation: done for NO₂/PM₂.₅ in Delhi; result is negative for the deployed path** (see `physics_hindcast_report.md`)
- Station-seeded (isolates transport): Delhi NO₂ +2 % / +4 % RMSE over not moving the field at 60 / 120 min; Mumbai inconclusive; PM₂.₅ no detectable skill. ✔
- Calibration on November, test on December: tuning K and τ changes nothing measurable. ✔
- Deployed satellite map: drifting the map degrades it by 12–16 % (improved and legacy solvers alike); a steady-source variant collapses to "no change". ✔
- Not done: held-out-station retraining of the surface model, multi-month/other-city runs, spatially varying wind.

**Phase D — Pollutants beyond NO₂: not started**
- SO₂, CO: add Sentinel-5P column products and a column-to-surface step before they can be an initial field.
- PM₂.₅, PM₁₀: no TROPOMI product. Initial field must come from a ground-observation model.
- O₃: secondary, photochemically produced. Advection alone is not defensible; stay `experimental`.

## 5. Known limits

- For a chemically conservative pollutant with no deposition (k = 0), "persistent" emission reduces to a free plume. PM and CO need an explicit emission term from an inventory.
- `pollutant_config.py` lists CO lifetime as 30 h; the real chemical lifetime is weeks to months. No effect at 2 h horizons, but the value is wrong.
- NO₂ still rescales total mass to its initial value each frame (original behaviour). That hides real decay and outflow.
- Annual city means (CPCB tables) can only give coarse plausibility bounds, not drive or validate a 30-minute model.
- The surface model's target is a 24 h level and it is in-sample at the OpenAQ stations for 2019–2022; both limit what any hourly hindcast can show.
- No absolute forecast accuracy is demonstrated, so none should be claimed.

## 6. Recommendations

1. Keep `solver="legacy"` as the default and treat short-horizon `/trends` values as unvalidated (they are not better than the unmoved map in the Delhi sample).
2. Retarget the surface model to overpass-hour values; the 24 h target caps hourly skill.
3. If a transport forecast is wanted, advect anomalies against an emission-locked baseline (`steady_source_forecast`) instead of the whole field.
4. Decide on NO₂ cubic advection and on the mass rescale only after step 2 and a held-out re-run of Protocol B.
5. Environment: in the Python 3.14 venv uv creates here, two tests abort the interpreter inside LightGBM prediction (`tests/test_pollutants_v2.py::test_pm25_model_prediction`, `::test_api_predict_pm25`; reproducible on `main`). The cause was not isolated; trying Python 3.12 or 3.13 is the first thing to test.
