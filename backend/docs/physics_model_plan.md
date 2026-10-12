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

## 4. Phases

**Phase A — Solver correctness (this branch, mostly done)**
- Pollutant-specific loss terms, deposition, mass policy, cubic advection. ✔
- Analytic benchmarks as regression tests. ✔
- Remaining: a conservative-pollutant emission term (see §5), time-varying boundary-layer height.

**Phase B — Unify and attach to the main model**
- Make the main path (`pipeline.py` / `service.py`) call `ImprovedPhysicsSolver` instead of the legacy `dispersion.py`, keeping the `/analyze` and simulator response contracts. Keep the legacy solver as a fallback until parity is shown.
- Adapter that turns the pipeline's coarse met dataset (`u10`, `v10`, `blh`) into the `wind_sequence` and `mixing_height_m` the solver takes, resampled to the fine grid.
- Ensemble: expose the physics output as the `physics` member in `ml_engine/ensemble.py` for every pollutant, with explicit fallback when inputs are missing.
- What-If Simulator: scenario controls (wind, boundary-layer height, emission scaling) map onto solver inputs; results labelled "modelled".

**Phase C — Validate against observations (started; see `physics_hindcast_report.md`)**
- Done: leave-stations-out hindcast harness, OpenAQ + ERA5-Land + GEOS-CF fetch. First result (Delhi, NO₂): +2 % / +4 % RMSE over not moving the field at 60 / 120 min; decay without emissions is harmful; Mumbai inconclusive (≈5 usable stations). Not yet done: hindcast from the satellite surface map, calibration, other pollutants.
- Original plan:
- Hindcast: start from the satellite-constrained map at overpass time, forecast +30/60/90/120 min, compare with hourly station readings (OpenAQ / CPCB).
- Report RMSE, MAE, R², bias and the persistence baseline per horizon, with sample counts and region; split by time and station.
- Calibrate K, τ_chem, v_d, and ensemble weights on a validation split only; report once on a held-out test.
- Acceptance: the physics member must beat persistence at some horizon, or it is reported as not adding value (SRS §17.4).

**Phase D — Pollutants beyond NO₂**
- SO₂, CO: add Sentinel-5P column products and a column-to-surface step before they can be an initial field.
- PM₂.₅, PM₁₀: no TROPOMI product. Initial field must come from a ground-observation model; physics gives transport and deposition only.
- O₃: secondary, photochemically produced. Advection alone is not defensible; stay `experimental` until a chemistry term and data exist.

## 5. Known limits

- For a chemically conservative pollutant with no deposition (k = 0), "persistent" emission reduces to a free plume. PM and CO need an explicit emission term from an inventory.
- `pollutant_config.py` lists CO lifetime as 30 h; the real chemical lifetime is weeks to months. No effect at 2 h horizons, but the value is wrong.
- NO₂ still rescales total mass to its initial value each frame (original behaviour). That hides real decay and outflow; revisit after validation.
- Annual city means (CPCB tables) can only give coarse plausibility bounds, not drive or validate a 30-minute model.
- Until Phase C is done, no accuracy claim should be made for the physics forecasts.

## 6. Needed from the team

- OpenAQ API key (or CPCB/CAAQMS hourly exports) for Phase C.
- Earth Engine access for Sentinel-5P SO₂/CO and GEOS-CF boundary-layer height (Phase D).
- Decision on whether NO₂ should move to cubic advection after validation.
