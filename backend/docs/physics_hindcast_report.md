# Physics Solver — Leave-Stations-Out Hindcast (NO₂)

**Branch**: `feat/physics-model` · **Window**: 2021-11-01 to 2022-02-28 · **Cities**: Mumbai, Delhi
**Code**: `ml_engine/forecasting/hindcast.py`, `hindcast_data.py` · **Tests**: `tests/test_hindcast.py`
**Reproduce**: `uv run python -m ml_engine.forecasting.hindcast --city delhi --start 2021-11-01 --end 2022-02-28`

## 1. What was tested

Whether moving a concentration field with the wind predicts the next 60 and 120 minutes at a station better than leaving the field where it is.

For each origin hour and each of 4 station folds: build the starting field by inverse-distance interpolation of the stations **outside** the fold, run the solver with hourly ERA5-Land wind, and score the forecast against the held-out stations' real readings. Held-out readings never enter the starting field (verified by `test_held_out_station_reading_cannot_reach_the_forecast`).

| Predictor | Meaning |
|---|---|
| `physics_bilinear` | `PhysicsConfig()` defaults: bilinear advection, 4 h lifetime, persistent sources, mass rescale |
| `physics_cubic` | as above with cubic advection |
| `physics_cubic_free` | cubic, **no** emissions, 4 h decay, no mass rescale |
| `no_transport` | the same interpolated field, not moved. **The fair baseline**: it has exactly the information the physics had |
| `own_persistence` | the held-out station's own reading at the origin. Uses information the physics lacks, so a reference ceiling, not a fair opponent |

`skill` = 1 − RMSE / RMSE(`no_transport`). The 95 % interval resamples whole days (1000 draws).

**Data**: OpenAQ v3 hourly NO₂ (µg/m³); ERA5-Land hourly 10 m wind (domain mean per hour); GEOS-CF boundary-layer height was downloaded but is not used for NO₂ (no deposition term). Readings ≤ 0 or > 500 µg/m³ are discarded.

## 2. Results (RMSE in µg/m³)

**Delhi** — 44 stations in the box, 46–47 days

| Horizon | n | `no_transport` | `physics_bilinear` | skill [95 % CI] | `physics_cubic_free` skill | `own_persistence` |
|---|---|---|---|---|---|---|
| 60 min | 516 | 61.3 | 60.0 | +2.1 % [+0.2, +3.8] | −10.6 % [−13.6, −8.2] | 24.9 |
| 120 min | 569 | 65.1 | 62.3 | +4.3 % [+2.0, +6.0] | −16.3 % [−20.4, −12.1] | 36.6 |

`physics_cubic` is indistinguishable from `physics_bilinear` (60.0 / 62.3).

**Mumbai** — about 5 stations with continuous data, 21–22 days

| Horizon | n | `no_transport` | `physics_bilinear` | skill [95 % CI] | `physics_cubic_free` skill | `own_persistence` |
|---|---|---|---|---|---|---|
| 60 min | 97 | 57.0 | 48.8 | +14.4 % [−3.2, +20.0] | +16.3 % [−4.8, +24.0] | 15.1 |
| 120 min | 101 | 45.9 | 43.5 | +5.2 % [+0.1, +7.7] | +0.2 % [−9.3, +4.3] | 20.3 |

Skill by wind speed (Delhi, `physics_bilinear`): calm (< 1.5 m/s) −1.8 % / +2.5 %; 1.5–3 m/s +4.6 % / +5.8 %; > 3 m/s +0.7 % / +3.4 % (60 / 120 min). Mumbai wind classes have n = 8–24 and are not interpretable.

## 3. What the results support

1. **Transport with persistent sources gives a small, statistically positive gain in Delhi** (2–4 % RMSE over not moving the field; the interval excludes zero at both horizons). It grows with lead time and is concentrated at moderate wind. This is the direction physics predicts.
2. **Decay without emissions is harmful** (−11 % to −16 %, intervals clearly below zero). Applying the NO₂ lifetime to a field with no ongoing emission removes mass that the real atmosphere keeps replenishing. The `emission_mode="none"` configuration should not be used for forecasts.
3. **Mumbai is inconclusive.** Several intervals include zero and the ranking of variants changes between horizons.
4. **Cubic advection showed no benefit here.** Its advantage is real against the analytic puff (`TestAnalyticPuff`), but interpolation error in the starting field is far larger than the numerical diffusion it removes.

## 4. What they do not support

- **No claim of forecast accuracy.** The starting field has an RMSE of about 60 µg/m³ and a bias of about −30 µg/m³ in Delhi. The physics gain is 1–3 µg/m³ on top of that. A station's own last reading beats everything shown (RMSE 25 vs 60 at 60 min).
- **This is not the deployed configuration.** The real model starts from a satellite-derived surface map, whose own blocked-CV R² is about 0.18 (`no2_surface_24h_report.json`). The deployed path has not been hindcast.
- **Mumbai provides almost no test.** Only about 5 of 19 stations report continuously; most report a few dozen hours in the window.
- **Wind is one domain-mean value per hour.** It cannot represent sea-breeze or other within-box gradients, and the 500 m grid is far finer than the station spacing.
- **Single winter, India only, 60/120 min only.** OpenAQ hourly data is hourly, so 30 and 90 min are not scored.
- **Tuning was not done.** Diffusivity, lifetime and source strength are the defaults. Nothing was calibrated, so there is no overfitting, and also no optimisation.

## 5. Status

| Question | Status |
|---|---|
| Solver implemented and unit-tested (analytic puff, loss terms, per-pollutant config) | **Yes** |
| Hindcast harness implemented, leakage-tested | **Yes** |
| Physics adds measurable skill over not moving the field (Delhi, NO₂, 60–120 min) | **Yes, small** (+2 to +4 %) |
| Physics forecast accurate in absolute terms | **Not shown** |
| Validated from the satellite-derived starting map (deployed path) | **No** |
| Validated for pollutants other than NO₂ | **No** |
| Ready to be reported as a validated forecast | **No** |

## 6. Next steps

1. Hindcast from the satellite surface map (the deployed starting field) and compare with the station-seeded result.
2. Calibrate source persistence (relaxation toward the starting field) and diffusivity on a validation split only; report once on a held-out period.
3. Spatially varying ERA5 wind instead of a domain mean.
4. Same protocol for PM₂.₅ and SO₂ once their starting fields exist.
