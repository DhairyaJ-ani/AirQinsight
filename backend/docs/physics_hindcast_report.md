# Physics Solver — Hindcast Validation Report

**Branch**: `feat/physics-model` · **Window**: 2021-11-01 to 2022-02-28 (OpenAQ's Indian hourly series end in 2022-10)
**Code**: `ml_engine/forecasting/hindcast.py`, `hindcast_data.py`, `hindcast_calibrate.py`, `hindcast_satellite.py`
**Tests**: `tests/test_hindcast.py` (harness, leakage, steady-source), `tests/test_physics_pollutants.py` (analytic puff, loss terms)

## 1. Headline

| Question | Answer |
|---|---|
| Does moving a station-interpolated NO₂ field with the wind beat leaving it in place? (Delhi) | **Yes, slightly**: +2 % / +4 % RMSE at 60 / 120 min, intervals above zero |
| Does tuning diffusivity and lifetime help? | **No**: the 9-point grid spans < 1 % RMSE; tuned and default skill agree within noise |
| Does the same hold for PM₂.₅? (Delhi) | **No detectable skill** (+1.9 % [−1.4, +5.8], +0.1 % [−2.0, +2.5]) |
| Does it improve the **deployed** map (satellite surface map) at +60/+120 min? | **No. Drifting the map makes it worse** (−14 % to −16 % at 60 min, intervals below zero), for both the improved and the production legacy solver |
| Is any absolute forecast accuracy demonstrated? | **No** |

Details, protocol and limits follow. Skill = 1 − RMSE / RMSE(`no_transport`); intervals are 95 % day-block bootstrap (1000 draws). Every sample size and day count is shown.

## 2. Protocol A: station-seeded (isolates transport)

For each origin hour and station fold: interpolate the readings of stations **outside** the fold (inverse-distance weighting) into a starting field, run the solver with hourly ERA5-Land wind, and compare the forecast at the held-out stations with their readings at +60 and +120 min. Held-out readings never enter the starting field (`test_held_out_station_reading_cannot_reach_the_forecast`).

Predictors: `physics_*` (solver variants); `no_transport` (the same field not moved: the fair baseline, same information); `own_persistence` (the station's own last reading: a reference ceiling that uses information the physics lacks).

Data: OpenAQ v3 hourly (readings ≤ 0 or above a pollutant cap discarded), ERA5-Land hourly 10 m wind (one domain mean per hour), GEOS-CF boundary-layer height (used for deposition of non-NO₂ pollutants).

### 2.1 NO₂, Delhi (44 stations in the box), RMSE µg/m³

| Horizon | n / days | `no_transport` | `physics_bilinear` | skill [95 % CI] | `physics_cubic_free` skill | `own_persistence` |
|---|---|---|---|---|---|---|
| 60 min | 516 / 47 | 61.3 | 60.0 | +2.1 % [+0.2, +3.8] | −10.6 % [−13.6, −8.2] | 24.9 |
| 120 min | 569 / 46 | 65.1 | 62.3 | +4.3 % [+2.0, +6.0] | −16.3 % [−20.4, −12.1] | 36.6 |

`physics_cubic` ≈ `physics_bilinear`. `cubic_free` = no emissions, 4 h decay, no mass rescale: clearly harmful, so `emission_mode="none"` should not be used for forecasts. Skill by wind (60 / 120 min): calm −1.8 % / +2.5 %, 1.5–3 m/s +4.6 % / +5.8 %, > 3 m/s +0.7 % / +3.4 %.

### 2.2 NO₂, Mumbai: inconclusive

Only about 5 of 19 stations report continuously. 97 / 101 samples over 21 / 22 days. `physics_bilinear` skill +14.4 % [−3.2, +20.0] at 60 min, +5.2 % [+0.1, +7.7] at 120 min; variants reorder between horizons. No conclusion.

### 2.3 Calibration (Delhi NO₂): tune on November, report on December

Grid: diffusivity {10, 50, 200} m²/s × lifetime {2, 4, 12} h, cubic advection. Chosen on origins before 2021-12-01 only (n = 682): the nine configurations span RMSE 66.2–66.6 µg/m³, a 0.6 % spread; best = K 200, τ 12 h. Reported once on December (n = 195 / 208, 19 days):

| Horizon | chosen (K200, τ12) skill | defaults (K50, τ4) skill |
|---|---|---|
| 60 min | +1.5 % [−0.9, +3.5] | +1.4 % [−0.6, +3.2] |
| 120 min | +4.7 % [+1.1, +8.0] | +4.2 % [+1.1, +6.7] |

Tuning changes nothing measurable: these parameters are not what limits skill. A January–February test was not possible: Delhi coverage collapses after December (4 samples).

### 2.4 PM₂.₅, Delhi: `PhysicsConfig.for_pollutant("pm25")`

Conservative, cubic advection, dry deposition with GEOS-CF boundary-layer height. n = 1622 / 1709 over 70–71 days. `no_transport` RMSE 58.7 / 70.4; physics skill +1.9 % [−1.4, +5.8] / +0.1 % [−2.0, +2.5]; best wind class 1.5–3 m/s: +6.0 % / +2.0 %. `own_persistence` RMSE 39.2 / 60.9. With no emission term PM₂.₅ physics is a free plume (see plan §5); no benefit is demonstrated.

## 3. Protocol B: deployed starting field (satellite surface map)

Run `outputs/runs/53cd121b80c1f929`: Delhi city box, 2021-12-02 to 12-31, S5P OFFL, pretrained surface model, 30 daily maps. Each map is the starting field at the overpass hour (≈ 13:30 local, rounded to the UTC hour), advected with ERA5-Land wind, and scored at the 42 stations inside the grid against their +60 / +120 min readings. n = 80 / 70, 20–21 days, RMSE µg/m³:

| Predictor | 60 min RMSE | skill [CI] | 120 min RMSE | skill [CI] |
|---|---|---|---|---|
| `no_transport` (the map as delivered) | 25.5 (bias +18.6) | | 27.1 (bias +17.1) | |
| `physics_default` (improved, NO₂ defaults) | 29.6 | −16.3 % [−32.1, −3.1] | 30.7 | −13.3 % [−28.8, +2.1] |
| `physics_legacy` (production `dispersion.py`) | 29.1 | −14.3 % [−29.0, −0.3] | 30.4 | −12.2 % [−29.6, +4.3] |
| `physics_free` (no emissions) | 27.4 | −7.4 % [−28.2, +12.4] | 35.0 | −29.2 % [−56.9, −3.4] |
| `physics_steady` (source holds the map fixed) | 25.9 | −1.6 % [−3.3, 0.0] | 27.9 | −3.0 % [−6.0, +0.1] |
| `own_persistence` | 8.4 | | 13.4 | |

**Reading.** Drifting the delivered map downwind degrades it, with both solvers. The map's pattern is built from land-use features (roads, built-up area), so it is locked to emission sources, not to a plume. `physics_steady` tests this: with a source term chosen so the starting map is an exact fixed point under constant wind (`steady_source_forecast`), the forecast collapses to the unmoved map (−1.6 % / −3.0 %, intervals touching zero). So at 1–2 h this map gains nothing from transport. The map's own error, a +18 µg/m³ bias against hourly readings, dominates and transport cannot remove it.

**Limits specific to this protocol**
- The surface model was fitted on OpenAQ India 2019–2022, which includes these stations and this month: the map is in-sample at the stations, so absolute errors are optimistic.
- The model is named `no2_surface_24h`; its target is a daily level, not an overpass-hour value, so part of the bias is the diurnal cycle.
- n = 80 / 70 from one month; intervals are wide. At 60 min only the two drifting variants (`physics_default`, `physics_legacy`) have intervals that exclude zero.

## 4. What is and is not supported

Supported
- Transport with persistent sources beats not moving a **station-interpolated** field by a small, significant margin in Delhi (NO₂, 60–120 min).
- Decay without emissions is harmful; the solver's analytic behaviour (puff spread, loss terms, mass handling) is verified by tests.
- Neither calibration nor cubic vs bilinear advection changes forecast skill at this noise floor.

Not supported
- Any absolute accuracy claim: the best station-seeded forecast has RMSE ≈ 60 µg/m³; a station's own last reading (25–37) beats it.
- Any benefit from transporting the deployed satellite map at 1–2 h; the evidence points the other way.
- PM₂.₅, SO₂, CO, O₃ physics skill; Mumbai; seasons other than winter; 30 / 90 min horizons (hourly readings only).
- Spatially varying wind: wind is one domain-mean value per hour.

## 5. Status

| Item | Status |
|---|---|
| Solver implemented, per-pollutant, unit-tested (analytic puff, loss, config) | **Yes** |
| Hindcast harness, leakage-tested, reproducible by CLI | **Yes** |
| Station-seeded NO₂ skill over no-transport (Delhi) | **Yes, small** |
| Deployed satellite-map forecast improved by physics | **No (degraded)** |
| Absolute forecast accuracy shown | **No** |
| Validated for other pollutants | **No** |
| Ready to ship as a validated forecast | **No** |

## 6. Recommendations

1. **Do not make the improved solver the default** for `forecast_point` (it is opt-in via `solver="improved"`). The production legacy solver is also worse than leaving the map unmoved over 1–2 h in this sample; treat short-horizon `/trends` values as unvalidated, or serve the map unchanged for horizons of a few hours.
2. Retrain or retarget the surface model on overpass-hour values (the plan already in `backend/data/README.md`); the current 24 h target caps what any hourly forecast can show.
3. If a transport forecast is wanted, advect **anomalies** against an emission-locked baseline (the `steady_source_forecast` form) instead of the whole field.
4. Re-run Protocol B on held-out stations (retrain the surface model without them) and on more than one month; add spatially varying wind.
5. Do not use the annual `/ensemble/{pollutant}` endpoint for forecasts: its "physics" value is supplied by the caller and its annual LightGBM members were trained on the corrupted NAMP CSV.
