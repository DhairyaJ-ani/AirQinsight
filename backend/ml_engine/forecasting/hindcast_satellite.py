"""Hindcast the physics solver from the **deployed** starting field: the satellite-derived surface map.

``hindcast.py`` seeds the solver from interpolated stations, which isolates transport. This module starts from
what production starts from - ``no2_surface_fine.nc`` of a pipeline run - at the satellite overpass hour, and
scores +60 / +120 min against hourly station readings. It answers a different question: does the solver improve
on the map as delivered?

Read the numbers with these limits in mind (they are repeated in the report):

* The pretrained surface model was fitted on OpenAQ India 2019-2022, which includes these stations and days.
  The map at a station pixel is therefore *in-sample*; absolute errors are optimistic. The comparison between
  ``physics_*`` and ``no_transport`` uses the same map and stays meaningful.
* The model is named ``no2_surface_24h``: its target is a daily level, not an overpass-hour value, so part of
  the error against hourly readings is the diurnal cycle, which no transport scheme can remove.

    uv run python -m ml_engine.forecasting.hindcast_satellite --run-dir outputs/runs/<id> --city delhi
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import DispersionConfig
from ..dispersion import AdvectionDiffusionSolver
from ..grid import GridSpec
from .config import PhysicsConfig
from .hindcast import HORIZONS_MIN, _met_at, summarize, by_wind_speed
from .physics import ImprovedPhysicsSolver

log = logging.getLogger(__name__)


def overpass_utc_hour(lon: float) -> float:
    """Sentinel-5P crosses at ~13:30 local solar time."""
    return (13.5 - lon / 15.0) % 24


def steady_source_forecast(solver: ImprovedPhysicsSolver, c0: np.ndarray, wind_at, horizons_min=HORIZONS_MIN) -> dict[int, np.ndarray]:
    """Forecast with a source that makes ``c0`` an exact steady state under the initial wind.

    The source is whatever one solver step removes from ``c0`` (advective export, diffusion and loss), re-added
    every step. With a constant wind the field then stays at ``c0``; it changes only where the wind changes.
    This models a map whose pattern is locked to its emission sources instead of drifting with the wind.
    ``wind_at(minutes)`` returns the (u, v) arrays at that many minutes after the origin.
    """
    import math

    dt = solver.cfg.dt_s
    decay = math.exp(-dt / (solver.cfg.lifetime_h * 3600.0))
    u0, v0 = wind_at(0.0)
    source = c0 - solver._step(c0, u0, v0, dt, 0.0, decay)
    c, t, out = c0.copy(), 0.0, {}
    while t < max(horizons_min) * 60 - 1e-6:
        u, v = wind_at(t / 60.0)
        c = solver._step(c, u, v, dt, source, decay)
        t += dt
        if abs(t / 60.0 - round(t / 60.0)) < 1e-6 and int(round(t / 60.0)) in horizons_min:
            out[int(round(t / 60.0))] = c.copy()
    return out


def run_satellite_hindcast(run_dir: str | Path, stations: pd.DataFrame, met: pd.DataFrame, *,
                           value_col: str = "no2", max_value: float = 500.0) -> pd.DataFrame:
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    fine = GridSpec(**report["grids"]["fine"])
    surface = xr.load_dataset(run_dir / "no2_surface_fine.nc", engine="h5netcdf")["no2_surface"]
    dx, dy = fine.pixel_size_m()
    kw = dict(max_horizon_min=max(HORIZONS_MIN))
    variants = {
        "physics_default": PhysicsConfig(**kw),
        "physics_free": PhysicsConfig(emission_mode="none", mass_policy="none", advection_order=3, **kw),
    }
    solvers = {n: ImprovedPhysicsSolver(fine.height, fine.width, dx, dy, c) for n, c in variants.items()}
    legacy = AdvectionDiffusionSolver(fine, DispersionConfig(horizons_h=tuple(m / 60.0 for m in HORIZONS_MIN)))

    st = stations.copy()
    st["t_end"] = pd.to_datetime(st["time_utc"]) + pd.Timedelta(hours=1)
    st = st[(st[value_col] > 0) & (st[value_col] <= max_value)]
    obs = st.set_index(["station_id", "t_end"])[value_col]
    obs = obs[~obs.index.duplicated(keep="first")]
    pos = st.drop_duplicates("station_id").set_index("station_id")[["lon", "lat"]]
    row, col, inside = fine.index_of(pos["lon"].values, pos["lat"].values)
    pix = {sid: (int(r), int(c)) for sid, r, c, ok in zip(pos.index, row, col, inside) if ok}
    met = met.assign(time_utc=pd.to_datetime(met["time_utc"])).dropna(subset=["u10", "v10"]).set_index("time_utc").sort_index()
    lon_mid = (fine.west + fine.east) / 2

    rows = []
    for day in surface["time"].values:
        d = pd.Timestamp(day).normalize()
        t0 = d + pd.Timedelta(hours=round(overpass_utc_hour(lon_mid)))
        if t0 < met.index[0] or t0 + pd.Timedelta(hours=2) > met.index[-1]:
            continue
        c0 = surface.sel(time=day).values.astype(np.float64)
        if not np.isfinite(c0).any():
            continue
        c0 = np.where(np.isfinite(c0), c0, np.nanmean(c0))
        wind = [tuple(np.full(c0.shape, w) for w in _met_at(met, t0 + pd.Timedelta(hours=k))) for k in range(3)]
        frames = {n: dict(s.forecast_frames(c0, wind)) for n, s in solvers.items()}
        # the production default: one constant wind (the first hour's), as service.forecast_point does
        leg = legacy.forecast(c0, wind[0][0], wind[0][1])
        frames["physics_steady"] = steady_source_forecast(
            solvers["physics_default"], c0, lambda m: tuple(np.full(c0.shape, w) for w in _met_at(met, t0 + pd.Timedelta(minutes=m))))
        frames["physics_legacy"] = {m: leg.sel(horizon_h=m / 60.0).values for m in HORIZONS_MIN}
        speed = float(np.hypot(*_met_at(met, t0)))
        for sid, (r, c) in pix.items():
            at0 = obs.get((sid, t0))
            for hmin in HORIZONS_MIN:
                truth = obs.get((sid, t0 + pd.Timedelta(minutes=hmin)))
                if truth is None or at0 is None or np.isnan(truth) or np.isnan(at0):
                    continue
                rows.append({"t0": t0, "station_id": sid, "horizon_min": hmin, "obs": float(truth), "own_persistence": float(at0),
                             "no_transport": float(c0[r, c]), "wind_speed": speed,
                             **{n: float(frames[n][hmin][r, c]) for n in frames}})
    df = pd.DataFrame(rows)
    log.info("satellite hindcast: %d days, %d stations in grid, %d samples", surface.sizes["time"], len(pix), len(df))
    return df


def main(argv=None) -> int:
    from .hindcast_data import CITY_BBOX, fetch_hourly_met, fetch_station_hours

    p = argparse.ArgumentParser(description="Hindcast the physics solver from a pipeline run's satellite surface maps")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--city", default="delhi", choices=sorted(CITY_BBOX))
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2022-02-28")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    bbox, cache = CITY_BBOX[args.city], f"cache/hindcast/{args.city}"
    st = fetch_station_hours(bbox, args.start, args.end, cache)
    met = fetch_hourly_met(bbox, args.start, args.end, cache)
    df = run_satellite_hindcast(args.run_dir, st, met)
    df.to_csv(Path(cache) / "hindcast_satellite_samples.csv", index=False)
    summ = summarize(df)
    summ.to_csv(Path(cache) / "hindcast_satellite_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
        print(by_wind_speed(df, "physics_default").to_string(index=False))
        print(by_wind_speed(df, "physics_legacy").to_string(index=False))
        print(by_wind_speed(df, "physics_steady").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
