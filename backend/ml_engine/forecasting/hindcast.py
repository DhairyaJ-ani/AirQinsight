"""Leave-stations-out hindcast of the physics solver against hourly station observations.

Question answered: *does transporting a concentration field with the wind predict the next 1-2 hours at a
station better than leaving the field where it is?*

Protocol
--------
For each hour ``T`` and each fold of stations:

1. Build the initial field on a regular grid by interpolating the readings of the stations **outside** the
   fold (inverse-distance weighting). The held-out stations contribute nothing to it.
2. Run the physics solver from that field with the hourly wind for ``T, T+1h, T+2h``.
3. Compare the forecast at each held-out station with its reading at ``T+60`` and ``T+120`` min.

Three predictors are scored on identical samples:

* ``physics_*``   - the solver (one column per configuration);
* ``no_transport`` - the same interpolated field without moving it. This is the fair "do nothing" baseline:
  it has exactly the information the physics had, so any gain is due to transport;
* ``own_persistence`` - the held-out station's own reading at ``T``. It uses information the physics does
  not have, so it is a reference ceiling for "just repeat the last value", not a fair opponent.

Hourly readings are averages over the hour *ending* at the timestamp used here (OpenAQ gives the start).
Wind is one domain-mean value per hour, so sea-breeze gradients inside the box are not represented.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PhysicsConfig
from .physics import ImprovedPhysicsSolver

log = logging.getLogger(__name__)

HORIZONS_MIN = (60, 120)
IDW_NUGGET_M = 1500.0  # softens bullseyes around single stations


@dataclass
class Variant:
    name: str
    cfg: PhysicsConfig


def default_variants() -> list[Variant]:
    kw = dict(max_horizon_min=max(HORIZONS_MIN))
    return [
        Variant("physics_bilinear", PhysicsConfig(**kw)),
        Variant("physics_cubic", PhysicsConfig(advection_order=3, **kw)),
        Variant("physics_cubic_free", PhysicsConfig(advection_order=3, emission_mode="none", mass_policy="none", **kw)),
    ]


@dataclass
class Grid:
    h: int
    w: int
    dx: float
    dy: float
    west: float
    north: float
    dlon: float
    dlat: float

    def index(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.clip(np.round((np.asarray(lon) - self.west) / self.dlon - 0.5).astype(int), 0, self.w - 1)
        row = np.clip(np.round((self.north - np.asarray(lat)) / self.dlat - 0.5).astype(int), 0, self.h - 1)
        return row, col


def build_grid(bbox: tuple[float, float, float, float], res_m: float) -> Grid:
    west, south, east, north = bbox
    mid = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320.0 * math.cos(mid)
    height_m = (north - south) * 110_574.0
    w, h = max(2, round(width_m / res_m)), max(2, round(height_m / res_m))
    return Grid(h, w, width_m / w, height_m / h, west, north, (east - west) / w, (north - south) / h)


def idw_field(grid: Grid, lon: np.ndarray, lat: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Inverse-distance-squared interpolation of point readings onto the grid (with a distance nugget)."""
    cols = (np.arange(grid.w) + 0.5) * grid.dx
    rows = (np.arange(grid.h) + 0.5) * grid.dy
    mid = math.radians(grid.north - (grid.h * grid.dlat) / 2)
    sx = (np.asarray(lon) - grid.west) * 111_320.0 * math.cos(mid)
    sy = (grid.north - np.asarray(lat)) * 110_574.0
    d2 = (cols[None, None, :] - sx[:, None, None]) ** 2 + (rows[None, :, None] - sy[:, None, None]) ** 2
    wgt = 1.0 / (d2 + IDW_NUGGET_M ** 2)
    return (wgt * np.asarray(values)[:, None, None]).sum(0) / wgt.sum(0)


def _met_at(met: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float]:
    """Domain-mean wind at an arbitrary time, linearly interpolated between the hourly values.

    Stations stamp their hours on :00 or :30 UTC depending on the network, so exact matches are rare.
    """
    x = met.index.values.astype("datetime64[s]").astype(np.int64)
    tt = np.datetime64(t, "s").astype(np.int64)
    return float(np.interp(tt, x, met["u10"].values)), float(np.interp(tt, x, met["v10"].values))


def _wind_sequence(met: pd.DataFrame, t0: pd.Timestamp, shape: tuple[int, int], n_hours: int = 3):
    seq = []
    for k in range(n_hours):
        u, v = _met_at(met, t0 + pd.Timedelta(hours=k))
        seq.append((np.full(shape, u, dtype=np.float64), np.full(shape, v, dtype=np.float64)))
    return seq


def run_hindcast(
    stations: pd.DataFrame,
    met: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    *,
    res_m: float = 500.0,
    n_folds: int = 4,
    variants: list[Variant] | None = None,
    t_hours_local: tuple[int, ...] = (8, 10, 12, 14, 16, 18),
    tz_offset_h: float = 5.5,
    min_train_stations: int = 5,
    max_origins: int | None = None,
) -> pd.DataFrame:
    """Return one row per (origin hour, held-out station, horizon) with every predictor's value."""
    variants = variants or default_variants()
    grid = build_grid(bbox, res_m)
    solvers = {v.name: ImprovedPhysicsSolver(grid.h, grid.w, grid.dx, grid.dy, v.cfg) for v in variants}

    st = stations.copy()
    st["t_end"] = pd.to_datetime(st["time_utc"]) + pd.Timedelta(hours=1)
    st = st[(st.no2 > 0) & (st.no2 <= 500)]   # exact zeros are missing-value codes, not clean air
    obs = st.set_index(["station_id", "t_end"])["no2"]
    obs = obs[~obs.index.duplicated(keep="first")]
    ids = sorted(st.station_id.unique())
    fold_of = {sid: i % n_folds for i, sid in enumerate(ids)}
    pos = st.drop_duplicates("station_id").set_index("station_id")[["lon", "lat"]]
    met = met.assign(time_utc=pd.to_datetime(met["time_utc"])).dropna(subset=["u10", "v10"]).set_index("time_utc").sort_index()

    origins = sorted(st.t_end.unique())
    origins = [pd.Timestamp(t) for t in origins if ((pd.Timestamp(t) + pd.Timedelta(hours=tz_offset_h)).hour in t_hours_local)]
    if max_origins:
        origins = origins[:: max(1, len(origins) // max_origins)]

    rows = []
    for t0 in origins:
        if t0 < met.index[0] or t0 + pd.Timedelta(hours=2) > met.index[-1]:
            continue
        at0 = {sid: obs.get((sid, t0)) for sid in ids}
        have0 = [sid for sid, v in at0.items() if v is not None and not np.isnan(v)]
        for fold in range(n_folds):
            held = [s for s in have0 if fold_of[s] == fold]
            train = [s for s in have0 if fold_of[s] != fold]
            if not held or len(train) < min_train_stations:
                continue
            c0 = idw_field(grid, pos.loc[train, "lon"].values, pos.loc[train, "lat"].values, np.array([at0[s] for s in train]))
            wind = _wind_sequence(met, t0, (grid.h, grid.w))
            frames = {name: dict(sv.forecast_frames(c0, wind)) for name, sv in solvers.items()}
            r_idx, c_idx = grid.index(pos.loc[held, "lon"].values, pos.loc[held, "lat"].values)
            speed = float(np.hypot(*_met_at(met, t0)))
            for hmin in HORIZONS_MIN:
                t1 = t0 + pd.Timedelta(minutes=hmin)
                for k, sid in enumerate(held):
                    truth = obs.get((sid, t1))
                    if truth is None or np.isnan(truth):
                        continue
                    row = {"t0": t0, "station_id": sid, "horizon_min": hmin, "obs": float(truth), "own_persistence": float(at0[sid]),
                           "no_transport": float(c0[r_idx[k], c_idx[k]]), "wind_speed": speed}
                    for name in frames:
                        row[name] = float(frames[name][hmin][r_idx[k], c_idx[k]])
                    rows.append(row)
    df = pd.DataFrame(rows)
    log.info("hindcast: %d origins, %d samples", len(origins), len(df))
    return df


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - obs
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "bias": float(err.mean()),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}


def summarize(df: pd.DataFrame, predictors: list[str] | None = None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per horizon and predictor: n, RMSE, MAE, bias, R², and RMSE skill against ``no_transport``.

    ``skill`` = 1 - RMSE/RMSE(no_transport); a 95 % interval comes from resampling whole days, since
    hours within a day are not independent.
    """
    predictors = predictors or [c for c in df.columns if c.startswith("physics_")] + ["no_transport", "own_persistence"]
    rng = np.random.default_rng(seed)
    out = []
    for hmin, g in df.groupby("horizon_min"):
        days = g["t0"].dt.date.values
        udays = np.unique(days)
        idx_by_day = {d: np.where(days == d)[0] for d in udays}
        base = (g["no_transport"].values - g["obs"].values) ** 2
        for p in predictors:
            m = _metrics(g["obs"].values, g[p].values)
            sq = (g[p].values - g["obs"].values) ** 2
            skill, lo, hi = float("nan"), float("nan"), float("nan")
            if p != "no_transport":
                skill = 1 - math.sqrt(sq.mean()) / math.sqrt(base.mean())
                boots = []
                for _ in range(n_boot):
                    pick = np.concatenate([idx_by_day[d] for d in rng.choice(udays, len(udays))])
                    boots.append(1 - math.sqrt(sq[pick].mean()) / math.sqrt(base[pick].mean()))
                lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"horizon_min": hmin, "predictor": p, "n": len(g), "days": len(udays), **m,
                        "skill_vs_no_transport": skill, "skill_ci_lo": lo, "skill_ci_hi": hi})
    return pd.DataFrame(out)


def by_wind_speed(df: pd.DataFrame, predictor: str, bins=(0, 1.5, 3.0, 99)) -> pd.DataFrame:
    """RMSE skill of ``predictor`` against ``no_transport`` within wind-speed classes (m/s)."""
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    cls = pd.cut(df.wind_speed, list(bins), right=False, labels=labels)
    rows = []
    for (hmin, c), g in df.groupby(["horizon_min", cls], observed=True):
        a = math.sqrt(np.mean((g[predictor] - g["obs"]) ** 2))
        b = math.sqrt(np.mean((g["no_transport"] - g["obs"]) ** 2))
        rows.append({"horizon_min": hmin, "wind_m_s": c, "n": len(g), "rmse_physics": a, "rmse_no_transport": b, "skill": 1 - a / b})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from .hindcast_data import CITY_BBOX, fetch_hourly_met, fetch_station_hours

    p = argparse.ArgumentParser(description="Leave-stations-out physics hindcast against hourly OpenAQ NO2")
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2022-02-28")
    p.add_argument("--res-m", type=float, default=500.0)
    p.add_argument("--city", default="mumbai", choices=sorted(CITY_BBOX))
    p.add_argument("--cache-dir", default=None, help="default: cache/hindcast/<city>")
    p.add_argument("--max-origins", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    bbox, cache = CITY_BBOX[args.city], args.cache_dir or f"cache/hindcast/{args.city}"
    st = fetch_station_hours(bbox, args.start, args.end, cache)
    met = fetch_hourly_met(bbox, args.start, args.end, cache)
    df = run_hindcast(st, met, bbox, res_m=args.res_m, max_origins=args.max_origins)
    out = Path(cache)
    df.to_csv(out / "hindcast_samples.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "hindcast_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
        print(by_wind_speed(df, "physics_cubic").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
