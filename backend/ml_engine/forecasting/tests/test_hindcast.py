"""The hindcast harness must credit real transport and must not leak the held-out station's reading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.forecasting.hindcast import Variant, build_grid, idw_field, run_hindcast, summarize
from ml_engine.forecasting.config import PhysicsConfig

BBOX = (72.70, 18.90, 73.00, 19.25)
U = 1.5  # m/s eastward -> 5.4 km per hour


def _truth(lon, lat, hour):
    """A smooth plume drifting east at U on a 20 ug/m3 background."""
    x = (lon - BBOX[0]) * 111_320.0 * np.cos(np.radians(19.07))
    y = (lat - BBOX[1]) * 110_574.0
    return 20.0 + 60.0 * np.exp(-((x - 5000.0 - U * 3600.0 * hour) ** 2 + (y - 19000.0) ** 2) / (2 * 6000.0 ** 2))


@pytest.fixture
def synthetic():
    lons = np.linspace(BBOX[0] + 0.01, BBOX[2] - 0.01, 5)
    lats = np.linspace(BBOX[1] + 0.01, BBOX[3] - 0.01, 4)
    pts = [(lo, la) for lo in lons for la in lats]
    hours = pd.date_range("2022-01-01 00:00", periods=8, freq="h")   # short: the plume must stay inside the box
    rows = []
    for sid, (lo, la) in enumerate(pts):
        for k, t in enumerate(hours):
            rows.append((sid, la, lo, t, float(_truth(lo, la, k + 1))))   # value for the hour ending at t+1h
    st = pd.DataFrame(rows, columns=["station_id", "lat", "lon", "time_utc", "no2"])
    met = pd.DataFrame({"time_utc": pd.date_range("2022-01-01 00:00", periods=20, freq="h"), "u10": U, "v10": 0.0, "blh": 800.0})
    return st, met


def _variants():
    kw = dict(max_horizon_min=120, enable_terrain_effect=False, diffusivity_m2_s=20.0,
              emission_mode="none", mass_policy="none", lifetime_h=float("inf"))
    return [Variant("physics_cubic_free", PhysicsConfig(advection_order=3, **kw))]


def _run(st, met):
    return run_hindcast(st, met, BBOX, res_m=1500.0, n_folds=4, variants=_variants(),
                        t_hours_local=tuple(range(24)), tz_offset_h=0.0, min_train_stations=6)


def test_transport_beats_no_transport_when_the_plume_really_moves(synthetic):
    df = _run(*synthetic)
    assert len(df) > 100
    s = summarize(df, ["physics_cubic_free", "no_transport"], n_boot=200).set_index(["horizon_min", "predictor"])
    for h in (60, 120):
        assert s.loc[(h, "physics_cubic_free"), "rmse"] < s.loc[(h, "no_transport"), "rmse"]
        assert s.loc[(h, "physics_cubic_free"), "skill_vs_no_transport"] > 0.01
    # The longer the lead, the more the plume has moved, so transport should matter more.
    assert s.loc[(120, "physics_cubic_free"), "skill_vs_no_transport"] > s.loc[(60, "physics_cubic_free"), "skill_vs_no_transport"]


def test_held_out_station_reading_cannot_reach_the_forecast(synthetic):
    st, met = synthetic
    base = _run(st, met)
    # Wreck one station's reading at the origin hours; it is held out in exactly one fold per origin.
    victim = 0
    t0s = base["t0"].unique()
    bad = st.copy()
    mask = (bad.station_id == victim) & ((bad.time_utc + pd.Timedelta(hours=1)).isin(t0s))
    bad.loc[mask, "no2"] += 100.0   # stays inside the harness QC range (0-500)
    altered = _run(bad, met)

    key = ["t0", "station_id", "horizon_min"]
    a = base.merge(altered, on=key, suffixes=("_a", "_b"))
    own = a[a.station_id == victim]
    assert len(own) > 0
    # Its own persistence changes (it is the reading), but the physics forecast *at that station* does not.
    assert (own["own_persistence_a"] != own["own_persistence_b"]).all()
    np.testing.assert_allclose(own["physics_cubic_free_a"], own["physics_cubic_free_b"], rtol=0, atol=1e-9)


def test_idw_reproduces_a_constant_field_and_stays_in_range():
    grid = build_grid(BBOX, 1000.0)
    lon = np.array([72.82, 72.9, 72.93])
    lat = np.array([19.02, 19.1, 19.13])
    np.testing.assert_allclose(idw_field(grid, lon, lat, np.array([30.0, 30.0, 30.0])), 30.0, rtol=1e-9)
    f = idw_field(grid, lon, lat, np.array([10.0, 50.0, 90.0]))
    assert 10.0 <= f.min() and f.max() <= 90.0
