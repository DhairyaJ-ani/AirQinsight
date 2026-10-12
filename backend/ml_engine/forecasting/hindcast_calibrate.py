"""Calibrate solver parameters on a validation period and report once on a held-out test period.

The grid is searched on origins **before** ``--split`` only. The chosen configuration, the defaults and the
``no_transport`` baseline are then scored on origins **from** ``--split`` on, which no choice has seen.

    uv run python -m ml_engine.forecasting.hindcast_calibrate --city delhi --split 2022-01-01
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd

from .config import PhysicsConfig
from .hindcast import HORIZONS_MIN, Variant, run_hindcast, summarize
from .hindcast_data import CITY_BBOX, fetch_hourly_met, fetch_station_hours

log = logging.getLogger(__name__)

DIFFUSIVITY_GRID = (10.0, 50.0, 200.0)   # m2/s
LIFETIME_GRID = (2.0, 4.0, 12.0)         # h


def grid_variants() -> list[Variant]:
    kw = dict(max_horizon_min=max(HORIZONS_MIN), advection_order=3)
    return [Variant(f"K{k:g}_tau{t:g}", PhysicsConfig(diffusivity_m2_s=k, lifetime_h=t, **kw))
            for k, t in itertools.product(DIFFUSIVITY_GRID, LIFETIME_GRID)]


def default_variant() -> Variant:
    return Variant("defaults", PhysicsConfig(max_horizon_min=max(HORIZONS_MIN)))


def pick_best(df: pd.DataFrame, names: list[str]) -> tuple[str, pd.DataFrame]:
    """Lowest RMSE averaged over both horizons, on the validation samples only."""
    rows = []
    for n in names:
        per_h = df.groupby("horizon_min").apply(lambda g: ((g[n] - g["obs"]) ** 2).mean() ** 0.5, include_groups=False)
        rows.append({"variant": n, "rmse_mean": float(per_h.mean()), **{f"rmse_{h}": float(per_h[h]) for h in per_h.index}})
    table = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    return table.loc[0, "variant"], table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Calibrate on a validation period, report on a held-out test period")
    p.add_argument("--city", default="delhi", choices=sorted(CITY_BBOX))
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2022-02-28")
    p.add_argument("--split", default="2022-01-01", help="origins before this date calibrate; from it on test")
    p.add_argument("--res-m", type=float, default=1000.0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    bbox, cache = CITY_BBOX[args.city], f"cache/hindcast/{args.city}"
    st = fetch_station_hours(bbox, args.start, args.end, cache)
    met = fetch_hourly_met(bbox, args.start, args.end, cache)
    variants = grid_variants() + [default_variant()]
    names = [v.name for v in variants]
    cut = pd.Timestamp(args.split)

    val = run_hindcast(st, met, bbox, res_m=args.res_m, variants=variants, t_max=cut)
    best, table = pick_best(val, [n for n in names if n != "defaults"])
    log.info("validation samples: %d; best = %s", len(val), best)

    keep = [best, "defaults"]
    test = run_hindcast(st, met, bbox, res_m=args.res_m, variants=[v for v in variants if v.name in keep], t_min=cut)
    summ = summarize(test, keep + ["no_transport", "own_persistence"])
    out = Path(cache)
    table.to_csv(out / "calibration_validation_grid.csv", index=False)
    summ.to_csv(out / "calibration_test_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(f"\nVALIDATION (origins < {args.split}), n={len(val)}: RMSE by configuration\n{table.to_string(index=False)}")
        print(f"\nTEST (origins >= {args.split}) with the chosen configuration '{best}'\n{summ.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
