"""Hourly observations for physics hindcasts: OpenAQ station NO₂ and ERA5-Land / GEOS-CF meteorology.

Everything is returned in **UTC**. OpenAQ hours come back as local start-of-hour time; ``tz_offset_h``
converts them (India: 5.5, no DST). Downloads are cached under ``cache_dir`` and resumable.

    uv run python -m ml_engine.forecasting.hindcast_data --start 2021-11-01 --end 2022-02-28
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MUMBAI_BBOX = (72.77, 18.88, 73.12, 19.32)  # west, south, east, north (matches the stored pipeline runs)
DELHI_BBOX = (76.84, 28.40, 77.35, 28.88)
CITY_BBOX = {"mumbai": MUMBAI_BBOX, "delhi": DELHI_BBOX}
SENSOR_LIST = Path(__file__).resolve().parents[2] / "data" / "openaq_india_no2_sensors.csv"
ERA5_HOURLY = "ECMWF/ERA5_LAND/HOURLY"


def select_sensors(bbox: tuple[float, float, float, float], start: str, end: str, sensor_csv: Path = SENSOR_LIST) -> pd.DataFrame:
    """NO₂ sensors inside ``bbox`` whose reporting period overlaps [start, end]."""
    west, south, east, north = bbox
    s = pd.read_csv(sensor_csv)
    s = s[s.lon.between(west, east) & s.lat.between(south, north)]
    s = s[s["first"].notna() & (s["first"] <= end) & (s["last"] >= start)]
    return s.reset_index(drop=True)


def fetch_station_hours(bbox, start: str, end: str, cache_dir: str | Path = "cache/hindcast", tz_offset_h: float = 5.5) -> pd.DataFrame:
    """Hourly NO₂ (µg/m³) for every sensor in ``bbox``: station_id, lat, lon, time_utc, no2."""
    from ..ingestion.openaq import OpenAQClient, fetch_sensor_hours

    cache = Path(cache_dir) / "openaq"
    cache.mkdir(parents=True, exist_ok=True)
    sensors = select_sensors(bbox, start, end)
    client = OpenAQClient()
    frames = []
    for i, s in enumerate(sensors.itertuples(), 1):
        path = cache / f"sensor_{s.sensor_id}_{start}_{end}.csv.gz"
        if path.exists():
            hours = pd.read_csv(path)
        else:
            hours = fetch_sensor_hours(client, int(s.sensor_id), start, end)
            hours.to_csv(path, index=False)
        log.info("[%d/%d] %-45s %6d hourly values", i, len(sensors), s.name[:45], len(hours))
        if len(hours):
            frames.append(hours.assign(station_id=int(s.sensor_id), lat=s.lat, lon=s.lon))
    out = pd.concat(frames, ignore_index=True)
    out["time_utc"] = pd.to_datetime(out["date"]) - pd.Timedelta(hours=tz_offset_h)
    return out[["station_id", "lat", "lon", "time_utc", "no2"]].sort_values(["station_id", "time_utc"]).reset_index(drop=True)


def fetch_hourly_met(bbox, start: str, end: str, cache_dir: str | Path = "cache/hindcast") -> pd.DataFrame:
    """Domain-mean hourly 10 m wind (m/s) from ERA5-Land and boundary-layer height (m) from GEOS-CF.

    Columns: time_utc, u10, v10, blh. The wind is one domain mean per hour; it does not resolve sea-breeze
    gradients inside the box. ``blh`` is NaN for hours GEOS-CF does not cover.
    """
    from ..ingestion import gee

    path = Path(cache_dir) / f"met_{start}_{end}.csv.gz"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time_utc"])
    path.parent.mkdir(parents=True, exist_ok=True)
    gee.initialize()
    import ee

    region = ee.Geometry.Rectangle(list(bbox))

    def monthly(coll: str, bands: list[str], scale: int) -> pd.DataFrame:
        frames = []
        edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), freq="10D")
        edges = list(edges) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
        for a, b in zip(edges[:-1], edges[1:]):
            if a >= b:
                continue
            col = ee.ImageCollection(coll).filterDate(a.isoformat(), b.isoformat()).select(bands)
            fc = col.map(lambda im: ee.Feature(None, im.reduceRegion(ee.Reducer.mean(), region, scale).set("t", im.date().millis())))
            feats = fc.getInfo()["features"]
            frames.append(pd.DataFrame([{"t": f["properties"].get("t"), **{k: f["properties"].get(k) for k in bands}} for f in feats]))
            log.info("%s %s..%s: %d hours", coll.split("/")[-1], a.date(), b.date(), len(feats))
        df = pd.concat(frames, ignore_index=True).dropna(subset=["t"])
        df["time_utc"] = pd.to_datetime(df["t"], unit="ms")
        return df.drop(columns="t")

    wind = monthly(ERA5_HOURLY, ["u_component_of_wind_10m", "v_component_of_wind_10m"], 9000).rename(
        columns={"u_component_of_wind_10m": "u10", "v_component_of_wind_10m": "v10"})
    try:
        blh = monthly(gee.GEOSCF_COLLECTION, [gee.GEOSCF_BLH_BAND], 25000).rename(columns={gee.GEOSCF_BLH_BAND: "blh"})
        # GEOS-CF tavg1hr stamps are the averaging-window start (half-hour centred); align to the hour.
        blh["time_utc"] = blh["time_utc"].dt.floor("h")
        met = wind.merge(blh.groupby("time_utc", as_index=False)["blh"].mean(), on="time_utc", how="left")
    except Exception as exc:  # GEOS-CF is optional: the solver falls back to its default mixing height
        log.warning("GEOS-CF boundary-layer height unavailable (%s); blh left empty", exc)
        met = wind.assign(blh=float("nan"))
    met = met.sort_values("time_utc").reset_index(drop=True)
    met.to_csv(path, index=False)
    return met


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download hourly stations + meteorology for a physics hindcast")
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2022-02-28")
    p.add_argument("--city", default="mumbai", choices=sorted(CITY_BBOX))
    p.add_argument("--cache-dir", default=None, help="default: cache/hindcast/<city>")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    bbox, cache = CITY_BBOX[args.city], args.cache_dir or f"cache/hindcast/{args.city}"
    met = fetch_hourly_met(bbox, args.start, args.end, cache)
    log.info("met: %d hours, blh available for %d", len(met), met.blh.notna().sum())
    st = fetch_station_hours(bbox, args.start, args.end, cache)
    log.info("stations: %d hourly rows, %d stations", len(st), st.station_id.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
