"""OpenAQ v3 multi-pollutant downloader.

Extends the existing NO₂-only ``ml_engine.ingestion.openaq`` client to support
all pollutants tracked by AirQ Insight: PM₂.₅, PM₁₀, NO₂, SO₂, CO and O₃.

The existing ``openaq.py`` module is **not modified** – this module imports the
shared ``OpenAQClient`` from it and adds multi-pollutant support.

Key design choices
------------------
* Uses the same per-sensor cache scheme as the NO₂ downloader so runs are
  resumable and re-downloads of identical periods are avoided.
* Mumbai is used as the default city for rapid testing (guaranteed sensors).
* Sensor listing is filtered by city bounding box if ``city_bbox`` is given.
* A unified station CSV is produced per pollutant: identical schema to the NO₂
  CSV (``station_id, name, lat, lon, date, <pollutant>``) so it can be fed
  directly into existing pipelines.

Usage::

    uv run python -m ml_engine.ingestion.openaq_multipollutant \\
        --pollutant pm25 --start 2023-01-01 --end 2023-12-31 \\
        --city mumbai -o data/openaq_mumbai_pm25.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from .openaq import OpenAQClient, _load_key, _covering_cache, PAGE

log = logging.getLogger(__name__)

# ── OpenAQ v3 parameter IDs ────────────────────────────────────────────────
# These are integer IDs used in the ``/locations`` and ``/sensors`` endpoints.
PARAMETER_IDS: dict[str, int] = {
    "pm25": 2,
    "pm10": 1,
    "no2": 5,
    "so2": 9,
    "co": 4,
    "o3": 7,
}

# ── Mumbai bounding box (west, south, east, north) for city filtering ──────
CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "mumbai": (72.77, 18.88, 73.12, 19.32),
    "delhi": (76.80, 28.40, 77.40, 28.90),
    "pune": (73.70, 18.40, 74.00, 18.70),
    "bangalore": (77.45, 12.85, 77.75, 13.15),
    "bengaluru": (77.45, 12.85, 77.75, 13.15),
    "chennai": (80.15, 12.95, 80.35, 13.20),
    "kolkata": (88.25, 22.40, 88.55, 22.70),
    "hyderabad": (78.30, 17.30, 78.65, 17.55),
    "ahmedabad": (72.45, 22.95, 72.70, 23.15),
}

MIN_HOUR_COMPLETENESS = 75.0


def list_pollutant_sensors(
    client: OpenAQClient,
    pollutant: str,
    country_code: str = "IN",
    city_bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Return one row per sensor reporting ``pollutant`` in ``country_code``.

    If ``city_bbox`` is provided, only sensors with coordinates inside the
    bounding box (west, south, east, north) are returned.
    """
    param_id = _resolve_param_id(pollutant)

    countries = client.get("/countries", {"limit": 300})["results"]
    try:
        country_id = next(c["id"] for c in countries if c["code"] == country_code)
    except StopIteration:
        raise ValueError(f"Country code {country_code!r} not found in OpenAQ")

    rows, page = [], 1
    while True:
        res = client.get(
            "/locations",
            {
                "countries_id": country_id,
                "parameters_id": param_id,
                "limit": PAGE,
                "page": page,
            },
        )["results"]
        for loc in res:
            coord = loc.get("coordinates") or {}
            lat = coord.get("latitude")
            lon = coord.get("longitude")
            if lat is None or lon is None:
                continue
            if city_bbox:
                west, south, east, north = city_bbox
                if not (west <= lon <= east and south <= lat <= north):
                    continue
            for s in loc["sensors"]:
                if s["parameter"]["id"] == param_id:
                    rows.append(
                        {
                            "location_id": loc["id"],
                            "sensor_id": s["id"],
                            "name": loc["name"],
                            "lat": lat,
                            "lon": lon,
                            "provider": (loc.get("provider") or {}).get("name", ""),
                        }
                    )
        if len(res) < PAGE:
            break
        page += 1

    df = pd.DataFrame(rows)
    log.info("%d %s sensors in %s%s", len(df), pollutant.upper(), country_code,
             f" (bbox={city_bbox})" if city_bbox else "")
    return df


def fetch_sensor_hours_pollutant(
    client: OpenAQClient,
    sensor_id: int,
    pollutant: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Hourly values for one sensor, one pollutant, over [start, end]."""
    colname = pollutant.lower()
    try:
        meta = client.get(f"/sensors/{sensor_id}")["results"][0]
    except (KeyError, IndexError, Exception) as exc:
        log.warning("Sensor %d metadata failed: %s", sensor_id, exc)
        return pd.DataFrame(columns=["date", colname])

    first = (meta.get("datetimeFirst") or {}).get("utc", "")[:10]
    last_ = (meta.get("datetimeLast") or {}).get("utc", "")[:10]
    lo = max(start, first or start)
    hi = min(end, last_ or end)
    if not first or lo > hi:
        return pd.DataFrame(columns=["date", colname])

    frames = []
    for q_start in pd.date_range(pd.Timestamp(lo).to_period("Q").start_time, hi, freq="QS"):
        a = max(lo, q_start.strftime("%Y-%m-%d"))
        b = min(hi, (q_start + pd.offsets.QuarterEnd(0)).strftime("%Y-%m-%d"))
        try:
            frames.append(_fetch_hours_range_pollutant(client, sensor_id, pollutant, a, b))
        except Exception as exc:
            log.warning("Sensor %d %s: skipping %s..%s (%s)", sensor_id, pollutant, a, b, exc)

    if not frames:
        return pd.DataFrame(columns=["date", colname])
    return pd.concat(frames, ignore_index=True)


def _fetch_hours_range_pollutant(
    client: OpenAQClient,
    sensor_id: int,
    pollutant: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    colname = pollutant.lower()
    rows, page = [], 1
    while True:
        res = client.get(
            f"/sensors/{sensor_id}/hours",
            {
                "datetime_from": f"{start}T00:00:00Z",
                "datetime_to": f"{end}T23:59:59Z",
                "limit": PAGE,
                "page": page,
            },
        )["results"]
        for r in res:
            if r.get("value") is None:
                continue
            complete = (r.get("coverage") or {}).get("percentComplete", 100.0)
            if complete is not None and complete < MIN_HOUR_COMPLETENESS:
                continue
            dt = r["period"]["datetimeFrom"]["local"][:16].replace("T", " ")
            rows.append((dt, float(r["value"])))
        if len(res) < PAGE:
            break
        page += 1
    return pd.DataFrame(rows, columns=["date", colname])


def _resolve_param_id(pollutant: str) -> int:
    key = pollutant.lower()
    if key not in PARAMETER_IDS:
        raise ValueError(f"Unknown pollutant {pollutant!r}. Supported: {list(PARAMETER_IDS)}")
    return PARAMETER_IDS[key]


def download_pollutant(
    pollutant: str,
    start: str,
    end: str,
    out_csv: str | Path,
    cache_dir: str | Path = "cache/openaq_multipollutant",
    country_code: str = "IN",
    city: str | None = None,
    city_bbox: tuple[float, float, float, float] | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download hourly ``pollutant`` data from OpenAQ for ``country_code``.

    Parameters
    ----------
    pollutant : str
        One of "pm25", "pm10", "no2", "so2", "co", "o3".
    start, end : str
        Date strings "YYYY-MM-DD".
    out_csv : str | Path
        Destination CSV file.
    cache_dir : str | Path
        Per-sensor cache directory.
    country_code : str
        ISO 3166-1 alpha-2 code (default "IN" for India).
    city : str | None
        City name for ``CITY_BBOXES`` lookup (e.g. ``"mumbai"``).
    city_bbox : tuple | None
        Explicit (west, south, east, north) bounding box.
    api_key : str | None
        Override ``OPENAQ_API_KEY``.
    """
    colname = pollutant.lower()
    client = OpenAQClient(api_key or _load_key())
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Resolve bounding box
    bbox = city_bbox
    if bbox is None and city:
        bbox = CITY_BBOXES.get(city.lower())
        if bbox is None:
            log.warning("City %r not in CITY_BBOXES; downloading all of %s", city, country_code)

    # Sensor discovery (cached per pollutant + bbox combination)
    bbox_tag = f"_{city.lower()}" if city else ""
    sensors_path = cache / f"sensors_{country_code}_{colname}{bbox_tag}.csv"
    if sensors_path.exists():
        sensors = pd.read_csv(sensors_path)
    else:
        sensors = list_pollutant_sensors(client, pollutant, country_code, bbox)
        sensors.to_csv(sensors_path, index=False)

    log.info("%d %s sensors to fetch", len(sensors), pollutant.upper())
    if sensors.empty:
        log.warning("No sensors found; returning empty DataFrame")
        return pd.DataFrame(columns=["station_id", "name", "lat", "lon", "date", colname])

    frames = []
    for i, s in enumerate(sensors.itertuples(), 1):
        path = cache / f"sensor_{s.sensor_id}_{colname}_{start}_{end}.csv.gz"
        wider = _covering_cache_pollutant(cache, int(s.sensor_id), colname, start, end)
        if path.exists():
            hours = pd.read_csv(path)
        elif wider is not None:
            hours = pd.read_csv(wider)
            hours = hours[(hours["date"] >= start) & (hours["date"] <= f"{end} 23:59")]
            hours.to_csv(path, index=False, compression="gzip")
        else:
            hours = fetch_sensor_hours_pollutant(client, int(s.sensor_id), pollutant, start, end)
            hours.to_csv(path, index=False, compression="gzip")

        log.info("[%d/%d] %-50s %6d hourly values", i, len(sensors), s.name, len(hours))
        if len(hours):
            frames.append(
                hours.assign(
                    station_id=f"OPENAQ_{s.location_id}_{s.sensor_id}",
                    name=s.name,
                    lat=s.lat,
                    lon=s.lon,
                )
            )

    if not frames:
        log.warning("All sensors returned 0 hourly values")
        return pd.DataFrame(columns=["station_id", "name", "lat", "lon", "date", colname])

    out = pd.concat(frames, ignore_index=True)[
        ["station_id", "name", "lat", "lon", "date", colname]
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    log.info("Wrote %d hourly rows for %d stations → %s", len(out), out.station_id.nunique(), out_csv)
    return out


def _covering_cache_pollutant(
    cache: Path, sensor_id: int, colname: str, start: str, end: str
) -> Path | None:
    """Like openaq._covering_cache but includes pollutant in the filename pattern."""
    for p in cache.glob(f"sensor_{sensor_id}_{colname}_*_*.csv.gz"):
        parts = p.name[: -len(".csv.gz")].split("_")
        if len(parts) >= 5:
            lo, hi = parts[-2], parts[-1]
            if lo <= start and hi >= end:
                return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Download hourly multi-pollutant data from OpenAQ"
    )
    p.add_argument("--pollutant", required=True,
                   choices=list(PARAMETER_IDS), help="Pollutant key")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--country", default="IN")
    p.add_argument("--city", default=None,
                   help="City name for bounding-box filter (e.g. mumbai)")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--cache-dir", default="cache/openaq_multipollutant")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    out = args.out or f"data/openaq_{args.city or args.country}_{args.pollutant}_{args.start[:4]}.csv"
    download_pollutant(
        pollutant=args.pollutant,
        start=args.start,
        end=args.end,
        out_csv=out,
        cache_dir=args.cache_dir,
        country_code=args.country,
        city=args.city,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
