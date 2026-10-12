"""Multi-pollutant data service for AirQ Insight v2.0.

Loads the NAMP yearly CSV and provides fast in-memory queries by city/state/year.
Also provides access to the trained PM2.5 model for inference.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Data paths (relative to backend/) ─────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_NAMP_CSV = _DATA_DIR / "namp_all_pollutants_clean.csv"
_NAMP_RAW_CSV = _DATA_DIR / "namp_all_pollutants.csv"  # fallback

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "pm25_lightgbm"

POLLUTANTS = ["so2", "no2", "pm10", "pm25"]


@lru_cache(maxsize=1)
def _load_namp() -> pd.DataFrame:
    """Load and cache the NAMP yearly dataset."""
    path = _NAMP_CSV if _NAMP_CSV.exists() else _NAMP_RAW_CSV
    if not path.exists():
        log.warning("NAMP CSV not found at %s — returning empty DataFrame", path)
        return pd.DataFrame(columns=["year", "state", "city", "so2", "no2", "pm10", "pm25", "lat", "lon"])
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["city_lower"] = df["city"].str.lower().str.strip()
    df["state_lower"] = df["state"].str.lower().str.strip()
    log.info("Loaded NAMP dataset: %d rows, years %d–%d", len(df), df["year"].min(), df["year"].max())
    return df


def get_yearly_data(
    city: Optional[str] = None,
    state: Optional[str] = None,
    pollutant: Optional[str] = None,
    year_from: int = 2015,
    year_to: int = 2023,
    limit: int = 500,
) -> list[dict]:
    """Query yearly pollutant data.

    Parameters
    ----------
    city : str, optional
        City name filter (case-insensitive, partial match).
    state : str, optional
        State name filter (case-insensitive, partial match).
    pollutant : str, optional
        Restrict to a single pollutant column ("so2", "no2", "pm10", "pm25").
    year_from, year_to : int
        Year range (inclusive).
    limit : int
        Maximum number of rows returned.
    """
    df = _load_namp()
    mask = (df["year"] >= year_from) & (df["year"] <= year_to)
    if city:
        mask &= df["city_lower"].str.contains(city.lower(), na=False)
    if state:
        mask &= df["state_lower"].str.contains(state.lower(), na=False)

    result = df[mask].copy()

    if pollutant and pollutant in POLLUTANTS:
        cols = ["year", "state", "city", "lat", "lon", pollutant]
        result = result[[c for c in cols if c in result.columns]]

    result = result.head(limit)

    # Convert NaN to None for JSON serialisability
    return result.where(pd.notna(result), other=None).to_dict(orient="records")


def get_mumbai_data(year_from: int = 2015, year_to: int = 2023) -> list[dict]:
    """Return all available Mumbai multi-pollutant data."""
    return get_yearly_data(city="mumbai", year_from=year_from, year_to=year_to, limit=100)


def get_city_trends(city: str, pollutant: str) -> dict:
    """Return year-over-year trend for one city and pollutant."""
    df = _load_namp()
    rows = df[df["city_lower"].str.contains(city.lower(), na=False)].sort_values("year")
    if rows.empty:
        return {"city": city, "pollutant": pollutant, "trend": []}

    series = []
    for _, r in rows.iterrows():
        val = r.get(pollutant)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            series.append({"year": int(r["year"]), "value": float(val)})

    # Simple linear trend slope (µg/m³ per year)
    slope = None
    if len(series) >= 3:
        years = [s["year"] for s in series]
        vals = [s["value"] for s in series]
        slope = float(np.polyfit(years, vals, 1)[0])

    from ml_engine.pollutant_config import get_pollutant as _get_cfg
    try:
        cfg = _get_cfg(pollutant)
        unit = cfg.unit
        naaqs = cfg.naaqs_annual
    except KeyError:
        unit = "µg/m³"
        naaqs = None

    return {
        "city": city,
        "pollutant": pollutant,
        "unit": unit,
        "naaqs_annual": naaqs,
        "trend": series,
        "slope_per_year": round(slope, 3) if slope is not None else None,
        "direction": (
            "improving" if slope is not None and slope < -0.5
            else "worsening" if slope is not None and slope > 0.5
            else "stable"
        ),
    }


def predict_pm25(
    city: str,
    state: Optional[str] = None,
    year: int = 2024,
    no2: Optional[float] = None,
    so2: Optional[float] = None,
    pm10: Optional[float] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> dict:
    """Predict PM₂.₅ for a city-year using the trained LightGBM model.

    Falls back to NAMP lookup if model is unavailable.
    """
    # Try NAMP lookup first
    df = _load_namp()
    namp_row = df[
        df["city_lower"].str.contains(city.lower(), na=False) & (df["year"] == year)
    ]
    if not namp_row.empty and not pd.isna(namp_row.iloc[0].get("pm25")):
        row = namp_row.iloc[0]
        from ml_engine.compliance import assess_point
        rec = assess_point("pm25", float(row["pm25"]), averaging_period="annual",
                           source="observation", location_name=city)
        return {
            "source": "namp_observation",
            "city": city,
            "year": year,
            "pm25_ugm3": float(row["pm25"]),
            "naaqs_status": rec.naaqs_status,
            "naaqs_threshold": rec.naaqs_threshold,
            "exceedance_fraction": rec.naaqs_exceedance_fraction,
        }

    # Try model prediction
    if not _MODEL_DIR.exists():
        return {
            "source": "unavailable",
            "city": city,
            "year": year,
            "error": "PM2.5 model not trained yet. Run: uv run python -m ml_engine.training.pm25_lightgbm",
        }

    try:
        import lightgbm as lgb

        meta = json.loads((_MODEL_DIR / "pm25_metadata.json").read_text())
        state_classes = json.loads((_MODEL_DIR / "pm25_state_enc.json").read_text())
        state_to_int = {s: i for i, s in enumerate(state_classes)}
        booster = lgb.Booster(model_file=str(_MODEL_DIR / "pm25_model.txt"))

        # Get lat/lon from NAMP if not provided
        if lat is None or lon is None:
            city_rows = df[df["city_lower"].str.contains(city.lower(), na=False)]
            if not city_rows.empty:
                lat = float(city_rows.iloc[0].get("lat") or 20.0)
                lon = float(city_rows.iloc[0].get("lon") or 78.0)
            else:
                lat, lon = 20.0, 78.0

        # Get co-pollutants from NAMP if not provided
        if no2 is None or so2 is None or pm10 is None:
            city_recent = df[df["city_lower"].str.contains(city.lower(), na=False)].sort_values("year")
            if not city_recent.empty:
                r = city_recent.iloc[-1]
                no2 = no2 or float(r.get("no2") or 25.0)
                so2 = so2 or float(r.get("so2") or 10.0)
                pm10 = pm10 or float(r.get("pm10") or 80.0)

        features = meta["features"]
        row_data = {
            "year": year,
            "so2": so2 or 10.0,
            "no2": no2 or 25.0,
            "pm10": pm10 or 80.0,
            "lat": lat,
            "lon": lon,
            "state_enc": state_to_int.get(state, 0),
        }
        X = np.array([[row_data.get(f, 0.0) for f in features]], dtype=np.float64)
        pm25_pred = float(max(0.0, booster.predict(X)[0]))

        from ml_engine.compliance import assess_point
        rec = assess_point("pm25", pm25_pred, averaging_period="annual",
                           source="model", location_name=city)
        return {
            "source": "lightgbm_model",
            "city": city,
            "year": year,
            "pm25_ugm3": round(pm25_pred, 2),
            "naaqs_status": rec.naaqs_status,
            "naaqs_threshold": rec.naaqs_threshold,
            "exceedance_fraction": rec.naaqs_exceedance_fraction,
            "model_r2_test": meta.get("independent_test", {}).get("r2"),
            "features_used": features,
        }

    except Exception as exc:
        log.warning("PM2.5 model prediction failed: %s", exc)
        return {"source": "error", "city": city, "year": year, "error": str(exc)}


def compliance_summary_all_pollutants(city: str) -> dict:
    """Return compliance summary for all pollutants for the latest available year."""
    from ml_engine.compliance import assess_point
    df = _load_namp()
    city_rows = df[df["city_lower"].str.contains(city.lower(), na=False)].sort_values("year")
    if city_rows.empty:
        return {"city": city, "error": f"No data found for city: {city}"}

    latest = city_rows.iloc[-1]
    year = int(latest["year"])
    results = {}
    for pol in ["so2", "no2", "pm10", "pm25"]:
        val = latest.get(pol)
        if isinstance(val, float) and not np.isnan(val):
            rec = assess_point(pol, val, averaging_period="annual",
                               source="namp_observation", location_name=city)
            results[pol] = {
                "concentration": val,
                "naaqs_status": rec.naaqs_status,
                "naaqs_threshold": rec.naaqs_threshold,
                "who_status": rec.who_status,
                "exceedance_fraction": rec.naaqs_exceedance_fraction,
            }
        else:
            results[pol] = {"concentration": None, "naaqs_status": "insufficient_data"}

    return {"city": city, "year": year, "pollutants": results}
