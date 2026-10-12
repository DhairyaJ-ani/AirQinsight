"""Pollutant-specific compliance engine for AirQ Insight v2.0.

Compares observed or predicted pollutant concentrations against India NAAQS
thresholds and WHO guidelines. Produces versioned compliance records with
exceedance magnitude, status classification and audit metadata.

This module is self-contained: no external network calls, no database access.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from .pollutant_config import POLLUTANTS, PollutantConfig, get_pollutant

log = logging.getLogger(__name__)

# ── WHO interim / guideline targets (µg/m³) ──────────────────────────────
# Source: WHO Air Quality Guidelines 2021
WHO_ANNUAL: dict[str, float] = {
    "pm25": 5.0,   # AQG annual mean
    "pm10": 15.0,  # AQG annual mean
    "no2": 10.0,   # AQG annual mean
    "so2": None,   # WHO uses 24h standard; no annual
    "co": None,    # WHO uses 8h standard
    "o3": None,    # WHO uses peak-season standard
}
WHO_24H: dict[str, float | None] = {
    "pm25": 15.0,
    "pm10": 45.0,
    "no2": 25.0,
    "so2": 40.0,
    "co": None,    # 8h: 4 mg/m³ — skip for now
    "o3": None,
}

ComplianceStatus = Literal["compliant", "approaching", "exceeded", "insufficient_data"]

APPROACHING_FRACTION = 0.80  # >80% of threshold = "approaching"


@dataclass
class ComplianceRecord:
    """One compliance assessment for a pollutant at a location / period."""
    pollutant: str
    averaging_period: str          # "annual", "24h"
    concentration_ugm3: float | None
    unit: str

    # India NAAQS
    naaqs_threshold: float | None
    naaqs_status: ComplianceStatus
    naaqs_exceedance_fraction: float | None  # concentration / threshold - 1

    # WHO guideline (informational)
    who_threshold: float | None
    who_status: ComplianceStatus

    # Metadata
    source: str = "observation"          # "observation" | "model" | "forecast"
    location_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    timestamp: str | None = None
    model_version: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyFlag:
    """Potential anomaly signal for manual follow-up."""
    pollutant: str
    location_name: str | None
    lat: float | None
    lon: float | None
    flag_type: str         # "sudden_increase", "spatial_gradient", "persistent_hotspot"
    severity: str          # "low", "medium", "high"
    description: str
    value_ugm3: float | None
    baseline_ugm3: float | None
    timestamp: str | None


def _classify(
    value: float | None,
    threshold: float | None,
    approaching_fraction: float = APPROACHING_FRACTION,
) -> tuple[ComplianceStatus, float | None]:
    """Return (status, exceedance_fraction)."""
    if value is None or np.isnan(value) or threshold is None:
        return "insufficient_data", None
    frac = value / threshold - 1.0
    if value > threshold:
        return "exceeded", round(frac, 3)
    if value >= threshold * approaching_fraction:
        return "approaching", round(frac, 3)
    return "compliant", round(frac, 3)


def assess_point(
    pollutant: str,
    concentration: float | None,
    averaging_period: str = "annual",
    source: str = "observation",
    location_name: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    timestamp: str | None = None,
    model_version: str | None = None,
) -> ComplianceRecord:
    """Assess compliance for a single concentration value.

    Parameters
    ----------
    pollutant : str
        One of the keys in ``POLLUTANTS`` (e.g. "pm25", "no2").
    concentration : float | None
        Measured or modelled concentration in the pollutant's native unit.
        Pass ``None`` if data is unavailable.
    averaging_period : str
        ``"annual"`` or ``"24h"``.
    """
    cfg = get_pollutant(pollutant)

    if averaging_period == "annual":
        naaqs_thr = cfg.naaqs_annual
        who_thr = WHO_ANNUAL.get(pollutant)
    elif averaging_period == "24h":
        naaqs_thr = cfg.naaqs_24h
        who_thr = WHO_24H.get(pollutant)
    else:
        raise ValueError(f"averaging_period must be 'annual' or '24h', got {averaging_period!r}")

    # Validate plausibility
    if concentration is not None and (
        concentration < cfg.min_valid or concentration > cfg.max_valid
    ):
        log.warning(
            "Concentration %.1f %s is outside valid range [%.1f, %.1f] for %s",
            concentration, cfg.unit, cfg.min_valid, cfg.max_valid, pollutant,
        )
        concentration = None

    naaqs_status, naaqs_exc = _classify(concentration, naaqs_thr)
    who_status, _ = _classify(concentration, who_thr)

    return ComplianceRecord(
        pollutant=pollutant,
        averaging_period=averaging_period,
        concentration_ugm3=round(float(concentration), 3) if concentration is not None else None,
        unit=cfg.unit,
        naaqs_threshold=naaqs_thr,
        naaqs_status=naaqs_status,
        naaqs_exceedance_fraction=naaqs_exc,
        who_threshold=who_thr,
        who_status=who_status,
        source=source,
        location_name=location_name,
        lat=lat,
        lon=lon,
        timestamp=timestamp or datetime.utcnow().isoformat(),
        model_version=model_version,
    )


def assess_dataframe(
    df: pd.DataFrame,
    pollutant: str,
    concentration_col: str | None = None,
    averaging_period: str = "annual",
    source: str = "observation",
) -> pd.DataFrame:
    """Assess compliance for every row in a DataFrame.

    Expected columns: concentration_col (defaults to pollutant key), optionally
    ``name``, ``lat``, ``lon``, ``year`` / ``date``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus ``naaqs_status``, ``naaqs_exceedance_fraction``,
        ``who_status``, ``naaqs_threshold``, ``who_threshold`` columns.
    """
    colname = concentration_col or pollutant.lower()
    if colname not in df.columns:
        raise ValueError(f"Column {colname!r} not found in DataFrame; available: {list(df.columns)}")

    cfg = get_pollutant(pollutant)
    naaqs_thr = cfg.naaqs_annual if averaging_period == "annual" else cfg.naaqs_24h
    who_thr = (WHO_ANNUAL if averaging_period == "annual" else WHO_24H).get(pollutant)

    result = df.copy()
    result["naaqs_threshold"] = naaqs_thr
    result["who_threshold"] = who_thr
    result["naaqs_status"] = "insufficient_data"
    result["naaqs_exceedance_fraction"] = np.nan
    result["who_status"] = "insufficient_data"

    valid = result[colname].notna()
    if naaqs_thr is not None:
        exc = result.loc[valid, colname] / naaqs_thr - 1.0
        result.loc[valid, "naaqs_exceedance_fraction"] = exc.round(3)
        result.loc[valid & (result[colname] > naaqs_thr), "naaqs_status"] = "exceeded"
        result.loc[valid & (result[colname] >= naaqs_thr * APPROACHING_FRACTION) & (result[colname] <= naaqs_thr), "naaqs_status"] = "approaching"
        result.loc[valid & (result[colname] < naaqs_thr * APPROACHING_FRACTION), "naaqs_status"] = "compliant"
    if who_thr is not None:
        result.loc[valid & (result[colname] > who_thr), "who_status"] = "exceeded"
        result.loc[valid & (result[colname] >= who_thr * APPROACHING_FRACTION) & (result[colname] <= who_thr), "who_status"] = "approaching"
        result.loc[valid & (result[colname] < who_thr * APPROACHING_FRACTION), "who_status"] = "compliant"

    return result


def screen_anomalies(
    df: pd.DataFrame,
    pollutant: str,
    value_col: str | None = None,
    baseline_col: str = "baseline",
    sudden_increase_threshold: float = 2.0,   # 2× baseline = anomaly
) -> list[AnomalyFlag]:
    """Screen a DataFrame of measurements for potential anomalies.

    This is a *screening* tool, not proof of emissions violations.

    Parameters
    ----------
    df : pd.DataFrame
        Rows with ``name`` (optional), ``lat``, ``lon``, the value column and
        optionally a ``baseline`` column.
    sudden_increase_threshold : float
        Flag if value > threshold × baseline.
    """
    col = value_col or pollutant.lower()
    flags: list[AnomalyFlag] = []

    for _, row in df.iterrows():
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        baseline = row.get(baseline_col)
        if baseline and not np.isnan(float(baseline)) and float(baseline) > 0:
            ratio = float(val) / float(baseline)
            if ratio > sudden_increase_threshold:
                flags.append(AnomalyFlag(
                    pollutant=pollutant,
                    location_name=row.get("name") or row.get("city"),
                    lat=row.get("lat"),
                    lon=row.get("lon"),
                    flag_type="sudden_increase",
                    severity="high" if ratio > 3.0 else "medium",
                    description=(
                        f"{pollutant.upper()} = {val:.1f} is {ratio:.1f}× above "
                        f"baseline ({baseline:.1f}). Requires investigation."
                    ),
                    value_ugm3=float(val),
                    baseline_ugm3=float(baseline),
                    timestamp=str(row.get("year") or row.get("date") or ""),
                ))

    return flags


def compliance_summary(
    df: pd.DataFrame,
    pollutant: str,
    averaging_period: str = "annual",
) -> dict:
    """Return a summary dict after calling ``assess_dataframe``."""
    assessed = assess_dataframe(df, pollutant, averaging_period=averaging_period)
    total = len(assessed)
    counts = assessed["naaqs_status"].value_counts().to_dict()
    cfg = get_pollutant(pollutant)

    return {
        "pollutant": pollutant,
        "display_name": cfg.display_name,
        "unit": cfg.unit,
        "averaging_period": averaging_period,
        "total_locations": total,
        "naaqs_threshold": cfg.naaqs_annual if averaging_period == "annual" else cfg.naaqs_24h,
        "who_threshold": (WHO_ANNUAL if averaging_period == "annual" else WHO_24H).get(pollutant),
        "status_counts": {
            "exceeded": int(counts.get("exceeded", 0)),
            "approaching": int(counts.get("approaching", 0)),
            "compliant": int(counts.get("compliant", 0)),
            "insufficient_data": int(counts.get("insufficient_data", 0)),
        },
        "exceedance_rate_pct": round(
            100 * counts.get("exceeded", 0) / total, 1) if total else 0.0,
        "mean_concentration": round(float(df[pollutant].mean()), 2) if pollutant in df else None,
        "max_concentration": round(float(df[pollutant].max()), 2) if pollutant in df else None,
    }
