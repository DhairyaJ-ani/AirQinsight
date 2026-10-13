"""Source assessment endpoint: local pollutant value at a source location + surrounding stats.

Reads from the same bundled GeoTIFF datasets served by ``/downscale/geotiff`` and
``/historical/geotiff``.  Provides the Analysis Agent's evidence-based interpretation
that is displayed in the Home page marker popup.

Scientific safeguards
---------------------
* Returns NaN / null when the source coordinate falls outside the raster extent.
* Returns null when every pixel in the neighbourhood is NoData (cloud-masked).
* Labels satellite column data distinctly from surface concentration.
* Never invents concentration values or attributes a facility as non-compliant from
  a satellite hotspot alone.
* O₃ and NO₂ secondary formation is explicitly noted where relevant.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_current_user

router = APIRouter()


# ── Raster helpers ─────────────────────────────────────────────────────────────

def _locate_raster(pollutant: str, date_str: str) -> Path | None:
    """Return the Path to a GeoTIFF for the given pollutant and date, or None."""
    pol_key = pollutant.lower()
    folder_map = {"pm25": "PM25", "pm10": "PM10", "so2": "SO2", "o3": "O3", "co": "CO"}

    if pol_key in folder_map:
        pol_folder = folder_map[pol_key]
        candidates = [
            Path(f"data/pollutant_samples/{pol_folder}/{pol_folder}_{date_str}.tif"),
            Path(f"backend/data/pollutant_samples/{pol_folder}/{pol_folder}_{date_str}.tif"),
        ]
    else:
        # NO₂ – fall back to Sentinel-5P test data
        candidates = [
            Path(f"data/test_data/no2_raw_coarse_{date_str}.tif"),
            Path(f"backend/data/test_data/no2_raw_coarse_{date_str}.tif"),
        ]

    # Fall back to any file with that date in the folder
    for c in candidates:
        if c.exists():
            return c

    # Last-resort: pick the nearest available date in the folder
    if pol_key in folder_map:
        pol_folder = folder_map[pol_key]
        dirs = [Path(f"data/pollutant_samples/{pol_folder}"), Path(f"backend/data/pollutant_samples/{pol_folder}")]
    else:
        dirs = [Path("data/test_data"), Path("backend/data/test_data")]

    for d in dirs:
        if d.exists():
            tifs = sorted(d.glob("*.tif"))
            if tifs:
                # Prefer closest date
                def _date_of(p: Path) -> str:
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
                    return m.group(1) if m else ""
                tifs_with_dates = [(p, _date_of(p)) for p in tifs if _date_of(p)]
                if tifs_with_dates:
                    closest = min(tifs_with_dates, key=lambda x: abs(x[1] > date_str))
                    return closest[0]
    return None


def _read_raster_with_transform(fpath: Path):
    """Open a GeoTIFF and return (array float32, transform, crs_str, nodata)."""
    import rasterio
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
    # Apply nodata mask
    if nodata is not None:
        arr[arr == nodata] = np.nan
    elif np.nanmax(arr) > 1e30:
        arr[arr > 1e30] = np.nan
    return arr, transform, crs


def _pixel_at(arr: np.ndarray, transform, lat: float, lon: float):
    """Return (row, col) for a geographic coordinate, or None if out of bounds."""
    from rasterio.transform import rowcol
    try:
        row, col = rowcol(transform, lon, lat)
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            return int(row), int(col)
    except Exception:
        pass
    return None


def _surrounding_stats(
    arr: np.ndarray,
    transform,
    lat: float,
    lon: float,
    radius_deg: float,
) -> dict:
    """
    Compute mean / median / max / valid_fraction across pixels within a geographic radius.
    Returns dict with all fields or empty-dict if no valid pixels.
    """
    height, width = arr.shape
    # Estimate pixel size from transform
    px_lat = abs(transform.e)  # degrees per row
    px_lon = abs(transform.a)  # degrees per col

    row_radius = max(1, int(math.ceil(radius_deg / px_lat)))
    col_radius = max(1, int(math.ceil(radius_deg / px_lon)))

    # Row / col of center
    from rasterio.transform import rowcol
    try:
        center_row, center_col = rowcol(transform, lon, lat)
    except Exception:
        return {}

    r0 = max(0, center_row - row_radius)
    r1 = min(height, center_row + row_radius + 1)
    c0 = max(0, center_col - col_radius)
    c1 = min(width, center_col + col_radius + 1)

    patch = arr[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch)]

    total_pixels = patch.size
    if total_pixels == 0:
        return {}
    valid_fraction = round(float(len(valid)) / total_pixels, 3)
    if len(valid) == 0:
        return {
            "valid_pixel_fraction": 0.0,
            "note": "All pixels in neighbourhood are NoData (cloud-masked or missing).",
        }

    return {
        "mean": round(float(np.mean(valid)), 2),
        "median": round(float(np.median(valid)), 2),
        "max": round(float(np.max(valid)), 2),
        "min": round(float(np.min(valid)), 2),
        "std": round(float(np.std(valid)), 2),
        "valid_pixel_count": int(len(valid)),
        "total_pixel_count": int(total_pixels),
        "valid_pixel_fraction": valid_fraction,
        "radius_deg": radius_deg,
        "analysis_method": "mean/median/max over valid (non-NaN) pixels in rectangular neighbourhood",
    }


# ── Pollutant metadata lookup ──────────────────────────────────────────────────

_POLLUTANT_META = {
    "no2": {
        "display": "NO₂",
        "units": "µmol/m²",
        "measurement_type": "satellite_column",
        "source": "Sentinel-5P TROPOMI OFFL L3",
        "naaqs_note": "Column density cannot be directly compared to CPCB 24-h surface NAAQS (80 µg/m³).",
        "formation_note": "NO₂ is primarily emitted from combustion (traffic, industry). Elevated satellite columns near a facility are consistent with emission but do not constitute a violation finding.",
    },
    "pm25": {
        "display": "PM₂.₅",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "source": "CAMS European/Global NRT Analysis (resampled to 0.035°)",
        "naaqs_note": "CPCB 24-h NAAQS for PM₂.₅: 60 µg/m³. WHO 24-h guideline: 15 µg/m³.",
        "formation_note": "PM₂.₅ has both primary emission (combustion, dust) and secondary formation (photochemical) sources. Attribution to a specific facility requires receptor modelling and ground measurements.",
    },
    "pm10": {
        "display": "PM₁₀",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "source": "CAMS European/Global NRT Analysis (resampled to 0.035°)",
        "naaqs_note": "CPCB 24-h NAAQS for PM₁₀: 100 µg/m³. WHO 24-h guideline: 45 µg/m³.",
        "formation_note": "PM₁₀ includes mineral dust, construction dust, and coarse combustion particles. Source attribution requires wind direction and dispersion modelling.",
    },
    "so2": {
        "display": "SO₂",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "source": "CAMS European/Global NRT Analysis (resampled to 0.035°)",
        "naaqs_note": "CPCB 24-h NAAQS for SO₂: 80 µg/m³. WHO 24-h guideline: 40 µg/m³.",
        "formation_note": "SO₂ is a primary indicator of high-sulphur fuel combustion (power plants, refineries, shipping). A nearby peak is a stronger indicator of industrial origin than for secondary species.",
    },
    "o3": {
        "display": "O₃",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "source": "CAMS European/Global NRT Analysis (resampled to 0.035°)",
        "naaqs_note": "CPCB 8-h NAAQS for O₃: 100 µg/m³. WHO 8-h guideline: 100 µg/m³.",
        "formation_note": "O₃ is a secondary pollutant formed from NOₓ and VOCs under sunlight. It is often LOWER near industrial sources (due to NO titration) and HIGHER in downwind suburban areas. Attributing O₃ elevation to a nearby facility without photochemical modelling is not scientifically valid.",
    },
    "co": {
        "display": "CO",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "source": "CAMS European/Global NRT Analysis (resampled to 0.035°)",
        "naaqs_note": "CPCB 8-h NAAQS for CO: 2 mg/m³ (2000 µg/m³).",
        "formation_note": "CO is a primary combustion product. Elevated levels near industrial or traffic sources are physically plausible but confirmation requires in-situ measurements.",
    },
}


# ── Analysis Agent interpretation ─────────────────────────────────────────────

def _build_agent_interpretation(
    pollutant: str,
    source_name: str,
    source_category: str,
    local_value: float | None,
    surrounding: dict,
    identification_status: str,
) -> dict:
    """
    Build an evidence-based, scientifically cautious interpretation for the given source.
    Returns structured interpretation dict.
    """
    meta = _POLLUTANT_META.get(pollutant, _POLLUTANT_META["no2"])
    pol_disp = meta["display"]
    units = meta["units"]

    evidence = []
    alternative_explanations = []
    confidence_level = "low"
    attribution_statement = "Insufficient evidence for attribution."
    limitations = [
        f"Raster resolution is ~3.9 km per pixel. The value represents the spatial mean of a coarse grid cell, not a direct measurement at the facility.",
        f"Measurement type: {meta['measurement_type']}. {meta['naaqs_note']}",
        meta["formation_note"],
    ]

    if local_value is not None and not math.isnan(local_value):
        evidence.append(f"Local {pol_disp} at source location: {local_value:.2f} {units} (coarse-cell mean, ~3.9 km).")
        vf = surrounding.get("valid_pixel_fraction", 0)
        if vf > 0:
            evidence.append(
                f"Neighbourhood mean: {surrounding.get('mean', 'N/A'):.2f} {units}, "
                f"max: {surrounding.get('max', 'N/A'):.2f} {units}, "
                f"valid pixel coverage: {vf*100:.0f}%."
            )

        if source_category in ("POWER_PLANT", "FACTORY"):
            if pollutant == "so2":
                attribution_statement = (
                    f"{pol_disp} is a direct emission tracer for high-sulphur combustion. "
                    f"Elevated {pol_disp} at {source_name} ({local_value:.1f} {units}) is "
                    f"physically consistent with this source category. Confirmatory receptor modelling is required."
                )
                confidence_level = "moderate"
                evidence.append("SO₂ emission is strongly associated with coal/oil combustion and refining.")
            elif pollutant == "no2":
                attribution_statement = (
                    f"Elevated {pol_disp} column ({local_value:.2f} {units}) over {source_name} is "
                    f"consistent with combustion emission from this facility. "
                    f"This is a satellite column observation; surface concentration and compliance assessment "
                    f"require in-situ measurement."
                )
                confidence_level = "low-moderate"
                evidence.append("NO₂ tropospheric column elevation is associated with combustion but also traffic and background.")
            elif pollutant == "o3":
                attribution_statement = (
                    f"O₃ concentration at {source_name} ({local_value:.1f} {units}) is present in the raster, "
                    f"but O₃ is a secondary photochemical pollutant. "
                    f"Elevated O₃ near an industrial source does NOT indicate local emission from this facility. "
                    f"O₃ may be higher downwind of the city, not at the source."
                )
                confidence_level = "insufficient"
                alternative_explanations.append("O₃ forms photochemically downwind, often away from primary emission sources.")
                alternative_explanations.append("NOₓ from industrial stacks can locally SUPPRESS O₃ through NO titration.")
            elif pollutant in ("pm25", "pm10"):
                attribution_statement = (
                    f"{pol_disp} level at {source_name} ({local_value:.1f} {units}) is noted. "
                    f"Industrial and combustion sources can emit primary PM, but secondary PM formation "
                    f"and regional transport are also major contributors. Attribution requires receptor modelling."
                )
                confidence_level = "low"
                alternative_explanations.append("Regional transport and boundary layer dynamics significantly affect surface PM.")
            elif pollutant == "co":
                attribution_statement = (
                    f"CO at {source_name} ({local_value:.1f} {units}). "
                    f"CO is a primary combustion tracer. Elevated levels near this source category are physically plausible "
                    f"but require ground-level monitoring for confirmation."
                )
                confidence_level = "low-moderate"

        elif source_category == "TRAFFIC_CORRIDOR":
            if pollutant in ("no2", "co"):
                attribution_statement = (
                    f"{pol_disp} at {source_name} ({local_value:.2f} {units}) is consistent "
                    f"with traffic corridor emissions (vehicles emit NO₂ and CO). "
                    f"Proximity is consistent with emission but coarse raster cannot resolve individual lanes."
                )
                confidence_level = "low-moderate"
            else:
                attribution_statement = (
                    f"{pol_disp} level at {source_name} ({local_value:.1f} {units}) noted. "
                    f"Traffic corridors primarily emit NOₓ and CO; {pol_disp} attribution to this source requires additional evidence."
                )
                confidence_level = "insufficient"
        else:
            attribution_statement = (
                f"{pol_disp}: {local_value:.2f} {units} at {source_name}. "
                f"Source identification status: {identification_status}. Interpretation requires additional data."
            )

    else:
        attribution_statement = (
            f"No valid {pol_disp} data available at this location. "
            f"The source coordinate may lie in a NoData region (cloud mask or outside raster extent)."
        )
        evidence.append("No valid raster data at source coordinates.")
        confidence_level = "insufficient"

    if not alternative_explanations:
        alternative_explanations.append("Background concentration and regional transport may dominate local levels.")
        alternative_explanations.append("Coarse raster resolution (3.9 km) cannot distinguish individual facilities.")

    return {
        "attribution_statement": attribution_statement,
        "confidence_level": confidence_level,
        "evidence": evidence,
        "alternative_explanations": alternative_explanations,
        "limitations": limitations,
        "recommended_follow_up": [
            "Deploy ground-level monitoring sensors at or near the facility.",
            "Request facility emissions stack reports from CPCB/state pollution control board.",
            "Apply receptor modelling (CMB, PMF) with measured speciated PM or trace gases.",
            "Cross-reference with historical satellite time series during shutdown periods.",
        ],
        "measurement_metadata": {
            "measurement_type": meta["measurement_type"],
            "units": units,
            "source_dataset": meta["source"],
            "spatial_resolution_km": 3.9,
            "coarse_cell_note": "Value represents a ~3.9 km grid cell mean, not a point measurement.",
        },
    }


# ── Main endpoint ──────────────────────────────────────────────────────────────

@router.get(
    "/source-assessment",
    summary="Local pollutant value + Analysis Agent interpretation at a source location",
    tags=["analyze"],
)
async def source_assessment(
    lat: float = Query(..., description="Source latitude"),
    lon: float = Query(..., description="Source longitude"),
    pollutant: str = Query("no2", description="Pollutant key (no2, pm25, pm10, so2, o3, co)"),
    date: str = Query(..., description="Observation date YYYY-MM-DD"),
    source_name: str = Query("Unknown source", description="Human-readable source name"),
    source_category: str = Query("UNKNOWN", description="Source category (FACTORY, POWER_PLANT, TRAFFIC_CORRIDOR, etc.)"),
    source_id: Optional[str] = Query(None, description="Source identifier"),
    radius_km: float = Query(5.0, description="Neighbourhood analysis radius in km", ge=0.5, le=50.0),
    identification_status: str = Query(
        "candidate_source",
        description="Identification status: known_source, candidate_source, candidate_location, insufficient_evidence",
    ),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Return local pollutant value at a source coordinate and surrounding statistics.
    The Analysis Agent evidence chain is included without fabricating attribution.
    """
    pol_key = pollutant.lower()
    if pol_key not in _POLLUTANT_META:
        raise HTTPException(status_code=422, detail=f"Unsupported pollutant '{pollutant}'. Use: no2, pm25, pm10, so2, o3, co")

    meta = _POLLUTANT_META[pol_key]

    # Locate raster
    fpath = _locate_raster(pol_key, date)
    if fpath is None:
        return {
            "source_id": source_id,
            "source_name": source_name,
            "source_category": source_category,
            "identification_status": identification_status,
            "selected_pollutant": pol_key,
            "pollutant_display": meta["display"],
            "dataset_id": None,
            "observation_date": date,
            "source_provenance": meta["source"],
            "local_value": None,
            "local_units": meta["units"],
            "measurement_type": meta["measurement_type"],
            "surrounding_statistics": {},
            "analysis_radius_km": radius_km,
            "agent_interpretation": _build_agent_interpretation(
                pol_key, source_name, source_category, None, {}, identification_status
            ),
            "data_availability": "unavailable",
            "data_availability_note": f"No raster file found for {pol_key} on {date}.",
        }

    try:
        arr, transform, crs = _read_raster_with_transform(fpath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read raster: {e}")

    # Local value at source point
    pixel = _pixel_at(arr, transform, lat, lon)
    local_value: float | None = None
    in_extent = False

    if pixel is not None:
        in_extent = True
        pval = float(arr[pixel[0], pixel[1]])
        local_value = None if math.isnan(pval) else round(pval, 3)
    else:
        in_extent = False

    # Surrounding statistics (radius in degrees; 1 km ≈ 0.009°)
    radius_deg = radius_km * 0.009
    surrounding = _surrounding_stats(arr, transform, lat, lon, radius_deg)

    # Agent interpretation
    interpretation = _build_agent_interpretation(
        pol_key, source_name, source_category, local_value, surrounding, identification_status
    )

    return {
        "source_id": source_id,
        "source_name": source_name,
        "source_category": source_category,
        "identification_status": identification_status,
        "selected_pollutant": pol_key,
        "pollutant_display": meta["display"],
        "dataset_id": f"{pol_key}_cams_61day_2025" if pol_key != "no2" else "no2_61day_2025",
        "observation_date": date,
        "source_location": {"lat": lat, "lon": lon},
        "source_provenance": meta["source"],
        "local_value": local_value,
        "local_units": meta["units"],
        "measurement_type": meta["measurement_type"],
        "in_raster_extent": in_extent,
        "surrounding_statistics": surrounding,
        "analysis_radius_km": radius_km,
        "raster_file": str(fpath),
        "crs": crs,
        "agent_interpretation": interpretation,
        "data_availability": "available" if in_extent and local_value is not None else (
            "nodata" if in_extent else "out_of_extent"
        ),
    }
