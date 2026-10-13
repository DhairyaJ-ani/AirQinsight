"""Historical raster dataset API — AirQ Insight.

Exposes the bundled 61-day NO₂ GeoTIFF dataset and any other validated historical
rasters through a typed REST interface.

Endpoints
---------
GET /api/v1/historical/datasets                          → list available bundled datasets
GET /api/v1/historical/{dataset_id}/manifest            → full dataset metadata
GET /api/v1/historical/{dataset_id}/{pollutant}/coverage → available dates + per-date stats
GET /api/v1/historical/{dataset_id}/{pollutant}/{date}/grid → JSON raster grid for a date
GET /api/v1/historical/{dataset_id}/{pollutant}/{date}/geotiff → raw GeoTIFF bytes

Design constraints
------------------
* Never mislabel measurement type. The NO₂ GeoTIFFs are tropospheric column data
  (µmol/m²), NOT surface concentration.  The API surface explicitly communicates this.
* Cloud-masked / missing pixels are returned as null in JSON, not as zero.
* The 61-day main dataset and the 8-image test samples remain separate datasets.
* Only NO₂ has real raster data for the Nov–Dec 2025 window.
  Other pollutants return availability=false + a clear explanation.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from ml_engine.pollutant_config import POLLUTANTS, get_pollutant

log = logging.getLogger(__name__)

router = APIRouter(tags=["historical"])

# ── Dataset registry ──────────────────────────────────────────────────────────

def _find_dir(*candidates: str) -> Path | None:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


# Each entry in BUNDLED_DATASETS describes one physical collection of rasters.
BUNDLED_DATASETS: list[dict[str, Any]] = [
    {
        "id": "no2_61day_2025",
        "display_name": "Sentinel-5P NO₂ — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Sentinel-5P TROPOMI tropospheric NO₂ column composites over Mumbai, "
            "1 November–31 December 2025. Pixels with cloud fraction ≥ 0.3 are masked to NaN. "
            "Native 0.035° resolution (~3.9 km). Measurement type: satellite atmospheric column."
        ),
        "pollutant": "no2",
        "measurement_type": "satellite_column",
        "units": "µmol/m²",
        "provider": "ESA Copernicus / Sentinel-5P TROPOMI OFFL L3",
        "product": "COPERNICUS/S5P/OFFL/L3_NO2",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.035,
        "native_resolution_km": 3.9,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (cloud-masked or missing)",
        "purpose": "main_historical",
        "data_dirs": ["data/test_data", "backend/data/test_data"],
        "file_pattern": "no2_raw_coarse_{date}.tif",
    },
    {
        "id": "no2_samples_8day",
        "display_name": "Sentinel-5P NO₂ — Mumbai (8 weekly samples)",
        "description": (
            "Eight weekly Sentinel-5P TROPOMI tropospheric NO₂ column samples over Mumbai, "
            "5 Nov–24 Dec 2025. Clear and partly cloudy conditions; largest cloud gap ~35%. "
            "Byte-for-byte copies of selected files from the 61-day dataset. "
            "Purpose: upload demonstration and pipeline testing only. "
            "Eight images is insufficient for training or establishing compliance."
        ),
        "pollutant": "no2",
        "measurement_type": "satellite_column",
        "units": "µmol/m²",
        "provider": "ESA Copernicus / Sentinel-5P TROPOMI OFFL L3",
        "product": "COPERNICUS/S5P/OFFL/L3_NO2",
        "date_start": "2025-11-05",
        "date_end": "2025-12-24",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.035,
        "native_resolution_km": 3.9,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (cloud-masked or missing)",
        "purpose": "test_upload_demo",
        "data_dirs": ["data/sentinel_test_samples", "backend/data/sentinel_test_samples"],
        "file_pattern": "no2_raw_coarse_{date}.tif",
        "manifest_csv": "manifest.csv",
    },
    {
        "id": "pm25_cams_61day_2025",
        "display_name": "CAMS PM₂.₅ Surface Analysis — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Copernicus Atmosphere Monitoring Service (CAMS) surface PM₂.₅ concentration "
            "analyses over Mumbai, 1 November–31 December 2025. 24-hour diurnal aggregated mean in µg/m³. "
            "Resampled to the Mumbai 0.035° grid (~3.9 km) from native ~0.4° CAMS global analysis."
        ),
        "pollutant": "pm25",
        "measurement_type": "surface_concentration",
        "units": "µg/m³",
        "provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "product": "CAMS European & Global Near-Real-Time Analysis",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.4,
        "native_resolution_km": 40.0,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (missing data preserved as NaN)",
        "purpose": "cams_surface_monitoring",
        "data_dirs": ["data/pollutant_samples/PM25", "backend/data/pollutant_samples/PM25"],
        "file_pattern": "PM25_{date}.tif",
        "manifest_csv": "../manifest.csv",
    },
    {
        "id": "pm10_cams_61day_2025",
        "display_name": "CAMS PM₁₀ Surface Analysis — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Copernicus Atmosphere Monitoring Service (CAMS) surface PM₁₀ concentration "
            "analyses over Mumbai, 1 November–31 December 2025. 24-hour diurnal aggregated mean in µg/m³. "
            "Resampled to the Mumbai 0.035° grid (~3.9 km) from native ~0.4° CAMS global analysis."
        ),
        "pollutant": "pm10",
        "measurement_type": "surface_concentration",
        "units": "µg/m³",
        "provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "product": "CAMS European & Global Near-Real-Time Analysis",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.4,
        "native_resolution_km": 40.0,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (missing data preserved as NaN)",
        "purpose": "cams_surface_monitoring",
        "data_dirs": ["data/pollutant_samples/PM10", "backend/data/pollutant_samples/PM10"],
        "file_pattern": "PM10_{date}.tif",
        "manifest_csv": "../manifest.csv",
    },
    {
        "id": "so2_cams_61day_2025",
        "display_name": "CAMS SO₂ Surface Analysis — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Copernicus Atmosphere Monitoring Service (CAMS) surface SO₂ concentration "
            "analyses over Mumbai, 1 November–31 December 2025. 24-hour diurnal aggregated mean in µg/m³. "
            "Resampled to the Mumbai 0.035° grid (~3.9 km) from native ~0.4° CAMS global analysis."
        ),
        "pollutant": "so2",
        "measurement_type": "surface_concentration",
        "units": "µg/m³",
        "provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "product": "CAMS European & Global Near-Real-Time Analysis",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.4,
        "native_resolution_km": 40.0,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (missing data preserved as NaN)",
        "purpose": "cams_surface_monitoring",
        "data_dirs": ["data/pollutant_samples/SO2", "backend/data/pollutant_samples/SO2"],
        "file_pattern": "SO2_{date}.tif",
        "manifest_csv": "../manifest.csv",
    },
    {
        "id": "o3_cams_61day_2025",
        "display_name": "CAMS Ground-Level O₃ Surface Analysis — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Copernicus Atmosphere Monitoring Service (CAMS) surface ground-level ozone (O₃) "
            "concentration analyses over Mumbai, 1 November–31 December 2025. 24-hour diurnal aggregated mean in µg/m³. "
            "Note: This is surface ground-level ozone, distinct from total atmospheric column ozone."
        ),
        "pollutant": "o3",
        "measurement_type": "surface_concentration",
        "units": "µg/m³",
        "provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "product": "CAMS European & Global Near-Real-Time Analysis",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.4,
        "native_resolution_km": 40.0,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (missing data preserved as NaN)",
        "purpose": "cams_surface_monitoring",
        "data_dirs": ["data/pollutant_samples/O3", "backend/data/pollutant_samples/O3"],
        "file_pattern": "O3_{date}.tif",
        "manifest_csv": "../manifest.csv",
    },
    {
        "id": "co_cams_61day_2025",
        "display_name": "CAMS CO Surface Analysis — Mumbai (Nov–Dec 2025)",
        "description": (
            "61 daily Copernicus Atmosphere Monitoring Service (CAMS) surface carbon monoxide (CO) "
            "concentration analyses over Mumbai, 1 November–31 December 2025. 24-hour diurnal aggregated mean in µg/m³. "
            "Resampled to the Mumbai 0.035° grid (~3.9 km) from native ~0.4° CAMS global analysis."
        ),
        "pollutant": "co",
        "measurement_type": "surface_concentration",
        "units": "µg/m³",
        "provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "product": "CAMS European & Global Near-Real-Time Analysis",
        "date_start": "2025-11-01",
        "date_end": "2025-12-31",
        "crs": "EPSG:4326",
        "native_resolution_deg": 0.4,
        "native_resolution_km": 40.0,
        "bounds": {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32},
        "grid": {"width": 10, "height": 13},
        "nodata_convention": "NaN (missing data preserved as NaN)",
        "purpose": "cams_surface_monitoring",
        "data_dirs": ["data/pollutant_samples/CO", "backend/data/pollutant_samples/CO"],
        "file_pattern": "CO_{date}.tif",
        "manifest_csv": "../manifest.csv",
    },
]

# Pollutants without raster data in the Nov–Dec 2025 window — honest status.
# If a pollutant has acquired GeoTIFFs on disk, it is dynamically detected as available.
UNAVAILABLE_POLLUTANTS: dict[str, dict[str, str]] = {}



def _get_dataset(dataset_id: str) -> dict[str, Any]:
    for ds in BUNDLED_DATASETS:
        if ds["id"] == dataset_id:
            return ds
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


def _resolve_dir(ds: dict[str, Any]) -> Path:
    for d in ds["data_dirs"]:
        p = Path(d)
        if p.exists():
            return p
    raise HTTPException(
        status_code=503,
        detail=f"Data directory for dataset '{ds['id']}' not found. "
               f"Searched: {ds['data_dirs']}",
    )


def _list_available_dates(ds: dict[str, Any]) -> list[str]:
    """Return sorted list of dates that have a GeoTIFF file."""
    d = _resolve_dir(ds)
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
    dates = set()
    for f in d.glob("*.tif"):
        m = pattern.search(f.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates)


def _read_tif(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single GeoTIFF, return (array float32 with NaN for nodata, profile dict)."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        # Convert masked array to NaN
        data = np.where(arr.mask if hasattr(arr, "mask") else np.isnan(arr.data),
                        np.nan, arr.data if hasattr(arr, "data") else arr)
        profile = {
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "bounds": {
                "west": src.bounds.left,
                "south": src.bounds.bottom,
                "east": src.bounds.right,
                "north": src.bounds.top,
            },
            "nodata": src.nodata,
            "units": src.tags().get("units", "umol m-2"),
            "long_name": src.tags().get("long_name", ""),
        }
        return data, profile


# ── Pydantic response models ──────────────────────────────────────────────────

class DatasetInfo(BaseModel):
    id: str
    display_name: str
    description: str
    pollutant: str
    measurement_type: str
    units: str
    provider: str
    date_start: str
    date_end: str
    crs: str
    native_resolution_deg: float
    native_resolution_km: float
    bounds: dict
    grid: dict
    nodata_convention: str
    purpose: str


class DateCoverageEntry(BaseModel):
    date: str
    available: bool
    valid_pixel_count: int | None
    total_pixels: int | None
    missing_fraction: float | None
    file_size_bytes: int | None


class CoverageReport(BaseModel):
    dataset_id: str
    pollutant: str
    total_days_in_window: int
    days_with_data: int
    days_without_data: int
    overall_missing_fraction: float
    dates: list[DateCoverageEntry]


class RasterGrid(BaseModel):
    dataset_id: str
    pollutant: str
    measurement_type: str
    units: str
    date: str
    crs: str
    bounds: dict
    width: int
    height: int
    native_resolution_deg: float
    valid_pixels: int
    total_pixels: int
    missing_fraction: float
    values: list[list[float | None]]  # row-major, None for NaN/cloud


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/datasets",
    response_model=list[DatasetInfo],
    summary="List available bundled historical datasets",
    description=(
        "Returns all bundled historical raster datasets with their metadata. "
        "Only datasets with files present on disk are listed. "
        "Each dataset specifies its actual measurement type and units."
    ),
)
async def list_datasets() -> list[DatasetInfo]:
    result = []
    for ds in BUNDLED_DATASETS:
        try:
            _resolve_dir(ds)
        except HTTPException:
            continue  # Skip if data dir missing
        result.append(DatasetInfo(
            id=ds["id"],
            display_name=ds["display_name"],
            description=ds["description"],
            pollutant=ds["pollutant"],
            measurement_type=ds["measurement_type"],
            units=ds["units"],
            provider=ds["provider"],
            date_start=ds["date_start"],
            date_end=ds["date_end"],
            crs=ds["crs"],
            native_resolution_deg=ds["native_resolution_deg"],
            native_resolution_km=ds["native_resolution_km"],
            bounds=ds["bounds"],
            grid=ds["grid"],
            nodata_convention=ds["nodata_convention"],
            purpose=ds["purpose"],
        ))
    return result


@router.get(
    "/pollutants/availability",
    summary="Check availability of each pollutant for Nov–Dec 2025 historical window",
)
async def pollutant_availability() -> dict:
    available = {}
    for pol, cfg in POLLUTANTS.items():
        has_data = any(
            ds["pollutant"] == pol
            for ds in BUNDLED_DATASETS
            if _find_dir(*ds["data_dirs"]) is not None
        )
        entry: dict[str, Any] = {
            "pollutant": pol,
            "display_name": cfg.display_name,
            "formula": cfg.formula,
            "units": cfg.unit,
            "has_raster_data_nov_dec_2025": has_data,
        }
        if not has_data and pol in UNAVAILABLE_POLLUTANTS:
            entry["unavailability_reason"] = UNAVAILABLE_POLLUTANTS[pol]["reason"]
            entry["acquisition_script"] = UNAVAILABLE_POLLUTANTS[pol]["acquisition_script"]
        elif has_data:
            entry["dataset_ids"] = [
                ds["id"] for ds in BUNDLED_DATASETS if ds["pollutant"] == pol
            ]
        available[pol] = entry
    return available


@router.get(
    "/{dataset_id}/manifest",
    summary="Full manifest for a bundled dataset",
)
async def dataset_manifest(dataset_id: str) -> dict:
    ds = _get_dataset(dataset_id)
    d = _resolve_dir(ds)
    dates = _list_available_dates(ds)

    coverage_entries = []
    for dt in dates:
        tif = d / ds["file_pattern"].format(date=dt)
        if tif.exists():
            data, _ = _read_tif(tif)
            total = data.size
            valid = int(np.isfinite(data).sum())
            coverage_entries.append({
                "date": dt,
                "valid_pixels": valid,
                "total_pixels": total,
                "missing_fraction": round(1.0 - valid / total, 4),
                "file_size_bytes": tif.stat().st_size,
            })

    manifest_extra = {}
    if ds.get("manifest_csv"):
        manifest_csv_path = d / ds["manifest_csv"]
        if manifest_csv_path.exists():
            import csv
            with open(manifest_csv_path) as f:
                manifest_extra["csv_manifest"] = list(csv.DictReader(f))

    return {
        **{k: v for k, v in ds.items() if k not in ("data_dirs", "file_pattern")},
        "dates_with_data": dates,
        "date_count": len(dates),
        "per_date_coverage": coverage_entries,
        **manifest_extra,
    }


@router.get(
    "/{dataset_id}/{pollutant}/coverage",
    response_model=CoverageReport,
    summary="Date-coverage report for a dataset and pollutant",
)
async def pollutant_coverage(
    dataset_id: str,
    pollutant: str,
) -> CoverageReport:
    ds = _get_dataset(dataset_id)

    if ds["pollutant"] != pollutant.lower():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{dataset_id}' contains {ds['pollutant'].upper()} data, "
                f"not {pollutant.upper()}. "
                + (UNAVAILABLE_POLLUTANTS.get(pollutant.lower(), {}).get("reason", ""))
            ),
        )

    d = _resolve_dir(ds)
    available_dates = set(_list_available_dates(ds))

    # Build full requested window
    start = date.fromisoformat(ds["date_start"])
    end = date.fromisoformat(ds["date_end"])
    total_days = (end - start).days + 1

    entries = []
    for i in range(total_days):
        dt_str = (start + timedelta(days=i)).isoformat()
        if dt_str in available_dates:
            tif = d / ds["file_pattern"].format(date=dt_str)
            if tif.exists():
                data, _ = _read_tif(tif)
                total = data.size
                valid = int(np.isfinite(data).sum())
                entries.append(DateCoverageEntry(
                    date=dt_str,
                    available=True,
                    valid_pixel_count=valid,
                    total_pixels=total,
                    missing_fraction=round(1.0 - valid / total, 4),
                    file_size_bytes=int(tif.stat().st_size),
                ))
            else:
                entries.append(DateCoverageEntry(
                    date=dt_str, available=False,
                    valid_pixel_count=None, total_pixels=None,
                    missing_fraction=None, file_size_bytes=None,
                ))
        else:
            entries.append(DateCoverageEntry(
                date=dt_str, available=False,
                valid_pixel_count=None, total_pixels=None,
                missing_fraction=None, file_size_bytes=None,
            ))

    days_with_data = sum(1 for e in entries if e.available)
    all_valid = [e.valid_pixel_count for e in entries if e.valid_pixel_count is not None]
    all_total = [e.total_pixels for e in entries if e.total_pixels is not None]
    overall_missing = (
        1.0 - sum(all_valid) / sum(all_total) if all_total else 1.0
    )

    return CoverageReport(
        dataset_id=dataset_id,
        pollutant=pollutant,
        total_days_in_window=total_days,
        days_with_data=days_with_data,
        days_without_data=total_days - days_with_data,
        overall_missing_fraction=round(overall_missing, 4),
        dates=entries,
    )


@router.get(
    "/{dataset_id}/{pollutant}/{date_str}/grid",
    response_model=RasterGrid,
    summary="JSON raster grid for a specific date",
    description=(
        "Returns the raster values as a 2-D JSON array (row-major). "
        "Cloud-masked and missing pixels are returned as null — never as zero. "
        "The measurement_type field indicates whether values are satellite column or surface concentration."
    ),
)
async def get_raster_grid(
    dataset_id: str,
    pollutant: str,
    date_str: str,
) -> RasterGrid:
    # Validate date format
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {date_str}. Use YYYY-MM-DD.")

    ds = _get_dataset(dataset_id)

    if ds["pollutant"] != pollutant.lower():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{dataset_id}' contains {ds['pollutant'].upper()}, not {pollutant.upper()}. "
                + UNAVAILABLE_POLLUTANTS.get(pollutant.lower(), {}).get("reason", "")
            ),
        )

    d = _resolve_dir(ds)
    tif_path = d / ds["file_pattern"].format(date=date_str)

    if not tif_path.exists():
        # Date is in window but no file — this is an "unavailable" date (full cloud or no observation)
        raise HTTPException(
            status_code=404,
            detail=(
                f"No observation available for {pollutant.upper()} on {date_str} in dataset '{dataset_id}'. "
                f"This date may be fully cloud-covered or outside the dataset's available dates. "
                f"Check /api/v1/historical/{dataset_id}/{pollutant}/coverage for available dates."
            ),
        )

    data, profile = _read_tif(tif_path)
    total = data.size
    valid = int(np.isfinite(data).sum())

    # Serialize — NaN → None for JSON
    rows: list[list[float | None]] = []
    for row in data:
        rows.append([
            None if not np.isfinite(v) else round(float(v), 4)
            for v in row
        ])

    return RasterGrid(
        dataset_id=dataset_id,
        pollutant=pollutant,
        measurement_type=ds["measurement_type"],
        units=ds["units"],
        date=date_str,
        crs=profile["crs"],
        bounds=profile["bounds"],
        width=profile["width"],
        height=profile["height"],
        native_resolution_deg=ds["native_resolution_deg"],
        valid_pixels=valid,
        total_pixels=total,
        missing_fraction=round(1.0 - valid / total, 4),
        values=rows,
    )


@router.get(
    "/{dataset_id}/{pollutant}/{date_str}/geotiff",
    summary="Raw GeoTIFF bytes for a specific date",
    description=(
        "Streams the original GeoTIFF file. NaN values represent cloud-masked or missing pixels. "
        "The file retains its original CRS, transform, and nodata metadata."
    ),
)
async def get_geotiff(
    dataset_id: str,
    pollutant: str,
    date_str: str,
):
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {date_str}. Use YYYY-MM-DD.")

    ds = _get_dataset(dataset_id)

    if ds["pollutant"] != pollutant.lower():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' has {ds['pollutant'].upper()}, not {pollutant.upper()}.",
        )

    d = _resolve_dir(ds)
    tif_path = d / ds["file_pattern"].format(date=date_str)

    if not tif_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No GeoTIFF available for {pollutant.upper()} on {date_str} in '{dataset_id}'.",
        )

    content = tif_path.read_bytes()
    # HTTP response headers must be ASCII-safe; replace Unicode units chars
    safe_units = ds["units"].replace("µ", "u").replace("²", "2").replace("³", "3")
    return Response(
        content=content,
        media_type="image/tiff",
        headers={
            "Content-Disposition": f"inline; filename={tif_path.name}",
            "X-Pollutant": pollutant,
            "X-Dataset": dataset_id,
            "X-Date": date_str,
            "X-Measurement-Type": ds["measurement_type"],
            "X-Units": safe_units,
        },
    )
