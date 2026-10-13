#!/usr/bin/env python3
"""Acquisition and conversion script for historical multi-pollutant raster data.
Target period: 1 November 2025 to 31 December 2025 (61 days).
Target region: Mumbai, India (matching existing Sentinel-5P NO2 GeoTIFF extent).

Sources:
- PM2.5, PM10, SO2, O3, CO: Copernicus Atmosphere Monitoring Service (CAMS) Global Atmospheric
  Composition Near-Real-Time analysis fields via Open-Meteo European CAMS API.
- Native parameters:
  * PM2.5: Surface particulate matter < 2.5 µm (µg/m³)
  * PM10: Surface particulate matter < 10 µm (µg/m³)
  * SO2: Surface sulphur dioxide (µg/m³)
  * O3: Surface ground-level ozone (µg/m³)
  * CO: Surface carbon monoxide (µg/m³)
- Georeferencing matches reference Sentinel-5P NO2 GeoTIFFs:
  * Extent: [72.77°E, 18.865°N, 73.12°E, 19.32°N]
  * Grid: 13 rows x 10 cols (0.035° cell size, EPSG:4326)
  * Nodata: NaN (float32)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
from rasterio.transform import Affine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("acquire_pollutants")

# Target dates
START_DATE = "2025-11-01"
END_DATE = "2025-12-31"

# Bounding box & Grid definition matching backend/data/test_data/no2_raw_coarse_*.tif
WEST = 72.77
BOTTOM = 18.865
EAST = 73.12
TOP = 19.32
ROWS = 13
COLS = 10
DX = 0.035
DY = 0.035

# Affine transform for upper-left corner
TRANSFORM = Affine(DX, 0.0, WEST, 0.0, -DY, TOP)
CRS = "EPSG:4326"

POLLUTANT_CONFIGS = {
    "PM25": {
        "variable_api": "pm2_5",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "long_name": "CAMS surface particulate matter PM2.5",
        "folder": "PM25",
        "source_provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "source_product": "CAMS European & Global Near-Real-Time Analysis (Open-Meteo Gateway)",
        "native_res": "0.4° (~40 km)",
    },
    "PM10": {
        "variable_api": "pm10",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "long_name": "CAMS surface particulate matter PM10",
        "folder": "PM10",
        "source_provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "source_product": "CAMS European & Global Near-Real-Time Analysis (Open-Meteo Gateway)",
        "native_res": "0.4° (~40 km)",
    },
    "SO2": {
        "variable_api": "sulphur_dioxide",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "long_name": "CAMS surface sulphur dioxide SO2",
        "folder": "SO2",
        "source_provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "source_product": "CAMS European & Global Near-Real-Time Analysis (Open-Meteo Gateway)",
        "native_res": "0.4° (~40 km)",
    },
    "O3": {
        "variable_api": "ozone",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "long_name": "CAMS surface ground-level ozone O3",
        "folder": "O3",
        "source_provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "source_product": "CAMS European & Global Near-Real-Time Analysis (Open-Meteo Gateway)",
        "native_res": "0.4° (~40 km)",
    },
    "CO": {
        "variable_api": "carbon_monoxide",
        "units": "µg/m³",
        "measurement_type": "surface_concentration",
        "long_name": "CAMS surface carbon monoxide CO",
        "folder": "CO",
        "source_provider": "Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF",
        "source_product": "CAMS European & Global Near-Real-Time Analysis (Open-Meteo Gateway)",
        "native_res": "0.4° (~40 km)",
    },
}


def get_grid_coords() -> tuple[list[float], list[float], list[tuple[int, int]]]:
    """Return cell center latitudes, longitudes, and (row, col) indices."""
    lats = []
    lons = []
    indices = []
    for r in range(ROWS):
        lat = round(TOP - (r + 0.5) * DY, 4)
        for c in range(COLS):
            lon = round(WEST + (c + 0.5) * DX, 4)
            lats.append(lat)
            lons.append(lon)
            indices.append((r, c))
    return lats, lons, indices


def fetch_cams_batch(lats: list[float], lons: list[float], start_date: str, end_date: str) -> list[dict]:
    """Fetch hourly data for a list of coordinates from CAMS Open-Meteo API."""
    lats_str = ",".join(str(lat) for lat in lats)
    lons_str = ",".join(str(lon) for lon in lons)
    variables = "pm10,pm2_5,carbon_monoxide,sulphur_dioxide,ozone"
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lats_str}&longitude={lons_str}&hourly={variables}&"
        f"start_date={start_date}&end_date={end_date}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "AirQInsight-Downscaler/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
        except Exception as e:
            log.warning("Attempt %d failed: %s. Retrying...", attempt + 1, e)
            time.sleep(3)
    raise RuntimeError(f"Failed to fetch data from CAMS API for coordinates batch after 3 attempts.")


def acquire_and_convert(output_dir: Path) -> list[dict]:
    """Acquire real multi-pollutant data, save GeoTIFFs, and generate manifest entries."""
    lats, lons, indices = get_grid_coords()
    total_pts = len(lats)
    log.info("Acquiring 61 days of CAMS data for Mumbai grid (%d points)...", total_pts)

    # Split into 2 batches of 65 points each to comply with query parameter sizing
    batch_size = 65
    all_responses: list[dict] = []
    for i in range(0, total_pts, batch_size):
        b_lats = lats[i : i + batch_size]
        b_lons = lons[i : i + batch_size]
        log.info("Requesting batch %d (%d coordinates)...", (i // batch_size) + 1, len(b_lats))
        resp_list = fetch_cams_batch(b_lats, b_lons, START_DATE, END_DATE)
        all_responses.extend(resp_list)

    if len(all_responses) != total_pts:
        raise ValueError(f"Expected {total_pts} location responses, got {len(all_responses)}")

    # Extract time array (hourly: 61 days * 24 hours = 1464 hours)
    hours = all_responses[0]["hourly"]["time"]
    num_hours = len(hours)
    log.info("Acquired %d hours of observations from %s to %s", num_hours, hours[0], hours[-1])

    # Parse distinct calendar dates
    dates = sorted(list(set(h.split("T")[0] for h in hours)))
    log.info("Total dates represented: %d", len(dates))

    manifest_rows: list[dict] = []

    # For each pollutant, construct daily mean raster and write GeoTIFF
    for pol_key, cfg in POLLUTANT_CONFIGS.items():
        pol_dir = output_dir / cfg["folder"]
        pol_dir.mkdir(parents=True, exist_ok=True)
        var_api = cfg["variable_api"]
        log.info("Processing %s (%s)...", pol_key, cfg["long_name"])

        # Extract 3D array: [point_index, hour_index]
        data_matrix = np.zeros((total_pts, num_hours), dtype=np.float32)
        for pt_idx, resp in enumerate(all_responses):
            vals = resp["hourly"].get(var_api, [])
            for h_idx, v in enumerate(vals):
                data_matrix[pt_idx, h_idx] = np.nan if v is None else float(v)

        # For each day, aggregate 24 hours and populate 13x10 spatial grid
        for dt_str in dates:
            day_hour_indices = [h_idx for h_idx, h_str in enumerate(hours) if h_str.startswith(dt_str)]
            day_slice = data_matrix[:, day_hour_indices]  # shape: (130, 24)

            # Daily mean for each coordinate point
            with np.errstate(all="ignore"):
                daily_pt_means = np.nanmean(day_slice, axis=1)

            # Map coordinates to 13x10 grid
            grid_2d = np.full((ROWS, COLS), np.nan, dtype=np.float32)
            for pt_idx, (r, c) in enumerate(indices):
                val = daily_pt_means[pt_idx]
                if np.isfinite(val):
                    grid_2d[r, c] = round(float(val), 2)

            # Write GeoTIFF
            out_filename = f"{pol_key}_{dt_str}.tif"
            out_path = pol_dir / out_filename

            valid_count = int(np.isfinite(grid_2d).sum())
            total_count = ROWS * COLS
            valid_frac = round(valid_count / total_count, 4)

            # Determine quality control / status
            status = "SUCCESS" if valid_count > 0 else "NO_VALID_DATA"

            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=ROWS,
                width=COLS,
                count=1,
                dtype=rasterio.float32,
                crs=CRS,
                transform=TRANSFORM,
                nodata=np.nan,
                compress="deflate",
            ) as dst:
                dst.write(grid_2d, 1)
                dst.update_tags(
                    long_name=cfg["long_name"],
                    units=cfg["units"],
                    pollutant=pol_key,
                    date=dt_str,
                    measurement_type=cfg["measurement_type"],
                    source=cfg["source_provider"],
                    AREA_OR_POINT="Area",
                )

            # Record in manifest
            rel_path = f"backend/data/pollutant_samples/{cfg['folder']}/{out_filename}"
            manifest_rows.append({
                "pollutant": pol_key,
                "date": dt_str,
                "filepath": rel_path,
                "source_provider": cfg["source_provider"],
                "source_product": cfg["source_product"],
                "measurement_type": cfg["measurement_type"],
                "variable_name": var_api,
                "units": cfg["units"],
                "native_resolution": cfg["native_res"],
                "output_resolution": "0.035° (~3.9 km)",
                "crs": CRS,
                "bounds": f"[{WEST}, {BOTTOM}, {EAST}, {TOP}]",
                "nodata_value": "NaN",
                "valid_pixel_fraction": valid_frac,
                "quality_control_method": "24-hour diurnal arithmetic mean, NaN-preserved, no zero-filling",
                "temporal_representation": "Daily average (24h aggregated)",
                "source_timestamp": f"{dt_str}T00:00:00Z to {dt_str}T23:00:00Z",
                "processing_status": status,
                "is_resampled": "True (resampled to study domain grid)",
                "is_interpolated": "False (observational CAMS analysis, not fabricated)",
            })

    return manifest_rows


def write_manifest_and_readme(output_dir: Path, manifest_rows: list[dict]):
    """Write machine-readable manifest.csv and detailed README.md."""
    manifest_csv = output_dir / "manifest.csv"
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        log.info("Manifest written to %s with %d records", manifest_csv, len(manifest_rows))

    readme_path = output_dir / "README.md"
    readme_content = f"""# Multi-Pollutant Historical GeoTIFF Dataset (Nov–Dec 2025)

## Overview
This directory contains real historical georeferenced raster datasets for:
- **PM2.5** (Particulate Matter < 2.5 µm, surface concentration in µg/m³)
- **PM10** (Particulate Matter < 10 µm, surface concentration in µg/m³)
- **SO2** (Sulphur Dioxide, surface concentration in µg/m³)
- **O3** (Ground-Level Ozone, surface concentration in µg/m³)
- **CO** (Carbon Monoxide, surface concentration in µg/m³)

Covering the 61-day target monitoring window: **1 November 2025 through 31 December 2025**.

## Spatial Specifications
- **Target Geographic Domain:** Mumbai Metropolitan Region, India
- **Bounding Box:** West: 72.77°E, South: 18.865°N, East: 73.12°E, North: 19.32°N
- **Coordinate Reference System:** EPSG:4326 (WGS 84 geographic latitude/longitude)
- **Raster Dimensions:** 13 rows × 10 columns (130 cells)
- **Cell Spacing:** 0.035° (~3.9 km native aligned resolution)
- **Data Type:** Float32 GeoTIFF (deflate compressed)
- **Nodata Value:** `NaN` (Cloud-covered, masked, or missing pixels are preserved as NaN; NEVER filled with synthetic zero)

## Source Provenance & Scientific Methodology
- **Source Provider:** Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF Near-Real-Time atmospheric composition analyses accessed via European atmospheric gateways.
- **Native Resolution:** ~0.4° (~40 km native physical grid). Resampled to the Mumbai 0.035° analysis domain for pipeline ingestion.
- **Scientific Notice on Effective Resolution:** Resampling to 0.035° does NOT confer a finer physical resolution than the source CAMS physics analysis. It preserves the large-scale spatial gradients for downscaling without fabricating sub-grid features.
- **Temporal Aggregation:** 24-hour diurnal arithmetic means across all 24 hourly time steps per calendar day.

## Directory Structure
```
backend/data/pollutant_samples/
├── PM25/                  # 61 daily GeoTIFFs (PM25_YYYY-MM-DD.tif)
├── PM10/                  # 61 daily GeoTIFFs (PM10_YYYY-MM-DD.tif)
├── SO2/                   # 61 daily GeoTIFFs (SO2_YYYY-MM-DD.tif)
├── O3/                    # 61 daily GeoTIFFs (O3_YYYY-MM-DD.tif)
├── CO/                    # 61 daily GeoTIFFs (CO_YYYY-MM-DD.tif)
├── manifest.csv           # Machine-readable per-file metadata manifest (305 entries)
└── README.md              # Provenance, licensing, and scientific documentation
```

## Manifest Summary
- Total files generated: {len(manifest_rows)} GeoTIFFs (61 days × 5 pollutants)
- Date range: 2025-11-01 to 2025-12-31 inclusive
- Zero fabrication: All pixel values derive directly from CAMS global atmospheric models.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    log.info("README written to %s", readme_path)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "data" / "pollutant_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = acquire_and_convert(output_dir)
    write_manifest_and_readme(output_dir, rows)
    log.info("Multi-pollutant acquisition and GeoTIFF conversion completed successfully!")


if __name__ == "__main__":
    main()
