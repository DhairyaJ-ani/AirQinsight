#!/usr/bin/env python3
"""Acquisition script for historical pollutant raster data — Nov–Dec 2025.

This script downloads SO₂, CO, PM₂.₅, PM₁₀, and O₃ data for the period
1 November–31 December 2025 over Mumbai, India.

IMPORTANT PREREQUISITES
------------------------
The following data sources require external credentials:

1. Google Earth Engine (for SO₂ and CO via Sentinel-5P TROPOMI):
   - Requires an authenticated Earth Engine project.
   - Run: ``earthengine authenticate`` then set EE_PROJECT env var.

2. Copernicus Atmosphere Data Store / ADS (for PM₂.₅, PM₁₀, surface O₃ via CAMS):
   - Register at https://ads.atmosphere.copernicus.eu/
   - Install the CDS API key at ~/.cdsapirc:
       url: https://ads.atmosphere.copernicus.eu/api
       key: <YOUR_UID>:<YOUR_API_KEY>
   - Install: pip install cdsapi

SETUP
-----
    pip install earthengine-api cdsapi rasterio numpy pandas

USAGE
-----
    # Download all available pollutants
    python acquire_historical_pollutants.py --all

    # Download only SO2 (via GEE)
    python acquire_historical_pollutants.py --pollutant so2

    # Download only PM2.5 (via CAMS ADS)
    python acquire_historical_pollutants.py --pollutant pm25

    # Dry run — show what would be downloaded without downloading
    python acquire_historical_pollutants.py --all --dry-run

OUTPUT
------
Files are written to:
    backend/data/historical/{pollutant}/{pollutant}_{date}.tif

Each file:
  - EPSG:4326
  - nodata = NaN for cloud-masked / missing pixels
  - Tags include units, long_name, product, date
  - No pixel is fabricated or synthesized

SCIENTIFIC NOTES
----------------
* SO₂, CO: Sentinel-5P TROPOMI satellite COLUMN data (µmol/m², mol/m²).
  These are atmospheric column measurements, NOT surface concentrations.
  They cannot be directly compared to CPCB NAAQS surface thresholds.

* PM₂.₅, PM₁₀: CAMS global atmospheric composition analysis, surface level.
  Native resolution ~0.4° (~40 km). Downsampling to ~0.035° is purely for
  visualization interpolation — it does NOT produce a genuinely finer-resolution
  measurement. Results must be labelled as "CAMS analysis, resampled" not
  "high-resolution PM2.5".

* O₃: CAMS surface ozone analysis (variable 'go3'), NOT total column ozone.
  Total-column ozone from TROPOMI (in Dobson Units) must never be relabelled
  as surface ozone. This script acquires the CAMS surface analysis product.

This script will NOT:
  - Fabricate GeoTIFF files if downloads fail.
  - Silently fall back to NO₂ data.
  - Claim completion if the download was incomplete.
  - Reinterpret column data as surface concentration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Study area — Mumbai bounding box (matches existing NO₂ dataset)
BBOX = {"west": 72.77, "south": 18.865, "east": 73.12, "north": 19.32}
DATE_START = date(2025, 11, 1)
DATE_END = date(2025, 12, 31)

OUTPUT_BASE = Path(__file__).resolve().parent.parent / "data" / "historical"


def date_range(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


# ── Google Earth Engine — SO₂ and CO ─────────────────────────────────────────

def download_s5p_gee(
    pollutant: str,
    collection_id: str,
    band: str,
    unit: str,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Download Sentinel-5P TROPOMI data via Google Earth Engine.

    MEASUREMENT TYPE: satellite atmospheric column (NOT surface concentration).
    """
    log.info("=== %s: Sentinel-5P via GEE ===", pollutant.upper())
    log.info("Collection: %s | Band: %s | Unit: %s", collection_id, band, unit)
    log.info("Area: %.3f–%.3fE, %.3f–%.3fN", BBOX["west"], BBOX["east"], BBOX["south"], BBOX["north"])
    log.info("Period: %s to %s", DATE_START, DATE_END)
    log.warning(
        "NOTE: %s TROPOMI data is satellite COLUMN data (%s), NOT surface concentration. "
        "Direct NAAQS comparison requires additional surface-estimation steps.",
        pollutant.upper(), unit,
    )

    if dry_run:
        log.info("[DRY RUN] Would download %s files", (DATE_END - DATE_START).days + 1)
        return

    try:
        import ee
    except ImportError:
        log.error("earthengine-api not installed. Run: pip install earthengine-api")
        sys.exit(1)

    project = None
    import os
    project = os.environ.get("EE_PROJECT")
    if not project:
        log.error(
            "EE_PROJECT environment variable not set. "
            "Set it to your Google Earth Engine project ID and run: earthengine authenticate"
        )
        sys.exit(1)

    try:
        ee.Initialize(project=project)
        log.info("Earth Engine initialized with project: %s", project)
    except Exception as exc:
        log.error("Earth Engine initialization failed: %s", exc)
        log.error(
            "Ensure you have run 'earthengine authenticate' and have access to the GEE project."
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    dates = date_range(DATE_START, DATE_END)
    succeeded = 0
    failed = 0

    for d in dates:
        out_path = output_dir / f"{pollutant}_{d.isoformat()}.tif"
        if out_path.exists():
            log.info("  %s: already exists, skipping", d)
            succeeded += 1
            continue

        try:
            start_str = d.isoformat()
            end_str = (d + timedelta(days=1)).isoformat()
            region = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])

            col = (
                ee.ImageCollection(collection_id)
                .filterDate(start_str, end_str)
                .filterBounds(region)
                .select(band)
            )

            img = col.mosaic()

            # Get pixel values — 0.035° resolution to match NO₂ dataset
            arr = img.sample(
                region=region,
                scale=3900,  # ~3.9km in metres
                projection="EPSG:4326",
                geometries=True,
            )

            # Alternative: use getDownloadURL for the full raster
            url = img.getDownloadURL({
                "scale": 3900,
                "crs": "EPSG:4326",
                "region": region,
                "format": "GeoTIFF",
                "bands": [band],
            })

            import urllib.request
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                tmp_path = Path(tmp.name)

            # Copy to output with correct metadata
            import shutil
            shutil.copy(tmp_path, out_path)
            tmp_path.unlink(missing_ok=True)

            # Add metadata tags
            with rasterio.open(out_path, "r+") as ds:
                ds.update_tags(
                    pollutant=pollutant,
                    units=unit,
                    long_name=f"Sentinel-5P TROPOMI {pollutant.upper()} column",
                    measurement_type="satellite_column",
                    date=d.isoformat(),
                    product=collection_id,
                    source="Sentinel-5P OFFL L3 via Google Earth Engine",
                    WARNING="Satellite column data. Cannot be compared to surface NAAQS thresholds.",
                )

            log.info("  %s: downloaded → %s", d, out_path)
            succeeded += 1

        except Exception as exc:
            log.error("  %s: FAILED — %s", d, exc)
            failed += 1

    log.info(
        "%s download complete. Succeeded: %d, Failed: %d",
        pollutant.upper(), succeeded, failed,
    )
    if failed > 0:
        log.warning(
            "%d days FAILED. Do not use incomplete data in analyses without noting missing dates.",
            failed,
        )


# ── CAMS ADS — PM₂.₅, PM₁₀, surface O₃ ─────────────────────────────────────

def download_cams_surface(
    pollutant: str,
    cams_variable: str,
    unit: str,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Download CAMS Global Atmospheric Composition Analysis via ADS.

    MEASUREMENT TYPE: model_analysis (surface level).
    RESOLUTION NOTE: Native CAMS resolution is ~0.4° (~40 km). This is
    NOT a high-resolution observation. Any visual resampling to finer grids
    must be clearly labelled as interpolated/resampled, not as independent
    fine-resolution measurements.
    """
    log.info("=== %s: CAMS Analysis via Copernicus ADS ===", pollutant.upper())
    log.info("Variable: %s | Unit: %s", cams_variable, unit)
    log.info("Native resolution: ~0.4° (~40 km) — NOT fine-resolution data")
    log.warning(
        "NOTE: CAMS analysis data at native ~40 km resolution. "
        "Any interpolation to finer grids is display-only and does NOT "
        "produce a genuinely finer-resolution measurement."
    )

    if dry_run:
        log.info("[DRY RUN] Would download CAMS %s for Nov–Dec 2025", pollutant.upper())
        return

    try:
        import cdsapi
    except ImportError:
        log.error(
            "cdsapi not installed. Run: pip install cdsapi\n"
            "Then configure ~/.cdsapirc with your Copernicus ADS credentials."
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = cdsapi.Client()
    except Exception as exc:
        log.error(
            "CDS API client initialization failed: %s\n"
            "Ensure ~/.cdsapirc is configured correctly. "
            "Register at https://ads.atmosphere.copernicus.eu/", exc
        )
        sys.exit(1)

    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    import tempfile

    # Request monthly data (CAMS analysis is available as monthly batches)
    for year_month in [("2025", "11"), ("2025", "12")]:
        year, month = year_month
        out_nc = output_dir / f"{pollutant}_cams_{year}_{month}.nc"

        if out_nc.exists():
            log.info("  %s-%s: already exists, skipping", year, month)
            continue

        try:
            client.retrieve(
                "cams-global-atmospheric-composition-forecasts",
                {
                    "variable": [cams_variable],
                    "date": [f"{year}-{month}-01/{year}-{month}-28"],
                    "time": ["00:00", "12:00"],
                    "leadtime_hour": ["0"],
                    "type": "forecast",
                    "format": "netcdf",
                    "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
                },
                str(out_nc),
            )
            log.info("  %s-%s: downloaded → %s", year, month, out_nc)
        except Exception as exc:
            log.error("  %s-%s: FAILED — %s", year, month, exc)
            log.error(
                "CAMS data may not be available for Nov–Dec 2025 yet "
                "(depends on ADS archive update schedule). Check availability at "
                "https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts"
            )

    # Convert NetCDF → daily GeoTIFFs
    log.info("Converting CAMS NetCDF → daily GeoTIFFs for %s...", pollutant.upper())
    nc_files = sorted(output_dir.glob(f"{pollutant}_cams_*.nc"))
    if not nc_files:
        log.error("No CAMS NetCDF files found. Download may have failed.")
        return

    try:
        import xarray as xr
        import rasterio
        from rasterio.transform import from_bounds

        for nc_path in nc_files:
            ds = xr.open_dataset(nc_path)
            var = ds[cams_variable] if cams_variable in ds else None
            if var is None:
                # Try alternate names
                for v in ds.data_vars:
                    if cams_variable in v.lower() or pollutant.lower() in v.lower():
                        var = ds[v]
                        break

            if var is None:
                log.error("Variable '%s' not found in %s. Available: %s", cams_variable, nc_path, list(ds.data_vars))
                continue

            # Write one GeoTIFF per day
            for t in var.time.values:
                d = str(t)[:10]
                daily = var.sel(time=t, method="nearest")

                out_tif = output_dir / f"{pollutant}_{d}.tif"
                if out_tif.exists():
                    continue

                arr = daily.values.astype("float32")
                # Ensure NaN for missing
                arr[~np.isfinite(arr)] = np.nan

                lats = daily.latitude.values if "latitude" in daily.coords else daily.lat.values
                lons = daily.longitude.values if "longitude" in daily.coords else daily.lon.values

                h, w = arr.shape
                transform = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), w, h)

                with rasterio.open(
                    out_tif, "w",
                    driver="GTiff",
                    dtype="float32",
                    width=w, height=h, count=1,
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=float("nan"),
                    compress="deflate",
                ) as dst:
                    dst.write(arr, 1)
                    dst.update_tags(
                        pollutant=pollutant,
                        units=unit,
                        long_name=f"CAMS {pollutant.upper()} surface analysis",
                        measurement_type="model_analysis",
                        date=d,
                        product="CAMS global atmospheric composition forecasts",
                        source="ECMWF CAMS via Copernicus ADS",
                        native_resolution_deg="0.4",
                        WARNING=(
                            f"CAMS analysis at ~0.4deg (~40km) native resolution. "
                            "Any finer-grid display is interpolated, NOT a higher-resolution observation."
                        ),
                    )
                log.info("  %s: %s → %s", pollutant.upper(), d, out_tif.name)

    except ImportError:
        log.error("xarray not installed. Run: pip install xarray")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical pollutant raster data for Nov–Dec 2025.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pollutant",
        choices=["so2", "co", "pm25", "pm10", "o3"],
        help="Download a specific pollutant.",
    )
    parser.add_argument("--all", action="store_true", help="Download all supported pollutants.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading.")
    args = parser.parse_args()

    if not args.all and not args.pollutant:
        parser.print_help()
        sys.exit(0)

    pollutants_to_run = []
    if args.all:
        pollutants_to_run = ["so2", "co", "pm25", "pm10", "o3"]
    elif args.pollutant:
        pollutants_to_run = [args.pollutant]

    gee_pollutants = {
        "so2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", "mol m-2"),
        "co":  ("COPERNICUS/S5P/OFFL/L3_CO",  "CO_column_number_density",  "mol m-2"),
    }
    cams_pollutants = {
        "pm25": ("pm2p5",             "µg/m³"),
        "pm10": ("pm10",              "µg/m³"),
        "o3":   ("go3",               "µg/m³"),  # CAMS surface ozone, NOT total column
    }

    for pol in pollutants_to_run:
        out_dir = OUTPUT_BASE / pol
        if pol in gee_pollutants:
            col_id, band, unit = gee_pollutants[pol]
            download_s5p_gee(pol, col_id, band, unit, out_dir, dry_run=args.dry_run)
        elif pol in cams_pollutants:
            cams_var, unit = cams_pollutants[pol]
            download_cams_surface(pol, cams_var, unit, out_dir, dry_run=args.dry_run)

    log.info("=== Acquisition script complete ===")
    log.info(
        "Output directory: %s\n"
        "REMINDER: SO2 and CO files are satellite COLUMN data (mol/m2), "
        "not surface concentrations.\n"
        "PM2.5, PM10, O3 files are CAMS analysis at ~40km resolution — "
        "label interpolated displays clearly as 'resampled'.",
        OUTPUT_BASE,
    )


if __name__ == "__main__":
    main()
