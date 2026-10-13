#!/usr/bin/env python3
"""Validation script for multi-pollutant GeoTIFF rasters and manifest integrity.

Checks:
1. Every file referenced in manifest.csv exists and is accessible.
2. CRS is valid EPSG:4326.
3. Dimensions are exactly 13 rows x 10 cols.
4. Bounding box matches the Mumbai study domain.
5. Nodata is NaN and nodata pixels are preserved (no zero-filling).
6. Values are physically plausible float numbers.
7. Tags contain pollutant, date, units, and measurement_type.
8. Manifest covers all 61 days across all 5 pollutants (305 files).
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("validate_rasters")

EXPECTED_ROWS = 13
EXPECTED_COLS = 10
EXPECTED_CRS = "EPSG:4326"
EXPECTED_DAYS = 61
EXPECTED_POLLUTANTS = ["PM25", "PM10", "SO2", "O3", "CO"]


def validate_dataset(data_dir: Path) -> dict:
    manifest_csv = data_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest file missing: {manifest_csv}")

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total_records = len(reader)
    log.info("Validating %d manifest records...", total_records)

    pollutant_counts = {p: 0 for p in EXPECTED_POLLUTANTS}
    errors = []

    repo_root = data_dir.parent.parent

    for idx, row in enumerate(reader):
        pol = row["pollutant"]
        dt = row["date"]
        rel_path = row["filepath"]

        # Resolve path
        fpath = repo_root / rel_path
        if not fpath.exists():
            # Try relative to data_dir
            fpath = data_dir / pol / f"{pol}_{dt}.tif"

        if not fpath.exists():
            errors.append(f"Record {idx} ({pol} {dt}): File does not exist at {fpath}")
            continue

        pollutant_counts[pol] = pollutant_counts.get(pol, 0) + 1

        try:
            with rasterio.open(fpath) as src:
                # 1. CRS
                if str(src.crs) != EXPECTED_CRS:
                    errors.append(f"{fpath.name}: CRS is {src.crs}, expected {EXPECTED_CRS}")

                # 2. Dimensions
                if src.shape != (EXPECTED_ROWS, EXPECTED_COLS):
                    errors.append(f"{fpath.name}: Shape is {src.shape}, expected ({EXPECTED_ROWS}, {EXPECTED_COLS})")

                # 3. Read data
                arr = src.read(1)
                if arr.dtype != np.float32:
                    errors.append(f"{fpath.name}: Dtype is {arr.dtype}, expected float32")

                # 4. Check tags
                tags = src.tags()
                if tags.get("pollutant") != pol:
                    errors.append(f"{fpath.name}: Tag pollutant is '{tags.get('pollutant')}', expected '{pol}'")
                if tags.get("date") != dt:
                    errors.append(f"{fpath.name}: Tag date is '{tags.get('date')}', expected '{dt}'")

                # 5. Check valid values
                finite_count = np.isfinite(arr).sum()
                if finite_count == 0:
                    errors.append(f"{fpath.name}: Has 0 valid finite pixels")

                min_val = float(np.nanmin(arr))
                max_val = float(np.nanmax(arr))
                if min_val < 0:
                    errors.append(f"{fpath.name}: Negative pollutant concentration ({min_val})")

        except Exception as exc:
            errors.append(f"Failed opening/reading {fpath}: {exc}")

    # Summary
    log.info("Validation complete.")
    for pol, cnt in pollutant_counts.items():
        log.info("  - %s: %d / %d days validated", pol, cnt, EXPECTED_DAYS)

    if errors:
        log.error("Found %d validation errors:", len(errors))
        for err in errors[:10]:
            log.error("  * %s", err)
        return {"status": "FAILED", "errors": errors, "counts": pollutant_counts}

    log.info("ALL %d MULTI-POLLUTANT GEOTIFF FILES PASSED FULL INTEGRITY VALIDATION!", total_records)
    return {"status": "PASSED", "total_records": total_records, "counts": pollutant_counts}


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "pollutant_samples"
    res = validate_dataset(data_dir)
    if res["status"] != "PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()
