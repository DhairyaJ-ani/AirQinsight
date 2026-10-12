# Sentinel-5P test samples (8 images)

Eight real Sentinel-5P tropospheric NO₂ images over Mumbai, for fast tests of the main model's input path
(loading, cloud-gap handling, grid checks). Not for training: 8 days is far below the 30-day window the
gap-filler and downscaler learn from. Use the full 61-day set in `../test_data/` for pipeline runs.

## Where they come from

Byte-for-byte copies of files in `backend/data/test_data/` (Sentinel-5P OFFL Level-3 NO₂,
`COPERNICUS/S5P/OFFL/L3_NO2`, daily composite at the ~13:30 local overpass, via Google Earth Engine; pixels with
cloud fraction ≥ 0.3 removed). A test (`tests/test_sentinel_samples.py`) checks they still match the originals.

## Which days and why

One image a week, 5 Nov – 24 Dec 2025, chosen to cover clear and partly cloudy conditions (see `manifest.csv`):

| Date | Cloud-masked | Mean NO₂ (µmol/m²) | Purpose |
|---|---|---|---|
| 2025-11-05 | 17 % | 133.2 | partly cloudy |
| 2025-11-12 | 8 % | 80.0 | low NO₂, slight cloud |
| 2025-11-19 | 0 % | 132.8 | clear |
| 2025-11-26 | 2 % | 160.9 | clear |
| 2025-12-02 | 35 % | 118.4 | gap-filling test (largest gap) |
| 2025-12-10 | 0 % | 124.4 | clear |
| 2025-12-17 | 0 % | 158.3 | clear |
| 2025-12-24 | 0 % | 114.8 | clear |

## Format

| Property | Value |
|---|---|
| Files | `no2_raw_coarse_<YYYY-MM-DD>.tif` |
| Quantity / units | tropospheric NO₂ column, µmol/m² |
| Grid | EPSG:4326, 0.035° pixels (~3.9 km), 10 columns × 13 rows, top row = north |
| Area | 72.77–73.12 °E, 18.865–19.32 °N |
| No data | NaN where clouds blocked the satellite |

## Use

```python
from ml_engine.ingestion.files import load_no2_geotiffs

dates, no2, grid = load_no2_geotiffs("data/sentinel_test_samples")
# The loader returns one layer per calendar day from the first to the last file: no2 is (50, 13, 10)
# for 2025-11-05..2025-12-24. The 8 sample days hold data (NaN under clouds); the other 42 days are all NaN,
# exactly like a fully cloudy day, so the gap-filler treats them as missing.
```

A pipeline run on files (needs an Earth Engine login for weather and land-use layers) is
`uv run python -m ml_engine --source files --input-dir data/sentinel_test_samples --ee-project <project>`. It has
not been run on this 8-day subset, and results from so few days are not representative.
