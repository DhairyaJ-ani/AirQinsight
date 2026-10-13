# Multi-Pollutant Historical GeoTIFF Dataset (Nov–Dec 2025)

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
- Total files generated: 305 GeoTIFFs (61 days × 5 pollutants)
- Date range: 2025-11-01 to 2025-12-31 inclusive
- Zero fabrication: All pixel values derive directly from CAMS global atmospheric models.
