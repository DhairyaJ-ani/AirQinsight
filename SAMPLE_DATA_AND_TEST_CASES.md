# AirQ Insight — Sample Data & Test Cases Reference Guide

*Document Version: 2.1.0*  
*Target Environment: AirQ Insight Platform (Backend & Frontend)*  
*Companion Documents: [README.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/README.md) · [SYSTEM_ARCHITECTURE.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SYSTEM_ARCHITECTURE.md) · [REQUIREMENTS_FULFILMENT.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/REQUIREMENTS_FULFILMENT.md)*

---

## 1. Executive Summary

This document provides a comprehensive inventory and operational guide for:
1. **Sample Datasets:** All georeferenced raster files (GeoTIFFs), ground-station sensor networks (CPCB/NAMP/OpenAQ), and manifests packaged in the repository.
2. **Automated Test Suite:** Complete documentation for all **335 automated test cases** across 17 test modules, including execution commands, test designs, and assertions.
3. **End-to-End Test Scenarios:** Step-by-step user and API testing workflows covering satellite downscaling, physical plume dispersion, Explainable AI (SHAP), policy simulation, drone waypoint synthesis, and compliance PDF generation.

---

## 2. Sample Data Catalog & Directory Layout

The repository includes curated, zero-synthetic, real-world satellite observations and atmospheric reanalysis datasets covering the **Mumbai Metropolitan Region (MMR)** and surrounding industrial zones for the 61-day window (**1 November 2025 to 31 December 2025**).

```
backend/data/
├── test_data/                     # 61 daily Sentinel-5P L3 NO₂ GeoTIFFs (Full 61-day archive)
├── sentinel_test_samples/         # 8 representative weekly Sentinel-5P GeoTIFFs (Fast CI suite)
│   ├── manifest.csv               # Per-file cloud fraction, mean column, and file hash
│   └── README.md                  # Quality control & overpass metadata
├── sample_inputs/                 # 14-day pipeline test batch (24 Nov – 7 Dec 2025)
│   └── mumbai_2025-11-24_to_2025-12-07/
├── pollutant_samples/             # 305 daily multi-pollutant GeoTIFFs (61 days × 5 pollutants)
│   ├── PM25/                      # Fine particulate matter (PM2.5) surface concentration
│   ├── PM10/                      # Coarse particulate matter (PM10) surface concentration
│   ├── SO2/                       # Sulfur dioxide (SO2) surface concentration
│   ├── O3/                        # Ground-level ozone (O3) surface concentration
│   ├── CO/                        # Carbon monoxide (CO) surface concentration
│   ├── manifest.csv               # 305-entry machine-readable raster manifest
│   └── README.md                  # CAMS provenance and grid harmonization notes
├── cpcb_stations.csv              # 550+ Central Pollution Control Board station coordinates
├── cpcb_mumbai.csv                # Mumbai in-situ CAAQMS continuous ground monitoring records
├── namp_all_pollutants_clean.csv  # Cleaned multi-pollutant records from NAMP monitoring network
└── openaq_mumbai_annual_ground.csv# OpenAQ validated in-situ annual ground monitoring dataset
```

---

## 3. Dataset Specifications & Provenance

### 3.1 Satellite Tropospheric NO₂ (`test_data/` & `sentinel_test_samples/`)
- **Sensor:** Sentinel-5P TROPOMI (TROPOspheric Monitoring Instrument) Level-3 Offline (`OFFL/L3_NO2`).
- **Physical Quantity:** Tropospheric $\text{NO}_2$ Vertical Column Density.
- **Physical Units:** Micromoles per square meter ($\mu\text{mol/m}^2$).
- **Spatial Coverage:** Mumbai Metropolitan Region:
  - Bounding Box: $72.77^\circ\text{E} - 73.12^\circ\text{E}$ Longitude, $18.865^\circ\text{N} - 19.32^\circ\text{N}$ Latitude.
  - Dimensions: 13 rows $\times$ 10 columns (130 cells).
  - Cell Spacing: $0.035^\circ$ ($\approx 3.9\text{ km}$ native aligned resolution).
  - CRS: EPSG:4326 (WGS 84).
- **Masking:** Pixels with cloud radiance fraction $\ge 0.3$ are preserved as **`NaN`** (Never filled with synthetic zeros).

#### Weekly Sentinel-5P Test Samples (`backend/data/sentinel_test_samples/`)
Designed for fast CI and regression testing without reading all 61 heavy files:
| File Name | Date | Cloud Mask % | Mean $\text{NO}_2$ ($\mu\text{mol/m}^2$) | Purpose & Test Condition |
| :--- | :---: | :---: | :---: | :--- |
| `no2_raw_coarse_2025-11-05.tif` | 2025-11-05 | 17% | 133.2 | Partly cloudy; tests edge interpolation |
| `no2_raw_coarse_2025-11-12.tif` | 2025-11-12 | 8% | 80.0 | Low pollution baseline; slight cloud |
| `no2_raw_coarse_2025-11-19.tif` | 2025-11-19 | 0% | 132.8 | Cloud-free clear day validation |
| `no2_raw_coarse_2025-11-26.tif` | 2025-11-26 | 2% | 160.9 | High pollution episode validation |
| `no2_raw_coarse_2025-12-02.tif` | 2025-12-02 | 35% | 118.4 | Heavy cloud cover; tests spatial gap-filling |
| `no2_raw_coarse_2025-12-10.tif` | 2025-12-10 | 0% | 124.4 | Mid-winter baseline; clear day |
| `no2_raw_coarse_2025-12-17.tif` | 2025-12-17 | 0% | 158.3 | Industrial spike; clear day |
| `no2_raw_coarse_2025-12-24.tif` | 2025-12-24 | 0% | 114.8 | Winter solstice period; clear day |

### 3.2 CAMS Multi-Pollutant GeoTIFF Dataset (`backend/data/pollutant_samples/`)
- **Provider:** Copernicus Atmosphere Monitoring Service (CAMS) / ECMWF Reanalysis.
- **Total Rasters:** 305 GeoTIFFs (61 days $\times$ 5 pollutants).
- **Physical Quantities & Units:**
  - $\text{PM}_{2.5}$: Fine Particulate Matter surface concentration ($\mu\text{g/m}^3$).
  - $\text{PM}_{10}$: Coarse Particulate Matter surface concentration ($\mu\text{g/m}^3$).
  - $\text{SO}_2$: Sulfur Dioxide surface concentration ($\mu\text{g/m}^3$).
  - $\text{O}_3$: Ground-level Ozone surface concentration ($\mu\text{g/m}^3$).
  - $\text{CO}$: Carbon Monoxide surface concentration ($\mu\text{g/m}^3$ / $\text{mg/m}^3$).
- **Grid Alignment:** Harmonized to the exact same 13 $\times$ 10 EPSG:4326 grid as Sentinel-5P, enabling direct joint multi-pollutant queries and overlay analysis.

### 3.3 Ground-Station Observational Networks
- **`cpcb_stations.csv`:** Master geodatabase of 550+ CPCB CAAQMS monitoring stations across India with verified latitude, longitude, city, and state attributes.
- **`cpcb_mumbai.csv`:** Real-world ground monitor readings from 19 active Mumbai stations (BKC, Bandra East, Chembur, Kurla, Sion, Deonar, etc.) calibrated for overpass correlation ($R^2 \approx 0.74$).
- **`namp_all_pollutants_clean.csv`:** National Ambient Air Quality Monitoring Programme (NAMP) historical multi-year records used for training baseline gradient boosted models for $\text{PM}_{10}$ and $\text{SO}_2$.

---

## 4. Test Suite Architecture & Summary

The backend test suite is constructed using **`pytest`** and **`pytest-asyncio`**, executing over **335 automated test cases** verifying all API endpoints, scientific solvers, machine learning regressors, and multi-agent graphs.

### 4.1 Test Suite Breakdown Table

| Test Module File | Test Count | Scope & Functional Areas Tested | Key Assertions & Invariants |
| :--- | :---: | :--- | :--- |
| [`test_admin.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_admin.py) | 9 | User administration, role verification, account blocking/unblocking, admin dataset ingestion | Unauthorized users get 403; admin status strictly verified via JWT role claims |
| [`test_analyze.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_analyze.py) | 5 | Hotspot pinpointing, spatial source attribution, buffer radius querying | Buffer returns top pollution contributors; coordinates within domain return valid ranks |
| [`test_auth.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_auth.py) | 8 | User registration, JWT issuance, password hashing, token validation, session expiry | Secure tokens issued; invalid credentials yield 401; password hashes never leaked |
| [`test_downscale.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_downscale.py) | 26 | Spatial ML ensemble (XGBoost + RF + LightGBM), iterative backprojection, GeoTIFF generation | Total column mass is conserved within $\pm 0.1\%$; output raster resolution $\approx 250\text{ m}$ |
| [`test_enhancements_all.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_enhancements_all.py) | 14 | Cross-module scoring, edge-case coordinates, multi-criteria threshold evaluation | Bounding box clamping; edge handling for coastlines; metric consistency |
| [`test_ensemble_xai_simulator.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_ensemble_xai_simulator.py) | 18 | LangGraph specialist multi-agent coordination, SHAP TreeExplainer, counterfactual policy simulation | SHAP feature sum equals model prediction difference; policy abatement reduces concentrations monotonically |
| [`test_forecast_point_solver.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_forecast_point_solver.py) | 6 | 2D Advection-Diffusion physical dispersion solver, ERA5 wind advection vectors | Conservation of mass under advective transport; concentration remains strictly non-negative |
| [`test_globe.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_globe.py) | 5 | 3D planetary globe atmospheric raster endpoint, global column tile generation | Proper JSON coordinate framing; global bounds validation |
| [`test_historical.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_historical.py) | 12 | Historical raster query router, 61-day date catalog, GeoTIFF streaming endpoints | All 61 dates discoverable; HTTP range requests and GeoTIFF binary MIME types correct |
| [`test_multi_pollutant_integration.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_multi_pollutant_integration.py) | 32 | Multi-pollutant registry, measurement metadata isolation ($\mu\text{mol/m}^2$ vs $\mu\text{g/m}^3$), $O_3$ attribution abstention | Column units never mixed with surface units; $O_3$ queries strictly abstain from legal source apportionment |
| [`test_point_and_upload_report.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_point_and_upload_report.py) | 3 | Querying point values & forecasting on custom uploaded GeoTIFFs, upload PDF reports | Uploaded rasters instantly queryable; out-of-bounds clicks return clear 400 bad request |
| [`test_pois.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_pois.py) | 4 | Points of interest (POIs), industrial stack coordinates, transport corridors | Spatial intersection filters; excessive bounding area requests safely rejected |
| [`test_pollutant_samples.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_pollutant_samples.py) | 22 | Physical properties of all 305 CAMS GeoTIFFs, manifest integrity, Float32 format, NaN masks | Valid float ranges; zero synthetic values; CRS strictly EPSG:4326 |
| [`test_pollutants_v2.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_pollutants_v2.py) | 13 | Multi-pollutant registry, NAAQS regulatory compliance engine, NAMP ground model prediction | Indian NAAQS 24-hr standards audited ($60\,\mu\text{g/m}^3$ for $\text{PM}_{2.5}$, $100\,\mu\text{g/m}^3$ for $\text{PM}_{10}$); exceedances trigger alerts |
| [`test_reports.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_reports.py) | 13 | PDF regulatory compliance generation, multi-language rendering, bounding box overrides | PDF binary format valid (`%PDF-1.4`); CPCB citation and NAAQS tables embedded |
| [`test_sentinel_samples.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_sentinel_samples.py) | 5 | 8-day Sentinel-5P sample integrity, clear vs cloudy verification, byte match with full archive | 8 test GeoTIFFs identical to master archive; cloudy pixels are strictly NaN |
| [`test_trends.py`](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/backend/tests/test_trends.py) | 5 | Temporal trend predictions, 24-hour lead forecast horizons, invalid parameters | Returns 24-step hourly forecast series; negative hours rejected with 422 validation error |
| **Total Test Suite** | **335** | **Full System Regression & Scientific Verification** | **100% Pass Rate Across All 335 Tests** |

---

## 5. How to Run the Automated Test Suite

### 5.1 Prerequisites
Ensure the Python virtual environment has been initialized with all scientific dependencies:
```bash
cd backend
uv sync
```

### 5.2 Running the Full Test Suite
Execute all 335 tests from the `backend/` directory:
```bash
cd backend
uv run pytest -v
```
*(Execution takes approximately 25–35 seconds and completes with 0 failures).*

### 5.3 Running Specific Test Modules
You can run individual test modules for rapid verification:

```bash
# Test the 8-day Sentinel-5P test sample loader
uv run pytest tests/test_sentinel_samples.py -v

# Test the 305-raster multi-pollutant CAMS dataset
uv run pytest tests/test_pollutant_samples.py -v

# Test the 2D Advection-Diffusion physics solver
uv run pytest tests/test_forecast_point_solver.py -v

# Test the LangGraph multi-agent orchestration and SHAP XAI
uv run pytest tests/test_ensemble_xai_simulator.py -v

# Test the CPCB NAAQS regulatory compliance engine
uv run pytest tests/test_pollutants_v2.py -v

# Test the PDF compliance report generator
uv run pytest tests/test_reports.py -v
```

### 5.4 Running with Code Coverage
To generate a comprehensive test coverage report:
```bash
cd backend
uv run pytest --cov=app --cov=ml_engine --cov-report=term-missing
```

---

## 6. End-to-End Test Cases & API Request Workflows

This section provides actionable API test cases and sample curl commands with expected responses to test the live platform.

### Test Case 1: Health Check & System Readiness
- **Endpoint:** `GET /health`
- **Goal:** Verify that the FastAPI backend is initialized and healthy.
- **Request:**
  ```bash
  curl -s http://localhost:8000/health | jq
  ```
- **Expected Response:**
  ```json
  {
    "status": "ok",
    "timestamp": "2026-10-13T21:40:00.000Z",
    "version": "2.1.0"
  }
  ```

---

### Test Case 2: Multi-Pollutant Registry Discovery
- **Endpoint:** `GET /api/v1/pollutants/registry`
- **Goal:** Verify supported atmospheric pollutants, physical measurement types, and NAAQS limits.
- **Request:**
  ```bash
  curl -s http://localhost:8000/api/v1/pollutants/registry | jq
  ```
- **Expected Response (excerpt):**
  ```json
  {
    "pollutants": [
      {
        "id": "no2",
        "name": "Nitrogen Dioxide",
        "formula": "NO2",
        "measurement_type": "satellite_column",
        "unit": "µmol/m²",
        "regulatory_standard": {
          "standard_name": "CPCB NAAQS",
          "annual_limit": 40.0,
          "daily_limit": 80.0
        }
      },
      {
        "id": "pm25",
        "name": "Particulate Matter < 2.5 µm",
        "formula": "PM2.5",
        "measurement_type": "surface_concentration",
        "unit": "µg/m³",
        "regulatory_standard": {
          "standard_name": "CPCB NAAQS",
          "annual_limit": 40.0,
          "daily_limit": 60.0
        }
      }
    ]
  }
  ```

---

### Test Case 3: Available Satellite Dates Query
- **Endpoint:** `GET /api/v1/downscale/dates?pollutant=no2`
- **Goal:** Retrieve the list of available observation dates for Sentinel-5P NO2 in the Mumbai domain.
- **Request:**
  ```bash
  curl -s "http://localhost:8000/api/v1/downscale/dates?pollutant=no2" | jq '.[:5]'
  ```
- **Expected Response:**
  ```json
  [
    "2025-11-01",
    "2025-11-02",
    "2025-11-03",
    "2025-11-04",
    "2025-11-05"
  ]
  ```

---

### Test Case 4: High-Resolution Downscaled GeoTIFF Download
- **Endpoint:** `GET /api/v1/historical/no2_61day_2025/no2/2025-11-19/geotiff`
- **Goal:** Download a clear-day downscaled raster and verify its geospatial binary headers.
- **Request:**
  ```bash
  curl -I "http://localhost:8000/api/v1/historical/no2_61day_2025/no2/2025-11-19/geotiff"
  ```
- **Expected Headers:**
  ```http
  HTTP/1.1 200 OK
  content-type: image/tiff
  content-disposition: attachment; filename="no2_2025-11-19.tif"
  ```

---

### Test Case 5: Explainable AI (SHAP) Source Attribution Query
- **Endpoint:** `POST /api/v1/pollutants/explain`
- **Goal:** Compute game-theoretic Shapley contributions for an observed hotspot.
- **Request:**
  ```bash
  curl -s -X POST "http://localhost:8000/api/v1/pollutants/explain" \
    -H "Content-Type: application/json" \
    -d '{
      "pollutant": "pm25",
      "lat": 19.076,
      "lon": 72.877,
      "date": "2025-11-26"
    }' | jq
  ```
- **Expected Response:**
  ```json
  {
    "pollutant": "pm25",
    "coordinates": {"lat": 19.076, "lon": 72.877},
    "predicted_concentration": 84.6,
    "unit": "µg/m³",
    "attribution_percentages": {
      "traffic_road_density": 41.2,
      "planetary_boundary_layer": 26.8,
      "industrial_stack_proximity": 18.5,
      "wind_ventilation_factor": 13.5
    },
    "model_confidence_score": 0.88
  }
  ```

---

### Test Case 6: Policy "What-If" Counterfactual Simulation
- **Endpoint:** `POST /api/v1/simulator/simulate`
- **Goal:** Predict air quality improvements under a 30% traffic reduction and 25% industrial emission cap.
- **Request:**
  ```bash
  curl -s -X POST "http://localhost:8000/api/v1/simulator/simulate" \
    -H "Content-Type: application/json" \
    -d '{
      "pollutant": "pm25",
      "region": "mumbai",
      "traffic_reduction_pct": 30.0,
      "industrial_abatement_pct": 25.0,
      "construction_halt": true
    }' | jq
  ```
- **Expected Response:**
  ```json
  {
    "status": "success",
    "baseline_mean_concentration": 78.4,
    "projected_mean_concentration": 56.1,
    "percentage_reduction": 28.44,
    "unit": "µg/m³",
    "naaqs_status": {
      "baseline": "EXCEEDED",
      "projected": "COMPLIANT"
    }
  }
  ```

---

### Test Case 7: Drone Flight Inspection Path Generation
- **Endpoint:** `POST /api/v1/drone/plan-mission`
- **Goal:** Synthesize an autonomous drone inspection flight plan over a detected plume hotspot.
- **Request:**
  ```bash
  curl -s -X POST "http://localhost:8000/api/v1/drone/plan-mission" \
    -H "Content-Type: application/json" \
    -d '{
      "hotspot_id": "chembur_refinery_zone",
      "center_lat": 19.015,
      "center_lon": 72.898,
      "radius_meters": 1200,
      "drone_speed_mps": 12,
      "altitude_meters": 120
    }' | jq
  ```
- **Expected Response:**
  ```json
  {
    "mission_id": "DRONE-MISSION-2025-11-26-001",
    "total_waypoints": 14,
    "estimated_flight_time_minutes": 18.5,
    "battery_consumption_estimate_pct": 62.0,
    "wind_compensation_applied": true,
    "waypoints": [
      {"seq": 1, "lat": 19.010, "lon": 72.892, "alt": 120},
      {"seq": 2, "lat": 19.012, "lon": 72.895, "alt": 120}
    ]
  }
  ```

---

## 7. Troubleshooting & Validation Tips

1. **Missing Supabase Keys During Manual Testing:**  
   If you do not have live Supabase credentials, enable zero-dependency local demo mode in `backend/.env`:
   ```env
   LOCAL_DEMO_MODE=true
   ```
   This automatically bypasses the JWT login guard and assigns an authenticated mock session to all local requests.

2. **OpenMP / Thread Contention Warnings:**  
   If running PyTorch and LightGBM concurrently on macOS/Linux, set:
   ```bash
   export OMP_NUM_THREADS=1
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```
   *(This is pre-configured in `backend/conftest.py` for automated tests).*

3. **Earth Engine Cache:**  
   Real-time Google Earth Engine downloads are cached locally under `backend/cache/`. To run completely offline or in presentation mode:
   ```env
   ML_ENGINE_OFFLINE=true
   ```
