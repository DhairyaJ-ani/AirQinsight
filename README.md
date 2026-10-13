# 🌍 AirQ Insight: Satellite-Based Air Quality Downscaling & Environmental Decision Support

[![Tests](https://img.shields.io/badge/Tests-335%20Passed-brightgreen.svg)](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SAMPLE_DATA_AND_TEST_CASES.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16.3-black.svg?logo=next.js&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-≥3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **AirQ Insight** bridges the gap between coarse satellite Earth observations (~2.5–3.5 km Sentinel-5P TROPOMI & CAMS reanalysis), atmospheric physics, and municipal regulatory enforcement. It transforms macro-scale atmospheric columns into high-resolution spatial heatmaps (~250 m), physical plume forward trajectories, explainable source attributions (SHAP), regulatory compliance audits (CPCB/NAAQS), and automated UAV/drone inspection flight plans.

---

## 📑 Table of Contents

- [1. Problem Statement & Mission](#1-problem-statement--mission)
- [2. Key Features & Capabilities](#2-key-features--capabilities)
- [3. System Architecture](#3-system-architecture)
- [4. Repository Structure](#4-repository-structure)
- [5. Quick Start & Setup Instructions](#5-quick-start--setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Zero-Dependency Local Demo Mode](#zero-dependency-local-demo-mode)
  - [Backend Setup (FastAPI)](#backend-setup-fastapi)
  - [Frontend Setup (Next.js)](#frontend-setup-nextjs)
- [6. Sample Data Catalog](#6-sample-data-catalog)
- [7. Automated Testing Suite (335 Tests)](#7-automated-testing-suite-335-tests)
- [8. API Reference](#8-api-reference)
- [9. Regulatory & Scientific Grounding](#9-regulatory--scientific-grounding)
- [10. Companion Documentation](#10-companion-documentation)

---

## 1. Problem Statement & Mission

Satellite-borne spectrometers like **Sentinel-5P (TROPOMI)** provide daily global observations of atmospheric pollutants such as Nitrogen Dioxide ($\text{NO}_2$), Sulfur Dioxide ($\text{SO}_2$), and Carbon Monoxide ($\text{CO}$). However, utilizing this data for municipal air quality enforcement faces severe operational challenges:
1. **Resolution Mismatch:** Native satellite pixels are coarse (~2.5 km to ~4.0 km), blurring individual industrial stacks, construction sites, and highway corridors into general regional haze.
2. **Column vs. Surface Disparity:** Satellites measure tropospheric vertical column density ($\mu\text{mol/m}^2$), whereas citizen health and Central Pollution Control Board (CPCB) standards govern ground-level breathing mass concentration ($\mu\text{g/m}^3$).
3. **Actionability Gap:** Regulators and State Pollution Control Boards (SPCBs) need actionable evidence: *Which specific source is driving this hotspot? Where will the plume drift in the next 2 hours? Where should a drone fly to collect legal air samples?*

**AirQ Insight solves this end-to-end** through machine learning downscaling, physics-based atmospheric advection-diffusion solvers, game-theoretic explainable AI (SHAP), counterfactual policy simulators, and autonomous multi-agent taskforces.

---

## 2. Key Features & Capabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AirQ Insight Platform                           │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ 🛰️ ML Downscaling    │ 💨 Plume Dispersion  │ 🎯 Explainable AI (SHAP)      │
│  2.5 km → 250 m      │  2D Advection-Diff   │  Game-theoretic drivers       │
│  Mass conservation   │  ERA5 wind coupling  │  Entropy-scaled confidence    │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 🧪 Multi-Pollutant   │ 🎮 What-If Simulator │ 🤖 Autonomous Swarm           │
│  NO₂, PM2.5, PM10,   │  Traffic/Industry    │  8 LangGraph specialist       │
│  SO₂, O₃, CO         │  Counterfactuals     │  agents with mission cards    │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 🚁 Drone Inspection  │ 📜 CPCB Compliance   │ 🌐 3D Atmospheric Globe       │
│  Wind-aware paths    │  Automated NAAQS     │  Three.js planetary view      │
│  Waypoint export     │  PDF audit briefs    │  Transboundary tracking       │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

- **Physics-Ensemble Satellite Downscaling:** Downscales regional satellite rasters from coarse resolution (~2.5 km) to street-level granularity (~250 m) using a blended ensemble of **XGBoost (50%)**, **Random Forest (30%)**, and **LightGBM (20%)** calibrated against CPCB ground monitoring networks. Enforces **iterative backprojection** to strictly preserve atmospheric column mass.
- **Unified Multi-Pollutant Framework:** Complete support for 6 key pollutants ($\text{NO}_2$, $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{SO}_2$, $\text{O}_3$, $\text{CO}$) while maintaining strict scientific separation between satellite column densities and breathing-level mass concentrations.
- **Physical Plume Advection & Spatiotemporal Forecasting:** Hybrid physics engine solving the 2D Advection-Diffusion Equation with real-time ERA5 wind vector fields, complemented by a spatiotemporal ConvLSTM neural network predicting short-term plume dispersion over 30- to 120-minute lead horizons.
- **Explainable AI (XAI) & Dynamic Attribution:** Employs SHAP TreeExplainers to decompose observed concentrations into feature-level drivers (road traffic density, boundary layer height, industrial proximity, wind stagnation) accompanied by entropy-grounded confidence scores.
- **Interactive Policy "What-If" Counterfactual Simulator:** Digital twin workbench allowing municipal authorities to simulate hypothetical interventions (e.g., 30% odd-even traffic reduction, 50% industrial stack abatement, construction bans) and project local concentration decreases.
- **Autonomous Multi-Agent Taskforce (LangGraph):** Orchestrates 8 specialist agents (Analysis, XAI, Forecast, Simulator, Compliance, Drone, Report, and Ensemble) that autonomously triage environmental hazards and synthesize PDF compliance briefs.
- **Precision Drone Flight Inspection:** Generates wind-compensated polygon-perimeter and crosshatch flight waypoints for tactical unmanned aerial vehicle (UAV) spot inspections.
- **3D Atmospheric Globe:** WebGL/Three.js-powered interactive globe visualizing transboundary atmospheric transport and global column densities.

---

## 3. System Architecture

AirQ Insight is engineered with a modular, loosely-coupled microservice-ready architecture comprising an asynchronous **FastAPI backend** and a high-performance **Next.js 16 (React 19) frontend** with client-side WebGL / MapLibre GL rendering.

```mermaid
flowchart TB
    subgraph ClientLayer ["Frontend Client Layer (Next.js 16 + React 19)"]
        UI_Home["Geospatial Map (/)"]
        UI_Plume["Plume Dispersion (/visualization)"]
        UI_Sim["Policy Simulator (/simulator)"]
        UI_Drone["Drone Mission Control (/drone)"]
        UI_Agent["AI Agent Assistant (/chatbot)"]
        UI_Reports["Compliance Reports (/reports)"]
        UI_Globe["3D Atmospheric Globe (/globe)"]
        UI_Upload["GeoTIFF Ingestion (/upload)"]
    end

    subgraph APILayer ["Backend API Gateway (FastAPI 0.115+)"]
        API_Downscale["/api/v1/downscale"]
        API_Pollutants["/api/v1/pollutants"]
        API_Historical["/api/v1/historical"]
        API_Trends["/api/v1/trends"]
        API_Simulator["/api/v1/simulator"]
        API_Drone["/api/v1/drone"]
        API_Agent["/api/v1/agent"]
        API_Reports["/api/v1/reports"]
        API_Globe["/api/v1/globe"]
        API_Auth["/api/v1/auth"]
    end

    subgraph AgentGraph ["Multi-Agent Orchestrator (LangGraph)"]
        MC["Mission Controller"]
        IntentParser["Intent & Entity Parser"]
        TriggerDetector["Autonomous Trigger Engine"]
        
        subgraph SpecialistAgents ["Specialist Agent Pipeline"]
            Node_Ens["Ensemble Confidence Node"]
            Agent_Analysis["Analysis Specialist"]
            Agent_XAI["XAI Specialist (SHAP)"]
            Agent_Forecast["Forecast Specialist"]
            Agent_Sim["Scenario Simulator Agent"]
            Agent_Compliance["Compliance Specialist (NAAQS)"]
            Agent_Drone["Drone Mission Planner"]
            Agent_Report["Report Generator (PDF/Docx)"]
        end
    end

    subgraph MLEngine ["AI/ML & Scientific Computing Engine"]
        Downscaler["Downscaling Ensemble\n(XGBoost 0.5 + RF 0.3 + LGBM 0.2)"]
        Backprojector["Iterative Mass Conservation\nBackprojection Filter"]
        PhysicsSolver["2D Advection-Diffusion Solver\n(ERA5 Wind Field Transport)"]
        ConvLSTM_Model["Spatiotemporal ConvLSTM\n(Dispersion Forecasting)"]
        SHAP_Engine["SHAP TreeExplainer &\nEntropy-Scaled Confidence"]
        PolicySimulator["Policy Abatement Engine"]
        SurfaceCalibrator["Boundary Layer Scaling &\nStation Calibration"]
    end

    subgraph DataStorage ["Data Stores & Geospatial Repositories"]
        GeoTIFF_Store["GeoTIFF Raster Store\n(61-Day NO₂ Sentinel-5P + CAMS Multipollutant)"]
        StationDB["Ground Station Database\n(CPCB, NAMP, OpenAQ In-Situ)"]
        Meteo_ERA5["ERA5 Reanalysis\n(u/v Winds, Temp, PBL Height)"]
        UserDB["User & Session Store\n(In-Memory / Persistent SQLite / JWT)"]
        ArtifactStore["Generated Artifacts\n(Flight Plans, PDF Reports, Mission Cards)"]
    end

    UI_Home --> API_Downscale
    UI_Home --> API_Pollutants
    UI_Plume --> API_Historical
    UI_Plume --> API_Trends
    UI_Sim --> API_Simulator
    UI_Drone --> API_Drone
    UI_Agent --> API_Agent
    UI_Reports --> API_Reports
    UI_Upload --> API_Historical
    UI_Globe --> API_Globe

    API_Downscale --> Downscaler
    API_Downscale --> Backprojector
    API_Historical --> GeoTIFF_Store
    API_Trends --> PhysicsSolver
    API_Trends --> ConvLSTM_Model
    API_Simulator --> PolicySimulator
    API_Agent --> AgentGraph
    API_Drone --> Agent_Drone
    API_Reports --> Agent_Report

    API_Agent --> IntentParser --> TriggerDetector --> Node_Ens
    Node_Ens --> Agent_Analysis --> Agent_XAI --> Agent_Forecast
    Agent_Forecast --> Agent_Sim --> Agent_Compliance --> Agent_Drone --> Agent_Report --> MC

    Downscaler --> SurfaceCalibrator
    Downscaler --> GeoTIFF_Store
    PhysicsSolver --> Meteo_ERA5
    ConvLSTM_Model --> GeoTIFF_Store
    SHAP_Engine --> Downscaler
    Agent_Compliance --> StationDB
    Agent_Report --> ArtifactStore
```

*For in-depth mathematical formulations and component specifications, see [SYSTEM_ARCHITECTURE.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SYSTEM_ARCHITECTURE.md) and [ARCHITECTURE.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/ARCHITECTURE.md).*

---

## 4. Repository Structure

```
AirQinsight/
├── README.md                      # Primary project documentation & quick start guide
├── ARCHITECTURE.md                # System architectural blueprint & subsystem portal
├── SYSTEM_ARCHITECTURE.md         # Comprehensive technical architecture specification (470+ lines)
├── REQUIREMENTS_FULFILMENT.md     # Requirements traceability matrix (46/54 verified)
├── SAMPLE_DATA_AND_TEST_CASES.md  # Detailed catalog of sample datasets and 335 test cases
│
├── backend/                       # Python FastAPI Backend & ML Engine
│   ├── pyproject.toml             # Python build configuration and dependencies (uv)
│   ├── app/
│   │   ├── main.py                # FastAPI ASGI application entrypoint & middleware
│   │   ├── dependencies.py        # Authentication & RBAC dependency injection
│   │   ├── core/                  # Settings (pydantic-settings), JWT security, DB factory
│   │   ├── api/v1/                # Modular REST route handlers (downscale, trends, agent...)
│   │   └── services/              # Business logic services (geospatial, report, auth...)
│   ├── ml_engine/                 # AI/ML & Scientific Computing Engine
│   │   ├── downscaling.py         # Multi-model spatial downscaler & backprojection
│   │   ├── dispersion.py          # 2D Advection-Diffusion physics solver
│   │   ├── forecasting/           # Spatiotemporal ConvLSTM deep learning forecaster
│   │   ├── xai/                   # SHAP TreeExplainer, attribution & confidence scaling
│   │   ├── simulator.py           # Counterfactual policy abatement engine
│   │   ├── drone/                 # Autonomous UAV flight path planning
│   │   ├── compliance.py          # CPCB NAAQS regulatory compliance auditor
│   │   └── agent/                 # LangGraph multi-agent environmental swarm
│   ├── data/                      # Georeferenced sample rasters & ground station CSVs
│   │   ├── test_data/             # 61 daily real Sentinel-5P NO2 GeoTIFFs (Nov-Dec 2025)
│   │   ├── sentinel_test_samples/ # 8 weekly Sentinel-5P GeoTIFF samples for CI
│   │   ├── sample_inputs/         # 14-day sample input pipeline batch
│   │   ├── pollutant_samples/     # 305 daily GeoTIFFs (PM2.5, PM10, SO2, O3, CO)
│   │   ├── cpcb_stations.csv      # 550+ CPCB CAAQMS station coordinates across India
│   │   └── cpcb_mumbai.csv        # Mumbai in-situ ground monitoring observations
│   └── tests/                     # Comprehensive test suite (335 automated pytest tests)
│
└── frontend/                      # Next.js 16 (React 19) Geospatial Web Application
    ├── package.json               # Frontend dependencies & scripts
    ├── next.config.ts             # Next.js build and Turbopack configuration
    ├── app/                       # Next.js App Router pages
    │   ├── page.tsx               # Home geospatial map dashboard
    │   ├── visualization/         # Plume flow, scrubbable timeline & point forecasting
    │   ├── simulator/             # Interactive policy What-If simulation workbench
    │   ├── drone/                 # Drone flight inspection mission control
    │   ├── chatbot/               # AI multi-agent environmental chat with Mission Cards
    │   ├── reports/               # CPCB NAAQS compliance audits & PDF download
    │   ├── globe/                 # Three.js 3D atmospheric globe view
    │   └── upload/                # GeoTIFF raster ingestion & validation
    ├── lib/                       # Geospatial utils, MapLibre helpers, upload context
    └── public/                    # Static assets, Web Workers, GeoJSON boundaries
```

---

## 5. Quick Start & Setup Instructions

### Prerequisites

| Component | Minimum Version | Recommended | Notes |
| :--- | :---: | :---: | :--- |
| **Python** | `3.10` | `3.12` / `3.14` | Required for backend & ML engine |
| **uv** | `0.4.0` | Latest | Extremely fast Python package manager |
| **Node.js** | `18.18` | `20+` LTS | Required for Next.js 16 frontend |
| **npm** | `9.0` | Latest | Frontend package management |

---

### Zero-Dependency Local Demo Mode

You can run AirQ Insight immediately **without any cloud account, Supabase project, or external API keys**. The backend includes a built-in `LOCAL_DEMO_MODE` that auto-authenticates local requests and serves packaged real-world historical satellite datasets.

1. In `backend/.env`, set:
   ```env
   LOCAL_DEMO_MODE=true
   SUPABASE_URL=https://demo.supabase.co
   SUPABASE_KEY=demo-anon-key
   SUPABASE_JWT_SECRET=demo-jwt-secret
   ```
2. In `frontend/.env.local`, set:
   ```env
   NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=AIzaSyBoQqBbWgSdkmDrHvPU7M022Y-fUhcRNp0
   ```

---

### Backend Setup (FastAPI)

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install all dependencies with `uv`:**
   ```bash
   uv sync
   ```
   *(This creates a dedicated `.venv` and installs PyTorch, XGBoost, LightGBM, Rasterio, LangGraph, and FastAPI in seconds).*

3. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   ```
   *(If you have Supabase credentials, fill in `SUPABASE_URL`, `SUPABASE_KEY`, and `SUPABASE_JWT_SECRET`. Otherwise, keep `LOCAL_DEMO_MODE=true` as configured above).*

4. **Launch the FastAPI Server:**
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Verify Backend Status:**
   - Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

### Frontend Setup (Next.js)

1. **Open a new terminal and navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Configure Frontend Environment Variables:**
   ```bash
   cp .env.example .env.local
   ```

4. **Start the Development Server:**
   ```bash
   npm run dev
   ```

5. **Access the Web Application:**
   Open your browser to [http://localhost:3000](http://localhost:3000).

6. **Build for Production (Optional):**
   ```bash
   npm run build
   npm run start
   ```

---

## 6. Sample Data Catalog

AirQ Insight comes pre-packaged with genuine, verified satellite and atmospheric rasters covering the Mumbai Metropolitan Region for the target monitoring window (**1 Nov 2025 to 31 Dec 2025**):

- **61-Day Full Sentinel-5P NO2 Archive (`backend/data/test_data/`):** 61 daily georeferenced GeoTIFFs at $0.035^\circ$ resolution with cloudy pixels preserved as `NaN`.
- **8-Day Fast Sample Subset (`backend/data/sentinel_test_samples/`):** Curated clear and cloudy sample rasters with complete checksum manifest (`manifest.csv`).
- **14-Day Sample Inputs (`backend/data/sample_inputs/mumbai_2025-11-24_to_2025-12-07/`):** Plug-and-play batch for executing ML downscaling and output generation.
- **305-Raster CAMS Multi-Pollutant Catalog (`backend/data/pollutant_samples/`):** 61 daily GeoTIFFs across $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{SO}_2$, $\text{O}_3$, and $\text{CO}$.
- **Ground Sensor Networks (`backend/data/cpcb_stations.csv` & `cpcb_mumbai.csv`):** 550+ CPCB CAAQMS monitoring station coordinates and hourly/daily ground truth readings.

*For full details on projections, pixel spacing, and data formats, see [SAMPLE_DATA_AND_TEST_CASES.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SAMPLE_DATA_AND_TEST_CASES.md).*

---

## 7. Automated Testing Suite (335 Tests)

The backend features an automated test suite containing **335 tests** that execute in ~30 seconds with a **100% pass rate**.

### Running All Tests

```bash
cd backend
uv run pytest -v
```

### Running Specific Test Categories

```bash
# Multi-pollutant registry and measurement metadata distinction
uv run pytest tests/test_multi_pollutant_integration.py -v

# 2D Advection-Diffusion physics solver & wind transport
uv run pytest tests/test_forecast_point_solver.py -v

# LangGraph specialist multi-agent coordination & SHAP XAI
uv run pytest tests/test_ensemble_xai_simulator.py -v

# CPCB NAAQS regulatory compliance engine
uv run pytest tests/test_pollutants_v2.py -v

# Downscaling ensemble and iterative backprojection mass conservation
uv run pytest tests/test_downscale.py -v

# Sentinel-5P sample loader and cloud masking
uv run pytest tests/test_sentinel_samples.py -v
```

---

## 8. API Reference

| HTTP Method | Route | Access | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | Public | System health check and service version |
| `GET` | `/api/v1/pollutants/registry` | Public | Catalog of supported pollutants, units, and NAAQS limits |
| `GET` | `/api/v1/downscale/dates` | Public | List available observation dates for a given pollutant |
| `GET` | `/api/v1/historical/{dataset_id}/{pollutant}/{date}/geotiff` | Public | Stream or download high-resolution GeoTIFF raster binary |
| `POST` | `/api/v1/pollutants/explain` | Authenticated | Game-theoretic SHAP source attribution and confidence score |
| `POST` | `/api/v1/trends/predict` | Authenticated | Spatiotemporal plume dispersion forecast (15–120 min) |
| `POST` | `/api/v1/simulator/simulate` | Authenticated | Counterfactual policy simulation (traffic, industry, construction) |
| `POST` | `/api/v1/drone/plan-mission` | Authenticated | Automated wind-aware drone inspection flight plan generation |
| `POST` | `/api/v1/agent/chat` | Authenticated | LangGraph multi-agent conversational environmental assistant |
| `POST` | `/api/v1/reports/generate` | Authenticated | Generate regulatory CPCB compliance PDF report |
| `GET` | `/api/v1/globe/snapshot` | Public | 3D atmospheric globe planetary data layer |

---

## 9. Regulatory & Scientific Grounding

AirQ Insight adheres to the **Central Pollution Control Board (CPCB) National Ambient Air Quality Standards (NAAQS)** notification:

| Pollutant | Measurement Basis | Averaging Period | Industrial / Residential Limit | Ecologically Sensitive Limit |
| :--- | :---: | :---: | :---: | :---: |
| **$\text{PM}_{2.5}$** | Surface Mass Concentration | 24 Hours | $60\,\mu\text{g/m}^3$ | $60\,\mu\text{g/m}^3$ |
| **$\text{PM}_{10}$** | Surface Mass Concentration | 24 Hours | $100\,\mu\text{g/m}^3$ | $100\,\mu\text{g/m}^3$ |
| **$\text{NO}_2$** | Surface Mass Concentration | 24 Hours | $80\,\mu\text{g/m}^3$ | $80\,\mu\text{g/m}^3$ |
| **$\text{SO}_2$** | Surface Mass Concentration | 24 Hours | $80\,\mu\text{g/m}^3$ | $80\,\mu\text{g/m}^3$ |
| **$\text{O}_3$** | Surface Mass Concentration | 8 Hours | $100\,\mu\text{g/m}^3$ | $100\,\mu\text{g/m}^3$ |
| **$\text{CO}$** | Surface Mass Concentration | 8 Hours | $2.0\,\text{mg/m}^3$ | $2.0\,\text{mg/m}^3$ |

### Physical Principles Enforced:
1. **Mass Conservation:** Iterative backprojection guarantees that downscaled high-resolution concentrations integrate to the parent coarse satellite cell observation ($\pm 0.1\%$).
2. **Column vs. Surface Distinction:** Satellite tropospheric vertical column densities ($\mu\text{mol/m}^2$) are never conflated with surface breathing mass concentrations ($\mu\text{g/m}^3$).
3. **Attribution Abstention:** Where quantitative chemical precursor inventories are missing (such as secondary photochemical ozone), the platform explicitly abstains from attributing legal liability to individual local polluters.

---

## 10. Companion Documentation

- 📐 **[SYSTEM_ARCHITECTURE.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SYSTEM_ARCHITECTURE.md):** 470+ lines of complete architectural specifications, mathematical formulations, and component designs.
- 🗺️ **[ARCHITECTURE.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/ARCHITECTURE.md):** Architectural summary blueprint and navigation portal.
- 📊 **[SAMPLE_DATA_AND_TEST_CASES.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SAMPLE_DATA_AND_TEST_CASES.md):** Detailed guide to all sample data rasters, ground station datasets, and the 335 automated test cases.
- ✅ **[REQUIREMENTS_FULFILMENT.md](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/REQUIREMENTS_FULFILMENT.md):** Comprehensive requirements audit (46 fulfilled, 7 partially fulfilled, 1 planned).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
