# Air Quality Downscaling & Analysis — Backend

FastAPI backend for the AI/ML Satellite-Based Air Quality Downscaling & Analysis System.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| [uv](https://github.com/astral-sh/uv) | ≥ 0.4.0 |
| Supabase project | with tables defined in the spec |

---

## Installation

Install all dependencies (including dev tools) with `uv`:

```bash
cd backend
uv sync
```

---

## Environment Setup

Copy the example environment file and fill in your Supabase credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret
```

> **Important**: The app will crash on startup with a clear validation error if any of these three variables are missing.

---

## Running the Development Server

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## Running Tests

```bash
uv run pytest tests/ -v
```

With coverage:

```bash
uv run pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Linting

```bash
uv run ruff check app/ tests/
```

Auto-fix where possible:

```bash
uv run ruff check --fix app/ tests/
```

---

## API Overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/auth/login` | None | Authenticate and get access token |
| `GET` | `/api/v1/downscale/map` | User | Get downscaled air quality map metadata |
| `GET` | `/api/v1/trends/predict` | User | Get NO2 trend predictions |
| `POST` | `/api/v1/analyze/pinpoint` | User | Identify top pollution sources near a point |
| `POST` | `/api/v1/reports/generate` | User | Download a PDF air quality report |
| `GET` | `/api/v1/admin/users` | Admin | List all registered users |
| `POST` | `/api/v1/admin/users/block` | Admin | Block or unblock a user |
| `POST` | `/api/v1/admin/data/upload` | Admin | Upload a geospatial dataset |
| `GET` | `/health` | None | Health check |

---

## Project Structure

```
backend/
├── pyproject.toml          # Project metadata and dependencies
├── .env.example            # Environment variable template
├── app/
│   ├── main.py             # FastAPI app, CORS, router registration
│   ├── dependencies.py     # get_current_user, require_admin
│   ├── core/
│   │   ├── config.py       # Settings (pydantic-settings) + Supabase factory
│   │   └── security.py     # JWT decode utility
│   ├── models/             # Pydantic request/response models
│   ├── services/           # Business logic (async, accept supabase client)
│   └── api/v1/             # Route handlers (thin, delegate to services)
└── tests/                  # pytest async test suite
```

---

---

## Multi-Pollutant GeoTIFF Datasets & Historical Endpoints

The backend bundles 61 daily georeferenced GeoTIFFs (1 November – 31 December 2025) across all 6 tracked pollutants for the Mumbai study area:
- **NO₂**: Sentinel-5P TROPOMI tropospheric column composites (µmol/m²), 0.035° resolution.
- **PM₂.₅, PM₁₀, SO₂, O₃, CO**: CAMS European/Global atmospheric analysis (µg/m³ surface concentration), resampled to the 0.035° domain with NaN-preserved cloud/missing masks.

Available endpoints:
- `GET /api/v1/historical/datasets`: List all bundled datasets with full CRS, grid, and resolution metadata.
- `GET /api/v1/historical/pollutants/availability`: Multi-pollutant capability and raster readiness matrix.
- `GET /api/v1/historical/coverage?dataset_id=...`: Temporal coverage audit and valid pixel statistics.
- `GET /api/v1/historical/manifest?dataset_id=...`: Machine-readable CSV/JSON raster manifest.
- `GET /api/v1/historical/grid?dataset_id=...&date=...`: Row-major JSON numerical grid with NaN-preservation.
- `GET /api/v1/historical/geotiff?dataset_id=...&date=...`: Direct Cloud-Optimized / LZW GeoTIFF byte streaming.
- `GET /api/v1/downscale/dates?pollutant=...`: Available observation timestamps for any pollutant.
- `GET /api/v1/downscale/geotiff?timestamp=...&pollutant=...`: Downscaled GeoTIFF raster for any pollutant.

---

## Notes

- CORS is configured to allow all origins (`*`) for development. Restrict `allow_origins` in production.
- Activity logging (`user_activity_logs`) is fault-tolerant — a logging failure will never break a user request.

