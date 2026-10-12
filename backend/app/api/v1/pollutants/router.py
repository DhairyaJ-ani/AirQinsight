"""Multi-pollutant API routes — AirQ Insight v2.0.

All routes are new additions; existing routes in other modules are untouched.

Endpoints
---------
GET  /api/v1/pollutants/                        → list supported pollutants
GET  /api/v1/pollutants/yearly                  → yearly city data
GET  /api/v1/pollutants/mumbai                  → Mumbai multi-pollutant data
GET  /api/v1/pollutants/{city}/trends/{pollutant} → year-over-year trend
GET  /api/v1/pollutants/{city}/compliance       → NAAQS compliance summary
GET  /api/v1/pollutants/{city}/predict-pm25     → PM₂.₅ model prediction
POST /api/v1/pollutants/openaq/fetch            → trigger OpenAQ data fetch
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.models.pollutants import (
    ComplianceSummary,
    CityTrend,
    PM25Prediction,
    PollutantInfo,
)
from app.services.pollutant_service import (
    compliance_summary_all_pollutants,
    get_city_trends,
    get_mumbai_data,
    get_yearly_data,
    predict_pm25,
)
from ml_engine.pollutant_config import list_pollutants

log = logging.getLogger(__name__)

router = APIRouter(tags=["pollutants"])


# ─────────────────────────────────────────────────────────────────────────────
# 1. List supported pollutants
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[PollutantInfo],
    summary="List supported pollutants",
    description=(
        "Returns all pollutants supported by the multi-pollutant pipeline: "
        "NO₂, PM₂.₅, PM₁₀, SO₂, CO, O₃. Includes NAAQS thresholds and data availability."
    ),
)
async def list_supported_pollutants() -> list[PollutantInfo]:
    return [PollutantInfo(**p) for p in list_pollutants()]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Yearly data browser
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/yearly",
    summary="Yearly city-level pollutant data (CPCB/NAMP)",
    description=(
        "Annual-average ambient-air-quality data from CPCB/NAMP yearly reports "
        "(2015–2023). Data is in µg/m³. 'NM' / missing values are returned as null."
    ),
)
async def yearly_data(
    city: Optional[str] = Query(None, description="City name filter (partial match, case-insensitive)"),
    state: Optional[str] = Query(None, description="State/UT filter (partial match)"),
    pollutant: Optional[str] = Query(None, description="One of: so2, no2, pm10, pm25"),
    year_from: int = Query(2015, ge=2011, le=2023),
    year_to: int = Query(2023, ge=2011, le=2023),
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict]:
    if pollutant and pollutant not in ("so2", "no2", "pm10", "pm25"):
        raise HTTPException(status_code=422, detail=f"pollutant must be one of: so2, no2, pm10, pm25")
    return get_yearly_data(
        city=city, state=state, pollutant=pollutant,
        year_from=year_from, year_to=year_to, limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mumbai quick-access endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/mumbai",
    summary="Mumbai multi-pollutant data (CPCB)",
    description=(
        "Annual-average SO₂, NO₂, PM₁₀, PM₂.₅ for Mumbai from CPCB/NAMP reports "
        "(2021–2023 available). Includes NAAQS status."
    ),
)
async def mumbai_data(
    year_from: int = Query(2015, ge=2011, le=2023),
    year_to: int = Query(2023, ge=2011, le=2023),
    include_openaq: bool = Query(True, description="Include live OpenAQ active monitor summary"),
) -> dict:
    rows = get_mumbai_data(year_from=year_from, year_to=year_to)
    from ml_engine.compliance import assess_point
    enriched = []
    for r in rows:
        entry = dict(r)
        for pol in ("so2", "no2", "pm10", "pm25"):
            val = r.get(pol)
            if val is not None:
                rec = assess_point(pol, float(val), averaging_period="annual",
                                   source="namp_observation", location_name="Mumbai")
                entry[f"{pol}_naaqs_status"] = rec.naaqs_status
                entry[f"{pol}_who_status"] = rec.who_status
        enriched.append(entry)

    openaq_summary = {}
    if include_openaq:
        try:
            from ml_engine.ingestion.openaq import OpenAQClient
            from ml_engine.ingestion.openaq_multipollutant import list_pollutant_sensors, CITY_BBOXES
            client = OpenAQClient()
            for pol in ("pm25", "pm10", "no2", "co"):
                df_s = list_pollutant_sensors(client, pol, city_bbox=CITY_BBOXES["mumbai"])
                openaq_summary[pol] = {
                    "active_sensors": int(len(df_s)),
                    "sample_locations": df_s["name"].head(3).tolist() if not df_s.empty else [],
                }
        except Exception as exc:
            openaq_summary = {"status": "unavailable", "detail": str(exc)}

    return {
        "city": "Mumbai",
        "data_source": "CPCB/NAMP Yearly Reports (integrated data, µg/m³)",
        "data": enriched,
        "openaq_ground_monitors": openaq_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3b. OpenAQ Live Mumbai Monitor Readings
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/openaq/mumbai/live",
    summary="Get live active OpenAQ sensor readings across Mumbai",
)
async def openaq_mumbai_live(
    pollutant: str = Query("pm25", regex="^(pm25|pm10|no2|co|so2|o3)$"),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Fetch live ground station observations from OpenAQ v3 for Mumbai."""
    from ml_engine.ingestion.openaq import OpenAQClient
    from ml_engine.ingestion.openaq_multipollutant import list_pollutant_sensors, CITY_BBOXES
    client = OpenAQClient()
    try:
        df_sensors = list_pollutant_sensors(client, pollutant, city_bbox=CITY_BBOXES["mumbai"])
        if df_sensors.empty:
            return {"city": "Mumbai", "pollutant": pollutant, "sensor_count": 0, "readings": []}

        # Fetch recent readings for top sensors
        readings = []
        for _, s in df_sensors.head(limit).iterrows():
            s_id = int(s["sensor_id"])
            try:
                meas = client.get(f"/sensors/{s_id}/measurements", {"limit": 1}).get("results", [])
                latest = meas[0] if meas else {}
                val = latest.get("value")
                dt = (latest.get("period") or {}).get("datetimeTo", {}).get("local")
                readings.append({
                    "location_name": s["name"],
                    "sensor_id": s_id,
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "provider": s.get("provider", ""),
                    "latest_value": val,
                    "unit": (latest.get("parameter") or {}).get("units", "µg/m³"),
                    "timestamp": dt,
                })
            except Exception:
                continue

        return {
            "city": "Mumbai",
            "pollutant": pollutant,
            "total_active_sensors": int(len(df_sensors)),
            "readings_fetched": len(readings),
            "readings": readings,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAQ live query failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. City trend
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{city}/trends/{pollutant}",
    response_model=CityTrend,
    summary="Year-over-year trend for a city and pollutant",
)
async def city_trend(
    city: str,
    pollutant: str,
) -> CityTrend:
    if pollutant not in ("so2", "no2", "pm10", "pm25"):
        raise HTTPException(status_code=422, detail="pollutant must be one of: so2, no2, pm10, pm25")
    result = get_city_trends(city, pollutant)
    return CityTrend(**result)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compliance summary
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{city}/compliance",
    summary="NAAQS and WHO compliance summary for a city (latest year)",
    description=(
        "Compares the latest available NAMP annual averages against India NAAQS 2009 "
        "and WHO AQG 2021 thresholds for all four pollutants (SO₂, NO₂, PM₁₀, PM₂.₅)."
    ),
)
async def city_compliance(city: str) -> dict:
    result = compliance_summary_all_pollutants(city)
    if "error" in result and result.get("pollutants") is None:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. PM2.5 prediction
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{city}/predict-pm25",
    response_model=PM25Prediction,
    summary="Predict annual PM₂.₅ for a city-year using the trained LightGBM model",
    description=(
        "Returns observed PM₂.₅ from NAMP if available, otherwise uses the trained "
        "LightGBM model (R²≈0.75 on 2022–2023 test set). Includes NAAQS compliance status."
    ),
)
async def predict_city_pm25(
    city: str,
    state: str = Query("Maharashtra", description="State name for model encoding"),
    year: int = Query(2023, ge=2015, le=2026),
    no2: Optional[float] = Query(None, description="Override NO₂ (µg/m³)"),
    so2: Optional[float] = Query(None, description="Override SO₂ (µg/m³)"),
    pm10: Optional[float] = Query(None, description="Override PM₁₀ (µg/m³)"),
    lat: Optional[float] = Query(None, description="Override latitude"),
    lon: Optional[float] = Query(None, description="Override longitude"),
) -> PM25Prediction:
    result = predict_pm25(city=city, state=state, year=year, no2=no2, so2=so2, pm10=pm10, lat=lat, lon=lon)
    return PM25Prediction(**result)


# ─────────────────────────────────────────────────────────────────────────────
# 7. OpenAQ fetch trigger (background)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAQFetchRequest(BaseModel):
    pollutant: str = "pm25"
    city: str = "mumbai"
    start: str = "2023-01-01"
    end: str = "2023-12-31"


@router.post(
    "/openaq/fetch",
    summary="Trigger OpenAQ data download for a city and pollutant",
    description=(
        "Starts a background download of hourly data from OpenAQ v3 API. "
        "The result CSV is saved to ``data/openaq_{city}_{pollutant}_{year}.csv``. "
        "Uses the ``OPENAQ_API_KEY`` from environment."
    ),
)
async def trigger_openaq_fetch(
    req: OpenAQFetchRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    supported = ("pm25", "pm10", "no2", "so2", "co", "o3")
    if req.pollutant not in supported:
        raise HTTPException(status_code=422, detail=f"pollutant must be one of {supported}")
    if req.city.lower() not in ("mumbai", "delhi", "pune", "bangalore", "bengaluru",
                                  "chennai", "kolkata", "hyderabad", "ahmedabad"):
        raise HTTPException(
            status_code=422,
            detail="City must be one of the supported Indian cities. Use 'mumbai' for fastest results.",
        )

    def _run_download():
        try:
            from ml_engine.ingestion.openaq_multipollutant import download_pollutant
            year = req.start[:4]
            out = f"data/openaq_{req.city}_{req.pollutant}_{year}.csv"
            download_pollutant(
                pollutant=req.pollutant,
                start=req.start,
                end=req.end,
                out_csv=out,
                city=req.city,
            )
            log.info("OpenAQ download complete: %s → %s", req.pollutant, out)
        except Exception as exc:
            log.error("OpenAQ download failed: %s", exc)

    background_tasks.add_task(_run_download)

    return {
        "status": "started",
        "message": (
            f"Downloading {req.pollutant.upper()} data for {req.city} "
            f"({req.start} → {req.end}) in background. "
            f"Output: data/openaq_{req.city}_{req.pollutant}_{req.start[:4]}.csv"
        ),
        "note": "Check server logs for progress.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model Registry status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/models",
    summary="List operational multi-pollutant models and readiness",
)
async def list_models() -> list[dict]:
    from ml_engine.registry import get_model_registry
    return get_model_registry().get_supported_pollutants()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Three-Component Ensemble Prediction
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{city}/ensemble/{pollutant}",
    summary="Three-component ensemble prediction (AI + LightGBM + Physics)",
)
async def predict_city_ensemble(
    city: str,
    pollutant: str,
    state: Optional[str] = Query(None),
    year: int = Query(2024),
    no2: Optional[float] = Query(None),
    so2: Optional[float] = Query(None),
    pm10: Optional[float] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    physics_estimate: Optional[float] = Query(None),
    ai_estimate: Optional[float] = Query(None),
) -> dict:
    from ml_engine.registry import get_model_registry
    pol = pollutant.lower()
    features = {
        "year": year,
        "city": city,
        "state": state or "",
        "no2": no2 or 25.0,
        "so2": so2 or 15.0,
        "pm10": pm10 or 80.0,
        "lat": lat or 19.0760,
        "lon": lon or 72.8777,
    }
    registry = get_model_registry()
    try:
        ens = registry.predict_ensemble(
            pol,
            features_dict=features,
            physics_estimate=physics_estimate,
            ai_estimate=ai_estimate,
        )
        return {"city": city, **ens.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Multi-Pollutant Explainable AI (TreeSHAP + Physics Sensitivity)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{city}/explain/{pollutant}",
    summary="XAI breakdown: TreeSHAP attribution and physics sensitivity",
)
async def explain_city_prediction(
    city: str,
    pollutant: str,
    state: Optional[str] = Query(None),
    year: int = Query(2024),
    no2: Optional[float] = Query(None),
    so2: Optional[float] = Query(None),
    pm10: Optional[float] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
) -> dict:
    from ml_engine.xai.multipollutant_xai import explain_ensemble_prediction
    pol = pollutant.lower()
    features = {
        "year": year,
        "city": city,
        "state": state or "",
        "no2": no2 or 25.0,
        "so2": so2 or 15.0,
        "pm10": pm10 or 80.0,
        "lat": lat or 19.0760,
        "lon": lon or 72.8777,
    }
    try:
        xai_res = explain_ensemble_prediction(pol, features_dict=features)
        return {"city": city, **xai_res}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
