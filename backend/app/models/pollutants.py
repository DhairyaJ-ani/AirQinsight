from pydantic import BaseModel, Field
from typing import Optional


class PollutantInfo(BaseModel):
    key: str
    display_name: str
    formula: str
    unit: str
    naaqs_annual: Optional[float]
    naaqs_24h: Optional[float]
    has_sentinel5p: bool
    description: str


class YearlyDataPoint(BaseModel):
    year: int
    state: Optional[str]
    city: str
    so2: Optional[float]
    no2: Optional[float]
    pm10: Optional[float]
    pm25: Optional[float]
    lat: Optional[float]
    lon: Optional[float]


class TrendPoint(BaseModel):
    year: int
    value: float


class CityTrend(BaseModel):
    city: str
    pollutant: str
    unit: str
    naaqs_annual: Optional[float]
    trend: list[TrendPoint]
    slope_per_year: Optional[float]
    direction: str  # "improving" | "worsening" | "stable"


class PollutantCompliance(BaseModel):
    concentration: Optional[float]
    naaqs_status: str   # "compliant" | "approaching" | "exceeded" | "insufficient_data"
    naaqs_threshold: Optional[float]
    who_status: Optional[str]
    exceedance_fraction: Optional[float]


class ComplianceSummary(BaseModel):
    city: str
    year: Optional[int]
    pollutants: dict[str, PollutantCompliance]
    error: Optional[str] = None


class PM25Prediction(BaseModel):
    source: str   # "namp_observation" | "lightgbm_model" | "error"
    city: str
    year: int
    pm25_ugm3: Optional[float] = None
    naaqs_status: Optional[str] = None
    naaqs_threshold: Optional[float] = None
    exceedance_fraction: Optional[float] = None
    model_r2_test: Optional[float] = None
    error: Optional[str] = None
