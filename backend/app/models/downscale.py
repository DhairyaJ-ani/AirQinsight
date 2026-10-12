from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DownscaleMapResponse(BaseModel):
    resolution: str
    bbox: List[float]
    timestamp: str
    grid_url: str
    format: str
    # Additional ML engine outputs (optional so existing clients are unaffected)
    date: Optional[str] = None
    units: Optional[str] = None
    raw_url: Optional[str] = None
    gapfilled_url: Optional[str] = None
    hazard_geojson_url: Optional[str] = None
    netcdf_url: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    # Ensemble downscaler fields
    ensemble_active: bool = False
    ensemble_metrics: Optional[Dict[str, Any]] = None
