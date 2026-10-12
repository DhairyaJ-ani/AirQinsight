"""Multi-pollutant configuration registry for AirQ Insight v2.0.

Each pollutant has a ``PollutantConfig`` describing:
* data sources
* target variable / units
* valid concentration ranges
* India NAAQS annual / 24-hour thresholds (µg/m³ or mg/m³)
* Sentinel-5P parameter name (where available)
* OpenAQ v3 parameter ID
* supported forecast horizons (minutes)

Usage::

    from ml_engine.pollutant_config import POLLUTANTS, get_pollutant

    cfg = get_pollutant("pm25")
    print(cfg.naaqs_annual)   # 40.0
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PollutantConfig:
    """Complete configuration for a single pollutant."""

    # Identifier used in API parameters and model artefacts
    key: str                     # e.g. "pm25"
    display_name: str            # e.g. "PM₂.₅"
    formula: str                 # e.g. "PM2.5"
    unit: str                    # measurement unit, e.g. "µg/m³"

    # ── Data availability ──────────────────────────────────────────────────
    has_sentinel5p: bool         # directly observable via TROPOMI
    s5p_variable: str | None     # xarray variable name in the GEE collection

    # OpenAQ v3 parameter ID (integer).  None = not available in OpenAQ.
    openaq_parameter_id: int | None

    # CPCB/NAMP column name in the parsed yearly PDF dataset
    namp_column: str | None      # e.g. "no2"

    # ── Physical plausibility bounds (µg/m³) ──────────────────────────────
    min_valid: float = 0.0
    max_valid: float = 1000.0    # clamp implausible readings

    # ── India NAAQS thresholds (µg/m³) ────────────────────────────────────
    naaqs_annual: float | None = None    # Annual mean standard
    naaqs_24h: float | None = None       # 24-hour mean standard

    # ── Forecast horizons (minutes ahead) ─────────────────────────────────
    forecast_horizons_min: tuple[int, ...] = (30, 60, 90, 120)

    # ── Physics / model parameters ────────────────────────────────────────
    diffusivity_m2_s: float = 50.0       # horizontal eddy diffusivity
    chemical_lifetime_h: float | None = None  # None = conservative (no decay)
    deposition_velocity_ms: float | None = None  # dry deposition

    # ── Ensemble weights (validation-calibrated; to be overridden per run) ─
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {"ai": 0.5, "lightgbm": 0.3, "physics": 0.2}
    )

    # ── Description ───────────────────────────────────────────────────────
    description: str = ""

    @property
    def naaqs_exceeded_label(self) -> str:
        return f"{self.display_name} NAAQS Exceeded"

    @property
    def approaching_label(self) -> str:
        return f"{self.display_name} Approaching NAAQS"


# ─────────────────────────────────────────────────────────────────────────────
# Pollutant registry
# ─────────────────────────────────────────────────────────────────────────────

_PM25 = PollutantConfig(
    key="pm25",
    display_name="PM₂.₅",
    formula="PM2.5",
    unit="µg/m³",
    has_sentinel5p=False,          # No direct TROPOMI product; use aerosol proxies
    s5p_variable=None,
    openaq_parameter_id=2,         # OpenAQ v3 parameter ID for PM2.5
    namp_column="pm25",
    min_valid=0.0,
    max_valid=500.0,
    naaqs_annual=40.0,             # India NAAQS 2009 annual mean
    naaqs_24h=60.0,                # India NAAQS 2009 24h mean
    diffusivity_m2_s=30.0,
    chemical_lifetime_h=None,      # PM2.5 is conservative at hourly scale
    deposition_velocity_ms=0.001,
    description=(
        "Fine particulate matter ≤2.5 µm aerodynamic diameter. "
        "Major health risk; estimated via MODIS/VIIRS aerosol optical depth "
        "and ERA5 meteorology."
    ),
)

_PM10 = PollutantConfig(
    key="pm10",
    display_name="PM₁₀",
    formula="PM10",
    unit="µg/m³",
    has_sentinel5p=False,
    s5p_variable=None,
    openaq_parameter_id=1,         # OpenAQ v3 parameter ID for PM10
    namp_column="pm10",
    min_valid=0.0,
    max_valid=1000.0,
    naaqs_annual=60.0,             # India NAAQS
    naaqs_24h=100.0,
    diffusivity_m2_s=40.0,
    chemical_lifetime_h=None,
    deposition_velocity_ms=0.005,
    description=(
        "Coarse particulate matter ≤10 µm aerodynamic diameter. "
        "Major dust and combustion product indicator."
    ),
)

_NO2 = PollutantConfig(
    key="no2",
    display_name="NO₂",
    formula="NO2",
    unit="µg/m³",
    has_sentinel5p=True,
    s5p_variable="tropospheric_NO2_column_number_density",
    openaq_parameter_id=5,         # OpenAQ v3 parameter ID for NO2 in µg/m³
    namp_column="no2",
    min_valid=0.0,
    max_valid=500.0,
    naaqs_annual=40.0,             # India NAAQS
    naaqs_24h=80.0,
    diffusivity_m2_s=50.0,
    chemical_lifetime_h=4.0,       # daytime OH oxidation
    deposition_velocity_ms=None,
    description=(
        "Nitrogen dioxide. Primary traffic and combustion indicator. "
        "Directly observable via Sentinel-5P TROPOMI. "
        "Existing pipeline is NO₂; this is the reference implementation."
    ),
)

_SO2 = PollutantConfig(
    key="so2",
    display_name="SO₂",
    formula="SO2",
    unit="µg/m³",
    has_sentinel5p=True,
    s5p_variable="SO2_column_number_density",   # TROPOMI SO2 product
    openaq_parameter_id=9,         # OpenAQ v3 parameter ID for SO2
    namp_column="so2",
    min_valid=0.0,
    max_valid=500.0,
    naaqs_annual=50.0,             # India NAAQS
    naaqs_24h=80.0,
    diffusivity_m2_s=50.0,
    chemical_lifetime_h=24.0,      # SO2 oxidation is slower than NO2
    deposition_velocity_ms=0.01,
    description=(
        "Sulfur dioxide. Primarily from power plants and industry. "
        "Sentinel-5P provides SO₂ column data."
    ),
)

_CO = PollutantConfig(
    key="co",
    display_name="CO",
    formula="CO",
    unit="mg/m³",                  # NAAQS uses mg/m³ for CO
    has_sentinel5p=True,
    s5p_variable="CO_column_number_density",    # TROPOMI CO product
    openaq_parameter_id=4,         # OpenAQ v3 parameter ID for CO
    namp_column=None,              # NAMP PDFs 2015–2023 do not report CO
    min_valid=0.0,
    max_valid=100.0,               # mg/m³
    naaqs_annual=None,             # No annual standard in India
    naaqs_24h=2.0,                 # India NAAQS 8-hour (proxy for 24h here)
    diffusivity_m2_s=60.0,
    chemical_lifetime_h=30.0,      # days at regional scale; ~30h urban
    deposition_velocity_ms=None,
    description=(
        "Carbon monoxide. Combustion tracer. No India annual NAAQS. "
        "Ground-level estimated from Sentinel-5P CO column."
    ),
)

_O3 = PollutantConfig(
    key="o3",
    display_name="O₃",
    formula="O3",
    unit="µg/m³",
    has_sentinel5p=True,
    s5p_variable="O3_column_number_density",    # TROPOMI ozone
    openaq_parameter_id=7,         # OpenAQ v3 parameter ID for O3
    namp_column=None,              # NAMP PDFs do not report O3
    min_valid=0.0,
    max_valid=500.0,
    naaqs_annual=None,
    naaqs_24h=180.0,               # India NAAQS 8-hour
    diffusivity_m2_s=40.0,
    chemical_lifetime_h=None,      # Photo-stationary, complex chemistry
    deposition_velocity_ms=0.005,
    description=(
        "Ozone. Complex photochemical formation from NOx/VOC precursors. "
        "Sentinel-5P total-column ozone used as a proxy."
    ),
)

# Public registry — ordered by priority (NO2 first, then PM, then SO2, CO, O3)
POLLUTANTS: dict[str, PollutantConfig] = {
    "no2": _NO2,
    "pm25": _PM25,
    "pm10": _PM10,
    "so2": _SO2,
    "co": _CO,
    "o3": _O3,
}


def get_pollutant(key: str) -> PollutantConfig:
    """Return config for ``key`` (case-insensitive). Raises KeyError if unknown."""
    cfg = POLLUTANTS.get(key.lower())
    if cfg is None:
        raise KeyError(f"Unknown pollutant {key!r}. Supported: {list(POLLUTANTS)}")
    return cfg


def list_pollutants() -> list[dict]:
    """Serialisable summary of all registered pollutants."""
    return [
        {
            "key": cfg.key,
            "display_name": cfg.display_name,
            "formula": cfg.formula,
            "unit": cfg.unit,
            "naaqs_annual": cfg.naaqs_annual,
            "naaqs_24h": cfg.naaqs_24h,
            "has_sentinel5p": cfg.has_sentinel5p,
            "description": cfg.description,
        }
        for cfg in POLLUTANTS.values()
    ]
