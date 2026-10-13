"""Central pollutant registry for AirQ Insight — shared by backend, agents, XAI, and reports.

This module extends ``ml_engine.pollutant_config`` with:
* measurement types and their scientific meaning
* data source availability matrix (raster vs. ground station vs. model)
* model capability matrix — which models are trained and validated
* regulatory standard configurations with full metadata
* data-quality rules

Use ``get_registry_entry(key)`` to get the full entry for any pollutant.
Use ``POLLUTANT_REGISTRY`` for the complete typed dict.

Backward compatibility: ``ml_engine.pollutant_config.POLLUTANTS`` and
``get_pollutant()`` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Measurement type taxonomy ──────────────────────────────────────────────────

MeasurementType = Literal[
    "surface_concentration",   # µg/m³ at breathing height — comparable to NAAQS
    "satellite_column",        # total or tropospheric atmospheric column (µmol/m², mol/m², DU)
    "model_analysis",          # model-assimilated surface field (CAMS, WRF, etc.)
    "ground_station",          # direct instrument measurement at station
    "estimated",               # downscaled or statistically derived surface estimate
]

SourceType = Literal[
    "sentinel5p_tropomi",
    "cams_analysis",
    "cpcb_namp",
    "openaq",
    "ground_station_assimilation",
    "model_estimate",
]

ModelCapability = Literal[
    "production",         # trained, validated, deployed
    "validated_cv",       # cross-validated but not independently evaluated
    "experimental",       # exploratory, not validated
    "unsupported",        # no model available
]


@dataclass
class RegulatoryStandard:
    """One pollutant standard (e.g. CPCB NAAQS 24-hour)."""
    name: str                        # "India NAAQS 2009"
    regulator: str                   # "CPCB"
    pollutant: str                   # "no2"
    averaging_period: str            # "24h" | "annual" | "8h"
    concentration_limit: float       # in `unit`
    unit: str                        # "µg/m³" | "mg/m³"
    measurement_type_required: MeasurementType = "surface_concentration"
    area_category: str | None = None # "industrial" | "residential" | None (all)
    notes: str = ""


@dataclass
class DataSourceCapability:
    """Describes one data source for a pollutant."""
    source_type: SourceType
    measurement_type: MeasurementType
    available_nov_dec_2025: bool
    units: str
    description: str
    acquisition_note: str = ""
    dataset_id: str | None = None    # links to historical API dataset_id if applicable


@dataclass
class ModelCapabilityEntry:
    """Describes model support for a pollutant."""
    model_type: str                  # "lightgbm" | "physics" | "neural" | "ensemble"
    capability: ModelCapability
    training_data: str               # what it was trained on
    validated_on: str                # validation set description
    known_limitations: list[str] = field(default_factory=list)


@dataclass
class RegistryEntry:
    """Full registry entry for a single pollutant."""
    key: str
    display_name: str
    formula: str
    canonical_units: str             # surface concentration unit for NAAQS comparison
    measurement_types_supported: list[MeasurementType]
    data_sources: list[DataSourceCapability]
    models: list[ModelCapabilityEntry]
    regulatory_standards: list[RegulatoryStandard]
    known_limitations: list[str]
    validation_status: str           # free-text status note

    def has_raster_data_nov_dec_2025(self) -> bool:
        return any(s.available_nov_dec_2025 for s in self.data_sources)

    def get_model(self, model_type: str) -> ModelCapabilityEntry | None:
        for m in self.models:
            if m.model_type == model_type:
                return m
        return None

    def best_model_capability(self) -> ModelCapability:
        order = {"production": 0, "validated_cv": 1, "experimental": 2, "unsupported": 3}
        best = min(self.models, key=lambda m: order.get(m.capability, 99), default=None)
        return best.capability if best else "unsupported"

    @property
    def model_capability(self) -> ModelCapability:
        return self.best_model_capability()

    def get_standard(self, averaging_period: str = "24h") -> RegulatoryStandard | None:
        for r in self.regulatory_standards:
            if r.averaging_period == averaging_period:
                return r
        return self.regulatory_standards[0] if self.regulatory_standards else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "formula": self.formula,
            "canonical_units": self.canonical_units,
            "has_raster_data_nov_dec_2025": self.has_raster_data_nov_dec_2025(),
            "best_model_capability": self.best_model_capability(),
            "regulatory_standards": [
                {
                    "regulator": r.regulator,
                    "averaging_period": r.averaging_period,
                    "concentration_limit": r.concentration_limit,
                    "unit": r.unit,
                    "measurement_type_required": r.measurement_type_required,
                    "notes": r.notes,
                }
                for r in self.regulatory_standards
            ],
            "known_limitations": self.known_limitations,
            "validation_status": self.validation_status,
        }


# ── Registry entries ──────────────────────────────────────────────────────────

_NO2_ENTRY = RegistryEntry(
    key="no2",
    display_name="NO₂",
    formula="NO2",
    canonical_units="µg/m³",
    measurement_types_supported=["satellite_column", "ground_station", "estimated"],
    data_sources=[
        DataSourceCapability(
            source_type="sentinel5p_tropomi",
            measurement_type="satellite_column",
            available_nov_dec_2025=True,
            units="µmol/m²",
            description=(
                "61 daily Sentinel-5P TROPOMI tropospheric NO₂ column composites, "
                "1 Nov–31 Dec 2025, over Mumbai. Native 0.035° resolution. "
                "Cloud fraction ≥ 0.3 masked to NaN."
            ),
            dataset_id="no2_61day_2025",
        ),
        DataSourceCapability(
            source_type="cpcb_namp",
            measurement_type="ground_station",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CPCB/NAMP annual mean ground station data, 2015–2023 (not Nov–Dec 2025).",
        ),
        DataSourceCapability(
            source_type="openaq",
            measurement_type="ground_station",
            available_nov_dec_2025=True,
            units="µg/m³",
            description="OpenAQ v3 live sensor readings (near-real-time, not historical archive).",
            acquisition_note="Requires OPENAQ_API_KEY environment variable.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="production",
            training_data="CPCB/NAMP 2015–2023 annual city averages",
            validated_on="2022–2023 hold-out test set (R²≈0.75)",
            known_limitations=[
                "Trained on annual city-level data, not daily raster data",
                "Predicts annual mean surface concentration, not column data",
            ],
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="production",
            training_data="Advection-diffusion solver calibrated on Mumbai geometry",
            validated_on="Hindcast against satellite observations",
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="production",
            training_data="ConvLSTM spatiotemporal model on NO2 raster sequences",
            validated_on="Holdout raster frames",
        ),
        ModelCapabilityEntry(
            model_type="ensemble",
            capability="production",
            training_data="Three-component: neural + LightGBM + physics",
            validated_on="Cross-validated on full dataset",
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="no2",
            averaging_period="annual",
            concentration_limit=40.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
            notes="Two-month satellite dataset cannot establish annual compliance.",
        ),
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="no2",
            averaging_period="24h",
            concentration_limit=80.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
            notes=(
                "Satellite column data (µmol/m²) cannot be directly compared to this threshold. "
                "Surface-level downscaling or ground measurements required."
            ),
        ),
    ],
    known_limitations=[
        "GeoTIFFs contain tropospheric column, NOT surface concentration.",
        "Column-to-surface conversion requires auxiliary meteorology and boundary layer data.",
        "3 of 61 days are fully cloud-masked (Nov 1–3).",
        "Native resolution is ~3.9 km; values represent ~15 km² grid cells.",
        "No CO, O₃, SO₂, PM2.5, or PM10 raster data exists for this period.",
    ],
    validation_status="Production: LightGBM, Physics, Neural ensemble operational for surface estimates.",
)

_PM25_ENTRY = RegistryEntry(
    key="pm25",
    display_name="PM₂.₅",
    formula="PM2.5",
    canonical_units="µg/m³",
    measurement_types_supported=["ground_station", "model_analysis", "estimated"],
    data_sources=[
        DataSourceCapability(
            source_type="cpcb_namp",
            measurement_type="ground_station",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CPCB/NAMP annual mean, 2015–2023 only.",
        ),
        DataSourceCapability(
            source_type="cams_analysis",
            measurement_type="model_analysis",
            available_nov_dec_2025=False,
            units="µg/m³",
            description=(
                "CAMS Global Atmospheric Composition Analysis (ECMWF). "
                "Variable 'pm2p5'. Requires Copernicus ADS credentials. "
                "Not downloaded for this repository."
            ),
            acquisition_note=(
                "Run scripts/acquire_historical_pollutants.py with valid ADS API key. "
                "Resolution: 0.4° (~40 km). Not a fine-resolution surface measurement."
            ),
        ),
        DataSourceCapability(
            source_type="openaq",
            measurement_type="ground_station",
            available_nov_dec_2025=True,
            units="µg/m³",
            description="OpenAQ v3 live sensor readings (near-real-time).",
            acquisition_note="Requires OPENAQ_API_KEY.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="production",
            training_data="CPCB/NAMP annual city averages 2015–2023",
            validated_on="2022–2023 test set (R²≈0.75)",
            known_limitations=[
                "Annual city-level prediction only",
                "No daily or spatial raster capability",
            ],
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="experimental",
            training_data="Advection-diffusion proxy based on PM2.5 config",
            validated_on="Not independently validated",
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="unsupported",
            training_data="N/A",
            validated_on="N/A",
            known_limitations=["No neural model trained for PM2.5"],
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="pm25",
            averaging_period="annual",
            concentration_limit=40.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="pm25",
            averaging_period="24h",
            concentration_limit=60.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
    ],
    known_limitations=[
        "No satellite product directly measures PM2.5.",
        "No PM2.5 raster data exists for Nov–Dec 2025 in this repository.",
        "CAMS analysis data would require separate download with ADS credentials.",
        "Resampled CAMS data at coarse resolution (~40 km) must not be presented as fine-resolution.",
    ],
    validation_status="LightGBM production for annual city estimates. Raster pipeline: data unavailable for Nov–Dec 2025.",
)

_PM10_ENTRY = RegistryEntry(
    key="pm10",
    display_name="PM₁₀",
    formula="PM10",
    canonical_units="µg/m³",
    measurement_types_supported=["ground_station", "model_analysis", "estimated"],
    data_sources=[
        DataSourceCapability(
            source_type="cpcb_namp",
            measurement_type="ground_station",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CPCB/NAMP annual mean, 2015–2023 only.",
        ),
        DataSourceCapability(
            source_type="cams_analysis",
            measurement_type="model_analysis",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CAMS PM10 analysis. Requires ADS credentials. Not downloaded.",
            acquisition_note="Run scripts/acquire_historical_pollutants.py.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="production",
            training_data="CPCB/NAMP annual city averages 2015–2023",
            validated_on="2022–2023 test set",
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="experimental",
            training_data="Advection-diffusion proxy",
            validated_on="Not independently validated",
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="unsupported",
            training_data="N/A",
            validated_on="N/A",
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="pm10",
            averaging_period="annual",
            concentration_limit=60.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="pm10",
            averaging_period="24h",
            concentration_limit=100.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
    ],
    known_limitations=[
        "No satellite product directly measures PM10.",
        "No PM10 raster data for Nov–Dec 2025 in this repository.",
    ],
    validation_status="LightGBM production for annual city estimates. Raster pipeline: data unavailable.",
)

_SO2_ENTRY = RegistryEntry(
    key="so2",
    display_name="SO₂",
    formula="SO2",
    canonical_units="µg/m³",
    measurement_types_supported=["satellite_column", "ground_station", "estimated"],
    data_sources=[
        DataSourceCapability(
            source_type="sentinel5p_tropomi",
            measurement_type="satellite_column",
            available_nov_dec_2025=False,
            units="µmol/m²",
            description=(
                "Sentinel-5P TROPOMI SO₂ OFFL L3 product available but not downloaded. "
                "Requires GEE project or CDSE credentials."
            ),
            acquisition_note="Run scripts/acquire_historical_pollutants.py with GEE or CDSE access.",
        ),
        DataSourceCapability(
            source_type="cpcb_namp",
            measurement_type="ground_station",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CPCB/NAMP annual mean SO₂, 2015–2023 only.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="validated_cv",
            training_data="CPCB/NAMP annual city averages 2015–2023",
            validated_on="Cross-validated; not independently evaluated",
            known_limitations=[
                "Annual city-level only",
                "CV performance lower than NO2/PM models",
            ],
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="experimental",
            training_data="Advection-diffusion with SO2 config (lifetime 24h)",
            validated_on="Not independently validated",
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="unsupported",
            training_data="N/A",
            validated_on="N/A",
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="so2",
            averaging_period="annual",
            concentration_limit=50.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="so2",
            averaging_period="24h",
            concentration_limit=80.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
        ),
    ],
    known_limitations=[
        "No SO₂ raster data for Nov–Dec 2025 exists in this repository.",
        "TROPOMI SO₂ column data requires acquisition with GEE or CDSE credentials.",
        "Column data cannot be compared directly to surface NAAQS thresholds.",
    ],
    validation_status="LightGBM validated_cv. Raster pipeline: data unavailable for Nov–Dec 2025.",
)

_CO_ENTRY = RegistryEntry(
    key="co",
    display_name="CO",
    formula="CO",
    canonical_units="mg/m³",
    measurement_types_supported=["satellite_column", "estimated"],
    data_sources=[
        DataSourceCapability(
            source_type="sentinel5p_tropomi",
            measurement_type="satellite_column",
            available_nov_dec_2025=False,
            units="mol/m²",
            description=(
                "Sentinel-5P TROPOMI CO OFFL L3 product available but not downloaded. "
                "Requires GEE project or CDSE credentials."
            ),
            acquisition_note="Run scripts/acquire_historical_pollutants.py.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="unsupported",
            training_data="N/A — NAMP PDFs do not report CO",
            validated_on="N/A",
            known_limitations=["No ground station training data available for CO"],
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="experimental",
            training_data="Advection-diffusion with CO config (lifetime ~30h urban)",
            validated_on="Not validated",
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="unsupported",
            training_data="N/A",
            validated_on="N/A",
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="co",
            averaging_period="8h",
            concentration_limit=2.0,
            unit="mg/m³",
            measurement_type_required="surface_concentration",
            notes=(
                "India NAAQS uses an 8-hour average. No annual standard. "
                "NAMP PDFs do not report CO. No trained model available."
            ),
        ),
    ],
    known_limitations=[
        "No CO raster data for Nov–Dec 2025 in this repository.",
        "No LightGBM model trained (NAMP data lacks CO).",
        "Physics model is experimental, unvalidated.",
        "TROPOMI CO is atmospheric column, not surface concentration.",
    ],
    validation_status="Experimental. No production model. No raster data for target window.",
)

_O3_ENTRY = RegistryEntry(
    key="o3",
    display_name="O₃",
    formula="O3",
    canonical_units="µg/m³",
    measurement_types_supported=["satellite_column", "model_analysis"],
    data_sources=[
        DataSourceCapability(
            source_type="sentinel5p_tropomi",
            measurement_type="satellite_column",
            available_nov_dec_2025=False,
            units="DU",
            description=(
                "WARNING: TROPOMI ozone is TOTAL COLUMN (Dobson Units), NOT surface ozone. "
                "This product must never be presented as surface concentration or compared to NAAQS. "
                "Not downloaded for this repository."
            ),
            acquisition_note="Surface O₃ requires CAMS surface analysis. See acquisition script.",
        ),
        DataSourceCapability(
            source_type="cams_analysis",
            measurement_type="model_analysis",
            available_nov_dec_2025=False,
            units="µg/m³",
            description="CAMS surface ozone analysis (variable 'go3'). Not downloaded.",
            acquisition_note="Run scripts/acquire_historical_pollutants.py with ADS credentials.",
        ),
    ],
    models=[
        ModelCapabilityEntry(
            model_type="lightgbm",
            capability="unsupported",
            training_data="N/A — NAMP PDFs do not report O₃",
            validated_on="N/A",
            known_limitations=["No ground station training data available"],
        ),
        ModelCapabilityEntry(
            model_type="physics",
            capability="experimental",
            training_data="Advection-diffusion proxy only; does not model photochemistry",
            validated_on="Not validated",
            known_limitations=["O₃ formation requires NOx/VOC photochemical model, not simple diffusion"],
        ),
        ModelCapabilityEntry(
            model_type="neural",
            capability="unsupported",
            training_data="N/A",
            validated_on="N/A",
        ),
    ],
    regulatory_standards=[
        RegulatoryStandard(
            name="India NAAQS 2009",
            regulator="CPCB",
            pollutant="o3",
            averaging_period="8h",
            concentration_limit=180.0,
            unit="µg/m³",
            measurement_type_required="surface_concentration",
            notes=(
                "CRITICAL: Total-column ozone (DU) must NOT be compared to this threshold. "
                "Only surface ozone from stations or CAMS analysis is applicable."
            ),
        ),
    ],
    known_limitations=[
        "TROPOMI ozone is total column (DU), NOT surface ozone — must never be mislabeled.",
        "No surface O₃ raster data for Nov–Dec 2025 in this repository.",
        "No trained model. Physics diffusion model is scientifically inappropriate for O₃.",
        "O₃ photochemical formation requires NOx/VOC chemistry not in current pipeline.",
    ],
    validation_status="Unsupported. No production or validated model. No appropriate raster data.",
)


# ── Public registry ───────────────────────────────────────────────────────────

POLLUTANT_REGISTRY: dict[str, RegistryEntry] = {
    "no2": _NO2_ENTRY,
    "pm25": _PM25_ENTRY,
    "pm10": _PM10_ENTRY,
    "so2": _SO2_ENTRY,
    "co": _CO_ENTRY,
    "o3": _O3_ENTRY,
}


def get_registry_entry(key: str) -> RegistryEntry:
    """Return the full registry entry for a pollutant key. Raises KeyError if unknown."""
    entry = POLLUTANT_REGISTRY.get(key.lower())
    if entry is None:
        raise KeyError(
            f"Unknown pollutant '{key}'. Supported: {list(POLLUTANT_REGISTRY)}"
        )
    return entry


def get_model_capability(pollutant: str, model_type: str) -> ModelCapability:
    """Return the model capability level for a pollutant + model type combination."""
    try:
        entry = get_registry_entry(pollutant)
        m = entry.get_model(model_type)
        return m.capability if m else "unsupported"
    except KeyError:
        return "unsupported"


def assert_surface_measurement(pollutant: str, measurement_type: str) -> None:
    """Raise ValueError if measurement_type is not surface_concentration.

    Used by compliance agent to block column data from being compared to NAAQS.
    """
    if measurement_type != "surface_concentration":
        raise ValueError(
            f"Compliance check for {pollutant.upper()} requires surface_concentration data. "
            f"Received measurement_type='{measurement_type}'. "
            "Satellite column or model-derived fields cannot be directly compared to NAAQS thresholds."
        )


def list_registry() -> list[dict]:
    """Serializable summary of all registry entries."""
    return [entry.to_dict() for entry in POLLUTANT_REGISTRY.values()]


def list_registry_entries() -> list[RegistryEntry]:
    """Return all registry entries."""
    return list(POLLUTANT_REGISTRY.values())


def get_standard(pollutant: str, averaging_period: str = "24h") -> RegulatoryStandard | None:
    """Return regulatory standard for a pollutant and averaging period (e.g. '24h')."""
    try:
        entry = get_registry_entry(pollutant)
        return entry.get_standard(averaging_period)
    except KeyError:
        return None

