"""Evidence schema and verification logic for the AirQ Insight Report Agent.

Provides typed data representations of scientific evidence to prevent
hallucinations, fabricated evaluation metrics, and unsupported policy claims.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class AudienceType(str, Enum):
    RESEARCHER = "researcher"
    GOVERNMENT = "government"
    PUBLIC = "public"


class ValidationStatus(str, Enum):
    PRELIMINARY = "preliminary"
    CALIBRATED = "calibrated"
    INDEPENDENTLY_VALIDATED = "independently_validated"
    UNVERIFIED = "unverified"


@dataclass
class EvalMetricsEvidence:
    """Rigorous evaluation metrics container.

    Distinguishes hold-out spatial/temporal validation from synthetic testing.
    """
    spatial_r2: float | None = None
    spatial_rmse: float | None = None
    persistence_skill_score: float | None = None  # 1.0 - (rmse_model / rmse_persistence)
    temporal_r2: float | None = None
    n_validation_samples: int = 0
    is_synthetic_baseline: bool = False
    validation_dataset_name: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spatial_r2": self.spatial_r2,
            "spatial_rmse": self.spatial_rmse,
            "persistence_skill_score": self.persistence_skill_score,
            "temporal_r2": self.temporal_r2,
            "n_validation_samples": self.n_validation_samples,
            "is_synthetic_baseline": self.is_synthetic_baseline,
            "validation_dataset_name": self.validation_dataset_name,
        }


@dataclass
class HotspotEvidence:
    """Evidence grounding an identified high-concentration cluster."""
    cluster_id: str
    lat: float
    lon: float
    peak_value: float
    unit: str
    nearest_known_source: str | None = None
    distance_to_source_km: float | None = None
    confidence_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportEvidence:
    """Rigorous structured evidence bundle passed to reporting agent narrative generators."""
    location_label: str
    observation_period: str
    data_sources: list[str]
    spatial_resolution_km: float
    model_version: str
    validation_status: ValidationStatus
    evaluation_metrics: EvalMetricsEvidence
    hotspots: list[HotspotEvidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_label": self.location_label,
            "observation_period": self.observation_period,
            "data_sources": self.data_sources,
            "spatial_resolution_km": self.spatial_resolution_km,
            "model_version": self.model_version,
            "validation_status": self.validation_status.value,
            "evaluation_metrics": self.evaluation_metrics.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "limitations": self.limitations,
            "caveats": self.caveats,
        }


def validate_evidence(evidence: ReportEvidence) -> tuple[bool, list[str]]:
    """Audit and validate an evidence bundle before feeding it to narrative generation.

    Rejects claims where:
    - Evaluation metrics claim high accuracy without any validation samples
    - Regulatory source attribution is made without proximity data
    - Synthetic inputs are claimed as independently validated observations

    Returns
    -------
    (is_valid, validation_errors)
    """
    errors: list[str] = []

    # Check metrics integrity
    m = evidence.evaluation_metrics
    if m.n_validation_samples <= 0 and (m.spatial_r2 is not None or m.temporal_r2 is not None):
        errors.append("Validation metrics present despite 0 reported validation samples.")

    if m.is_synthetic_baseline and evidence.validation_status == ValidationStatus.INDEPENDENTLY_VALIDATED:
        errors.append("Cannot claim independent validation status when evaluation uses synthetic baseline.")

    # Check hotspot causal attribution grounding
    for h in evidence.hotspots:
        if h.nearest_known_source is not None and h.distance_to_source_km is None:
            errors.append(f"Hotspot {h.cluster_id} names source '{h.nearest_known_source}' without proximity distance.")

    # Ensure limitations exist for preliminary models
    if evidence.validation_status == ValidationStatus.PRELIMINARY and len(evidence.limitations) == 0:
        errors.append("Preliminary models must explicitly list limitations.")

    return len(errors) == 0, errors
