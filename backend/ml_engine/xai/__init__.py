"""XAI package exports."""

from .charts import generate_bar_chart_base64, generate_waterfall_chart_base64
from .shap_service import (
    SHAPService,
    XAIExplanation,
    explain_ensemble_downscaler,
    get_shap_service,
)
from .summaries import (
    compute_relative_contributions,
    generate_detailed_narrative,
    generate_executive_explanation,
)

__all__ = [
    "SHAPService",
    "XAIExplanation",
    "get_shap_service",
    "explain_ensemble_downscaler",
    "generate_waterfall_chart_base64",
    "generate_bar_chart_base64",
    "compute_relative_contributions",
    "generate_executive_explanation",
    "generate_detailed_narrative",
]
