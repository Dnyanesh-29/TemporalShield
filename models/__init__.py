"""
TemporalShield — Multi-Tiered Fraud Detection Models Package
Exposes all 5 detection engines and the central SHAP evidence assembler.
"""

from .isolation_forest import TemporalInsiderDetector
from .autoencoder_role import RoleAnomalyDetector
from .lstm_structuring import StructuringDetector
from .xgboost_profile import ProfileMismatchDetector
from .shap_explainer import SHAPEvidenceAssembler

__all__ = [
    "TemporalInsiderDetector",
    "RoleAnomalyDetector",
    "StructuringDetector",
    "ProfileMismatchDetector",
    "SHAPEvidenceAssembler"
]
