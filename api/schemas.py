"""
TemporalShield / NexusTrace — API Data Contract & Schemas
Pydantic schemas establishing the strict contract between the FastAPI backend and frontend UI.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class IncidentBenchmark(BaseModel):
    incident_name: str
    quantum: str
    parallel_description: Optional[str] = None
    description: Optional[str] = None
    pmla_sections: Optional[str] = None
    regulatory_trigger: Optional[str] = None


class SHAPFeature(BaseModel):
    feature: str
    value: Optional[Union[float, int, str]] = None
    importance: Optional[Union[float, str]] = None
    reason: Optional[str] = None
    contribution: Optional[Union[float, str]] = None
    explanation: Optional[str] = None


class AlertSummary(BaseModel):
    alert_id: str
    account_id: str
    timestamp: str
    composite_risk_score: float = Field(..., ge=0.0, le=100.0)
    severity: str = Field(..., description="CRITICAL | HIGH | MEDIUM | LOW")
    status: str = Field(default="ACTIVE")
    alert_title: str
    summary: str
    fired_modules: List[str] = Field(default_factory=list)


class AlertDetail(AlertSummary):
    module_scores: Dict[str, float] = Field(default_factory=dict)
    top_shap_features: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_points: List[str] = Field(default_factory=list)
    historical_incident_parallel: Optional[Dict[str, Any]] = None
    recommended_action: str
    raw_event: Optional[Dict[str, Any]] = None


class EvidenceDossier(BaseModel):
    alert_id: str
    account_id: str
    timestamp: str
    composite_risk_score: float
    severity: str
    status: str
    alert_title: str
    summary: str
    modules: Dict[str, Any] = Field(default_factory=dict)
    top_shap_features: List[Dict[str, Any]] = Field(default_factory=list)
    historical_incident_parallel: Optional[Dict[str, Any]] = None
    historical_incident_benchmark: Optional[Dict[str, Any]] = None
    recommended_action: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    graph_neighborhood: Optional[Dict[str, Any]] = None


class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str = Field(..., description="employee | account | counterparty")
    group: Optional[str] = None
    color: Optional[str] = None
    size: Optional[int] = None
    role: Optional[str] = None
    branch: Optional[str] = None
    access_count: Optional[int] = None
    total_outflow: Optional[float] = None
    total_inflow: Optional[float] = None
    last_seen: Optional[str] = None


class GraphEdge(BaseModel):
    id: str
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    edge_type: str = Field(..., description="ACCESS | TRANSFER | OWNERSHIP")
    amount: Optional[float] = None
    timestamp: Optional[str] = None
    weight: Optional[float] = None
    transaction_type: Optional[str] = None
    scenario: Optional[str] = None

    class Config:
        populate_by_name = True


class GraphSnapshot(BaseModel):
    timestamp: str
    node_count: int
    edge_count: int
    displayed_node_count: int
    displayed_edge_count: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    temporal_window_minutes: float = 4.0


class ScenarioTriggerRequest(BaseModel):
    scenario: str = Field(..., description="temporal_link | structuring | circular_transfer | clean_busy")
    speed_multiplier: Optional[float] = Field(default=500.0, description="Demo replay acceleration")


class ScenarioTriggerResponse(BaseModel):
    status: str
    scenario_id: str
    title: Optional[str] = None
    historical_incident: Optional[str] = None
    speed_multiplier: float
    message: str


class ScenarioInfo(BaseModel):
    scenario_id: str
    title: str
    historical_incident: str
    quantum: str
    pmla_context: str
    description: str
    target_model: str
    expected_events: int
    recommended_speed: float
    detection_window: str


class StatsResponse(BaseModel):
    total_alerts: int
    total_access_events: int
    total_transactions: int
    patterns_detected: int
    false_positive_rate: float
    temporal_correlations_detected: int
    circular_rings_detected: int
    structuring_bursts_detected: int
    profile_anomalies_detected: int
    stream_status: str
    uptime_seconds: float


class ResetResponse(BaseModel):
    status: str
    message: str
    timestamp: str
