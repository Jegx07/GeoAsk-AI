"""GeoAsk AI — Pydantic schemas for the entire pipeline.

Every data structure exchanged between components is defined here so that
the architecture is fully typed and serialisable end-to-end.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.enums import (
    AnalysisStatus,
    ConfidenceLevel,
    EvidenceType,
    ImageFormat,
    ImageModality,
    InputMode,
    StepStatus,
    TaskType,
)


# ============================================================
# Image & Input Models
# ============================================================


class GeoMetadata(BaseModel):
    """Geospatial metadata extracted from a GeoTIFF."""

    crs: Optional[str] = None          # e.g. "EPSG:4326"
    bounds: Optional[list[float]] = None  # [west, south, east, north]
    resolution: Optional[list[float]] = None  # [x_res, y_res] in CRS units
    transform: Optional[list[float]] = None   # Affine transform coefficients
    width: int = 0
    height: int = 0


class ImageInput(BaseModel):
    """Represents a single uploaded / processed image."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    filename: str
    filepath: str                       # absolute path on server
    format: ImageFormat = ImageFormat.UNKNOWN
    modality: ImageModality = ImageModality.UNKNOWN
    band_count: int = 0
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    geo_metadata: Optional[GeoMetadata] = None
    thumbnail_path: Optional[str] = None  # path to RGB preview
    label: Optional[str] = None         # user label, e.g. "T1", "SAR"


class AnalysisRequest(BaseModel):
    """A user's complete request: query + images."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    query: str
    images: list[ImageInput] = []
    input_mode: InputMode = InputMode.SINGLE
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Query Understanding
# ============================================================


class ExtractedEntity(BaseModel):
    """An entity (object, location, concept) extracted from the user query."""

    text: str
    entity_type: str  # "object", "location", "temporal", "attribute"
    confidence: float = 1.0


class QueryIntent(BaseModel):
    """Structured understanding of the user's query."""

    primary_task: TaskType
    secondary_tasks: list[TaskType] = []
    entities: list[ExtractedEntity] = []
    requires_multiple_images: bool = False
    requires_temporal: bool = False
    requires_sar: bool = False
    raw_query: str = ""
    reformulated_query: str = ""
    reasoning: str = ""   # LLM's reasoning about the classification
    confidence: float = 0.0


# ============================================================
# Task Planning
# ============================================================


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    step_number: int
    task_type: TaskType
    specialist_name: str           # which specialist to invoke
    description: str               # human-readable description
    depends_on: list[str] = []     # step_ids this step depends on
    parameters: dict[str, Any] = {}
    status: StepStatus = StepStatus.PENDING


class ExecutionPlan(BaseModel):
    """Ordered execution plan produced by the Task Planner."""

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    steps: list[PlanStep] = []
    reasoning: str = ""            # planner's reasoning
    estimated_duration_seconds: float = 0.0


# ============================================================
# Specialist Results
# ============================================================


class SpecialistResult(BaseModel):
    """Output from a specialist model execution."""

    specialist_name: str
    task_type: TaskType
    answer: str = ""
    raw_output: Any = None         # model-specific raw output
    visual_outputs: list[str] = []  # paths to generated images
    spatial_outputs: list[dict[str, Any]] = []  # GeoJSON features
    metrics: dict[str, float] = {}  # e.g. {"change_percentage": 12.5}
    model_confidence: float = 0.0
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


# ============================================================
# Evidence
# ============================================================


class Evidence(BaseModel):
    """A single piece of evidence supporting the answer."""

    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    evidence_type: EvidenceType
    description: str
    source_step: str = ""          # which plan step produced this
    content: Optional[str] = None  # text content or file path
    metadata: dict[str, Any] = {}


class EvidenceBundle(BaseModel):
    """All evidence collected for an analysis."""

    items: list[Evidence] = []
    summary: str = ""


# ============================================================
# Validation
# ============================================================


class ValidationCheck(BaseModel):
    """Result of a single validation check."""

    check_name: str
    passed: bool
    score: float = 1.0
    message: str = ""


class ValidationResult(BaseModel):
    """Aggregated validation outcome."""

    checks: list[ValidationCheck] = []
    overall_passed: bool = True
    overall_score: float = 1.0
    warnings: list[str] = []


# ============================================================
# Confidence
# ============================================================


class ConfidenceBreakdown(BaseModel):
    """Individual factor contributing to confidence."""

    factor: str
    score: float
    weight: float = 1.0
    explanation: str = ""


class ConfidenceScore(BaseModel):
    """Multi-factor confidence estimation."""

    overall: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.LOW
    breakdown: list[ConfidenceBreakdown] = []
    explanation: str = ""

    @staticmethod
    def from_score(score: float, breakdown: list[ConfidenceBreakdown] | None = None) -> "ConfidenceScore":
        if score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        return ConfidenceScore(
            overall=round(score, 3),
            level=level,
            breakdown=breakdown or [],
        )


# ============================================================
# Execution Trace
# ============================================================


class TraceEvent(BaseModel):
    """A single event in the execution trace (audit trail)."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    step_id: str = ""
    event_type: str  # "step_start", "step_complete", "model_call", "decision", etc.
    message: str
    details: dict[str, Any] = {}
    duration_ms: Optional[float] = None


class ExecutionTrace(BaseModel):
    """Complete audit trail of the analysis."""

    events: list[TraceEvent] = []
    total_duration_seconds: float = 0.0

    def add(self, event_type: str, message: str, step_id: str = "", **details: Any) -> None:
        self.events.append(
            TraceEvent(
                event_type=event_type,
                message=message,
                step_id=step_id,
                details=details,
            )
        )


# ============================================================
# Final Response
# ============================================================


class AnalysisResponse(BaseModel):
    """Complete response returned to the user."""

    session_id: str
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    answer: str = ""
    confidence: ConfidenceScore = ConfidenceScore()
    evidence: EvidenceBundle = EvidenceBundle()
    validation: ValidationResult = ValidationResult()
    execution_trace: ExecutionTrace = ExecutionTrace()
    visual_outputs: list[str] = []    # paths to output images
    report_path: Optional[str] = None  # path to PDF report
    input_mode: InputMode = InputMode.SINGLE
    tasks_performed: list[TaskType] = []
    processing_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
