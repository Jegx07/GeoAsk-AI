"""GeoAsk AI — Enumeration types for the entire pipeline."""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """Types of analysis tasks the system can perform."""

    VQA = "vqa"
    CAPTIONING = "captioning"
    SCENE_CLASSIFICATION = "scene_classification"
    CHANGE_DETECTION = "change_detection"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    SAR_ANALYSIS = "sar_analysis"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    GROUNDING = "grounding"
    GENERAL = "general"


class InputMode(str, Enum):
    """How the user's uploaded images should be interpreted."""

    SINGLE = "single"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR = "optical_sar"
    MULTI_IMAGE = "multi_image"


class ImageModality(str, Enum):
    """Sensor modality of an uploaded image."""

    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    """Supported image file formats."""

    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence categorisation."""

    HIGH = "high"        # > 0.8
    MEDIUM = "medium"    # 0.5 – 0.8
    LOW = "low"          # < 0.5


class StepStatus(str, Enum):
    """Status of an individual execution step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceType(str, Enum):
    """Kind of evidence produced by the pipeline."""

    TEXTUAL = "textual"
    VISUAL = "visual"           # annotated image / heatmap
    SPATIAL = "spatial"         # GeoJSON region
    CHANGE_MASK = "change_mask"
    ATTENTION_MAP = "attention_map"
    FUSED_IMAGE = "fused_image"


class AnalysisStatus(str, Enum):
    """Overall status of an analysis session."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
