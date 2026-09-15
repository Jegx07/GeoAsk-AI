"""GeoAsk AI — Evidence Engine.

Collects and structures evidence from specialist outputs into a
unified EvidenceBundle for the final response.
"""

from __future__ import annotations

import logging

from app.models.enums import EvidenceType
from app.models.schemas import (
    Evidence,
    EvidenceBundle,
    ExecutionPlan,
    ImageInput,
    SpecialistResult,
)

logger = logging.getLogger(__name__)


class EvidenceEngine:
    """Aggregates evidence from specialist results."""

    def collect(
        self,
        specialist_results: list[SpecialistResult],
        images: list[ImageInput],
        plan: ExecutionPlan,
    ) -> EvidenceBundle:
        """Collect all evidence items from specialist outputs.

        Creates evidence items for:
        - Textual answers from each specialist
        - Visual outputs (annotated images, masks, heatmaps)
        - Spatial outputs (GeoJSON features)
        - Metrics and measurements
        """
        items: list[Evidence] = []

        for i, result in enumerate(specialist_results):
            step_id = plan.steps[i].step_id if i < len(plan.steps) else ""

            # Textual evidence: the specialist's answer
            if result.answer:
                items.append(
                    Evidence(
                        evidence_type=EvidenceType.TEXTUAL,
                        description=f"Analysis from {result.specialist_name}: {result.task_type.value}",
                        source_step=step_id,
                        content=result.answer,
                        metadata={
                            "specialist": result.specialist_name,
                            "task_type": result.task_type.value,
                            "confidence": result.model_confidence,
                        },
                    )
                )

            # Visual evidence: output images
            for path in result.visual_outputs:
                ev_type = self._classify_visual(path)
                items.append(
                    Evidence(
                        evidence_type=ev_type,
                        description=f"Visual output from {result.specialist_name}",
                        source_step=step_id,
                        content=path,
                        metadata={"filepath": path},
                    )
                )

            # Spatial evidence: GeoJSON
            for feature in result.spatial_outputs:
                items.append(
                    Evidence(
                        evidence_type=EvidenceType.SPATIAL,
                        description=f"Spatial feature from {result.specialist_name}",
                        source_step=step_id,
                        content=None,
                        metadata=feature,
                    )
                )

            # Metrics as textual evidence
            if result.metrics:
                metrics_text = ", ".join(
                    f"{k}: {v}" for k, v in result.metrics.items()
                )
                items.append(
                    Evidence(
                        evidence_type=EvidenceType.TEXTUAL,
                        description=f"Metrics from {result.specialist_name}",
                        source_step=step_id,
                        content=metrics_text,
                        metadata=result.metrics,
                    )
                )

        # Add input image evidence
        for img in images:
            if img.thumbnail_path:
                items.append(
                    Evidence(
                        evidence_type=EvidenceType.VISUAL,
                        description=f"Input image: {img.filename}",
                        content=img.thumbnail_path,
                        metadata={
                            "modality": img.modality.value,
                            "dimensions": f"{img.width}x{img.height}",
                        },
                    )
                )

        # Build summary
        summary = self._build_summary(items, specialist_results)

        logger.info("Evidence collected: %d items", len(items))
        return EvidenceBundle(items=items, summary=summary)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_visual(path: str) -> EvidenceType:
        """Classify a visual output by its filename."""
        lower = path.lower()
        if "mask" in lower:
            return EvidenceType.CHANGE_MASK
        if "heatmap" in lower or "attention" in lower:
            return EvidenceType.ATTENTION_MAP
        if "fusion" in lower or "fused" in lower:
            return EvidenceType.FUSED_IMAGE
        return EvidenceType.VISUAL

    @staticmethod
    def _build_summary(
        items: list[Evidence],
        results: list[SpecialistResult],
    ) -> str:
        """Build a human-readable evidence summary."""
        text_count = sum(1 for e in items if e.evidence_type == EvidenceType.TEXTUAL)
        visual_count = sum(1 for e in items if e.evidence_type in (
            EvidenceType.VISUAL, EvidenceType.CHANGE_MASK,
            EvidenceType.ATTENTION_MAP, EvidenceType.FUSED_IMAGE,
        ))
        spatial_count = sum(1 for e in items if e.evidence_type == EvidenceType.SPATIAL)

        specialists_used = list({r.specialist_name for r in results if not r.error})
        avg_conf = (
            sum(r.model_confidence for r in results if not r.error)
            / max(1, len([r for r in results if not r.error]))
        )

        return (
            f"Evidence from {len(specialists_used)} specialist(s): "
            f"{text_count} textual, {visual_count} visual, {spatial_count} spatial items. "
            f"Average model confidence: {avg_conf:.0%}."
        )
