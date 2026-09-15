"""GeoAsk AI — Response Engine.

Assembles the final AnalysisResponse and generates the downloadable PDF report.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.enums import AnalysisStatus, InputMode, TaskType
from app.models.schemas import (
    AnalysisResponse,
    ConfidenceScore,
    EvidenceBundle,
    ExecutionPlan,
    ExecutionTrace,
    QueryIntent,
    SpecialistResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ResponseEngine:
    """Assembles final response and generates reports."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def assemble(
        self,
        session_id: str,
        query: str,
        intent: QueryIntent,
        plan: ExecutionPlan,
        specialist_results: list[SpecialistResult],
        evidence: EvidenceBundle,
        validation: ValidationResult,
        confidence: ConfidenceScore,
        trace: ExecutionTrace,
        input_mode: InputMode,
        processing_time: float,
    ) -> AnalysisResponse:
        """Assemble the complete AnalysisResponse."""
        
        # Combine textual answers from specialists
        answers = [r.answer for r in specialist_results if r.answer]
        final_answer = "\n\n".join(answers) if answers else "No answer could be generated."

        # Collect visual outputs
        visuals = []
        for r in specialist_results:
            visuals.extend(r.visual_outputs)

        # Collect tasks performed
        tasks = [r.task_type for r in specialist_results if not r.error]

        # Generate report (fire-and-forget or async await)
        report_path = None
        try:
            report_path = self._generate_report(
                session_id=session_id,
                query=query,
                answer=final_answer,
                confidence=confidence,
                visuals=visuals,
                metrics=self._aggregate_metrics(specialist_results),
            )
        except Exception as exc:
            logger.warning("Failed to generate PDF report: %s", exc)

        return AnalysisResponse(
            session_id=session_id,
            status=AnalysisStatus.COMPLETED,
            answer=final_answer,
            confidence=confidence,
            evidence=evidence,
            validation=validation,
            execution_trace=trace,
            visual_outputs=visuals,
            report_path=report_path,
            input_mode=input_mode,
            tasks_performed=tasks,
            processing_time_seconds=round(processing_time, 2),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate_metrics(results: list[SpecialistResult]) -> dict[str, Any]:
        """Combine metrics from all specialists."""
        agg = {}
        for r in results:
            agg.update(r.metrics)
        return agg

    def _generate_report(
        self,
        session_id: str,
        query: str,
        answer: str,
        confidence: ConfidenceScore,
        visuals: list[str],
        metrics: dict[str, Any],
    ) -> str:
        """Generate a basic PDF report using reportlab.
        
        Returns the path to the generated PDF.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
        except ImportError:
            logger.warning("reportlab not installed, skipping PDF generation")
            return ""

        output_path = self.settings.reports_path / f"GeoAsk_Report_{session_id}.pdf"
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=0))
        
        Story = []
        
        # Header
        Story.append(Paragraph(f"<b>GeoAsk AI — Analysis Report</b>", styles["Heading1"]))
        Story.append(Paragraph(f"Session ID: {session_id}", styles["Normal"]))
        Story.append(Paragraph(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", styles["Normal"]))
        Story.append(Spacer(1, 20))
        
        # Query
        Story.append(Paragraph(f"<b>User Query:</b>", styles["Heading2"]))
        Story.append(Paragraph(query, styles["Normal"]))
        Story.append(Spacer(1, 15))
        
        # Answer
        Story.append(Paragraph(f"<b>Analysis Output:</b>", styles["Heading2"]))
        Story.append(Paragraph(answer.replace('\n', '<br/>'), styles["Normal"]))
        Story.append(Spacer(1, 15))
        
        # Confidence & Metrics
        Story.append(Paragraph(f"<b>Confidence Score:</b> {confidence.overall:.0%} ({confidence.level.value.upper()})", styles["Normal"]))
        if metrics:
            Story.append(Spacer(1, 10))
            Story.append(Paragraph(f"<b>Extracted Metrics:</b>", styles["Heading3"]))
            for k, v in metrics.items():
                Story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        
        Story.append(Spacer(1, 20))
        
        # Visuals
        if visuals:
            Story.append(Paragraph(f"<b>Visual Evidence:</b>", styles["Heading2"]))
            Story.append(Spacer(1, 10))
            for vis_path in visuals:
                if Path(vis_path).exists():
                    try:
                        # Add image with max width of 400
                        img = Image(vis_path, width=400, height=300)
                        img.preserveAspectRatio = True
                        Story.append(img)
                        Story.append(Spacer(1, 15))
                    except Exception as e:
                        logger.warning(f"Could not embed image {vis_path}: {e}")

        # Build PDF
        doc.build(Story)
        logger.info(f"Generated PDF report: {output_path}")
        
        return str(output_path)
