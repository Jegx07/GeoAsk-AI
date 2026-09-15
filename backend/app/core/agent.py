"""GeoAsk AI — Agentic Orchestrator.

The central "brain" that coordinates the full analysis pipeline:
query understanding → task planning → specialist execution →
evidence fusion → validation → confidence estimation → response.

Every decision is logged to an ExecutionTrace for full auditability.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.config import get_settings
from app.core.input_manager import InputManager
from app.core.model_router import ModelRouter
from app.core.query_understanding import QueryUnderstanding
from app.core.rs_processor import RSProcessor
from app.core.task_planner import TaskPlanner
from app.engines.confidence import ConfidenceEstimator
from app.engines.evidence import EvidenceEngine
from app.engines.response import ResponseEngine
from app.engines.validation import ValidationEngine
from app.models.enums import AnalysisStatus, InputMode, StepStatus
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ExecutionPlan,
    ExecutionTrace,
    ImageInput,
    PlanStep,
    SpecialistResult,
)

logger = logging.getLogger(__name__)


class GeoAskAgent:
    """The agentic orchestrator that drives the entire analysis pipeline.

    This is a stateful, single-request agent — each call to ``run()``
    processes one user request end-to-end and returns a complete
    AnalysisResponse with execution trace.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.input_manager = InputManager()
        self.rs_processor = RSProcessor()
        self.query_understanding = QueryUnderstanding()
        self.task_planner = TaskPlanner()
        self.model_router = ModelRouter()
        self.evidence_engine = EvidenceEngine()
        self.validation_engine = ValidationEngine()
        self.confidence_estimator = ConfidenceEstimator()
        self.response_engine = ResponseEngine()

    async def run(
        self,
        request: AnalysisRequest,
        progress_callback: Optional[Any] = None,
    ) -> AnalysisResponse:
        """Execute the full analysis pipeline for a user request.

        Parameters
        ----------
        request : AnalysisRequest
            The user's query + images.
        progress_callback : callable, optional
            Called with (event_type, message) for real-time progress updates.

        Returns
        -------
        AnalysisResponse with answer, evidence, confidence, and trace.
        """
        trace = ExecutionTrace()
        t_start = time.time()

        def _emit(event_type: str, msg: str, step_id: str = "", **kw: Any) -> None:
            trace.add(event_type, msg, step_id=step_id, **kw)
            if progress_callback:
                try:
                    progress_callback(event_type, msg)
                except Exception:
                    pass

        try:
            return await self._run_pipeline(request, trace, _emit, t_start)
        except Exception as exc:
            logger.exception("Agent pipeline failed")
            _emit("error", f"Pipeline failed: {exc}")
            return AnalysisResponse(
                session_id=request.session_id,
                status=AnalysisStatus.FAILED,
                error=str(exc),
                execution_trace=trace,
                processing_time_seconds=time.time() - t_start,
            )

    async def _run_pipeline(
        self,
        request: AnalysisRequest,
        trace: ExecutionTrace,
        emit: Any,
        t_start: float,
    ) -> AnalysisResponse:
        """Core pipeline logic, separated for clean error handling."""
        images = request.images
        input_mode = request.input_mode

        # ── Step 1: Input Understanding ──────────────────────────────
        emit("pipeline_start", f"Starting analysis for query: {request.query[:120]}")
        emit("step_start", "Classifying input mode and validating images")

        if not images:
            emit("warning", "No images provided — proceeding with text-only query")

        emit(
            "step_complete",
            f"Input mode: {input_mode.value} | Images: {len(images)}",
            input_mode=input_mode.value,
            image_count=len(images),
        )

        # ── Step 2: Query Understanding ──────────────────────────────
        emit("step_start", "Analysing user query to determine intent")

        intent = await self.query_understanding.analyze(
            query=request.query,
            images=images,
            input_mode=input_mode,
        )

        emit(
            "step_complete",
            f"Query classified as '{intent.primary_task.value}' "
            f"(confidence: {intent.confidence:.0%})",
            primary_task=intent.primary_task.value,
            secondary_tasks=[t.value for t in intent.secondary_tasks],
            reasoning=intent.reasoning,
        )

        # ── Step 3: Task Planning ────────────────────────────────────
        emit("step_start", "Creating execution plan")

        plan = self.task_planner.plan(intent, images, input_mode)

        emit(
            "step_complete",
            f"Plan created with {len(plan.steps)} step(s)",
            steps=[
                {"step": s.step_number, "task": s.task_type.value, "specialist": s.specialist_name}
                for s in plan.steps
            ],
        )

        # ── Step 4: Specialist Execution ─────────────────────────────
        emit("step_start", "Executing specialist models")

        specialist_results: list[SpecialistResult] = []

        for step in plan.steps:
            result = await self._execute_step(step, images, input_mode, trace, emit)
            specialist_results.append(result)

        # ── Step 5: Evidence Fusion ──────────────────────────────────
        emit("step_start", "Collecting and fusing evidence")

        evidence_bundle = self.evidence_engine.collect(
            specialist_results=specialist_results,
            images=images,
            plan=plan,
        )

        emit("step_complete", f"Collected {len(evidence_bundle.items)} evidence items")

        # ── Step 6: Validation ───────────────────────────────────────
        emit("step_start", "Validating results for consistency")

        validation = self.validation_engine.validate(
            specialist_results=specialist_results,
            evidence=evidence_bundle,
            intent=intent,
        )

        emit(
            "step_complete",
            f"Validation {'PASSED' if validation.overall_passed else 'FAILED'} "
            f"(score: {validation.overall_score:.2f})",
        )

        # ── Step 7: Confidence Estimation ────────────────────────────
        emit("step_start", "Estimating confidence")

        confidence = self.confidence_estimator.estimate(
            specialist_results=specialist_results,
            validation=validation,
            intent=intent,
        )

        emit(
            "step_complete",
            f"Confidence: {confidence.level.value} ({confidence.overall:.0%})",
        )

        # ── Step 8: Response Assembly ────────────────────────────────
        emit("step_start", "Assembling final response")

        total_time = time.time() - t_start

        response = await self.response_engine.assemble(
            session_id=request.session_id,
            query=request.query,
            intent=intent,
            plan=plan,
            specialist_results=specialist_results,
            evidence=evidence_bundle,
            validation=validation,
            confidence=confidence,
            trace=trace,
            input_mode=input_mode,
            processing_time=total_time,
        )

        emit(
            "pipeline_complete",
            f"Analysis complete in {total_time:.1f}s | "
            f"Confidence: {confidence.level.value}",
        )

        trace.total_duration_seconds = total_time
        response.execution_trace = trace

        return response

    async def _execute_step(
        self,
        step: PlanStep,
        images: list[ImageInput],
        input_mode: InputMode,
        trace: ExecutionTrace,
        emit: Any,
    ) -> SpecialistResult:
        """Execute a single plan step via the appropriate specialist."""
        step.status = StepStatus.RUNNING
        emit(
            "specialist_start",
            f"Step {step.step_number}: {step.description}",
            step_id=step.step_id,
            specialist=step.specialist_name,
        )

        t0 = time.time()

        try:
            specialist = self.model_router.get_specialist(step.specialist_name)
            result = await specialist.execute(
                images=images,
                query=step.parameters.get("query", ""),
                context=step.parameters,
            )
            result.processing_time_seconds = time.time() - t0
            step.status = StepStatus.COMPLETED

            emit(
                "specialist_complete",
                f"Step {step.step_number} complete ({result.processing_time_seconds:.1f}s)",
                step_id=step.step_id,
                answer_preview=result.answer[:200] if result.answer else "",
            )

        except Exception as exc:
            step.status = StepStatus.FAILED
            result = SpecialistResult(
                specialist_name=step.specialist_name,
                task_type=step.task_type,
                error=str(exc),
                processing_time_seconds=time.time() - t0,
            )
            emit(
                "specialist_failed",
                f"Step {step.step_number} failed: {exc}",
                step_id=step.step_id,
            )
            logger.error("Specialist %s failed: %s", step.specialist_name, exc)

        return result
