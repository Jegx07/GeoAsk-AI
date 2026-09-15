"""GeoAsk AI — Task Planner.

Decomposes a classified query intent into an ordered execution plan
of specialist model invocations.
"""

from __future__ import annotations

import logging

from app.models.enums import InputMode, TaskType, StepStatus
from app.models.schemas import ExecutionPlan, ImageInput, PlanStep, QueryIntent

logger = logging.getLogger(__name__)

# Mapping from task type to the specialist that handles it
_TASK_SPECIALIST_MAP: dict[TaskType, str] = {
    TaskType.VQA: "vqa_specialist",
    TaskType.CAPTIONING: "captioning_specialist",
    TaskType.SCENE_CLASSIFICATION: "captioning_specialist",  # reuse captioning
    TaskType.CHANGE_DETECTION: "change_detection_specialist",
    TaskType.CHANGE_DESCRIPTION: "change_detection_specialist",
    TaskType.CHANGE_VQA: "change_detection_specialist",
    TaskType.SAR_ANALYSIS: "sar_optical_specialist",
    TaskType.OPTICAL_SAR_FUSION: "sar_optical_specialist",
    TaskType.GROUNDING: "vqa_specialist",  # grounding via VQA with spatial prompt
    TaskType.GENERAL: "vqa_specialist",
}


class TaskPlanner:
    """Produces an ExecutionPlan from a QueryIntent."""

    def plan(
        self,
        intent: QueryIntent,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> ExecutionPlan:
        """Create an execution plan for the given intent.

        The plan may contain multiple steps if the query requires
        chaining specialists (e.g. change detection → change description).
        """
        steps: list[PlanStep] = []
        step_num = 0

        # Gather all tasks to execute (primary + secondary)
        all_tasks = [intent.primary_task] + intent.secondary_tasks
        # Remove duplicates while preserving order
        seen = set()
        unique_tasks = []
        for t in all_tasks:
            if t not in seen:
                seen.add(t)
                unique_tasks.append(t)

        prev_step_id: str | None = None

        for task in unique_tasks:
            step_num += 1
            specialist = _TASK_SPECIALIST_MAP.get(task, "vqa_specialist")
            description = self._describe_step(task, images, input_mode)

            step = PlanStep(
                step_number=step_num,
                task_type=task,
                specialist_name=specialist,
                description=description,
                depends_on=[prev_step_id] if prev_step_id else [],
                parameters=self._build_parameters(task, intent, images, input_mode),
                status=StepStatus.PENDING,
            )
            steps.append(step)
            prev_step_id = step.step_id

        # Estimate duration
        est = len(steps) * 5.0  # rough: 5s per step

        plan = ExecutionPlan(
            steps=steps,
            reasoning=self._build_reasoning(intent, unique_tasks),
            estimated_duration_seconds=est,
        )

        logger.info(
            "Created execution plan: %d steps for query '%s'",
            len(steps),
            intent.raw_query[:80],
        )
        return plan

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _describe_step(
        task: TaskType,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> str:
        """Generate a human-readable description of a plan step."""
        descriptions: dict[TaskType, str] = {
            TaskType.VQA: "Answer the user's question about the remote-sensing image",
            TaskType.CAPTIONING: "Generate a detailed scene description of the image",
            TaskType.SCENE_CLASSIFICATION: "Classify the scene type in the image",
            TaskType.CHANGE_DETECTION: "Detect and localise changes between the two temporal images",
            TaskType.CHANGE_DESCRIPTION: "Describe the detected changes in natural language",
            TaskType.CHANGE_VQA: "Answer the user's question about changes between images",
            TaskType.SAR_ANALYSIS: "Analyse the SAR image for features and properties",
            TaskType.OPTICAL_SAR_FUSION: "Perform joint analysis of the optical and SAR image pair",
            TaskType.GROUNDING: "Locate and highlight the requested objects in the image",
            TaskType.GENERAL: "Process the general query about the imagery",
        }
        return descriptions.get(task, f"Execute {task.value} task")

    @staticmethod
    def _build_parameters(
        task: TaskType,
        intent: QueryIntent,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> dict:
        """Build task-specific parameters for a specialist."""
        params: dict = {
            "query": intent.reformulated_query or intent.raw_query,
            "input_mode": input_mode.value,
        }

        if task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA):
            params["temporal"] = True
            if len(images) >= 2:
                params["t1_image_id"] = images[0].id
                params["t2_image_id"] = images[1].id

        if task in (TaskType.SAR_ANALYSIS, TaskType.OPTICAL_SAR_FUSION):
            params["sar_mode"] = True

        if intent.entities:
            params["entities"] = [e.text for e in intent.entities]

        return params

    @staticmethod
    def _build_reasoning(intent: QueryIntent, tasks: list[TaskType]) -> str:
        """Build reasoning string for the plan."""
        task_names = ", ".join(t.value for t in tasks)
        return (
            f"Query classified as '{intent.primary_task.value}'. "
            f"Planned tasks: [{task_names}]. "
            f"Intent reasoning: {intent.reasoning}"
        )
