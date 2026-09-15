"""GeoAsk AI — Model Router.

Maintains a registry of available specialist models and selects the
best one for a given task based on capabilities, availability, and
the current hardware environment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import get_settings
from app.models.enums import TaskType

if TYPE_CHECKING:
    from app.specialists.base import BaseSpecialist

logger = logging.getLogger(__name__)


class ModelRouter:
    """Selects and instantiates specialist models for execution."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._registry: dict[str, type["BaseSpecialist"]] = {}
        self._instances: dict[str, "BaseSpecialist"] = {}
        self._register_defaults()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def _register_defaults(self) -> None:
        """Register all built-in specialist classes."""
        from app.specialists.vqa import VQASpecialist
        from app.specialists.captioning import CaptioningSpecialist
        from app.specialists.change_detection import ChangeDetectionSpecialist
        from app.specialists.sar_optical import SAROpticalSpecialist

        self._registry["vqa_specialist"] = VQASpecialist
        self._registry["captioning_specialist"] = CaptioningSpecialist
        self._registry["change_detection_specialist"] = ChangeDetectionSpecialist
        self._registry["sar_optical_specialist"] = SAROpticalSpecialist

    def register(self, name: str, specialist_class: type["BaseSpecialist"]) -> None:
        """Register a custom specialist."""
        self._registry[name] = specialist_class
        logger.info("Registered specialist: %s", name)

    # ------------------------------------------------------------------ #
    # Selection & instantiation
    # ------------------------------------------------------------------ #

    def get_specialist(self, name: str) -> "BaseSpecialist":
        """Get or create a specialist instance by name.

        Raises KeyError if the name is not registered.
        """
        if name not in self._instances:
            if name not in self._registry:
                raise KeyError(
                    f"Unknown specialist '{name}'. "
                    f"Available: {list(self._registry.keys())}"
                )
            self._instances[name] = self._registry[name]()
            logger.info("Instantiated specialist: %s", name)

        return self._instances[name]

    def select_for_task(self, task_type: TaskType) -> "BaseSpecialist":
        """Select the best specialist for a given task type.

        Uses the default mapping, with fallback to VQA specialist.
        """
        mapping: dict[TaskType, str] = {
            TaskType.VQA: "vqa_specialist",
            TaskType.CAPTIONING: "captioning_specialist",
            TaskType.SCENE_CLASSIFICATION: "captioning_specialist",
            TaskType.CHANGE_DETECTION: "change_detection_specialist",
            TaskType.CHANGE_DESCRIPTION: "change_detection_specialist",
            TaskType.CHANGE_VQA: "change_detection_specialist",
            TaskType.SAR_ANALYSIS: "sar_optical_specialist",
            TaskType.OPTICAL_SAR_FUSION: "sar_optical_specialist",
            TaskType.GROUNDING: "vqa_specialist",
            TaskType.GENERAL: "vqa_specialist",
        }
        name = mapping.get(task_type, "vqa_specialist")
        return self.get_specialist(name)

    def list_available(self) -> list[dict[str, str]]:
        """List all registered specialists and their capabilities."""
        result = []
        for name, cls in self._registry.items():
            instance = self.get_specialist(name)
            caps = [t.value for t in instance.get_capabilities()]
            result.append({
                "name": name,
                "class": cls.__name__,
                "capabilities": ", ".join(caps),
            })
        return result
