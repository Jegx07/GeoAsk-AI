"""GeoAsk AI — Base Specialist Interface.

All specialist models must inherit from BaseSpecialist and implement
the execute(), get_capabilities(), and estimate_confidence() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.enums import TaskType
from app.models.schemas import ImageInput, SpecialistResult


class BaseSpecialist(ABC):
    """Abstract base class for all specialist models.

    A specialist encapsulates a specific AI capability
    (VQA, captioning, change detection, SAR analysis, etc.)
    and provides a uniform interface for the agent to invoke it.
    """

    @abstractmethod
    async def execute(
        self,
        images: list[ImageInput],
        query: str,
        context: dict,
    ) -> SpecialistResult:
        """Execute the specialist's analysis.

        Parameters
        ----------
        images : list[ImageInput]
            The uploaded image(s).
        query : str
            The user's query (reformulated by query understanding).
        context : dict
            Additional context from the plan step (parameters, prior results).

        Returns
        -------
        SpecialistResult with answer, visual outputs, and confidence.
        """

    @abstractmethod
    def get_capabilities(self) -> list[TaskType]:
        """Return the list of task types this specialist can handle."""

    @abstractmethod
    def estimate_confidence(self, result: SpecialistResult) -> float:
        """Estimate confidence in the specialist's result.

        Returns a float in [0, 1].
        """
