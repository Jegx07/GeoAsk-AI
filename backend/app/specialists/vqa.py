"""GeoAsk AI — VQA Specialist.

Visual Question Answering for single remote-sensing images.
Uses Google Gemini Vision API with remote-sensing-adapted prompting.
"""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path

from app.config import get_settings
from app.models.enums import TaskType
from app.models.schemas import ImageInput, SpecialistResult
from app.specialists.base import BaseSpecialist
from app.utils.image import image_to_pil, pil_to_bytes

logger = logging.getLogger(__name__)

# Remote-sensing-specific system prompt for VQA
_RS_VQA_SYSTEM_PROMPT = """\
You are an expert remote-sensing image analyst. You are viewing a satellite or aerial image
from a nadir (top-down) perspective.

When answering questions about this image, consider:
- This is a remote-sensing image, NOT a ground-level photograph
- Objects appear from above (rooftops, not facades)
- Scale: individual pixels may represent metres or tens of metres
- Common features: buildings, roads, vegetation, water bodies, agricultural fields,
  bare soil, urban areas, industrial zones, forests, rivers, coastlines
- Spectral information: different band combinations reveal different information
- SAR images appear as greyscale with speckle noise; bright areas = high backscatter

Provide specific, evidence-based answers. If you can estimate counts, areas, or proportions,
do so. If uncertain, state your uncertainty clearly.

Always structure your response as:
1. Direct answer to the question
2. Supporting evidence from the image
3. Any relevant caveats or uncertainties
"""


class VQASpecialist(BaseSpecialist):
    """Single-image Visual Question Answering using Gemini Vision."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def execute(
        self,
        images: list[ImageInput],
        query: str,
        context: dict,
    ) -> SpecialistResult:
        """Answer a question about a remote-sensing image."""
        if not images:
            return SpecialistResult(
                specialist_name="vqa_specialist",
                task_type=TaskType.VQA,
                answer="No image was provided to analyse.",
                model_confidence=0.0,
                error="No images provided",
            )

        image_input = images[0]

        try:
            answer = await self._query_gemini(image_input, query)
            confidence = self.estimate_confidence(
                SpecialistResult(
                    specialist_name="vqa_specialist",
                    task_type=TaskType.VQA,
                    answer=answer,
                )
            )
            return SpecialistResult(
                specialist_name="vqa_specialist",
                task_type=TaskType.VQA,
                answer=answer,
                model_confidence=confidence,
            )
        except Exception as exc:
            logger.error("VQA failed: %s", exc)
            # Fallback to a descriptive error response
            return SpecialistResult(
                specialist_name="vqa_specialist",
                task_type=TaskType.VQA,
                answer=f"I was unable to fully analyse this image due to a processing error. "
                       f"Based on the available information: the image appears to be a "
                       f"{image_input.modality.value} remote-sensing image "
                       f"({image_input.width}x{image_input.height} pixels, "
                       f"{image_input.band_count} bands).",
                model_confidence=0.2,
                error=str(exc),
            )

    def get_capabilities(self) -> list[TaskType]:
        return [TaskType.VQA, TaskType.GROUNDING, TaskType.GENERAL]

    def estimate_confidence(self, result: SpecialistResult) -> float:
        """Heuristic confidence based on answer quality."""
        if result.error:
            return 0.2
        answer = result.answer
        if not answer:
            return 0.1

        score = 0.6  # base confidence for any answer

        # Boost for longer, more detailed answers
        if len(answer) > 200:
            score += 0.1
        if len(answer) > 500:
            score += 0.05

        # Boost for evidence language
        evidence_phrases = ["i can see", "the image shows", "visible in", "appears to be",
                            "based on", "evidence", "observation"]
        for phrase in evidence_phrases:
            if phrase in answer.lower():
                score += 0.03

        # Penalty for uncertainty language
        uncertain_phrases = ["i'm not sure", "unclear", "cannot determine", "hard to tell"]
        for phrase in uncertain_phrases:
            if phrase in answer.lower():
                score -= 0.05

        return max(0.1, min(1.0, score))

    # ------------------------------------------------------------------ #
    # Private methods
    # ------------------------------------------------------------------ #

    async def _query_gemini(self, image_input: ImageInput, query: str) -> str:
        """Send the image + query to Gemini Vision API."""
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_vision_model)

        # Load image as PIL
        pil_img = image_to_pil(image_input.filepath)

        # Build prompt
        prompt = f"{_RS_VQA_SYSTEM_PROMPT}\n\nUser question: {query}"

        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, [prompt, pil_img]),
            timeout=45,
        )
        return response.text.strip()
