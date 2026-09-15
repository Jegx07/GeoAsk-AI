"""GeoAsk AI — Captioning Specialist.

Generates structured scene descriptions and captions for
remote-sensing imagery using Gemini Vision.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.models.enums import TaskType
from app.models.schemas import ImageInput, SpecialistResult
from app.specialists.base import BaseSpecialist
from app.utils.image import image_to_pil

logger = logging.getLogger(__name__)

_RS_CAPTIONING_PROMPT = """\
You are an expert remote-sensing image analyst. Generate a comprehensive,
structured description of this satellite/aerial image.

Your description MUST include:

1. **Scene Overview** (1-2 sentences): What type of area is this? (urban, rural, coastal, forest, etc.)

2. **Land Cover Analysis**: List all visible land cover types with approximate proportions:
   - Built-up / Urban areas
   - Vegetation (forest, cropland, grassland)
   - Water bodies
   - Bare soil / Rock
   - Roads / Infrastructure
   - Other notable features

3. **Key Features**: Describe specific notable features:
   - Buildings (density, arrangement, size)
   - Road network (grid pattern, highways, etc.)
   - Water features (rivers, lakes, coastline)
   - Vegetation patterns (regular cropland vs natural)
   - Industrial or special-purpose areas

4. **Spatial Patterns**: Describe spatial organisation:
   - Is the area developed uniformly or has distinct zones?
   - What is the dominant pattern? (grid, organic, radial)

5. **Image Properties**: Note the approximate spatial resolution,
   season indicators (if visible), and any atmospheric effects.

Be specific and quantitative where possible. Avoid vague descriptions.
"""

_RS_BRIEF_CAPTION_PROMPT = """\
You are an expert remote-sensing analyst. Write a single concise caption
(1-2 sentences) describing this satellite/aerial image. Focus on the most
prominent features and land cover types. Be specific.
"""


class CaptioningSpecialist(BaseSpecialist):
    """Scene captioning and description for remote-sensing images."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def execute(
        self,
        images: list[ImageInput],
        query: str,
        context: dict,
    ) -> SpecialistResult:
        """Generate a scene description for the image."""
        if not images:
            return SpecialistResult(
                specialist_name="captioning_specialist",
                task_type=TaskType.CAPTIONING,
                answer="No image was provided for captioning.",
                model_confidence=0.0,
                error="No images provided",
            )

        image_input = images[0]
        q_lower = query.lower() if query else ""

        # Decide between brief caption and detailed description
        brief = any(kw in q_lower for kw in ["brief", "short", "caption", "one sentence"])

        try:
            answer = await self._generate_caption(image_input, brief=brief)
            confidence = self.estimate_confidence(
                SpecialistResult(
                    specialist_name="captioning_specialist",
                    task_type=TaskType.CAPTIONING,
                    answer=answer,
                )
            )
            return SpecialistResult(
                specialist_name="captioning_specialist",
                task_type=TaskType.CAPTIONING,
                answer=answer,
                model_confidence=confidence,
            )
        except Exception as exc:
            logger.error("Captioning failed: %s", exc)
            return SpecialistResult(
                specialist_name="captioning_specialist",
                task_type=TaskType.CAPTIONING,
                answer=f"This is a {image_input.modality.value} remote-sensing image "
                       f"({image_input.width}×{image_input.height} pixels, "
                       f"{image_input.band_count} band(s)). "
                       f"Detailed captioning is temporarily unavailable.",
                model_confidence=0.2,
                error=str(exc),
            )

    def get_capabilities(self) -> list[TaskType]:
        return [TaskType.CAPTIONING, TaskType.SCENE_CLASSIFICATION]

    def estimate_confidence(self, result: SpecialistResult) -> float:
        if result.error:
            return 0.2
        answer = result.answer
        if not answer:
            return 0.1

        score = 0.7

        # Detailed descriptions → higher confidence
        if len(answer) > 300:
            score += 0.1
        if len(answer) > 800:
            score += 0.05

        # Structure indicators
        structure_markers = ["**", "1.", "2.", "- ", "scene overview", "land cover"]
        for marker in structure_markers:
            if marker.lower() in answer.lower():
                score += 0.02

        return max(0.1, min(1.0, score))

    async def _generate_caption(self, image_input: ImageInput, brief: bool = False) -> str:
        """Call Gemini Vision to generate the caption."""
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_vision_model)

        pil_img = image_to_pil(image_input.filepath)
        prompt = _RS_BRIEF_CAPTION_PROMPT if brief else _RS_CAPTIONING_PROMPT

        response = model.generate_content([prompt, pil_img])
        return response.text.strip()
