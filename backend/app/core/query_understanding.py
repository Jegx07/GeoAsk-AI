"""GeoAsk AI — Query Understanding.

Uses an LLM to classify user intent, extract entities, and determine
what task(s) the user is requesting from their natural-language query.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.config import get_settings
from app.models.enums import InputMode, TaskType
from app.models.schemas import ExtractedEntity, ImageInput, QueryIntent

logger = logging.getLogger(__name__)

# System prompt that instructs the LLM to classify remote-sensing queries
_CLASSIFICATION_PROMPT = """\
You are a remote-sensing AI assistant that classifies user queries about satellite/aerial imagery.

Given a user query and information about uploaded images, determine:
1. The PRIMARY task type the user wants.
2. Any SECONDARY tasks needed to fulfill the request.
3. Key ENTITIES mentioned (objects, locations, temporal references).

TASK TYPES (choose from these exact values):
- "vqa"               → Visual Question Answering about a single image
- "captioning"        → Describe/caption the scene in an image
- "scene_classification" → Classify the scene type
- "change_detection"  → Detect changes between two temporal images
- "change_description" → Describe what changed between temporal images
- "change_vqa"        → Answer questions about changes between images
- "sar_analysis"      → Analyse a SAR image specifically
- "optical_sar_fusion" → Joint analysis of optical + SAR image pair
- "grounding"         → Locate/ground specific objects in an image
- "general"           → General query that doesn't fit above categories

RULES:
- If the query asks "what changed" or "compare these images" with temporal images → change_detection + change_description
- If the query asks about a SAR+optical pair → optical_sar_fusion
- If the query asks to describe or caption → captioning
- If the query asks a specific question about one image → vqa
- If query asks to find or locate something → grounding
- If ambiguous, default to "vqa" for single image, "change_detection" for pairs

Respond ONLY with valid JSON matching this schema:
{
  "primary_task": "<task_type>",
  "secondary_tasks": ["<task_type>", ...],
  "entities": [{"text": "...", "entity_type": "object|location|temporal|attribute"}],
  "requires_multiple_images": true/false,
  "requires_temporal": true/false,
  "requires_sar": true/false,
  "reformulated_query": "<clearer version of the query>",
  "reasoning": "<brief explanation of your classification>"
}
"""


class QueryUnderstanding:
    """Classifies user queries into structured intents using an LLM."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def analyze(
        self,
        query: str,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> QueryIntent:
        """Analyse the user's query and return a structured intent.

        Falls back to rule-based classification if LLM is unavailable.
        """
        try:
            return await self._llm_classify(query, images, input_mode)
        except Exception as exc:
            logger.warning("LLM classification failed (%s), using rule-based fallback", exc)
            return self._rule_based_classify(query, images, input_mode)

    # ------------------------------------------------------------------ #
    # LLM-based classification
    # ------------------------------------------------------------------ #

    async def _llm_classify(
        self,
        query: str,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> QueryIntent:
        """Use Google Gemini to classify the query."""
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)

        # Build context about the uploaded images
        image_context = self._build_image_context(images, input_mode)

        prompt = (
            f"{_CLASSIFICATION_PROMPT}\n\n"
            f"--- IMAGE CONTEXT ---\n{image_context}\n\n"
            f"--- USER QUERY ---\n{query}\n\n"
            f"Respond with JSON only."
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        entities = [
            ExtractedEntity(**e) for e in data.get("entities", [])
        ]

        return QueryIntent(
            primary_task=TaskType(data["primary_task"]),
            secondary_tasks=[TaskType(t) for t in data.get("secondary_tasks", [])],
            entities=entities,
            requires_multiple_images=data.get("requires_multiple_images", False),
            requires_temporal=data.get("requires_temporal", False),
            requires_sar=data.get("requires_sar", False),
            raw_query=query,
            reformulated_query=data.get("reformulated_query", query),
            reasoning=data.get("reasoning", ""),
            confidence=0.85,
        )

    # ------------------------------------------------------------------ #
    # Rule-based fallback
    # ------------------------------------------------------------------ #

    def _rule_based_classify(
        self,
        query: str,
        images: list[ImageInput],
        input_mode: InputMode,
    ) -> QueryIntent:
        """Simple keyword-based classification when LLM is unavailable."""
        q = query.lower()

        # Input-mode driven defaults
        if input_mode == InputMode.OPTICAL_SAR:
            primary = TaskType.OPTICAL_SAR_FUSION
        elif input_mode == InputMode.BI_TEMPORAL:
            primary = TaskType.CHANGE_DETECTION
        else:
            primary = TaskType.VQA

        # Keyword overrides
        change_keywords = ["change", "differ", "before and after", "temporal", "transform"]
        caption_keywords = ["describe", "caption", "tell me about", "what is in", "scene"]
        sar_keywords = ["sar", "radar", "backscatter", "synthetic aperture"]
        ground_keywords = ["find", "locate", "where is", "point to", "show me"]
        count_keywords = ["count", "how many", "number of"]

        secondary: list[TaskType] = []

        if any(kw in q for kw in change_keywords):
            primary = TaskType.CHANGE_DETECTION
            secondary.append(TaskType.CHANGE_DESCRIPTION)
        elif any(kw in q for kw in caption_keywords):
            primary = TaskType.CAPTIONING
        elif any(kw in q for kw in sar_keywords):
            if input_mode == InputMode.OPTICAL_SAR:
                primary = TaskType.OPTICAL_SAR_FUSION
            else:
                primary = TaskType.SAR_ANALYSIS
        elif any(kw in q for kw in ground_keywords):
            primary = TaskType.GROUNDING
        elif any(kw in q for kw in count_keywords):
            primary = TaskType.VQA

        return QueryIntent(
            primary_task=primary,
            secondary_tasks=secondary,
            entities=[],
            requires_multiple_images=input_mode != InputMode.SINGLE,
            requires_temporal=input_mode == InputMode.BI_TEMPORAL,
            requires_sar=input_mode == InputMode.OPTICAL_SAR,
            raw_query=query,
            reformulated_query=query,
            reasoning="Rule-based classification (LLM unavailable)",
            confidence=0.6,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_image_context(images: list[ImageInput], input_mode: InputMode) -> str:
        """Build a text description of the uploaded images for the LLM."""
        lines = [f"Input mode: {input_mode.value}", f"Number of images: {len(images)}", ""]
        for i, img in enumerate(images):
            lines.append(
                f"Image {i+1}: filename={img.filename}, "
                f"modality={img.modality.value}, "
                f"bands={img.band_count}, "
                f"size={img.width}x{img.height}, "
                f"format={img.format.value}, "
                f"label={img.label or 'none'}"
            )
            if img.geo_metadata and img.geo_metadata.crs:
                lines.append(f"  CRS: {img.geo_metadata.crs}")
        return "\n".join(lines)
