"""GeoAsk AI — Confidence Estimator.

Calculates a multi-factor confidence score based on model outputs,
evidence strength, validation results, and query complexity.
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    ConfidenceBreakdown,
    ConfidenceScore,
    QueryIntent,
    SpecialistResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Estimates overall confidence from multiple signals."""

    def estimate(
        self,
        specialist_results: list[SpecialistResult],
        validation: ValidationResult,
        intent: QueryIntent,
    ) -> ConfidenceScore:
        """Calculate the final confidence score."""
        breakdown: list[ConfidenceBreakdown] = []

        # Signal 1: Specialist reported confidences
        model_score = self._score_model_confidence(specialist_results)
        breakdown.append(
            ConfidenceBreakdown(
                factor="Model Confidence",
                score=model_score,
                weight=0.5,
                explanation="Aggregated confidence reported by specialist models.",
            )
        )

        # Signal 2: Validation results
        val_score = validation.overall_score
        breakdown.append(
            ConfidenceBreakdown(
                factor="Validation Checks",
                score=val_score,
                weight=0.3,
                explanation="Result consistency and completeness checks.",
            )
        )

        # Signal 3: Query complexity
        complexity_score = self._score_query_complexity(intent)
        breakdown.append(
            ConfidenceBreakdown(
                factor="Query Complexity",
                score=complexity_score,
                weight=0.2,
                explanation="Based on query length, entities, and task type.",
            )
        )

        # Weighted sum
        total_weight = sum(b.weight for b in breakdown)
        overall = sum(b.score * b.weight for b in breakdown) / total_weight

        # Penalty for validation failure
        if not validation.overall_passed:
            overall *= 0.7
            breakdown.append(
                ConfidenceBreakdown(
                    factor="Validation Penalty",
                    score=0.7,
                    weight=0.0,
                    explanation="Penalty applied due to failed validation checks.",
                )
            )

        overall = max(0.0, min(1.0, overall))

        # Generate explanation
        explanation = self._generate_explanation(overall, breakdown)

        score = ConfidenceScore.from_score(overall, breakdown)
        score.explanation = explanation

        logger.info("Estimated confidence: %.3f (%s)", score.overall, score.level.value)
        return score

    # ------------------------------------------------------------------ #
    # Signal calculations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_model_confidence(results: list[SpecialistResult]) -> float:
        """Average of valid specialist confidence scores."""
        scores = [r.model_confidence for r in results if r.model_confidence > 0]
        if not scores:
            return 0.5  # Neutral default
        return sum(scores) / len(scores)

    @staticmethod
    def _score_query_complexity(intent: QueryIntent) -> float:
        """Estimate how 'complex' the query is (lower complexity -> higher confidence)."""
        score = 1.0

        # Long queries are often more complex/ambiguous
        if len(intent.raw_query) > 100:
            score -= 0.1

        # Multiple entities increase complexity
        if len(intent.entities) > 2:
            score -= 0.1
        elif len(intent.entities) > 5:
            score -= 0.2

        # Multiple tasks decrease confidence slightly
        if intent.secondary_tasks:
            score -= 0.05 * len(intent.secondary_tasks)

        # Task-specific baselines
        from app.models.enums import TaskType
        if intent.primary_task == TaskType.VQA:
            pass  # Baseline 1.0
        elif intent.primary_task == TaskType.CHANGE_DETECTION:
            score -= 0.1  # Inherently more complex
        elif intent.primary_task == TaskType.OPTICAL_SAR_FUSION:
            score -= 0.15  # Cross-modal is complex

        return max(0.2, min(1.0, score))

    @staticmethod
    def _generate_explanation(overall: float, breakdown: list[ConfidenceBreakdown]) -> str:
        """Generate a human-readable explanation of the confidence score."""
        if overall > 0.8:
            return "High confidence based on strong model agreement and successful validation."
        elif overall > 0.5:
            return "Moderate confidence. The model produced an answer but there is some uncertainty or complexity."
        else:
            factors = [b.factor for b in breakdown if b.score < 0.5]
            if factors:
                return f"Low confidence primarily due to: {', '.join(factors)}."
            return "Low confidence in the generated analysis."
