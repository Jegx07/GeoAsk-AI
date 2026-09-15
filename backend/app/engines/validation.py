"""GeoAsk AI — Validation Engine.

Cross-checks specialist results for consistency and correctness.
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    EvidenceBundle,
    QueryIntent,
    SpecialistResult,
    ValidationCheck,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Validates specialist outputs for consistency and quality."""

    def validate(
        self,
        specialist_results: list[SpecialistResult],
        evidence: EvidenceBundle,
        intent: QueryIntent,
    ) -> ValidationResult:
        """Run all validation checks and return aggregated result."""
        checks: list[ValidationCheck] = []

        # Check 1: At least one specialist produced an answer
        checks.append(self._check_has_answer(specialist_results))

        # Check 2: No specialist errors
        checks.append(self._check_no_errors(specialist_results))

        # Check 3: Answer relevance (does it address the query?)
        checks.append(self._check_relevance(specialist_results, intent))

        # Check 4: Evidence exists
        checks.append(self._check_evidence_exists(evidence))

        # Check 5: Confidence above minimum threshold
        checks.append(self._check_minimum_confidence(specialist_results))

        # Check 6: Cross-specialist consistency (if multiple results)
        if len(specialist_results) > 1:
            checks.append(self._check_cross_consistency(specialist_results))

        # Aggregate
        passed_checks = [c for c in checks if c.passed]
        overall_score = sum(c.score for c in checks) / max(1, len(checks))
        overall_passed = len(passed_checks) >= len(checks) * 0.5  # 50% pass rate

        warnings = [c.message for c in checks if not c.passed]

        result = ValidationResult(
            checks=checks,
            overall_passed=overall_passed,
            overall_score=round(overall_score, 3),
            warnings=warnings,
        )

        logger.info(
            "Validation: %d/%d checks passed, score=%.3f",
            len(passed_checks),
            len(checks),
            overall_score,
        )
        return result

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_has_answer(results: list[SpecialistResult]) -> ValidationCheck:
        has = any(r.answer for r in results)
        return ValidationCheck(
            check_name="has_answer",
            passed=has,
            score=1.0 if has else 0.0,
            message="" if has else "No specialist produced a textual answer",
        )

    @staticmethod
    def _check_no_errors(results: list[SpecialistResult]) -> ValidationCheck:
        errors = [r for r in results if r.error]
        score = 1.0 - (len(errors) / max(1, len(results)))
        return ValidationCheck(
            check_name="no_errors",
            passed=len(errors) == 0,
            score=score,
            message="" if not errors else f"{len(errors)} specialist(s) reported errors",
        )

    @staticmethod
    def _check_relevance(
        results: list[SpecialistResult],
        intent: QueryIntent,
    ) -> ValidationCheck:
        """Check that at least one result addresses the intended task."""
        for r in results:
            if r.task_type == intent.primary_task and r.answer:
                return ValidationCheck(
                    check_name="relevance",
                    passed=True,
                    score=1.0,
                    message="Primary task addressed",
                )
        return ValidationCheck(
            check_name="relevance",
            passed=False,
            score=0.3,
            message=f"Primary task '{intent.primary_task.value}' not directly addressed",
        )

    @staticmethod
    def _check_evidence_exists(evidence: EvidenceBundle) -> ValidationCheck:
        count = len(evidence.items)
        return ValidationCheck(
            check_name="evidence_exists",
            passed=count > 0,
            score=min(1.0, count / 3.0),  # full score at 3+ items
            message="" if count > 0 else "No evidence items collected",
        )

    @staticmethod
    def _check_minimum_confidence(results: list[SpecialistResult]) -> ValidationCheck:
        confidences = [r.model_confidence for r in results if r.model_confidence > 0]
        if not confidences:
            return ValidationCheck(
                check_name="minimum_confidence",
                passed=False,
                score=0.0,
                message="No confidence scores available",
            )
        avg = sum(confidences) / len(confidences)
        return ValidationCheck(
            check_name="minimum_confidence",
            passed=avg >= 0.3,
            score=avg,
            message="" if avg >= 0.3 else f"Average confidence ({avg:.2f}) is below threshold",
        )

    @staticmethod
    def _check_cross_consistency(results: list[SpecialistResult]) -> ValidationCheck:
        """Check that multiple specialist outputs don't contradict each other.

        Simple heuristic: if confidence scores are within a reasonable range
        of each other, consider them consistent.
        """
        confidences = [r.model_confidence for r in results if r.model_confidence > 0]
        if len(confidences) < 2:
            return ValidationCheck(
                check_name="cross_consistency",
                passed=True,
                score=1.0,
                message="Insufficient data for cross-check",
            )

        spread = max(confidences) - min(confidences)
        consistent = spread < 0.5
        return ValidationCheck(
            check_name="cross_consistency",
            passed=consistent,
            score=1.0 - spread,
            message="" if consistent else f"Confidence spread ({spread:.2f}) suggests inconsistency",
        )
