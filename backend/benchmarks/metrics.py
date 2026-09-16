"""Lightweight, dependency-free metrics for text benchmark outputs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_text(value: str) -> list[str]:
    """Tokenize text consistently for answer-overlap metrics."""
    return re.findall(r"[a-z0-9]+", value.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 for a generated answer and reference."""
    predicted = Counter(normalize_text(prediction))
    expected = Counter(normalize_text(reference))
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(predicted.values()))
    recall = overlap / max(1, sum(expected.values()))
    return 2 * precision * recall / (precision + recall)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate success rate, answer F1, and confidence."""
    successful = [item for item in results if item.get("status") == "success"]
    f1_scores = [float(item["token_f1"]) for item in successful if "token_f1" in item]
    confidences = [float(item["confidence"]) for item in successful if "confidence" in item]
    return {
        "sample_count": len(results),
        "success_count": len(successful),
        "success_rate": round(len(successful) / max(1, len(results)), 4),
        "mean_token_f1": round(sum(f1_scores) / max(1, len(f1_scores)), 4),
        "mean_confidence": round(sum(confidences) / max(1, len(confidences)), 4),
    }
