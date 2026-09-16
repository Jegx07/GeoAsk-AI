"""Dataset loading helpers for GeoAsk benchmark evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_REQUIRED_FIELDS = {"image_path", "query", "ground_truth"}


def load_vqa_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSON or JSONL VQA dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("samples", [])

    if not isinstance(records, list):
        raise ValueError("Dataset must contain a list of samples or a 'samples' list.")

    for index, sample in enumerate(records):
        if not isinstance(sample, dict) or not _REQUIRED_FIELDS.issubset(sample):
            missing = _REQUIRED_FIELDS - set(sample) if isinstance(sample, dict) else _REQUIRED_FIELDS
            raise ValueError(f"Sample {index} is missing required fields: {sorted(missing)}")

    return records
