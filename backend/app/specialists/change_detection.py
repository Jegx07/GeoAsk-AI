"""GeoAsk AI — Change Detection Specialist.

Bi-temporal change analysis: detects, localises, and describes
changes between two co-registered remote-sensing images.

Uses an algorithmic pipeline (image differencing + adaptive threshold
+ morphological analysis) that works without GPU, then feeds the
change mask and images to Gemini Vision for natural-language description.
"""

from __future__ import annotations

import logging
import asyncio
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.models.enums import TaskType
from app.models.schemas import ImageInput, SpecialistResult
from app.specialists.base import BaseSpecialist
from app.utils.image import image_to_pil, pil_to_bytes

logger = logging.getLogger(__name__)


class ChangeDetectionSpecialist(BaseSpecialist):
    """Bi-temporal change detection, description, and change-VQA."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def execute(
        self,
        images: list[ImageInput],
        query: str,
        context: dict,
    ) -> SpecialistResult:
        """Detect and describe changes between two temporal images."""
        if len(images) < 2:
            return SpecialistResult(
                specialist_name="change_detection_specialist",
                task_type=TaskType.CHANGE_DETECTION,
                answer="Change detection requires two images (T1 and T2). "
                       "Please upload a pair of bi-temporal images.",
                model_confidence=0.0,
                error="Insufficient images for change detection",
            )

        img_t1 = images[0]
        img_t2 = images[1]

        try:
            # Step 1: Compute change mask using algorithmic approach
            change_mask, change_pct, diff_map = self._compute_change_mask(img_t1, img_t2)

            # Step 2: Generate change visualisation
            visual_paths = self._generate_visualisations(
                img_t1, img_t2, change_mask, diff_map
            )

            # Step 3: Generate natural-language description via VLM
            description = await self._describe_changes(
                img_t1, img_t2, change_mask, change_pct, query
            )

            confidence = min(0.85, 0.5 + change_pct / 100.0)  # higher change → higher confidence

            return SpecialistResult(
                specialist_name="change_detection_specialist",
                task_type=TaskType.CHANGE_DETECTION,
                answer=description,
                visual_outputs=visual_paths,
                metrics={
                    "change_percentage": round(change_pct, 2),
                    "changed_pixels": int(np.sum(change_mask > 0)),
                    "total_pixels": int(change_mask.size),
                },
                model_confidence=confidence,
            )

        except Exception as exc:
            logger.error("Change detection failed: %s", exc)
            return SpecialistResult(
                specialist_name="change_detection_specialist",
                task_type=TaskType.CHANGE_DETECTION,
                answer="Change detection analysis encountered an error. "
                       "Please ensure both images cover the same area.",
                model_confidence=0.1,
                error=str(exc),
            )

    def get_capabilities(self) -> list[TaskType]:
        return [TaskType.CHANGE_DETECTION, TaskType.CHANGE_DESCRIPTION, TaskType.CHANGE_VQA]

    def estimate_confidence(self, result: SpecialistResult) -> float:
        if result.error:
            return 0.15
        return result.model_confidence

    # ------------------------------------------------------------------ #
    # Algorithmic Change Detection Pipeline
    # ------------------------------------------------------------------ #

    def _compute_change_mask(
        self,
        img_t1: ImageInput,
        img_t2: ImageInput,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute a binary change mask using image differencing.

        Pipeline:
        1. Load both images as greyscale
        2. Resize to same dimensions
        3. Compute absolute difference
        4. Adaptive threshold (Otsu or percentile-based)
        5. Morphological cleanup (opening to remove noise)

        Returns (change_mask, change_percentage, diff_map).
        """
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_opening, binary_closing

        # Load as greyscale arrays
        t1 = np.array(image_to_pil(img_t1.filepath).convert("L"), dtype=np.float64)
        t2 = np.array(image_to_pil(img_t2.filepath).convert("L"), dtype=np.float64)

        # Resize to common dimensions
        min_h = min(t1.shape[0], t2.shape[0])
        min_w = min(t1.shape[1], t2.shape[1])
        t1 = t1[:min_h, :min_w]
        t2 = t2[:min_h, :min_w]

        # Absolute difference
        diff = np.abs(t1 - t2)

        # Normalise diff to [0, 255]
        if diff.max() > 0:
            diff_norm = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff_norm = diff.astype(np.uint8)

        # Adaptive threshold
        try:
            thresh = threshold_otsu(diff_norm)
        except ValueError:
            thresh = 30  # fallback
        thresh = max(thresh, 15)  # minimum threshold to avoid noise

        binary_mask = (diff_norm > thresh).astype(np.uint8)

        # Morphological cleanup
        struct = np.ones((3, 3))
        binary_mask = binary_opening(binary_mask, structure=struct, iterations=2).astype(np.uint8)
        binary_mask = binary_closing(binary_mask, structure=struct, iterations=1).astype(np.uint8)

        # Statistics
        change_pct = (np.sum(binary_mask) / binary_mask.size) * 100

        logger.info(
            "Change detection: threshold=%d, change=%.2f%%",
            thresh,
            change_pct,
        )

        return binary_mask, change_pct, diff_norm

    def _generate_visualisations(
        self,
        img_t1: ImageInput,
        img_t2: ImageInput,
        change_mask: np.ndarray,
        diff_map: np.ndarray,
    ) -> list[str]:
        """Generate visualisation images and save to disk."""
        output_dir = self.settings.upload_path / "results"
        output_dir.mkdir(exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        paths: list[str] = []

        # 1. Change mask as red overlay on T2
        try:
            t2_pil = image_to_pil(img_t2.filepath).convert("RGB")
            t2_arr = np.array(t2_pil)

            # Resize mask to match t2
            mask_resized = np.array(
                Image.fromarray(change_mask * 255).resize(
                    (t2_arr.shape[1], t2_arr.shape[0]),
                    Image.Resampling.NEAREST,
                )
            )

            # Overlay red where change detected
            overlay = t2_arr.copy()
            change_pixels = mask_resized > 127
            overlay[change_pixels, 0] = np.clip(
                overlay[change_pixels, 0].astype(int) + 100, 0, 255
            ).astype(np.uint8)
            overlay[change_pixels, 1] = (overlay[change_pixels, 1] * 0.5).astype(np.uint8)
            overlay[change_pixels, 2] = (overlay[change_pixels, 2] * 0.5).astype(np.uint8)

            overlay_path = str(output_dir / f"change_overlay_{uid}.png")
            Image.fromarray(overlay).save(overlay_path)
            paths.append(overlay_path)
        except Exception as exc:
            logger.warning("Failed to generate change overlay: %s", exc)

        # 2. Difference heatmap
        try:
            from PIL import ImageOps

            diff_img = Image.fromarray(diff_map).convert("L")
            # Apply a colormap-like effect: low=blue, high=red
            diff_colored = Image.merge("RGB", (
                diff_img,
                Image.fromarray(np.zeros_like(diff_map, dtype=np.uint8)),
                ImageOps.invert(diff_img),
            ))
            heatmap_path = str(output_dir / f"change_heatmap_{uid}.png")
            diff_colored.save(heatmap_path)
            paths.append(heatmap_path)
        except Exception as exc:
            logger.warning("Failed to generate heatmap: %s", exc)

        # 3. Binary change mask
        try:
            mask_path = str(output_dir / f"change_mask_{uid}.png")
            Image.fromarray(change_mask * 255).save(mask_path)
            paths.append(mask_path)
        except Exception as exc:
            logger.warning("Failed to save change mask: %s", exc)

        return paths

    async def _describe_changes(
        self,
        img_t1: ImageInput,
        img_t2: ImageInput,
        change_mask: np.ndarray,
        change_pct: float,
        query: str,
    ) -> str:
        """Use Gemini Vision to describe the detected changes."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(self.settings.gemini_vision_model)

            pil_t1 = image_to_pil(img_t1.filepath)
            pil_t2 = image_to_pil(img_t2.filepath)

            prompt = (
                "You are a remote-sensing change detection expert. "
                "You are shown two satellite images of the same area at different times.\n\n"
                f"Image 1 is the EARLIER image (T1).\n"
                f"Image 2 is the LATER image (T2).\n\n"
                f"Algorithmic change detection found that approximately {change_pct:.1f}% "
                f"of the area has changed.\n\n"
                f"User query: {query}\n\n"
                "Provide a detailed analysis of:\n"
                "1. What areas have changed\n"
                "2. The nature of the changes (construction, demolition, vegetation change, etc.)\n"
                "3. The spatial distribution of changes\n"
                "4. Potential causes or significance of the changes\n"
            )

            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, [prompt, pil_t1, pil_t2]),
                timeout=60,
            )
            return response.text.strip()

        except Exception as exc:
            logger.warning("VLM change description failed: %s", exc)
            # Fallback to algorithmic-only description
            return (
                f"Change detection analysis between the two images reveals that "
                f"approximately {change_pct:.1f}% of the area has undergone changes. "
                f"The change mask shows {int(np.sum(change_mask > 0)):,} changed pixels "
                f"out of {change_mask.size:,} total pixels. "
                f"Areas of change are highlighted in the overlay visualisation."
            )
