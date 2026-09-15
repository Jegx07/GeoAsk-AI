"""GeoAsk AI — SAR-Optical Fusion Specialist.

Cross-modal analysis of optical and SAR image pairs.
Performs feature extraction, PCA-based fusion, and VLM reasoning
over the combined representation.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from PIL import Image

from app.config import get_settings
from app.models.enums import TaskType
from app.models.schemas import ImageInput, SpecialistResult
from app.specialists.base import BaseSpecialist
from app.utils.image import image_to_pil

logger = logging.getLogger(__name__)


class SAROpticalSpecialist(BaseSpecialist):
    """Cross-modal analysis of optical + SAR image pairs."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def execute(
        self,
        images: list[ImageInput],
        query: str,
        context: dict,
    ) -> SpecialistResult:
        """Perform joint analysis of optical and SAR images."""
        if len(images) < 2:
            # Single SAR image analysis
            if images and images[0].modality.value == "sar":
                return await self._analyze_single_sar(images[0], query)
            return SpecialistResult(
                specialist_name="sar_optical_specialist",
                task_type=TaskType.OPTICAL_SAR_FUSION,
                answer="SAR-Optical fusion requires two images (one optical, one SAR).",
                model_confidence=0.0,
                error="Insufficient images for cross-modal analysis",
            )

        # Identify which is optical and which is SAR
        optical_img, sar_img = self._identify_modalities(images)

        try:
            # Step 1: Feature extraction & fusion
            fused_path, stats = self._fuse_images(optical_img, sar_img)

            # Step 2: VLM-based cross-modal reasoning
            analysis = await self._cross_modal_analysis(
                optical_img, sar_img, stats, query
            )

            visual_paths = [fused_path] if fused_path else []

            return SpecialistResult(
                specialist_name="sar_optical_specialist",
                task_type=TaskType.OPTICAL_SAR_FUSION,
                answer=analysis,
                visual_outputs=visual_paths,
                metrics=stats,
                model_confidence=0.7,
            )

        except Exception as exc:
            logger.error("SAR-Optical fusion failed: %s", exc)
            return SpecialistResult(
                specialist_name="sar_optical_specialist",
                task_type=TaskType.OPTICAL_SAR_FUSION,
                answer="SAR-Optical fusion analysis encountered an error.",
                model_confidence=0.1,
                error=str(exc),
            )

    def get_capabilities(self) -> list[TaskType]:
        return [TaskType.SAR_ANALYSIS, TaskType.OPTICAL_SAR_FUSION]

    def estimate_confidence(self, result: SpecialistResult) -> float:
        if result.error:
            return 0.15
        return result.model_confidence

    # ------------------------------------------------------------------ #
    # Internal Methods
    # ------------------------------------------------------------------ #

    def _identify_modalities(
        self, images: list[ImageInput]
    ) -> tuple[ImageInput, ImageInput]:
        """Identify which image is optical and which is SAR."""
        sar_idx = None
        for i, img in enumerate(images):
            if img.modality.value == "sar":
                sar_idx = i
                break

        if sar_idx is not None:
            optical_idx = 1 - sar_idx if len(images) == 2 else 0
            return images[optical_idx], images[sar_idx]

        # Default: first is optical, second is SAR
        return images[0], images[1]

    def _fuse_images(
        self,
        optical_img: ImageInput,
        sar_img: ImageInput,
    ) -> tuple[str | None, dict[str, float]]:
        """Perform PCA-based feature fusion of optical and SAR images.

        Creates a fused visualisation and computes correlation statistics.
        """
        from sklearn.decomposition import PCA

        output_dir = self.settings.upload_path / "results"
        output_dir.mkdir(exist_ok=True)
        uid = uuid.uuid4().hex[:8]

        # Load images as greyscale arrays
        opt_arr = np.array(image_to_pil(optical_img.filepath).convert("L"), dtype=np.float64)
        sar_arr = np.array(image_to_pil(sar_img.filepath).convert("L"), dtype=np.float64)

        # Resize to common dimensions
        min_h = min(opt_arr.shape[0], sar_arr.shape[0])
        min_w = min(opt_arr.shape[1], sar_arr.shape[1])
        opt_arr = opt_arr[:min_h, :min_w]
        sar_arr = sar_arr[:min_h, :min_w]

        # Normalise
        opt_norm = (opt_arr - opt_arr.mean()) / (opt_arr.std() + 1e-8)
        sar_norm = (sar_arr - sar_arr.mean()) / (sar_arr.std() + 1e-8)

        # Correlation
        correlation = float(np.corrcoef(opt_norm.ravel(), sar_norm.ravel())[0, 1])

        # PCA fusion
        stacked = np.stack([opt_norm.ravel(), sar_norm.ravel()], axis=1)  # (N, 2)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(stacked)

        # First principal component → fused image
        pc1 = pca_result[:, 0].reshape(min_h, min_w)
        pc1_norm = ((pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8) * 255).astype(np.uint8)

        # Create a false-colour composite: R=optical, G=fused, B=SAR
        opt_uint8 = ((opt_arr - opt_arr.min()) / (opt_arr.max() - opt_arr.min() + 1e-8) * 255).astype(np.uint8)
        sar_uint8 = ((sar_arr - sar_arr.min()) / (sar_arr.max() - sar_arr.min() + 1e-8) * 255).astype(np.uint8)

        false_color = np.stack([opt_uint8, pc1_norm, sar_uint8], axis=-1)
        fused_path = str(output_dir / f"sar_optical_fusion_{uid}.png")
        Image.fromarray(false_color).save(fused_path)

        stats = {
            "correlation": round(correlation, 4),
            "pca_variance_ratio_1": round(float(pca.explained_variance_ratio_[0]), 4),
            "pca_variance_ratio_2": round(float(pca.explained_variance_ratio_[1]), 4),
            "optical_mean": round(float(opt_arr.mean()), 2),
            "sar_mean": round(float(sar_arr.mean()), 2),
            "optical_std": round(float(opt_arr.std()), 2),
            "sar_std": round(float(sar_arr.std()), 2),
        }

        logger.info(
            "SAR-Optical fusion: correlation=%.3f, PCA variance=[%.3f, %.3f]",
            correlation,
            pca.explained_variance_ratio_[0],
            pca.explained_variance_ratio_[1],
        )

        return fused_path, stats

    async def _cross_modal_analysis(
        self,
        optical_img: ImageInput,
        sar_img: ImageInput,
        stats: dict[str, float],
        query: str,
    ) -> str:
        """Use Gemini Vision for cross-modal reasoning."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(self.settings.gemini_vision_model)

            opt_pil = image_to_pil(optical_img.filepath)
            sar_pil = image_to_pil(sar_img.filepath)

            correlation = stats.get("correlation", 0)

            prompt = (
                "You are a remote-sensing expert skilled in multi-modal analysis. "
                "You are shown two images of the same area:\n\n"
                "Image 1: OPTICAL satellite image (visible light bands)\n"
                "Image 2: SAR (Synthetic Aperture Radar) image\n\n"
                f"Statistical analysis shows a correlation of {correlation:.3f} between the two.\n\n"
                f"User query: {query}\n\n"
                "Provide a comprehensive cross-modal analysis:\n"
                "1. **Optical Image Analysis**: What features are visible in the optical image?\n"
                "2. **SAR Image Analysis**: What does the SAR backscatter reveal? "
                "(bright areas = high backscatter = rough surfaces, buildings, ships; "
                "dark areas = smooth surfaces, calm water)\n"
                "3. **Complementary Information**: What does each modality reveal "
                "that the other cannot?\n"
                "4. **Joint Interpretation**: Combining both sources, what can you conclude?\n"
                "5. **Practical Significance**: Why is this multi-modal view useful?\n"
            )

            response = model.generate_content([prompt, opt_pil, sar_pil])
            return response.text.strip()

        except Exception as exc:
            logger.warning("VLM cross-modal analysis failed: %s", exc)
            correlation = stats.get("correlation", 0)
            return (
                f"Cross-modal analysis of the optical and SAR image pair:\n\n"
                f"**Statistical Correlation**: {correlation:.3f} — "
                f"{'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'} "
                f"correlation between modalities.\n\n"
                f"**Optical Image**: {optical_img.width}×{optical_img.height} pixels, "
                f"mean intensity: {stats.get('optical_mean', 0):.1f}\n"
                f"**SAR Image**: {sar_img.width}×{sar_img.height} pixels, "
                f"mean intensity: {stats.get('sar_mean', 0):.1f}\n\n"
                f"**PCA Fusion**: The first principal component captures "
                f"{stats.get('pca_variance_ratio_1', 0)*100:.1f}% of the joint variance, "
                f"indicating {'strong' if stats.get('pca_variance_ratio_1', 0) > 0.8 else 'moderate'} "
                f"shared information between modalities.\n\n"
                f"A false-colour composite has been generated (R=Optical, G=Fused, B=SAR) "
                f"to visualise the complementary information from both sensors."
            )

    async def _analyze_single_sar(self, sar_img: ImageInput, query: str) -> SpecialistResult:
        """Analyse a single SAR image."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel(self.settings.gemini_vision_model)

            pil_img = image_to_pil(sar_img.filepath)

            prompt = (
                "You are a SAR (Synthetic Aperture Radar) image analysis expert.\n\n"
                "This is a SAR image. Key properties of SAR:\n"
                "- Bright areas = high backscatter (rough surfaces, buildings, vegetation, ships)\n"
                "- Dark areas = low backscatter (smooth surfaces, calm water, roads, runways)\n"
                "- Speckle noise is inherent to SAR imagery\n"
                "- SAR can see through clouds and works day/night\n\n"
                f"User query: {query}\n\n"
                "Provide a detailed SAR image analysis including:\n"
                "1. Identification of features based on backscatter patterns\n"
                "2. Land cover interpretation\n"
                "3. Any notable structures or anomalies\n"
            )

            response = model.generate_content([prompt, pil_img])
            return SpecialistResult(
                specialist_name="sar_optical_specialist",
                task_type=TaskType.SAR_ANALYSIS,
                answer=response.text.strip(),
                model_confidence=0.65,
            )

        except Exception as exc:
            logger.error("Single SAR analysis failed: %s", exc)
            return SpecialistResult(
                specialist_name="sar_optical_specialist",
                task_type=TaskType.SAR_ANALYSIS,
                answer=f"This SAR image ({sar_img.width}×{sar_img.height} pixels) shows "
                       f"radar backscatter patterns. Detailed analysis is temporarily unavailable.",
                model_confidence=0.2,
                error=str(exc),
            )
