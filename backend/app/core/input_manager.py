"""GeoAsk AI — Input Manager.

Handles file upload validation, metadata extraction, modality detection,
and input mode classification.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.enums import ImageFormat, ImageModality, InputMode
from app.models.schemas import GeoMetadata, ImageInput
from app.utils.geo import extract_geo_metadata, get_raster_band_count
from app.utils.image import (
    detect_image_format,
    generate_thumbnail,
    get_image_dimensions,
)

logger = logging.getLogger(__name__)

# SAR-related filename keywords for modality heuristics
_SAR_KEYWORDS = {"sar", "radar", "sentinel1", "s1", "asar", "radarsat", "gf3", "gaofen3"}
_OPTICAL_KEYWORDS = {"optical", "sentinel2", "s2", "landsat", "spot", "worldview", "rgb"}


class InputManager:
    """Validates, analyses and classifies user-uploaded imagery."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def process_upload(
        self,
        filepath: str,
        original_filename: str,
        label: Optional[str] = None,
    ) -> ImageInput:
        """Process a single uploaded file and return structured metadata.

        Parameters
        ----------
        filepath : str
            Absolute path to the saved upload.
        original_filename : str
            The filename as provided by the user (used for heuristics).
        label : str, optional
            User-supplied label (e.g. "T1", "SAR").
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Upload not found: {filepath}")

        file_size = path.stat().st_size
        if file_size > self.settings.max_upload_bytes:
            raise ValueError(
                f"File too large ({file_size / 1e6:.1f} MB). "
                f"Max: {self.settings.max_upload_size_mb} MB."
            )

        # --- Format detection ---
        fmt_str = detect_image_format(filepath)
        fmt = _str_to_format(fmt_str)

        # --- Dimensions & bands ---
        width, height = get_image_dimensions(filepath)
        band_count = get_raster_band_count(filepath)

        # --- Geo metadata ---
        geo_raw = extract_geo_metadata(filepath)
        geo_meta = GeoMetadata(**geo_raw) if geo_raw else None

        # --- Modality detection ---
        modality = self._detect_modality(original_filename, band_count, label)

        # --- Thumbnail ---
        thumb_dir = self.settings.upload_path / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = str(thumb_dir / f"thumb_{path.stem}.png")
        try:
            generate_thumbnail(filepath, thumb_path)
        except Exception as exc:
            logger.warning("Thumbnail generation failed: %s", exc)
            thumb_path = None

        image_input = ImageInput(
            filename=original_filename,
            filepath=filepath,
            format=fmt,
            modality=modality,
            band_count=band_count,
            width=width,
            height=height,
            file_size_bytes=file_size,
            geo_metadata=geo_meta,
            thumbnail_path=thumb_path,
            label=label,
        )

        logger.info(
            "Processed upload: %s | format=%s | modality=%s | %dx%d | %d bands",
            original_filename,
            fmt.value,
            modality.value,
            width,
            height,
            band_count,
        )
        return image_input

    def classify_input_mode(self, images: list[ImageInput]) -> InputMode:
        """Determine the input mode from the set of uploaded images.

        Rules:
        - 1 image → SINGLE
        - 2 images, one SAR + one optical → OPTICAL_SAR
        - 2 images, same modality → BI_TEMPORAL
        - 3+ images → MULTI_IMAGE
        """
        if len(images) == 0:
            return InputMode.SINGLE
        if len(images) == 1:
            return InputMode.SINGLE
        if len(images) == 2:
            modalities = {img.modality for img in images}
            if ImageModality.SAR in modalities and (
                ImageModality.OPTICAL in modalities or ImageModality.MULTISPECTRAL in modalities
            ):
                return InputMode.OPTICAL_SAR
            return InputMode.BI_TEMPORAL
        return InputMode.MULTI_IMAGE

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_modality(
        filename: str,
        band_count: int,
        label: Optional[str] = None,
    ) -> ImageModality:
        """Heuristic modality detection from filename, bands, and label."""
        lower = filename.lower()
        if label:
            lower += " " + label.lower()

        # Check keywords
        for kw in _SAR_KEYWORDS:
            if kw in lower:
                return ImageModality.SAR
        for kw in _OPTICAL_KEYWORDS:
            if kw in lower:
                return ImageModality.OPTICAL

        # Band-count heuristics
        if band_count == 1:
            return ImageModality.SAR  # single-band is often SAR or panchromatic
        if band_count in (3, 4):
            return ImageModality.OPTICAL
        if band_count > 4:
            return ImageModality.MULTISPECTRAL

        return ImageModality.UNKNOWN


def _str_to_format(s: str) -> ImageFormat:
    """Map a format string to the ImageFormat enum."""
    mapping = {
        "geotiff": ImageFormat.GEOTIFF,
        "tiff": ImageFormat.TIFF,
        "png": ImageFormat.PNG,
        "jpeg": ImageFormat.JPEG,
    }
    return mapping.get(s, ImageFormat.UNKNOWN)
