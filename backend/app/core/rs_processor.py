"""GeoAsk AI — Remote-Sensing Processor.

Pre-processing pipeline for remote-sensing imagery:
band normalisation, RGB compositing, SAR speckle filtering,
and image co-registration for bi-temporal pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.models.schemas import ImageInput

logger = logging.getLogger(__name__)


class RSProcessor:
    """Remote-sensing image pre-processing pipeline."""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def normalize_bands(self, array: np.ndarray) -> np.ndarray:
        """Per-band percentile normalisation to [0, 1].

        Parameters
        ----------
        array : np.ndarray
            Shape (bands, H, W).
        """
        out = np.empty_like(array, dtype=np.float32)
        for i in range(array.shape[0]):
            band = array[i].astype(np.float32)
            p2, p98 = np.percentile(band, (2, 98))
            if p98 - p2 < 1e-6:
                out[i] = 0.0
            else:
                out[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        return out

    def make_rgb_composite(
        self,
        array: np.ndarray,
        bands: tuple[int, int, int] = (0, 1, 2),
    ) -> np.ndarray:
        """Create an RGB uint8 composite from selected bands.

        Parameters
        ----------
        array : np.ndarray
            Shape (bands, H, W), float32 in [0, 1].
        bands : tuple
            Indices of bands to use as R, G, B.

        Returns
        -------
        np.ndarray of shape (H, W, 3), dtype uint8.
        """
        normed = self.normalize_bands(array)
        r, g, b = [normed[i] for i in bands]
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).clip(0, 255).astype(np.uint8)

    def apply_lee_filter(
        self,
        array: np.ndarray,
        window_size: int = 7,
    ) -> np.ndarray:
        """Apply Lee speckle filter to a single-band SAR image.

        A simple local-statistics adaptive filter:
            filtered = mean + k * (pixel - mean)
        where k = 1 - (var_noise / var_local).

        Parameters
        ----------
        array : np.ndarray
            2-D SAR intensity array.
        window_size : int
            Size of the sliding window (must be odd).
        """
        from scipy.ndimage import uniform_filter

        img = array.astype(np.float64)
        mean = uniform_filter(img, size=window_size)
        sq_mean = uniform_filter(img ** 2, size=window_size)
        var = sq_mean - mean ** 2
        var = np.maximum(var, 0)

        # Estimate noise variance as the mean of local variances
        noise_var = np.mean(var)
        if noise_var < 1e-10:
            return img

        k = np.where(var > noise_var, 1.0 - noise_var / var, 0.0)
        return mean + k * (img - mean)

    def coregister_pair(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Simple co-registration of target image to reference.

        Uses phase correlation for sub-pixel shift estimation,
        then applies the shift via Fourier transform.

        Parameters
        ----------
        reference, target : np.ndarray
            2-D or 3-D arrays. If 3-D, uses the first band for alignment.

        Returns
        -------
        Shifted target array.
        """
        from scipy.ndimage import shift as ndimage_shift

        # Work with 2-D representations
        ref_2d = reference[0] if reference.ndim == 3 else reference
        tgt_2d = target[0] if target.ndim == 3 else target

        # Ensure same size
        min_h = min(ref_2d.shape[0], tgt_2d.shape[0])
        min_w = min(ref_2d.shape[1], tgt_2d.shape[1])
        ref_crop = ref_2d[:min_h, :min_w].astype(np.float64)
        tgt_crop = tgt_2d[:min_h, :min_w].astype(np.float64)

        # Phase correlation
        f_ref = np.fft.fft2(ref_crop)
        f_tgt = np.fft.fft2(tgt_crop)
        cross_power = (f_ref * np.conj(f_tgt)) / (np.abs(f_ref * np.conj(f_tgt)) + 1e-10)
        correlation = np.abs(np.fft.ifft2(cross_power))
        max_loc = np.unravel_index(np.argmax(correlation), correlation.shape)

        # Convert peak location to shift
        dy = max_loc[0] if max_loc[0] < min_h // 2 else max_loc[0] - min_h
        dx = max_loc[1] if max_loc[1] < min_w // 2 else max_loc[1] - min_w

        logger.info("Co-registration shift: dy=%d, dx=%d", dy, dx)

        # Apply shift to all bands
        if target.ndim == 3:
            result = np.empty_like(target[:, :min_h, :min_w])
            for i in range(target.shape[0]):
                result[i] = ndimage_shift(target[i, :min_h, :min_w], (dy, dx), mode="reflect")
            return result
        else:
            return ndimage_shift(target[:min_h, :min_w], (dy, dx), mode="reflect")

    def compute_difference_map(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
    ) -> np.ndarray:
        """Compute absolute difference map between two images.

        Parameters
        ----------
        img_t1, img_t2 : np.ndarray
            Same-shape 2-D arrays (single band) or 3-D (multi-band).

        Returns
        -------
        np.ndarray : Absolute difference, same shape as inputs.
        """
        t1 = img_t1.astype(np.float64)
        t2 = img_t2.astype(np.float64)

        # If multi-band, compute per-band diff then average
        if t1.ndim == 3 and t2.ndim == 3:
            diff = np.mean(np.abs(t1 - t2), axis=0)
        else:
            diff = np.abs(t1 - t2)

        return diff

    def prepare_for_model(
        self,
        image_input: ImageInput,
        target_size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """Load and prepare an image for model input.

        Returns a PIL RGB image resized to target_size.
        """
        from app.utils.image import image_to_pil

        img = image_to_pil(image_input.filepath)
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
