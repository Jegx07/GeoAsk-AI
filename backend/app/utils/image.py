"""GeoAsk AI — Image processing utility helpers.

Provides functions for image format detection, thumbnail generation,
normalisation, and conversion.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Magic byte signatures for image format detection
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "tiff": [b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"],  # little-endian, big-endian
    "png": [b"\x89\x50\x4e\x47"],
    "jpeg": [b"\xff\xd8\xff"],
}


def detect_format_from_bytes(header: bytes) -> str:
    """Detect image format from the first few bytes of a file."""
    for fmt, sigs in _MAGIC_SIGNATURES.items():
        for sig in sigs:
            if header[: len(sig)] == sig:
                return fmt
    return "unknown"


def detect_image_format(filepath: str) -> str:
    """Detect image format by magic bytes, falling back to extension."""
    path = Path(filepath)

    # Try magic bytes first
    try:
        with open(filepath, "rb") as f:
            header = f.read(16)
        fmt = detect_format_from_bytes(header)
        if fmt != "unknown":
            # Distinguish GeoTIFF from plain TIFF
            if fmt == "tiff":
                try:
                    import rasterio

                    with rasterio.open(filepath) as ds:
                        if ds.crs is not None:
                            return "geotiff"
                except Exception:
                    pass
            return fmt
    except Exception:
        pass

    # Fallback to extension
    ext = path.suffix.lower()
    ext_map = {
        ".tif": "tiff",
        ".tiff": "tiff",
        ".geotiff": "geotiff",
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
    }
    return ext_map.get(ext, "unknown")


def generate_thumbnail(
    filepath: str,
    output_path: str,
    max_size: int = 512,
) -> str:
    """Generate an RGB thumbnail for display.

    Handles multi-band rasters by compositing the first 3 bands as RGB,
    or using a single band as greyscale.
    """
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            bands = ds.count
            if bands >= 3:
                # Use bands 1-3 as RGB
                r = ds.read(1)
                g = ds.read(2)
                b = ds.read(3)
                rgb = np.stack([r, g, b], axis=-1)
            else:
                data = ds.read(1)
                rgb = np.stack([data, data, data], axis=-1)

            # Normalise to 0-255
            rgb = _normalize_to_uint8(rgb)
            img = Image.fromarray(rgb)
    except Exception:
        img = Image.open(filepath).convert("RGB")

    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    img.save(output_path, "PNG")
    return output_path


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 range using 2-98 percentile stretch."""
    arr = arr.astype(np.float64)
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 - p2 < 1e-6:
        p2 = arr.min()
        p98 = arr.max()
    if p98 - p2 < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
    return arr.astype(np.uint8)


def image_to_pil(filepath: str) -> Image.Image:
    """Load any supported image format as a PIL RGB image."""
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            if ds.count >= 3:
                rgb = np.stack([ds.read(i) for i in range(1, 4)], axis=-1)
            else:
                data = ds.read(1)
                rgb = np.stack([data, data, data], axis=-1)
            rgb = _normalize_to_uint8(rgb)
            return Image.fromarray(rgb)
    except Exception:
        return Image.open(filepath).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def get_image_dimensions(filepath: str) -> tuple[int, int]:
    """Return (width, height) of an image."""
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            return ds.width, ds.height
    except Exception:
        img = Image.open(filepath)
        return img.size
