"""GeoAsk AI — Geospatial utility helpers.

Provides functions for extracting metadata from GeoTIFF files,
coordinate transforms, and spatial operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_geo_metadata(filepath: str) -> dict[str, Any] | None:
    """Extract geospatial metadata from a raster file using rasterio.

    Returns None if the file has no geospatial information or rasterio
    is not available.
    """
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            crs = str(ds.crs) if ds.crs else None
            bounds = list(ds.bounds) if ds.bounds else None
            resolution = list(ds.res) if ds.res else None
            transform = list(ds.transform)[:6] if ds.transform else None

            return {
                "crs": crs,
                "bounds": bounds,
                "resolution": resolution,
                "transform": transform,
                "width": ds.width,
                "height": ds.height,
            }
    except ImportError:
        logger.warning("rasterio not installed — skipping geo metadata extraction")
        return None
    except Exception as exc:
        logger.warning("Failed to extract geo metadata from %s: %s", filepath, exc)
        return None


def read_raster_bands(filepath: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read all bands of a raster file into a numpy array.

    Returns
    -------
    (bands, profile) where bands has shape (band_count, height, width).
    """
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            bands = ds.read()  # shape: (bands, H, W)
            profile = dict(ds.profile)
            return bands, profile
    except ImportError:
        # Fallback: use PIL for non-geospatial images
        from PIL import Image

        img = Image.open(filepath)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return arr, {"width": arr.shape[2], "height": arr.shape[1], "count": arr.shape[0]}


def bounds_to_geojson(bounds: list[float]) -> dict[str, Any]:
    """Convert [west, south, east, north] bounds to a GeoJSON Polygon."""
    w, s, e, n = bounds
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        },
        "properties": {},
    }


def get_raster_band_count(filepath: str) -> int:
    """Return the number of bands in a raster file."""
    try:
        import rasterio

        with rasterio.open(filepath) as ds:
            return ds.count
    except Exception:
        from PIL import Image

        img = Image.open(filepath)
        arr = np.array(img)
        if arr.ndim == 2:
            return 1
        return arr.shape[2]
