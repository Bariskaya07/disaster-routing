from __future__ import annotations

from typing import Any

from .constants import EPSG_4326
from .types import GeorefBundle

try:
    from rasterio.transform import from_bounds, rowcol, xy
except ModuleNotFoundError:  # pragma: no cover - exercised only in underspecified local envs
    from_bounds = None
    rowcol = None
    xy = None


def _require_rasterio() -> None:
    if from_bounds is None or rowcol is None or xy is None:
        raise RuntimeError("rasterio is required for georeferencing utilities.")


def build_georef(bbox: tuple[float, float, float, float], width: int, height: int) -> GeorefBundle:
    _require_rasterio()
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    return GeorefBundle(
        bbox=bbox,
        crs=EPSG_4326,
        transform=transform,
        width=int(width),
        height=int(height),
    )


def pixel_to_latlon(georef: GeorefBundle, row: int, col: int) -> tuple[float, float]:
    _require_rasterio()
    lon, lat = xy(georef.transform, row, col)
    return float(lat), float(lon)


def latlon_to_pixel(georef: GeorefBundle, lat: float, lon: float) -> tuple[int, int]:
    _require_rasterio()
    row, col = rowcol(georef.transform, lon, lat)
    return int(row), int(col)


def clamp_pixel_indices(georef: GeorefBundle, row: int, col: int) -> tuple[int, int] | None:
    if row < 0 or col < 0 or row >= georef.height or col >= georef.width:
        return None
    return row, col


def image_bounds_for_folium(georef: GeorefBundle) -> list[list[float]]:
    west, south, east, north = georef.bbox
    return [[south, west], [north, east]]
