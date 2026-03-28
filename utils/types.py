from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class GeorefBundle:
    bbox: tuple[float, float, float, float]
    crs: str
    transform: Any
    width: int
    height: int


@dataclass(slots=True)
class RasterBundle:
    array: np.ndarray
    georef: GeorefBundle | None


@dataclass(slots=True)
class InferenceArtifacts:
    pre_image: np.ndarray
    post_image: np.ndarray
    building_mask: np.ndarray
    damage_mask: np.ndarray
    heavy_damage_mask: np.ndarray
    damage_raster: RasterBundle
    damage_probabilities: np.ndarray | None
    localization_probability: np.ndarray | None
    georef: GeorefBundle | None
    used_fallback: bool
    device: str
    warnings: list[str] = field(default_factory=list)
    localization_weights: Path | None = None
    damage_weights: Path | None = None


@dataclass(slots=True)
class GraphBundle:
    graph_latlon: Any
    graph_proj: Any
    graph_crs: str
    to_wgs84: Any
    from_wgs84: Any
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class SnapResult:
    start_node: int
    goal_node: int
    start_snap_m: float
    goal_snap_m: float
    snapped_start: tuple[float, float] | None = None
    snapped_goal: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RouteBundle:
    shortest_nodes: list[int]
    safest_nodes: list[int]
    shortest_length_m: float
    safest_length_m: float
    shortest_cost: float
    safest_cost: float
    shortest_geometry: list[tuple[float, float]]
    safest_geometry: list[tuple[float, float]]


@dataclass(slots=True)
class OperationsBaseCandidate:
    lat: float
    lon: float
    support_score: float
    zone_area_px: int
    road_snap_m: float
    road_class: str
    damage_clearance: float = 0.0
