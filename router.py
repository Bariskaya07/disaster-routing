from __future__ import annotations

import copy
import math
from typing import Any

import networkx as nx
import numpy as np

from utils.constants import EPSG_4326
from utils.geospatial import clamp_pixel_indices, latlon_to_pixel, pixel_to_latlon
from utils.types import GraphBundle, OperationsBaseCandidate, RasterBundle, RouteBundle, SnapResult

try:
    import osmnx as ox
except ModuleNotFoundError:  # pragma: no cover - depends on deployment environment
    ox = None

try:
    from pyproj import Transformer
except ModuleNotFoundError:  # pragma: no cover - depends on deployment environment
    Transformer = None

try:
    from shapely.geometry import LineString
except ModuleNotFoundError:  # pragma: no cover - depends on deployment environment
    LineString = None


def _require_router_dependencies() -> None:
    if ox is None or Transformer is None or LineString is None:
        raise RuntimeError("router.py requires osmnx, pyproj, and shapely to be installed.")


def load_graph(
    bbox: tuple[float, float, float, float],
    *,
    network_type: str = "drive_service",
) -> GraphBundle:
    _require_router_dependencies()
    west, south, east, north = bbox
    graph_latlon = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type=network_type,
        simplify=True,
        retain_all=False,
    )
    if graph_latlon.number_of_nodes() == 0:
        raise ValueError("No routable roads found within the given bounding box.")
    graph_proj = ox.project_graph(graph_latlon)
    if graph_proj.number_of_nodes() == 0:
        raise ValueError("Projected road graph is empty for the given bounding box.")
    graph_crs = str(graph_proj.graph["crs"])
    to_wgs84 = Transformer.from_crs(graph_crs, EPSG_4326, always_xy=True)
    from_wgs84 = Transformer.from_crs(EPSG_4326, graph_crs, always_xy=True)

    graph_proj.graph["_to_wgs84"] = to_wgs84
    graph_proj.graph["_from_wgs84"] = from_wgs84
    graph_proj.graph["_bbox"] = bbox

    return GraphBundle(
        graph_latlon=graph_latlon,
        graph_proj=graph_proj,
        graph_crs=graph_crs,
        to_wgs84=to_wgs84,
        from_wgs84=from_wgs84,
        bbox=bbox,
    )


def _edge_geometry(graph_proj: nx.MultiDiGraph, u: int, v: int, key: int, data: dict[str, Any]) -> Any:
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry
    return LineString(
        [
            (graph_proj.nodes[u]["x"], graph_proj.nodes[u]["y"]),
            (graph_proj.nodes[v]["x"], graph_proj.nodes[v]["y"]),
        ]
    )


def _sample_line(geometry: Any, spacing_m: float) -> list[tuple[float, float]]:
    length = max(float(geometry.length), 0.0)
    if length == 0:
        point = geometry.interpolate(0.0)
        return [(float(point.x), float(point.y))]
    num_points = max(2, int(math.ceil(length / spacing_m)) + 1)
    distances = np.linspace(0.0, length, num_points)
    samples = [geometry.interpolate(float(distance)) for distance in distances]
    return [(float(point.x), float(point.y)) for point in samples]


def _edge_length_m(graph_proj: nx.MultiDiGraph, u: int, v: int, data: dict[str, Any]) -> float:
    if "length" in data and data["length"] is not None:
        return float(data["length"])
    geometry = _edge_geometry(graph_proj, u, v, 0, data)
    return float(geometry.length)


def _connected_components(mask: np.ndarray) -> list[dict[str, float | int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, float | int]] = []
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or visited[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            area = 0
            row_sum = 0.0
            col_sum = 0.0

            while stack:
                row, col = stack.pop()
                area += 1
                row_sum += float(row)
                col_sum += float(col)

                for delta_row, delta_col in neighbors:
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if next_row < 0 or next_col < 0 or next_row >= height or next_col >= width:
                        continue
                    if visited[next_row, next_col] or not mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            components.append(
                {
                    "area": area,
                    "centroid_row": row_sum / area,
                    "centroid_col": col_sum / area,
                }
            )

    return components


def _coerce_highway_values(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, (list, tuple, set)):
        return [str(value) for value in raw_value]
    return [str(raw_value)]


def _node_best_road_class(graph_proj: nx.MultiDiGraph, node_id: int) -> str:
    road_priority = {
        "motorway": 6,
        "trunk": 5,
        "primary": 4,
        "secondary": 3,
        "tertiary": 2,
        "residential": 1,
        "service": 1,
        "unclassified": 1,
    }
    best_class = "local-road"
    best_score = 0

    for _, _, data in graph_proj.edges(node_id, data=True):
        for highway in _coerce_highway_values(data.get("highway")):
            score = road_priority.get(highway, 0)
            if score > best_score:
                best_score = score
                best_class = highway

    return best_class


def recommend_operations_base(
    graph_proj: nx.MultiDiGraph,
    georef: Any,
    building_mask: np.ndarray,
    damage_raster_array: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    *,
    min_zone_pixels: int = 64,
    openness_window_radius: int = 24,
    road_distance_cap_m: float = 250.0,
    max_candidate_road_snap_m: float = 85.0,
) -> OperationsBaseCandidate | None:
    if georef is None:
        return None

    from_wgs84 = graph_proj.graph.get("_from_wgs84")
    if from_wgs84 is None:
        raise ValueError("Projected graph is missing the WGS84-to-graph transformer.")

    open_mask = building_mask == 0
    if exclusion_mask is not None:
        open_mask = open_mask & (~exclusion_mask.astype(bool))
    components = [component for component in _connected_components(open_mask) if int(component["area"]) >= min_zone_pixels]
    if not components:
        return None
    max_area = max(int(component["area"]) for component in components)
    road_priority = {
        "motorway": 1.0,
        "trunk": 0.95,
        "primary": 0.9,
        "secondary": 0.82,
        "tertiary": 0.72,
        "residential": 0.6,
        "service": 0.5,
        "unclassified": 0.45,
        "local-road": 0.4,
    }

    best_candidate: OperationsBaseCandidate | None = None

    for component in components:
        area = int(component["area"])
        centroid_row = int(round(float(component["centroid_row"])))
        centroid_col = int(round(float(component["centroid_col"])))

        row_start = max(0, centroid_row - openness_window_radius)
        row_end = min(building_mask.shape[0], centroid_row + openness_window_radius + 1)
        col_start = max(0, centroid_col - openness_window_radius)
        col_end = min(building_mask.shape[1], centroid_col + openness_window_radius + 1)
        local_window = building_mask[row_start:row_end, col_start:col_end] == 0
        if exclusion_mask is not None:
            local_window = local_window & (~exclusion_mask[row_start:row_end, col_start:col_end].astype(bool))
        local_open_area = int(local_window.sum())
        local_damage_window = damage_raster_array[row_start:row_end, col_start:col_end] > 0
        damage_clearance = max(0.0, 1.0 - float(local_damage_window.mean())) if local_damage_window.size else 1.0

        lat, lon = pixel_to_latlon(georef, centroid_row, centroid_col)
        x_coord, y_coord = from_wgs84.transform(lon, lat)
        nearest_node, snap_distance_m = _bruteforce_nearest_node(graph_proj, x_coord, y_coord)
        if snap_distance_m > max_candidate_road_snap_m:
            continue
        road_class = _node_best_road_class(graph_proj, nearest_node)
        area_norm = max(local_open_area, area) / max_area if max_area else 0.0
        access_norm = max(0.0, 1.0 - min(float(snap_distance_m), road_distance_cap_m) / road_distance_cap_m)
        road_class_norm = road_priority.get(road_class, road_priority["local-road"])
        support_score = float(0.45 * area_norm + 0.25 * access_norm + 0.15 * road_class_norm + 0.15 * damage_clearance)

        candidate = OperationsBaseCandidate(
            lat=float(lat),
            lon=float(lon),
            support_score=support_score,
            zone_area_px=max(local_open_area, area),
            road_snap_m=float(snap_distance_m),
            road_class=road_class,
            damage_clearance=damage_clearance,
        )
        if best_candidate is None or candidate.support_score > best_candidate.support_score:
            best_candidate = candidate

    return best_candidate


def _sample_damage_value(
    raster: np.ndarray,
    row: int,
    col: int,
    *,
    neighborhood_radius_px: int,
) -> int:
    row_start = max(0, row - neighborhood_radius_px)
    row_end = min(raster.shape[0], row + neighborhood_radius_px + 1)
    col_start = max(0, col - neighborhood_radius_px)
    col_end = min(raster.shape[1], col + neighborhood_radius_px + 1)
    if row_start >= row_end or col_start >= col_end:
        return 0
    return int(raster[row_start:row_end, col_start:col_end].max())


def apply_damage_cost(
    graph_proj: nx.MultiDiGraph,
    damage_raster: RasterBundle,
    *,
    sample_spacing_m: float = 10.0,
    damage_exponent: float = 3.5,
    neighborhood_radius_px: int = 4,
) -> nx.MultiDiGraph:
    if damage_raster.georef is None:
        raise ValueError("Damage raster georeferencing metadata is required for routing.")

    graph = nx.MultiDiGraph()
    graph.graph.update(graph_proj.graph)
    graph.add_nodes_from((node, copy.copy(data)) for node, data in graph_proj.nodes(data=True))
    for u, v, key, data in graph_proj.edges(keys=True, data=True):
        graph.add_edge(u, v, key=key, **copy.copy(data))
    to_wgs84 = graph.graph.get("_to_wgs84")
    if to_wgs84 is None:
        raise ValueError("Projected graph is missing the WGS84 transformer.")

    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = _edge_geometry(graph, u, v, key, data)
        samples_proj = _sample_line(geometry, sample_spacing_m)
        sample_risks: list[int] = []
        for x_coord, y_coord in samples_proj:
            lon, lat = to_wgs84.transform(x_coord, y_coord)
            row, col = latlon_to_pixel(damage_raster.georef, lat, lon)
            pixel = clamp_pixel_indices(damage_raster.georef, row, col)
            if pixel is None:
                continue
            pixel_row, pixel_col = pixel
            sample_risks.append(
                _sample_damage_value(
                    damage_raster.array,
                    pixel_row,
                    pixel_col,
                    neighborhood_radius_px=neighborhood_radius_px,
                )
            )

        damage_raw = 0.0 if not sample_risks else float(np.percentile(sample_risks, 85))
        damage_norm = damage_raw / 255.0
        length_m = _edge_length_m(graph, u, v, data)
        safe_weight = float(length_m * math.exp(damage_exponent * damage_norm))

        data["damage_score"] = damage_raw
        data["damage_norm"] = damage_norm
        data["safe_weight"] = safe_weight
        data["length"] = length_m

    return graph


def snap_points(
    graph_proj: nx.MultiDiGraph,
    start_latlon: tuple[float, float],
    goal_latlon: tuple[float, float],
    *,
    warning_threshold_m: float = 150.0,
) -> SnapResult:
    _require_router_dependencies()
    from_wgs84 = graph_proj.graph.get("_from_wgs84")
    to_wgs84 = graph_proj.graph.get("_to_wgs84")
    if from_wgs84 is None:
        raise ValueError("Projected graph is missing the WGS84-to-graph transformer.")
    if to_wgs84 is None:
        raise ValueError("Projected graph is missing the graph-to-WGS84 transformer.")

    start_lon, start_lat = float(start_latlon[1]), float(start_latlon[0])
    goal_lon, goal_lat = float(goal_latlon[1]), float(goal_latlon[0])
    start_x, start_y = from_wgs84.transform(start_lon, start_lat)
    goal_x, goal_y = from_wgs84.transform(goal_lon, goal_lat)

    warnings: list[str] = []
    try:
        start_node, start_dist = ox.distance.nearest_nodes(graph_proj, start_x, start_y, return_dist=True)
        goal_node, goal_dist = ox.distance.nearest_nodes(graph_proj, goal_x, goal_y, return_dist=True)
    except Exception as exc:
        if "scipy" not in str(exc).lower():
            raise
        start_node, start_dist = _bruteforce_nearest_node(graph_proj, start_x, start_y)
        goal_node, goal_dist = _bruteforce_nearest_node(graph_proj, goal_x, goal_y)
        warnings.append(
            "SciPy bulunamadığı için yol düğümü sabitleme işlemi yedek modda yapıldı; rota geçerlidir ancak biraz daha yavaş hesaplanır."
        )

    if start_dist > warning_threshold_m:
        warnings.append(
            f"Başlangıç noktası en yakın yol düğümüne {start_dist:.1f} m öteden sabitlendi."
        )
    if goal_dist > warning_threshold_m:
        warnings.append(
            f"Hedef noktası en yakın yol düğümüne {goal_dist:.1f} m öteden sabitlendi."
        )

    start_node_data = graph_proj.nodes[int(start_node)]
    goal_node_data = graph_proj.nodes[int(goal_node)]
    start_lon_snapped, start_lat_snapped = to_wgs84.transform(float(start_node_data["x"]), float(start_node_data["y"]))
    goal_lon_snapped, goal_lat_snapped = to_wgs84.transform(float(goal_node_data["x"]), float(goal_node_data["y"]))

    return SnapResult(
        start_node=int(start_node),
        goal_node=int(goal_node),
        start_snap_m=float(start_dist),
        goal_snap_m=float(goal_dist),
        snapped_start=(float(start_lat_snapped), float(start_lon_snapped)),
        snapped_goal=(float(goal_lat_snapped), float(goal_lon_snapped)),
        warnings=warnings,
    )


def _bruteforce_nearest_node(
    graph_proj: nx.MultiDiGraph,
    target_x: float,
    target_y: float,
) -> tuple[int, float]:
    nearest_node: int | None = None
    nearest_distance_sq: float | None = None

    for node_id, data in graph_proj.nodes(data=True):
        node_x = float(data["x"])
        node_y = float(data["y"])
        distance_sq = (node_x - target_x) ** 2 + (node_y - target_y) ** 2
        if nearest_distance_sq is None or distance_sq < nearest_distance_sq:
            nearest_node = int(node_id)
            nearest_distance_sq = distance_sq

    if nearest_node is None or nearest_distance_sq is None:
        raise ValueError("Projected graph has no nodes available for snapping.")
    return nearest_node, math.sqrt(nearest_distance_sq)


def _projected_distance_heuristic(graph_proj: nx.MultiDiGraph, source: int, target: int) -> float:
    source_x = float(graph_proj.nodes[source]["x"])
    source_y = float(graph_proj.nodes[source]["y"])
    target_x = float(graph_proj.nodes[target]["x"])
    target_y = float(graph_proj.nodes[target]["y"])
    return math.hypot(target_x - source_x, target_y - source_y)


def recommend_demo_route_points(
    graph_proj: nx.MultiDiGraph,
    graph_latlon: nx.MultiDiGraph,
    *,
    side_candidate_count: int = 8,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    nodes = [
        (int(node_id), float(data["x"]), float(data["y"]))
        for node_id, data in graph_latlon.nodes(data=True)
        if "x" in data and "y" in data
    ]
    if len(nodes) < 2:
        return None

    west_nodes = [node_id for node_id, _, _ in sorted(nodes, key=lambda item: item[1])[:side_candidate_count]]
    east_nodes = [node_id for node_id, _, _ in sorted(nodes, key=lambda item: item[1], reverse=True)[:side_candidate_count]]
    south_nodes = [node_id for node_id, _, _ in sorted(nodes, key=lambda item: item[2])[:side_candidate_count]]
    north_nodes = [node_id for node_id, _, _ in sorted(nodes, key=lambda item: item[2], reverse=True)[:side_candidate_count]]

    candidate_pairs = [
        (start_node, goal_node)
        for start_side, goal_side in ((west_nodes, east_nodes), (south_nodes, north_nodes))
        for start_node in start_side
        for goal_node in goal_side
        if start_node != goal_node
    ]
    if not candidate_pairs:
        return None

    best_pair: tuple[int, int] | None = None
    best_length = -1.0
    for start_node, goal_node in candidate_pairs:
        try:
            path_length = float(nx.shortest_path_length(graph_proj, start_node, goal_node, weight="length"))
        except Exception:
            continue
        if path_length > best_length:
            best_length = path_length
            best_pair = (start_node, goal_node)

    if best_pair is None:
        return None

    start_node, goal_node = best_pair
    start_data = graph_latlon.nodes[start_node]
    goal_data = graph_latlon.nodes[goal_node]
    start = (float(start_data["y"]), float(start_data["x"]))
    goal = (float(goal_data["y"]), float(goal_data["x"]))
    return start, goal


def _best_edge_data(graph: nx.MultiDiGraph, u: int, v: int, weight: str) -> dict[str, Any]:
    edges = graph.get_edge_data(u, v)
    if not edges:
        raise KeyError(f"No edge found between {u} and {v}.")
    _, data = min(edges.items(), key=lambda item: float(item[1].get(weight, float("inf"))))
    return data


def _best_edge_key(graph: nx.MultiDiGraph, u: int, v: int, weight: str) -> int:
    edges = graph.get_edge_data(u, v)
    if not edges:
        raise KeyError(f"No edge found between {u} and {v}.")
    key, _ = min(edges.items(), key=lambda item: float(item[1].get(weight, float("inf"))))
    return int(key)


def _path_length(graph: nx.MultiDiGraph, nodes: list[int], weight: str) -> float:
    total = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        total += float(_best_edge_data(graph, u, v, weight).get(weight, 0.0))
    return total


def _path_to_geometry(
    graph: nx.MultiDiGraph,
    nodes: list[int],
    *,
    metric_graph: nx.MultiDiGraph | None = None,
    weight: str,
) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        selector_graph = metric_graph if metric_graph is not None else graph
        edge_key = _best_edge_key(selector_graph, u, v, weight)
        edge_dict = graph.get_edge_data(u, v)
        if not edge_dict:
            raise KeyError(f"No edge found between {u} and {v}.")
        edge_data = edge_dict.get(edge_key)
        if edge_data is None:
            edge_data = _best_edge_data(graph, u, v, "length")
        geometry = edge_data.get("geometry")
        if geometry is None:
            segment = [
                (float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])),
                (float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])),
            ]
        else:
            segment = [(float(lat), float(lon)) for lon, lat in geometry.coords]
        if coordinates:
            coordinates.extend(segment[1:])
        else:
            coordinates.extend(segment)
    return coordinates


def _edge_to_geometry(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
) -> list[tuple[float, float]]:
    edge_dict = graph.get_edge_data(u, v)
    if not edge_dict:
        raise KeyError(f"No edge found between {u} and {v}.")
    edge_data = edge_dict.get(key)
    if edge_data is None:
        edge_data = next(iter(edge_dict.values()))
    geometry = edge_data.get("geometry")
    if geometry is None:
        return [
            (float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])),
            (float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])),
        ]
    return [(float(lat), float(lon)) for lon, lat in geometry.coords]


def extract_high_damage_segments(
    graph_proj: nx.MultiDiGraph,
    *,
    graph_latlon: nx.MultiDiGraph | None = None,
    min_damage_threshold: float = 0.18,
    percentile: float = 82.0,
    min_length_m: float = 12.0,
    min_segments: int = 12,
    limit: int = 80,
) -> list[dict[str, Any]]:
    route_graph = graph_latlon if graph_latlon is not None else ox.project_graph(graph_proj, to_latlong=True)
    candidates: list[dict[str, Any]] = []

    for u, v, key, data in graph_proj.edges(keys=True, data=True):
        damage_norm = float(data.get("damage_norm", 0.0))
        length_m = float(data.get("length", 0.0))
        if damage_norm <= 0.0 or length_m < min_length_m:
            continue
        try:
            geometry = _edge_to_geometry(route_graph, u, v, key)
        except KeyError:
            continue
        candidates.append(
            {
                "geometry": geometry,
                "damage_norm": damage_norm,
                "length_m": length_m,
            }
        )

    if not candidates:
        return []

    damage_values = np.asarray([item["damage_norm"] for item in candidates], dtype=np.float32)
    adaptive_threshold = max(float(min_damage_threshold), float(np.percentile(damage_values, percentile)))
    selected = [item for item in candidates if item["damage_norm"] >= adaptive_threshold]

    candidates.sort(key=lambda item: (item["damage_norm"], item["length_m"]), reverse=True)
    if len(selected) < min_segments:
        selected = candidates[: min(min_segments, len(candidates))]
    else:
        selected.sort(key=lambda item: (item["damage_norm"], item["length_m"]), reverse=True)

    return selected[:limit]


def find_routes(
    graph_proj: nx.MultiDiGraph,
    start_node: int,
    goal_node: int,
    *,
    baseline_graph_proj: nx.MultiDiGraph | None = None,
    graph_latlon: nx.MultiDiGraph | None = None,
) -> RouteBundle:
    if start_node == goal_node:
        raise ValueError("Start and goal snapped to the same road node; they are too close to route.")

    baseline_graph = baseline_graph_proj if baseline_graph_proj is not None else graph_proj

    try:
        safest_nodes = nx.astar_path(
            graph_proj,
            start_node,
            goal_node,
            heuristic=lambda u, v: _projected_distance_heuristic(graph_proj, u, v),
            weight="safe_weight",
        )
    except Exception:
        safest_nodes = nx.shortest_path(graph_proj, start_node, goal_node, weight="safe_weight")

    shortest_nodes = nx.shortest_path(baseline_graph, start_node, goal_node, weight="length")

    route_graph = graph_latlon if graph_latlon is not None else ox.project_graph(graph_proj, to_latlong=True)

    return RouteBundle(
        shortest_nodes=[int(node) for node in shortest_nodes],
        safest_nodes=[int(node) for node in safest_nodes],
        shortest_length_m=_path_length(baseline_graph, shortest_nodes, "length"),
        safest_length_m=_path_length(graph_proj, safest_nodes, "length"),
        shortest_cost=_path_length(graph_proj, shortest_nodes, "safe_weight"),
        safest_cost=_path_length(graph_proj, safest_nodes, "safe_weight"),
        shortest_geometry=_path_to_geometry(route_graph, shortest_nodes, metric_graph=baseline_graph, weight="length"),
        safest_geometry=_path_to_geometry(route_graph, safest_nodes, metric_graph=graph_proj, weight="safe_weight"),
    )
