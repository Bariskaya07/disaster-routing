from __future__ import annotations

import unittest
from unittest import mock

import networkx as nx
import numpy as np

import router
from utils.types import GeorefBundle, RasterBundle


class _Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _Line:
    def __init__(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        self.start = start
        self.end = end
        self.length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        self.coords = [start, end]

    def interpolate(self, distance: float) -> _Point:
        if self.length == 0:
            return _Point(*self.start)
        ratio = max(0.0, min(1.0, distance / self.length))
        x = self.start[0] + (self.end[0] - self.start[0]) * ratio
        y = self.start[1] + (self.end[1] - self.start[1]) * ratio
        return _Point(x, y)


class _IdentityTransformer:
    def transform(self, x: float, y: float) -> tuple[float, float]:
        return x, y


class RouterTests(unittest.TestCase):
    def test_snap_points_falls_back_without_scipy(self) -> None:
        graph = nx.MultiDiGraph()
        graph.graph["_from_wgs84"] = _IdentityTransformer()
        graph.graph["_to_wgs84"] = _IdentityTransformer()
        graph.add_node(10, x=0.0, y=0.0)
        graph.add_node(20, x=100.0, y=0.0)
        ox_stub = mock.Mock()
        ox_stub.distance.nearest_nodes.side_effect = RuntimeError(
            "scipy must be installed as an optional dependency to search a projected graph"
        )

        with mock.patch("router._require_router_dependencies"), mock.patch.object(router, "ox", ox_stub):
            snap = router.snap_points(graph, (1.0, 2.0), (3.0, 90.0), warning_threshold_m=1000.0)

        self.assertEqual(snap.start_node, 10)
        self.assertEqual(snap.goal_node, 20)
        self.assertEqual(snap.snapped_start, (0.0, 0.0))
        self.assertEqual(snap.snapped_goal, (0.0, 100.0))
        self.assertTrue(any("SciPy" in warning for warning in snap.warnings))

    def test_apply_damage_cost_penalizes_edge_near_damaged_pixels(self) -> None:
        graph = nx.MultiDiGraph()
        graph.graph["_to_wgs84"] = _IdentityTransformer()
        graph.add_node(1, x=0.0, y=0.0)
        graph.add_node(2, x=10.0, y=0.0)
        graph.add_node(3, x=0.0, y=10.0)
        graph.add_node(4, x=10.0, y=10.0)
        graph.add_edge(1, 2, key=0, length=10.0, geometry=_Line((0.0, 0.0), (10.0, 0.0)))
        graph.add_edge(1, 3, key=0, length=10.0, geometry=_Line((0.0, 10.0), (10.0, 10.0)))

        damage_raster = np.zeros((20, 20), dtype=np.uint8)
        damage_raster[0:2, 0:12] = 255
        georef = GeorefBundle(bbox=(0.0, 0.0, 20.0, 20.0), crs="EPSG:4326", transform=None, width=20, height=20)
        bundle = RasterBundle(array=damage_raster, georef=georef)

        with mock.patch("router.latlon_to_pixel", side_effect=lambda georef, lat, lon: (int(round(lat)), int(round(lon)))), mock.patch(
            "router.clamp_pixel_indices", side_effect=lambda georef, row, col: (row, col)
            if 0 <= row < georef.height and 0 <= col < georef.width
            else None
        ):
            scored = router.apply_damage_cost(graph, bundle, sample_spacing_m=5.0, neighborhood_radius_px=0)

        risky = scored.edges[(1, 2, 0)]
        safe = scored.edges[(1, 3, 0)]
        self.assertGreater(risky["damage_score"], safe["damage_score"])
        self.assertGreater(risky["safe_weight"], safe["safe_weight"])

    def test_find_routes_prefers_safe_detour(self) -> None:
        graph = nx.MultiDiGraph()
        for node_id, x_coord, y_coord in [
            (1, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 2.0, 0.0),
            (4, 0.0, 1.0),
            (5, 1.0, 1.0),
            (6, 2.0, 1.0),
        ]:
            graph.add_node(node_id, x=x_coord, y=y_coord)

        graph.add_edge(1, 2, key=0, length=5.0, safe_weight=50.0)
        graph.add_edge(2, 3, key=0, length=5.0, safe_weight=50.0)
        graph.add_edge(1, 4, key=0, length=8.0, safe_weight=8.0)
        graph.add_edge(4, 5, key=0, length=8.0, safe_weight=8.0)
        graph.add_edge(5, 6, key=0, length=8.0, safe_weight=8.0)
        graph.add_edge(6, 3, key=0, length=8.0, safe_weight=8.0)

        route = router.find_routes(graph, 1, 3, graph_latlon=graph)
        self.assertEqual(route.shortest_nodes, [1, 2, 3])
        self.assertEqual(route.safest_nodes, [1, 4, 5, 6, 3])
        self.assertGreater(route.shortest_cost, route.safest_cost)

    def test_find_routes_uses_baseline_graph_for_reference_shortest_path(self) -> None:
        baseline_graph = nx.MultiDiGraph()
        risk_graph = nx.MultiDiGraph()

        for graph in (baseline_graph, risk_graph):
            for node_id, x_coord, y_coord in [
                (1, 0.0, 0.0),
                (2, 1.0, 0.0),
                (3, 2.0, 0.0),
                (4, 0.0, 1.0),
                (5, 1.0, 1.0),
                (6, 2.0, 1.0),
            ]:
                graph.add_node(node_id, x=x_coord, y=y_coord)

        baseline_graph.add_edge(1, 2, key=0, length=5.0)
        baseline_graph.add_edge(2, 3, key=0, length=5.0)
        baseline_graph.add_edge(1, 4, key=0, length=8.0)
        baseline_graph.add_edge(4, 5, key=0, length=8.0)
        baseline_graph.add_edge(5, 6, key=0, length=8.0)
        baseline_graph.add_edge(6, 3, key=0, length=8.0)

        risk_graph.add_edge(1, 2, key=0, length=5.0, safe_weight=50.0)
        risk_graph.add_edge(2, 3, key=0, length=5.0, safe_weight=50.0)
        risk_graph.add_edge(1, 4, key=0, length=8.0, safe_weight=8.0)
        risk_graph.add_edge(4, 5, key=0, length=8.0, safe_weight=8.0)
        risk_graph.add_edge(5, 6, key=0, length=8.0, safe_weight=8.0)
        risk_graph.add_edge(6, 3, key=0, length=8.0, safe_weight=8.0)

        route = router.find_routes(risk_graph, 1, 3, baseline_graph_proj=baseline_graph, graph_latlon=risk_graph)
        self.assertEqual(route.shortest_nodes, [1, 2, 3])
        self.assertEqual(route.safest_nodes, [1, 4, 5, 6, 3])

    def test_find_routes_uses_metric_graph_edge_key_for_geometry(self) -> None:
        graph_proj = nx.MultiDiGraph()
        graph_latlon = nx.MultiDiGraph()

        for graph in (graph_proj, graph_latlon):
            graph.add_node(1, x=0.0, y=0.0)
            graph.add_node(2, x=1.0, y=1.0)

        graph_proj.add_edge(1, 2, key=0, length=5.0, safe_weight=50.0)
        graph_proj.add_edge(1, 2, key=1, length=6.0, safe_weight=6.0)

        graph_latlon.add_edge(1, 2, key=0, length=5.0, geometry=_Line((0.0, 0.0), (1.0, 0.0)))
        graph_latlon.add_edge(1, 2, key=1, length=6.0, geometry=_Line((0.0, 0.0), (1.0, 1.0)))

        route = router.find_routes(graph_proj, 1, 2, graph_latlon=graph_latlon)
        self.assertEqual(route.shortest_geometry, [(0.0, 0.0), (0.0, 1.0)])
        self.assertEqual(route.safest_geometry, [(0.0, 0.0), (1.0, 1.0)])

    def test_find_routes_rejects_same_start_and_goal_node(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node(1, x=0.0, y=0.0)
        with self.assertRaises(ValueError):
            router.find_routes(graph, 1, 1, graph_latlon=graph)

    def test_extract_high_damage_segments_uses_adaptive_selection(self) -> None:
        graph_proj = nx.MultiDiGraph()
        graph_latlon = nx.MultiDiGraph()

        for graph in (graph_proj, graph_latlon):
            graph.add_node(1, x=0.0, y=0.0)
            graph.add_node(2, x=1.0, y=0.0)
            graph.add_node(3, x=2.0, y=0.0)

        graph_proj.add_edge(1, 2, key=0, length=20.0, damage_norm=0.22)
        graph_proj.add_edge(2, 3, key=0, length=20.0, damage_norm=0.19)

        graph_latlon.add_edge(1, 2, key=0, length=20.0, geometry=_Line((0.0, 0.0), (1.0, 0.0)))
        graph_latlon.add_edge(2, 3, key=0, length=20.0, geometry=_Line((1.0, 0.0), (2.0, 0.0)))

        segments = router.extract_high_damage_segments(
            graph_proj,
            graph_latlon=graph_latlon,
            min_damage_threshold=0.5,
            min_segments=2,
        )
        self.assertEqual(len(segments), 2)

    def test_recommend_demo_route_points_uses_real_graph_nodes(self) -> None:
        graph_proj = nx.MultiDiGraph()
        graph_latlon = nx.MultiDiGraph()

        for graph in (graph_proj, graph_latlon):
            graph.add_node(1, x=0.0, y=0.0)
            graph.add_node(2, x=5.0, y=0.0)
            graph.add_node(3, x=10.0, y=0.0)
            graph.add_node(4, x=0.0, y=8.0)
            graph.add_node(5, x=5.0, y=8.0)
            graph.add_node(6, x=10.0, y=8.0)

        for graph in (graph_proj, graph_latlon):
            graph.add_edge(1, 2, key=0, length=5.0)
            graph.add_edge(2, 3, key=0, length=5.0)
            graph.add_edge(4, 5, key=0, length=5.0)
            graph.add_edge(5, 6, key=0, length=5.0)
            graph.add_edge(1, 4, key=0, length=8.0)
            graph.add_edge(2, 5, key=0, length=8.0)
            graph.add_edge(3, 6, key=0, length=8.0)

        points = router.recommend_demo_route_points(graph_proj, graph_latlon, side_candidate_count=3)

        self.assertIsNotNone(points)
        assert points is not None
        start, goal = points
        self.assertNotEqual(start, goal)
        self.assertIn(start, [(0.0, 0.0), (8.0, 0.0), (0.0, 10.0), (8.0, 10.0)])
        self.assertIn(goal, [(0.0, 0.0), (8.0, 0.0), (0.0, 10.0), (8.0, 10.0)])

    def test_recommend_operations_base_prefers_large_open_zone_near_better_road(self) -> None:
        graph = nx.MultiDiGraph()
        graph.graph["_from_wgs84"] = _IdentityTransformer()
        graph.add_node(1, x=2.0, y=2.0)
        graph.add_node(2, x=8.0, y=8.0)
        graph.add_edge(1, 2, key=0, highway="primary", length=10.0, safe_weight=10.0)

        graph.add_node(3, x=18.0, y=18.0)
        graph.add_node(4, x=19.0, y=18.0)
        graph.add_edge(3, 4, key=0, highway="service", length=2.0, safe_weight=2.0)

        building_mask = np.ones((24, 24), dtype=np.uint8)
        damage_raster = np.zeros((24, 24), dtype=np.uint8)

        building_mask[0:9, 0:9] = 0
        damage_raster[0:9, 0:9] = 0

        building_mask[15:19, 15:19] = 0
        damage_raster[15:19, 15:19] = 255

        with mock.patch("router.pixel_to_latlon", side_effect=lambda georef, row, col: (float(row), float(col))):
            base = router.recommend_operations_base(graph, object(), building_mask, damage_raster, min_zone_pixels=3)

        self.assertIsNotNone(base)
        assert base is not None
        self.assertLess(base.road_snap_m, 5.0)
        self.assertEqual(base.road_class, "primary")
        self.assertGreaterEqual(base.zone_area_px, 20)
        self.assertGreater(base.damage_clearance, 0.5)

    def test_recommend_operations_base_respects_exclusion_mask(self) -> None:
        graph = nx.MultiDiGraph()
        graph.graph["_from_wgs84"] = _IdentityTransformer()
        graph.add_node(1, x=2.0, y=2.0)
        graph.add_node(2, x=20.0, y=20.0)
        graph.add_edge(1, 2, key=0, highway="primary", length=30.0, safe_weight=30.0)

        building_mask = np.ones((32, 32), dtype=np.uint8)
        damage_raster = np.zeros((32, 32), dtype=np.uint8)
        exclusion_mask = np.zeros((32, 32), dtype=bool)

        building_mask[0:12, 0:12] = 0
        building_mask[18:28, 18:28] = 0
        exclusion_mask[0:12, 0:12] = True

        with mock.patch("router.pixel_to_latlon", side_effect=lambda georef, row, col: (float(row), float(col))):
            base = router.recommend_operations_base(
                graph,
                object(),
                building_mask,
                damage_raster,
                exclusion_mask=exclusion_mask,
                min_zone_pixels=8,
            )

        self.assertIsNotNone(base)
        assert base is not None
        self.assertGreaterEqual(base.lat, 18.0)
        self.assertGreaterEqual(base.lon, 18.0)

    def test_recommend_operations_base_rejects_candidates_too_far_from_road(self) -> None:
        graph = nx.MultiDiGraph()
        graph.graph["_from_wgs84"] = _IdentityTransformer()
        graph.add_node(1, x=0.0, y=0.0)
        graph.add_node(2, x=2.0, y=0.0)
        graph.add_edge(1, 2, key=0, highway="primary", length=2.0, safe_weight=2.0)

        building_mask = np.ones((64, 64), dtype=np.uint8)
        damage_raster = np.zeros((64, 64), dtype=np.uint8)
        building_mask[40:56, 40:56] = 0

        with mock.patch("router.pixel_to_latlon", side_effect=lambda georef, row, col: (float(row), float(col))):
            base = router.recommend_operations_base(
                graph,
                object(),
                building_mask,
                damage_raster,
                min_zone_pixels=16,
                max_candidate_road_snap_m=20.0,
            )

        self.assertIsNone(base)


if __name__ == "__main__":
    unittest.main()
