from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DISASTERS = (
    "santa-rosa-wildfire",
    "palu-tsunami",
    "hurricane-harvey",
    "mexico-earthquake",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top-ranked xView2 hold scenes into a local demo library."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("hold_images_labels_targets.tar"),
        help="Path to the hold split archive.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/demo_library"),
        help="Destination directory for extracted demo candidates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of scenes to keep per disaster.",
    )
    parser.add_argument(
        "--disaster",
        action="append",
        dest="disasters",
        help="Disaster slug to include. Can be repeated. Defaults to the four requested demo groups.",
    )
    parser.add_argument(
        "--activate-scene",
        type=str,
        default=None,
        help="Optional scene id to copy into data/demo as the active one-button demo.",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/demo"),
        help="Active demo directory used by the Streamlit demo button.",
    )
    return parser.parse_args()


def _parse_wkt_pairs(wkt: str) -> list[tuple[float, float]]:
    pattern = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    return [(float(a), float(b)) for a, b in re.findall(pattern, wkt)]


def _collect_feature_pairs(label_data: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    xy_by_uid: dict[str, list[tuple[float, float]]] = {}
    for feature in label_data.get("features", {}).get("xy", []):
        uid = (feature.get("properties") or {}).get("uid")
        if not uid:
            continue
        xy_by_uid[uid] = _parse_wkt_pairs(feature.get("wkt", ""))

    pairs: list[tuple[float, float, float, float]] = []
    for feature in label_data.get("features", {}).get("lng_lat", []):
        uid = (feature.get("properties") or {}).get("uid")
        if not uid or uid not in xy_by_uid:
            continue
        xy_points = xy_by_uid[uid]
        lng_lat_points = _parse_wkt_pairs(feature.get("wkt", ""))
        for (x, y), (lon, lat) in zip(xy_points, lng_lat_points):
            pairs.append((x, y, lon, lat))
    return pairs


def _collect_lng_lat_points(label_data: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for feature in label_data.get("features", {}).get("lng_lat", []):
        points.extend(_parse_wkt_pairs(feature.get("wkt", "")))
    return points


def _score_post_label(label_data: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for feature in label_data.get("features", {}).get("xy", []):
        subtype = (feature.get("properties") or {}).get("subtype", "unknown")
        counts[subtype] += 1

    destroyed = counts.get("destroyed", 0)
    major = counts.get("major-damage", 0)
    minor = counts.get("minor-damage", 0)
    total = len(label_data.get("features", {}).get("xy", []))
    score = destroyed * 3 + major * 2 + minor
    metadata = label_data.get("metadata", {})

    return {
        "score": score,
        "destroyed": destroyed,
        "major": major,
        "minor": minor,
        "no_damage": counts.get("no-damage", 0),
        "unclassified": counts.get("un-classified", 0),
        "total": total,
        "disaster": metadata.get("disaster", ""),
        "disaster_type": metadata.get("disaster_type", ""),
        "width": int(metadata.get("width", 1024)),
        "height": int(metadata.get("height", 1024)),
    }


def _estimate_bbox(pre_label: dict[str, Any], post_label: dict[str, Any]) -> tuple[list[float], dict[str, float | str]]:
    pairs = _collect_feature_pairs(pre_label) + _collect_feature_pairs(post_label)
    metadata = post_label.get("metadata", {}) or pre_label.get("metadata", {})
    width = float(metadata.get("width", 1024))
    height = float(metadata.get("height", 1024))
    if len(pairs) >= 3:
        points = np.asarray(pairs, dtype=np.float64)
        design = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
        lon_coef, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        lat_coef, *_ = np.linalg.lstsq(design, points[:, 3], rcond=None)

        pred_lon = design @ lon_coef
        pred_lat = design @ lat_coef
        rmse_lon = float(np.sqrt(np.mean((pred_lon - points[:, 2]) ** 2)))
        rmse_lat = float(np.sqrt(np.mean((pred_lat - points[:, 3]) ** 2)))

        corners = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [width, 0.0, 1.0],
                [width, height, 1.0],
                [0.0, height, 1.0],
            ],
            dtype=np.float64,
        )
        corner_lons = corners @ lon_coef
        corner_lats = corners @ lat_coef
        bbox = [
            float(corner_lons.min()),
            float(corner_lats.min()),
            float(corner_lons.max()),
            float(corner_lats.max()),
        ]
        diagnostics = {
            "method": "affine-fit-from-xview2-label-pairs",
            "fit_rmse_lon": rmse_lon,
            "fit_rmse_lat": rmse_lat,
            "point_pairs": float(len(points)),
        }
        return bbox, diagnostics

    if len(pairs) >= 2:
        points = np.asarray(pairs, dtype=np.float64)
        x_design = np.c_[points[:, 0], np.ones(len(points))]
        y_design = np.c_[points[:, 1], np.ones(len(points))]
        lon_coef, *_ = np.linalg.lstsq(x_design, points[:, 2], rcond=None)
        lat_coef, *_ = np.linalg.lstsq(y_design, points[:, 3], rcond=None)

        pred_lon = x_design @ lon_coef
        pred_lat = y_design @ lat_coef
        rmse_lon = float(np.sqrt(np.mean((pred_lon - points[:, 2]) ** 2)))
        rmse_lat = float(np.sqrt(np.mean((pred_lat - points[:, 3]) ** 2)))

        corner_x = np.asarray([0.0, width], dtype=np.float64)
        corner_y = np.asarray([0.0, height], dtype=np.float64)
        corner_lons = corner_x * lon_coef[0] + lon_coef[1]
        corner_lats = corner_y * lat_coef[0] + lat_coef[1]
        bbox = [
            float(corner_lons.min()),
            float(corner_lats.min()),
            float(corner_lons.max()),
            float(corner_lats.max()),
        ]
        diagnostics = {
            "method": "axis-aligned-fit-from-xview2-label-pairs",
            "fit_rmse_lon": rmse_lon,
            "fit_rmse_lat": rmse_lat,
            "point_pairs": float(len(points)),
        }
        return bbox, diagnostics

    lng_lat_points = _collect_lng_lat_points(pre_label) + _collect_lng_lat_points(post_label)
    if len(lng_lat_points) >= 2:
        lon_values = [point[0] for point in lng_lat_points]
        lat_values = [point[1] for point in lng_lat_points]
        min_lon = min(lon_values)
        max_lon = max(lon_values)
        min_lat = min(lat_values)
        max_lat = max(lat_values)
        lon_pad = max((max_lon - min_lon) * 0.15, 1e-4)
        lat_pad = max((max_lat - min_lat) * 0.15, 1e-4)
        bbox = [min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad]
        diagnostics = {
            "method": "expanded-building-envelope-from-lng-lat-labels",
            "fit_rmse_lon": 0.0,
            "fit_rmse_lat": 0.0,
            "point_pairs": 0.0,
        }
        return bbox, diagnostics

    raise ValueError("Not enough label geometry to estimate bbox.")


def _default_route_points(bbox: list[float]) -> tuple[list[float], list[float]]:
    west, south, east, north = bbox
    lon_span = east - west
    lat_span = north - south
    start = [south + lat_span * 0.18, west + lon_span * 0.18]
    goal = [north - lat_span * 0.18, east - lon_span * 0.18]
    return start, goal


def _pair_key(stem: str) -> str:
    return re.sub(r"_(pre|post)_disaster$", "", stem)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _extract_member(tf: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member = tf.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Archive member could not be read: {member_name}")
    _write_bytes(destination, member.read())


def _activate_scene(scene_dir: Path, demo_dir: Path) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "pre_disaster.png").write_bytes((scene_dir / "pre_disaster.png").read_bytes())
    (demo_dir / "post_disaster.png").write_bytes((scene_dir / "post_disaster.png").read_bytes())
    (demo_dir / "metadata.json").write_text(
        (scene_dir / "metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    archive_path = args.archive.resolve()
    out_dir = args.out_dir.resolve()
    disasters = tuple(args.disasters or DEFAULT_DISASTERS)

    scene_assets: dict[str, dict[str, str]] = defaultdict(dict)
    post_label_payloads: dict[str, dict[str, Any]] = {}
    pre_label_payloads: dict[str, dict[str, Any]] = {}
    skipped_scenes: list[dict[str, str]] = []

    with tarfile.open(archive_path) as tf:
        members = [member for member in tf.getmembers() if member.isfile()]
        for member in members:
            name = member.name
            filename = Path(name).name
            stem = filename
            if filename.endswith(".png"):
                stem = filename[:-4]
            elif filename.endswith(".json"):
                stem = filename[:-5]

            if name.startswith("hold/images/"):
                scene_assets[_pair_key(stem)][f"image::{stem}"] = name
            elif name.startswith("hold/targets/"):
                scene_assets[_pair_key(stem.replace("_target", ""))][f"target::{stem}"] = name
            elif name.startswith("hold/labels/"):
                scene_assets[_pair_key(stem)][f"label::{stem}"] = name
                with tf.extractfile(member) as handle:
                    if handle is None:
                        continue
                    payload = json.load(handle)
                if stem.endswith("_post_disaster"):
                    post_label_payloads[_pair_key(stem)] = payload
                elif stem.endswith("_pre_disaster"):
                    pre_label_payloads[_pair_key(stem)] = payload

        ranked_by_disaster: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for scene_id, post_label in post_label_payloads.items():
            pre_label = pre_label_payloads.get(scene_id)
            assets = scene_assets.get(scene_id, {})
            required_keys = [
                f"image::{scene_id}_pre_disaster",
                f"image::{scene_id}_post_disaster",
                f"label::{scene_id}_pre_disaster",
                f"label::{scene_id}_post_disaster",
            ]
            if pre_label is None or not all(key in assets for key in required_keys):
                continue

            metrics = _score_post_label(post_label)
            if metrics["disaster"] not in disasters:
                continue

            try:
                bbox, diagnostics = _estimate_bbox(pre_label, post_label)
            except ValueError as exc:
                skipped_scenes.append(
                    {
                        "scene_id": scene_id,
                        "disaster": metrics["disaster"],
                        "reason": str(exc),
                    }
                )
                continue
            start, goal = _default_route_points(bbox)
            ranked_by_disaster[metrics["disaster"]].append(
                {
                    "scene_id": scene_id,
                    "bbox": bbox,
                    "start": start,
                    "goal": goal,
                    "diagnostics": diagnostics,
                    **metrics,
                }
            )

        manifest = {
            "source_archive": str(archive_path),
            "selection_limit_per_disaster": args.limit,
            "scenes": [],
            "skipped_scenes": skipped_scenes,
        }

        for disaster in disasters:
            ranked = sorted(
                ranked_by_disaster.get(disaster, []),
                key=lambda item: (item["score"], item["destroyed"], item["major"], item["minor"]),
                reverse=True,
            )[: args.limit]

            for scene in ranked:
                scene_id = scene["scene_id"]
                scene_dir = out_dir / disaster / scene_id
                scene_dir.mkdir(parents=True, exist_ok=True)

                files_to_copy = {
                    f"image::{scene_id}_pre_disaster": scene_dir / "pre_disaster.png",
                    f"image::{scene_id}_post_disaster": scene_dir / "post_disaster.png",
                    f"label::{scene_id}_pre_disaster": scene_dir / "pre_label.json",
                    f"label::{scene_id}_post_disaster": scene_dir / "post_label.json",
                }
                optional_targets = {
                    f"target::{scene_id}_pre_disaster_target": scene_dir / "pre_target.png",
                    f"target::{scene_id}_post_disaster_target": scene_dir / "post_target.png",
                }

                assets = scene_assets[scene_id]
                for asset_key, destination in files_to_copy.items():
                    _extract_member(tf, assets[asset_key], destination)
                for asset_key, destination in optional_targets.items():
                    if asset_key in assets:
                        _extract_member(tf, assets[asset_key], destination)

                metadata_payload = {
                    "scene_id": scene_id,
                    "disaster": disaster,
                    "bbox": scene["bbox"],
                    "start": scene["start"],
                    "goal": scene["goal"],
                    "crs": "EPSG:4326",
                    "bbox_estimation": scene["diagnostics"],
                }
                (scene_dir / "metadata.json").write_text(
                    json.dumps(metadata_payload, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )

                manifest["scenes"].append(
                    {
                        "scene_id": scene_id,
                        "disaster": disaster,
                        "score": scene["score"],
                        "destroyed": scene["destroyed"],
                        "major": scene["major"],
                        "minor": scene["minor"],
                        "no_damage": scene["no_damage"],
                        "unclassified": scene["unclassified"],
                        "total": scene["total"],
                        "bbox": scene["bbox"],
                        "start": scene["start"],
                        "goal": scene["goal"],
                        "relative_dir": str((Path(disaster) / scene_id).as_posix()),
                        "bbox_fit_rmse_lon": scene["diagnostics"]["fit_rmse_lon"],
                        "bbox_fit_rmse_lat": scene["diagnostics"]["fit_rmse_lat"],
                    }
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        if args.activate_scene:
            selected = [scene for scene in manifest["scenes"] if scene["scene_id"] == args.activate_scene]
            if not selected:
                raise ValueError(f"Scene not found in extracted library: {args.activate_scene}")
            scene_dir = out_dir / selected[0]["relative_dir"]
            _activate_scene(scene_dir, args.demo_dir.resolve())
            print(f"Activated demo scene: {args.activate_scene}")

    print(f"Prepared {len(manifest['scenes'])} scenes under {out_dir}")
    for disaster in disasters:
        disaster_count = sum(1 for scene in manifest["scenes"] if scene["disaster"] == disaster)
        print(f"  - {disaster}: {disaster_count}")
    if skipped_scenes:
        print(f"Skipped scenes without usable geospatial geometry: {len(skipped_scenes)}")


if __name__ == "__main__":
    main()
