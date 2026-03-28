from __future__ import annotations

import argparse
import json
from pathlib import Path

from router import load_graph, recommend_demo_route_points
from utils.metadata import parse_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite demo metadata start/goal points using real OSM road nodes."
    )
    parser.add_argument(
        "--demo-library",
        type=Path,
        default=Path("data/demo_library"),
        help="Root directory containing prepared demo scenes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print proposed updates without writing metadata files.",
    )
    return parser.parse_args()


def _scene_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*") if path.is_dir())


def main() -> None:
    args = _parse_args()
    root = args.demo_library.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Demo library not found: {root}")

    updated_count = 0
    skipped_count = 0

    for scene_dir in _scene_dirs(root):
        metadata_path = scene_dir / "metadata.json"
        if not metadata_path.exists():
            skipped_count += 1
            print(f"SKIP {scene_dir.name}: metadata.json missing")
            continue

        metadata = parse_metadata(metadata_path)
        bbox = metadata.get("bbox")
        if bbox is None:
            skipped_count += 1
            print(f"SKIP {scene_dir.name}: bbox missing")
            continue

        try:
            graph_bundle = load_graph(tuple(bbox))
            points = recommend_demo_route_points(graph_bundle.graph_proj, graph_bundle.graph_latlon)
        except Exception as exc:
            skipped_count += 1
            print(f"SKIP {scene_dir.name}: {exc}")
            continue

        if points is None:
            skipped_count += 1
            print(f"SKIP {scene_dir.name}: no route points found")
            continue

        start, goal = points
        metadata["start"] = [float(start[0]), float(start[1])]
        metadata["goal"] = [float(goal[0]), float(goal[1])]
        metadata["route_points_source"] = "real-road-nodes"

        if not args.dry_run:
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        updated_count += 1
        print(
            f"{'PLAN' if args.dry_run else 'OK  '} {scene_dir.name}: "
            f"start={metadata['start']} goal={metadata['goal']}"
        )

    print(f"Updated: {updated_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
