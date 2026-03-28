from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate one prepared demo scene into data/demo.")
    parser.add_argument("scene_dir", type=Path, help="Scene directory under data/demo_library/<disaster>/<scene_id>.")
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/demo"),
        help="Destination directory used by the Streamlit demo button.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    demo_dir = args.demo_dir.resolve()

    required = ("pre_disaster.png", "post_disaster.png", "metadata.json")
    missing = [name for name in required if not (scene_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Scene directory is missing required files: {', '.join(missing)}")

    demo_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(scene_dir / name, demo_dir / name)

    print(f"Activated {scene_dir.name} into {demo_dir}")


if __name__ == "__main__":
    main()
