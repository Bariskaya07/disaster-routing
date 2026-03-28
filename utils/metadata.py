from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _coerce_bbox(raw_bbox: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("Metadata must include bbox as [min_lon, min_lat, max_lon, max_lat].")
    west, south, east, north = (float(value) for value in raw_bbox)
    if not (west < east and south < north):
        raise ValueError("Invalid bbox ordering. Expected min_lon < max_lon and min_lat < max_lat.")
    return west, south, east, north


def parse_metadata(metadata: str | bytes | Path | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        parsed = dict(metadata)
    elif isinstance(metadata, Path):
        parsed = json.loads(metadata.read_text(encoding="utf-8"))
    elif isinstance(metadata, bytes):
        parsed = json.loads(metadata.decode("utf-8"))
    elif isinstance(metadata, str):
        text = metadata.strip()
        if not text:
            return {}
        candidate = Path(text)
        if candidate.exists():
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            parsed = json.loads(text)
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)!r}")

    if "bbox" in parsed:
        parsed["bbox"] = _coerce_bbox(parsed["bbox"])

    for key in ("start", "goal"):
        if key in parsed and parsed[key] is not None:
            point = parsed[key]
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Metadata field '{key}' must be [lat, lon].")
            parsed[key] = (float(point[0]), float(point[1]))

    parsed.setdefault("crs", "EPSG:4326")
    return parsed
