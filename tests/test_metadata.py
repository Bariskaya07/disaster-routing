from __future__ import annotations

import unittest

from utils.metadata import parse_metadata


class MetadataTests(unittest.TestCase):
    def test_parse_metadata_normalizes_bbox_and_points(self) -> None:
        parsed = parse_metadata(
            {
                "bbox": [32.8, 39.85, 32.81, 39.86],
                "start": [39.851, 32.801],
                "goal": [39.859, 32.809],
            }
        )
        self.assertEqual(parsed["bbox"], (32.8, 39.85, 32.81, 39.86))
        self.assertEqual(parsed["start"], (39.851, 32.801))
        self.assertEqual(parsed["goal"], (39.859, 32.809))
        self.assertEqual(parsed["crs"], "EPSG:4326")

    def test_parse_metadata_rejects_invalid_bbox(self) -> None:
        with self.assertRaises(ValueError):
            parse_metadata({"bbox": [32.81, 39.86, 32.8, 39.85]})


if __name__ == "__main__":
    unittest.main()
