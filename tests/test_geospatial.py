from __future__ import annotations

import importlib.util
import unittest


RASTERIO_AVAILABLE = importlib.util.find_spec("rasterio") is not None

if RASTERIO_AVAILABLE:
    from utils.geospatial import build_georef, latlon_to_pixel, pixel_to_latlon


@unittest.skipUnless(RASTERIO_AVAILABLE, "rasterio is not installed in the current environment.")
class GeospatialTests(unittest.TestCase):
    def test_affine_roundtrip_stays_inside_scene(self) -> None:
        georef = build_georef((32.8, 39.85, 32.81, 39.86), 100, 100)
        row, col = latlon_to_pixel(georef, 39.855, 32.805)
        lat, lon = pixel_to_latlon(georef, row, col)
        self.assertTrue(39.85 <= lat <= 39.86)
        self.assertTrue(32.8 <= lon <= 32.81)


if __name__ == "__main__":
    unittest.main()
