from __future__ import annotations

import numpy as np


def colorize_mask(mask: np.ndarray, building_mask: np.ndarray | None = None) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)

    if building_mask is not None:
        intact = (building_mask > 0) & (mask == 0)
        colored[intact] = np.array([34, 197, 94], dtype=np.uint8)

    colored[mask == 1] = np.array([250, 204, 21], dtype=np.uint8)
    colored[mask == 2] = np.array([249, 115, 22], dtype=np.uint8)
    colored[mask == 3] = np.array([220, 38, 38], dtype=np.uint8)
    return colored


def colorize_damage_presence(mask: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    damaged = mask > 0
    rgba[damaged, 0] = 139
    rgba[damaged, 1] = 92
    rgba[damaged, 2] = 246
    rgba[damaged, 3] = int(255 * alpha)
    return rgba


def overlay_mask(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32)
    overlay = mask_rgb.astype(np.float32)
    blended = base * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
