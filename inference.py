from __future__ import annotations

import argparse
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from models import dpn_seamese_unet_shared, dpn_unet
from utils.constants import (
    DAMAGE_CHECKPOINT_NAME,
    DEFAULT_DAMAGE_WEIGHTS_ENV,
    DEFAULT_LOCALIZATION_WEIGHTS_ENV,
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    LOCALIZATION_CHECKPOINT_NAME,
)
from utils.geospatial import build_georef
from utils.metadata import parse_metadata
from utils.types import InferenceArtifacts, RasterBundle
from utils.weights import resolve_weight_path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_rgb_array(image: np.ndarray | Image.Image | bytes | str | Path) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)
        raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _normalize_chw(image_rgb: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image, (2, 0, 1))
    mean_arr = np.asarray(mean[: chw.shape[0]], dtype=np.float32)[:, None, None]
    std_arr = np.asarray(std[: chw.shape[0]], dtype=np.float32)[:, None, None]
    return (chw - mean_arr) / std_arr


def _pad_to_stride(chw: np.ndarray, stride: int = 32) -> tuple[np.ndarray, tuple[slice, slice]]:
    _, height, width = chw.shape
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    pad_mode = "reflect"
    if height <= 1 or width <= 1 or top >= height or bottom >= height or left >= width or right >= width:
        pad_mode = "edge"

    padded = np.pad(chw, ((0, 0), (top, bottom), (left, right)), mode=pad_mode)
    crop_slices = (slice(top, top + height), slice(left, left + width))
    return padded, crop_slices


def _load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(checkpoint)!r}")
    return {key.replace("module.", ""): value for key, value in checkpoint.items()}


def _build_damage_raster(heavy_damage_mask: np.ndarray) -> np.ndarray:
    seed = (heavy_damage_mask > 0).astype(np.uint8) * 255
    if seed.max() == 0:
        return seed

    cleaned = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    inflated = cv2.dilate(cleaned, np.ones((5, 5), dtype=np.uint8), iterations=1)
    blurred = cv2.GaussianBlur(inflated, (0, 0), sigmaX=9, sigmaY=9)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def _build_fallback_outputs(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    metadata: dict[str, Any],
    warning: str,
) -> InferenceArtifacts:
    pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(pre_gray, post_gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    damage_mask = np.zeros_like(diff, dtype=np.uint8)
    damage_mask[diff > 50] = 1
    damage_mask[diff > 110] = 2
    damage_mask[diff > 180] = 3

    heavy_damage_mask = np.isin(damage_mask, (2, 3)).astype(np.uint8)
    building_mask = np.ones_like(diff, dtype=np.uint8)
    damage_raster = _build_damage_raster(heavy_damage_mask)

    georef = None
    bbox = metadata.get("bbox")
    if bbox:
        try:
            georef = build_georef(bbox, diff.shape[1], diff.shape[0])
        except Exception as exc:  # pragma: no cover - depends on optional GIS deps
            warning = f"{warning} Georeferencing fallback failed: {exc}"

    return InferenceArtifacts(
        pre_image=pre_image,
        post_image=post_image,
        building_mask=building_mask,
        damage_mask=damage_mask,
        heavy_damage_mask=heavy_damage_mask,
        damage_raster=RasterBundle(array=damage_raster, georef=georef),
        damage_probabilities=None,
        localization_probability=None,
        georef=georef,
        used_fallback=True,
        device="cpu-fallback",
        warnings=[warning],
    )


class InferenceEngine:
    def __init__(
        self,
        *,
        localization_weights: Path,
        damage_weights: Path,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or select_device()
        self.localization_weights = localization_weights
        self.damage_weights = damage_weights

        self.localization_model = dpn_unet(seg_classes=1, backbone_arch="dpn92")
        self.localization_model.load_state_dict(_load_checkpoint_state_dict(localization_weights), strict=True)
        self.localization_model.to(self.device)
        self.localization_model.eval()

        self.damage_model = dpn_seamese_unet_shared(seg_classes=5, backbone_arch="dpn92")
        self.damage_model.load_state_dict(_load_checkpoint_state_dict(damage_weights), strict=True)
        self.damage_model.to(self.device)
        self.damage_model.eval()

    def _predict_localization(self, image_rgb: np.ndarray) -> np.ndarray:
        chw = _normalize_chw(image_rgb, DEFAULT_NORMALIZE_MEAN[:3], DEFAULT_NORMALIZE_STD[:3])
        chw, crop_slices = _pad_to_stride(chw, stride=32)
        tensor = torch.from_numpy(chw[None, ...]).to(self.device).float()
        with torch.no_grad():
            logits = self.localization_model(tensor)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        row_slice, col_slice = crop_slices
        return probability[row_slice, col_slice].astype(np.float32)

    def _predict_damage(self, pre_image_rgb: np.ndarray, post_image_rgb: np.ndarray) -> np.ndarray:
        merged = np.concatenate([pre_image_rgb, post_image_rgb], axis=-1)
        chw = _normalize_chw(merged, DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD)
        chw, crop_slices = _pad_to_stride(chw, stride=32)
        tensor = torch.from_numpy(chw[None, ...]).to(self.device).float()
        with torch.no_grad():
            logits = self.damage_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        row_slice, col_slice = crop_slices
        return probabilities[:, row_slice, col_slice].astype(np.float32)

    def run(self, pre_image: np.ndarray, post_image: np.ndarray, metadata: dict[str, Any] | None = None) -> InferenceArtifacts:
        warnings: list[str] = []
        metadata = metadata or {}

        if pre_image.shape[:2] != post_image.shape[:2]:
            post_image = cv2.resize(post_image, (pre_image.shape[1], pre_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            warnings.append("Afet sonrası görüntü, afet öncesi görüntü boyutuna uyarlandı.")

        georef = None
        if metadata.get("bbox"):
            try:
                georef = build_georef(metadata["bbox"], pre_image.shape[1], pre_image.shape[0])
            except Exception as exc:  # pragma: no cover - depends on optional GIS deps
                warnings.append(f"Coğrafi hizalama kullanılamıyor: {exc}")

        localization_probability = self._predict_localization(pre_image)
        building_mask = (localization_probability > 0.25).astype(np.uint8)

        damage_probabilities = self._predict_damage(pre_image, post_image)
        damage_argmax = np.argmax(damage_probabilities, axis=0).astype(np.uint8)
        damage_mask = np.where(damage_argmax <= 1, 0, damage_argmax - 1).astype(np.uint8)
        damage_mask = np.where(building_mask > 0, damage_mask, 0).astype(np.uint8)
        heavy_damage_mask = np.isin(damage_mask, (2, 3)).astype(np.uint8)
        damage_raster = _build_damage_raster(heavy_damage_mask)

        return InferenceArtifacts(
            pre_image=pre_image,
            post_image=post_image,
            building_mask=building_mask,
            damage_mask=damage_mask,
            heavy_damage_mask=heavy_damage_mask,
            damage_raster=RasterBundle(array=damage_raster, georef=georef),
            damage_probabilities=damage_probabilities,
            localization_probability=localization_probability,
            georef=georef,
            used_fallback=False,
            device=str(self.device),
            warnings=warnings,
            localization_weights=self.localization_weights,
            damage_weights=self.damage_weights,
        )


@lru_cache(maxsize=4)
def _get_cached_engine(
    localization_path: str | None,
    damage_path: str | None,
    hf_model_repo_id: str | None,
    hf_revision: str | None,
    hf_token: str | None,
) -> InferenceEngine:
    localization_weights = resolve_weight_path(
        LOCALIZATION_CHECKPOINT_NAME,
        explicit_path=localization_path,
        hf_model_repo_id=hf_model_repo_id,
        hf_revision=hf_revision,
        hf_token=hf_token,
    )
    damage_weights = resolve_weight_path(
        DAMAGE_CHECKPOINT_NAME,
        explicit_path=damage_path,
        hf_model_repo_id=hf_model_repo_id,
        hf_revision=hf_revision,
        hf_token=hf_token,
    )
    return InferenceEngine(
        localization_weights=localization_weights,
        damage_weights=damage_weights,
        device=select_device(),
    )


def run_inference(
    pre_image: np.ndarray | Image.Image | str | Path,
    post_image: np.ndarray | Image.Image | str | Path,
    metadata: dict[str, Any] | str | Path | None = None,
    *,
    localization_weights_path: str | None = None,
    damage_weights_path: str | None = None,
    hf_model_repo_id: str | None = None,
    hf_revision: str | None = None,
    hf_token: str | None = None,
) -> InferenceArtifacts:
    metadata_dict = parse_metadata(metadata)
    pre_array = _ensure_rgb_array(pre_image)
    post_array = _ensure_rgb_array(post_image)
    localization_weights_path = localization_weights_path or os.getenv(DEFAULT_LOCALIZATION_WEIGHTS_ENV)
    damage_weights_path = damage_weights_path or os.getenv(DEFAULT_DAMAGE_WEIGHTS_ENV)

    try:
        engine = _get_cached_engine(
            localization_weights_path,
            damage_weights_path,
            hf_model_repo_id,
            hf_revision,
            hf_token,
        )
        return engine.run(pre_array, post_array, metadata_dict)
    except Exception as exc:  # pragma: no cover - fallback path depends on runtime deployment state
        return _build_fallback_outputs(
            pre_array,
            post_array,
            metadata_dict,
            warning=f"Model çıkarımı başarısız olduğu için fark-temelli yedek moda geçildi: {exc}",
        )


def _write_outputs(artifacts: InferenceArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "building_mask.png"), artifacts.building_mask * 255)
    cv2.imwrite(str(output_dir / "damage_mask.png"), artifacts.damage_mask)
    cv2.imwrite(str(output_dir / "damage_raster.png"), artifacts.damage_raster.array)
    summary = {
        "used_fallback": artifacts.used_fallback,
        "device": artifacts.device,
        "warnings": artifacts.warnings,
        "bbox": artifacts.georef.bbox if artifacts.georef else None,
        "localization_weights": str(artifacts.localization_weights) if artifacts.localization_weights else None,
        "damage_weights": str(artifacts.damage_weights) if artifacts.damage_weights else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TUA disaster inference on pre/post satellite imagery.")
    parser.add_argument("--pre", required=True, help="Path to the pre-disaster RGB image.")
    parser.add_argument("--post", required=True, help="Path to the post-disaster RGB image.")
    parser.add_argument("--metadata", help="Path to metadata JSON or an inline JSON string.")
    parser.add_argument("--output-dir", default="outputs/inference", help="Directory for serialized inference outputs.")
    parser.add_argument(
        "--localization-weights",
        default=None,
        help=f"Optional override for {DEFAULT_LOCALIZATION_WEIGHTS_ENV}.",
    )
    parser.add_argument(
        "--damage-weights",
        default=None,
        help=f"Optional override for {DEFAULT_DAMAGE_WEIGHTS_ENV}.",
    )
    args = parser.parse_args()

    artifacts = run_inference(
        args.pre,
        args.post,
        args.metadata,
        localization_weights_path=args.localization_weights,
        damage_weights_path=args.damage_weights,
    )
    _write_outputs(artifacts, Path(args.output_dir))


if __name__ == "__main__":
    main()
