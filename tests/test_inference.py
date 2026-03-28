from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from inference import InferenceEngine, _pad_to_stride, run_inference, select_device
from utils.constants import DAMAGE_CHECKPOINT_NAME, LOCALIZATION_CHECKPOINT_NAME, PROJECT_ROOT


class DeviceSelectionTests(unittest.TestCase):
    def test_prefers_cuda_when_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            self.assertEqual(str(select_device()), "cuda")

    def test_uses_cpu_when_no_accelerator_is_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            self.assertEqual(str(select_device()), "cpu")


class InferenceSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        (PROJECT_ROOT / LOCALIZATION_CHECKPOINT_NAME).exists() and (PROJECT_ROOT / DAMAGE_CHECKPOINT_NAME).exists(),
        "Local checkpoints are not present.",
    )
    def test_run_inference_uses_real_models_without_bbox(self) -> None:
        pre = np.zeros((64, 64, 3), dtype=np.uint8)
        post = np.zeros((64, 64, 3), dtype=np.uint8)
        artifacts = run_inference(pre, post, {})
        self.assertFalse(artifacts.used_fallback)
        self.assertEqual(artifacts.building_mask.shape, (64, 64))
        self.assertEqual(artifacts.damage_mask.shape, (64, 64))
        self.assertEqual(artifacts.damage_raster.array.shape, (64, 64))

    def test_predict_damage_pads_and_crops_back_to_original_shape(self) -> None:
        class DummyDamageModel:
            def __init__(self) -> None:
                self.last_input_shape = None

            def __call__(self, tensor):
                self.last_input_shape = tuple(tensor.shape)
                _, _, height, width = tensor.shape
                output = torch.zeros((1, 5, height, width), dtype=torch.float32)
                output[:, 4, :, :] = 1.0
                return output

        engine = object.__new__(InferenceEngine)
        engine.device = torch.device("cpu")
        engine.damage_model = DummyDamageModel()

        pre = np.zeros((33, 35, 3), dtype=np.uint8)
        post = np.zeros((33, 35, 3), dtype=np.uint8)
        output = InferenceEngine._predict_damage(engine, pre, post)

        self.assertEqual(engine.damage_model.last_input_shape, (1, 6, 64, 64))
        self.assertEqual(output.shape, (5, 33, 35))

    def test_pad_to_stride_returns_original_crop_window(self) -> None:
        chw = np.zeros((6, 33, 35), dtype=np.float32)
        padded, (row_slice, col_slice) = _pad_to_stride(chw, stride=32)
        self.assertEqual(padded.shape, (6, 64, 64))
        self.assertEqual(padded[:, row_slice, col_slice].shape, chw.shape)

    def test_run_inference_reads_weight_paths_from_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "LOCALIZATION_WEIGHTS_PATH": "/tmp/localization.ckpt",
                "DAMAGE_WEIGHTS_PATH": "/tmp/damage.ckpt",
            },
            clear=False,
        ):
            with mock.patch("inference._get_cached_engine") as get_engine:
                fake_engine = mock.Mock()
                fake_engine.run.return_value = mock.Mock()
                get_engine.return_value = fake_engine
                run_inference(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8), {})
                args = get_engine.call_args[0]
                self.assertEqual(args[0], "/tmp/localization.ckpt")
                self.assertEqual(args[1], "/tmp/damage.ckpt")


if __name__ == "__main__":
    unittest.main()
