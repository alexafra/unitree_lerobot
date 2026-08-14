from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np
from PIL import Image
from lerobot.datasets.video_utils import get_video_pixel_channels


CONVERTER_PATH = Path(__file__).parents[1] / "unitree_lerobot" / "utils" / "convert_unitree_json_to_lerobot.py"


class _FakeLeRobotDataset:
    @classmethod
    def create(cls, **_kwargs):
        return cls

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        self.base_calls.append((video_key, episode_index))
        return Path("base-video.mp4")


def _load_converter_with_lerobot_stubs():
    lerobot = types.ModuleType("lerobot")
    lerobot_utils = types.ModuleType("lerobot.utils")
    lerobot_constants = types.ModuleType("lerobot.utils.constants")
    lerobot_constants.HF_LEROBOT_HOME = Path("/tmp/fake-lerobot-home")
    lerobot_datasets = types.ModuleType("lerobot.datasets")
    lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")
    lerobot_dataset.LeRobotDataset = _FakeLeRobotDataset
    dataset_utils = types.ModuleType("lerobot.datasets.utils")
    dataset_utils.write_info = mock.Mock()
    video_utils = types.ModuleType("lerobot.datasets.video_utils")
    video_utils.encode_video_frames = mock.Mock()
    cv2 = types.ModuleType("cv2")
    tyro = types.ModuleType("tyro")

    stubs = {
        "lerobot": lerobot,
        "lerobot.utils": lerobot_utils,
        "lerobot.utils.constants": lerobot_constants,
        "lerobot.datasets": lerobot_datasets,
        "lerobot.datasets.lerobot_dataset": lerobot_dataset,
        "lerobot.datasets.utils": dataset_utils,
        "lerobot.datasets.video_utils": video_utils,
        "cv2": cv2,
        "tyro": tyro,
    }
    module_name = "_surface_normal_converter_under_test"
    spec = importlib.util.spec_from_file_location(module_name, CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)

    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class SurfaceNormalConverterVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = _load_converter_with_lerobot_stubs()

    def test_geometry_views_use_lossless_rgb_h264_without_changing_ego(self):
        dataset = object.__new__(self.converter.GeometryVideoLeRobotDataset)
        dataset.fps = 30
        dataset.base_calls = []

        with tempfile.TemporaryDirectory() as root:
            dataset.root = Path(root)
            image_dir = Path(root) / "frames"
            image_dir.mkdir()
            dataset._get_image_file_dir = mock.Mock(return_value=image_dir)

            with (
                mock.patch.object(
                    self.converter,
                    "encode_lossless_geometry_video",
                ) as encoder,
                mock.patch.object(self.converter.shutil, "rmtree") as remove_tree,
            ):
                normals_output = dataset._encode_temporary_episode_video(
                    "observation.images.surface_normals_view",
                    3,
                )
                depth_output = dataset._encode_temporary_episode_video(
                    "observation.images.depth_gray_view",
                    4,
                )

            self.assertEqual(encoder.call_count, 2)
            encoder.assert_has_calls(
                [
                    mock.call(image_dir, normals_output, 30),
                    mock.call(image_dir, depth_output, 30),
                ]
            )
            self.assertEqual(remove_tree.call_count, 2)
            remove_tree.assert_has_calls([mock.call(image_dir), mock.call(image_dir)])
            self.assertEqual(
                normals_output.name,
                "observation.images.surface_normals_view_003.mp4",
            )
            self.assertEqual(
                depth_output.name,
                "observation.images.depth_gray_view_004.mp4",
            )

        result = dataset._encode_temporary_episode_video("observation.images.ego_view", 5)
        self.assertEqual(result, Path("base-video.mp4"))
        self.assertEqual(dataset.base_calls, [("observation.images.ego_view", 5)])

    def test_lossless_rgb_h264_round_trip_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as root:
            image_dir = Path(root) / "frames"
            image_dir.mkdir()
            expected = []
            for frame_index in range(3):
                rng = np.random.default_rng(frame_index)
                image = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
                image[0, :, :] = 0
                expected.append(image)
                Image.fromarray(image, mode="RGB").save(
                    image_dir / f"frame-{frame_index:06d}.png"
                )

            output = Path(root) / "geometry.mp4"
            self.converter.encode_lossless_geometry_video(image_dir, output, 30)

            with av.open(str(output)) as container:
                decoded = [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]

            self.assertEqual(len(decoded), len(expected))
            for actual, wanted in zip(decoded, expected, strict=True):
                np.testing.assert_array_equal(actual, wanted)

    def test_lerobot_recognizes_planar_gbr_as_three_channels(self):
        self.assertEqual(get_video_pixel_channels("gbrp"), 3)

    def test_geometry_dataset_is_selected_for_depth_or_normals(self):
        common = {
            "repo_id": "owner/dataset",
            "robot_type": "Unitree_G1_Dex3_HeadOnly",
            "mode": "video",
        }

        depth_dataset = self.converter.create_empty_dataset(
            **common,
            include_depth=True,
        )
        normals_dataset = self.converter.create_empty_dataset(
            **common,
            include_surface_normals=True,
        )
        rgb_dataset = self.converter.create_empty_dataset(**common)

        self.assertIs(depth_dataset, self.converter.GeometryVideoLeRobotDataset)
        self.assertIs(normals_dataset, self.converter.GeometryVideoLeRobotDataset)
        self.assertIs(rgb_dataset, _FakeLeRobotDataset)

    def test_normals_contract_is_explicitly_written_before_finalize(self):
        metadata = types.SimpleNamespace(info={"features": {}}, root=Path("/tmp/fake-dataset"))
        dataset = types.SimpleNamespace(
            meta=metadata,
            finalize=mock.Mock(),
            push_to_hub=mock.Mock(),
        )

        with (
            mock.patch.object(self.converter, "create_empty_dataset", return_value=dataset),
            mock.patch.object(self.converter, "populate_dataset", return_value=dataset),
            mock.patch.object(self.converter, "write_info") as write_info,
        ):
            self.converter.json_to_lerobot(
                raw_dir=Path("/tmp/raw"),
                repo_id="owner/dataset",
                robot_type="Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
            )

        contract = dataset.meta.info["surface_normals_encoding"]
        self.assertEqual(contract["feature_key"], "observation.images.surface_normals_view")
        self.assertEqual(contract["encoding"], "camera_xyz_uint8")
        write_info.assert_called_once_with(dataset.meta.info, dataset.meta.root)
        dataset.finalize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
