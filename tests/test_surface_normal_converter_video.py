from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


CONVERTER_PATH = Path(__file__).parents[1] / "unitree_lerobot" / "utils" / "convert_unitree_json_to_lerobot.py"


class _FakeLeRobotDataset:
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

    def test_normals_use_near_lossless_h264_444_without_changing_other_views(self):
        dataset = object.__new__(self.converter.SurfaceNormalsLeRobotDataset)
        dataset.fps = 30
        dataset.base_calls = []

        with tempfile.TemporaryDirectory() as root:
            dataset.root = Path(root)
            image_dir = Path(root) / "frames"
            image_dir.mkdir()
            dataset._get_image_file_dir = mock.Mock(return_value=image_dir)

            with (
                mock.patch.object(self.converter, "encode_video_frames") as encoder,
                mock.patch.object(self.converter.shutil, "rmtree") as remove_tree,
            ):
                output = dataset._encode_temporary_episode_video(
                    "observation.images.surface_normals_view",
                    3,
                )

            encoder.assert_called_once_with(
                image_dir,
                output,
                30,
                vcodec="h264",
                pix_fmt="yuv444p",
                crf=0,
                overwrite=True,
            )
            remove_tree.assert_called_once_with(image_dir)
            self.assertEqual(output.name, "observation.images.surface_normals_view_003.mp4")

        result = dataset._encode_temporary_episode_video("observation.images.ego_view", 4)
        self.assertEqual(result, Path("base-video.mp4"))
        self.assertEqual(dataset.base_calls, [("observation.images.ego_view", 4)])

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
