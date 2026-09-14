from __future__ import annotations

import copy
import importlib.util
import json
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
from unitree_lerobot.utils.camera_calibration import calibration_fingerprint


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
    cv2.IMREAD_UNCHANGED = -1
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
            mock.patch.object(
                self.converter,
                "JsonDataset",
                return_value=types.SimpleNamespace(
                    surface_normal_intrinsics=(
                        self.converter.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480
                    ),
                    camera_calibration=None,
                    camera_calibration_identity=None,
                ),
            ),
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
        self.assertNotIn("camera_calibration", contract)
        self.assertNotIn("camera_calibration", dataset.meta.info)
        write_info.assert_called_once_with(dataset.meta.info, dataset.meta.root)
        dataset.finalize.assert_called_once_with()

    def test_recorded_calibration_is_homogeneous_and_drives_surface_normal_k(self):
        calibration = self.converter.NAMED_CAMERA_CALIBRATIONS["d435i-254322071415"]
        with tempfile.TemporaryDirectory() as root:
            for index in range(2):
                episode = Path(root) / f"episode_{index:04d}"
                episode.mkdir()
                (episode / "data.json").write_text(
                    json.dumps({"info": {"depth": {"calibration": calibration}}, "data": []}),
                    encoding="utf-8",
                )

            dataset = self.converter.JsonDataset(
                Path(root),
                "Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
            )

        self.assertEqual(
            dataset.surface_normal_intrinsics.fx,
            calibration["color"]["fx"],
        )
        self.assertEqual(
            dataset.camera_calibration_identity.source,
            "episode.info.depth.calibration",
        )
        self.assertEqual(
            dataset.camera_calibration_identity.fingerprint,
            calibration["fingerprint"],
        )

    def test_recorded_calibration_rejects_conflicting_anonymous_intrinsics(self):
        calibration = self.converter.NAMED_CAMERA_CALIBRATIONS["d435i-254322071415"]
        recorded_intrinsics = self.converter._intrinsics_from_calibration(calibration)
        conflicting_intrinsics = self.converter.PinholeIntrinsics(
            width=640,
            height=480,
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
        )
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0000"
            episode.mkdir()
            (episode / "data.json").write_text(
                json.dumps({"info": {"depth": {"calibration": calibration}}, "data": []}),
                encoding="utf-8",
            )

            exact = self.converter.JsonDataset(
                Path(root),
                "Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
                surface_normal_intrinsics=recorded_intrinsics,
            )
            self.assertEqual(exact.surface_normal_intrinsics, recorded_intrinsics)

            with self.assertRaisesRegex(
                ValueError,
                "Recorded camera calibration conflicts.*anonymous",
            ):
                self.converter.JsonDataset(
                    Path(root),
                    "Unitree_G1_Dex3_HeadOnly",
                    include_surface_normals=True,
                    surface_normal_intrinsics=conflicting_intrinsics,
                )

    def test_normals_contract_embeds_recorded_calibration_identity_and_provenance(self):
        calibration = self.converter.NAMED_CAMERA_CALIBRATIONS["d435i-254322071415"]
        identity = self.converter.calibration_identity(
            calibration,
            source="episode.info.depth.calibration",
        )
        color = calibration["color"]
        intrinsics = self.converter.PinholeIntrinsics(
            width=color["width"],
            height=color["height"],
            fx=color["fx"],
            fy=color["fy"],
            cx=color["cx"],
            cy=color["cy"],
        )
        metadata = types.SimpleNamespace(info={"features": {}}, root=Path("/tmp/fake-dataset"))
        dataset = types.SimpleNamespace(
            meta=metadata,
            finalize=mock.Mock(),
            push_to_hub=mock.Mock(),
        )
        json_dataset = types.SimpleNamespace(
            surface_normal_intrinsics=intrinsics,
            camera_calibration=calibration,
            camera_calibration_identity=identity,
        )

        with (
            mock.patch.object(self.converter, "JsonDataset", return_value=json_dataset),
            mock.patch.object(self.converter, "create_empty_dataset", return_value=dataset),
            mock.patch.object(self.converter, "populate_dataset", return_value=dataset),
            mock.patch.object(self.converter, "write_info"),
        ):
            self.converter.json_to_lerobot(
                raw_dir=Path("/tmp/raw"),
                repo_id="owner/dataset",
                robot_type="Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
            )

        contract = dataset.meta.info["surface_normals_encoding"]
        self.assertEqual(contract["camera_calibration"], identity.to_metadata())
        self.assertEqual(contract["intrinsics"]["fx"], color["fx"])
        self.assertEqual(dataset.meta.info["camera_calibration"], calibration)
        self.assertEqual(
            dataset.meta.info["camera_calibration"]["fingerprint"],
            contract["camera_calibration"]["fingerprint"],
        )

    def test_legacy_raw_remains_old_default_and_untagged(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0000"
            episode.mkdir()
            (episode / "data.json").write_text(
                json.dumps({"info": {"depth": {}}, "data": []}),
                encoding="utf-8",
            )

            dataset = self.converter.JsonDataset(
                Path(root),
                "Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
            )

        self.assertEqual(
            dataset.surface_normal_intrinsics,
            self.converter.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        )
        self.assertIsNone(dataset.camera_calibration_identity)

    def test_named_profile_tags_legacy_raw_with_replacement_camera(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0000"
            episode.mkdir()
            (episode / "data.json").write_text(
                json.dumps({"info": {"depth": {}}, "data": []}),
                encoding="utf-8",
            )

            dataset = self.converter.JsonDataset(
                Path(root),
                "Unitree_G1_Dex3_HeadOnly",
                include_surface_normals=True,
                camera_calibration_profile="d435i-254322071415",
            )

        self.assertEqual(dataset.surface_normal_intrinsics.fx, 609.3858642578125)
        self.assertEqual(
            dataset.camera_calibration_identity.source,
            "converter.profile.d435i-254322071415",
        )

    def test_converter_rejects_mixed_or_different_recorded_calibrations(self):
        calibration = self.converter.NAMED_CAMERA_CALIBRATIONS["d435i-254322071415"]
        paths = [Path("episode_0000"), Path("episode_0001")]
        tagged = {"info": {"depth": {"calibration": calibration}}}
        legacy = {"info": {"depth": {}}}
        with self.assertRaisesRegex(ValueError, "mixes calibration-tagged and legacy"):
            self.converter._resolve_recorded_camera_calibration([tagged, legacy], paths)

        different = copy.deepcopy(calibration)
        different["camera"]["serial"] = "other"
        fingerprint_payload = {
            key: value for key, value in different.items() if key != "fingerprint"
        }
        different["fingerprint"] = calibration_fingerprint(fingerprint_payload)
        with self.assertRaisesRegex(ValueError, "heterogeneous camera calibrations"):
            self.converter._resolve_recorded_camera_calibration(
                [tagged, {"info": {"depth": {"calibration": different}}}],
                paths,
            )

    def test_explicit_profile_rejects_conflicting_recorded_calibration(self):
        calibration = copy.deepcopy(
            self.converter.NAMED_CAMERA_CALIBRATIONS["d435i-254322071415"]
        )
        calibration["camera"]["serial"] = "other"
        fingerprint_payload = {
            key: value for key, value in calibration.items() if key != "fingerprint"
        }
        calibration["fingerprint"] = calibration_fingerprint(fingerprint_payload)
        with tempfile.TemporaryDirectory() as root:
            episode = Path(root) / "episode_0000"
            episode.mkdir()
            (episode / "data.json").write_text(
                json.dumps({"info": {"depth": {"calibration": calibration}}, "data": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conflicts with recorded calibration"):
                self.converter.JsonDataset(
                    Path(root),
                    "Unitree_G1_Dex3_HeadOnly",
                    include_surface_normals=True,
                    camera_calibration_profile="d435i-254322071415",
                )

    def test_depth_read_retries_transient_failure_without_real_sleep(self):
        dataset = object.__new__(self.converter.JsonDataset)
        dataset.camera_to_image_key = {
            self.converter.DEPTH_COLOR_SOURCE_KEY: "observation.images.ego_view"
        }
        dataset.depth_near_m = 0.25
        dataset.depth_far_m = 1.0
        expected_depth = np.array([[250, 500], [750, 1_000]], dtype=np.uint16)
        episode_data = {
            "info": {"depth": {"scale_m_per_unit": 0.001}},
            "data": [{"idx": 0, "depths": {self.converter.DEPTH_SOURCE_KEY: "depth.png"}}],
        }

        with (
            mock.patch.object(
                self.converter.cv2,
                "imread",
                side_effect=[None, None, expected_depth],
                create=True,
            ) as imread,
            mock.patch.object(self.converter.time, "sleep") as sleep,
        ):
            images = dataset._parse_depth_derived_images(
                "/raw/episode_0000",
                episode_data,
                include_depth=True,
                include_surface_normals=False,
            )

        self.assertEqual(imread.call_count, 3)
        imread.assert_called_with("/raw/episode_0000/depth.png", -1)
        self.assertEqual(sleep.call_args_list, [mock.call(0.05), mock.call(0.05)])
        self.assertEqual(len(images[self.converter.DEPTH_OUTPUT_KEY]), 1)

    def test_converter_validates_sdk_scale_but_encodes_with_exact_canonical_constant(self):
        dataset = object.__new__(self.converter.JsonDataset)
        dataset.camera_to_image_key = {
            self.converter.DEPTH_COLOR_SOURCE_KEY: "observation.images.ego_view"
        }
        dataset.depth_near_m = 0.25
        dataset.depth_far_m = 1.0
        depth = np.array([[250, 625], [1000, 0]], dtype=np.uint16)
        episode_data = {
            "info": {
                "depth": {
                    "scale_m_per_unit": 0.001,
                    "scale_reported_m_per_unit": 0.0010000000474974513,
                }
            },
            "data": [{"idx": 0, "depths": {self.converter.DEPTH_SOURCE_KEY: "depth.png"}}],
        }
        encoded = np.zeros((2, 2, 3), dtype=np.uint8)

        with (
            mock.patch.object(self.converter.cv2, "imread", return_value=depth, create=True),
            mock.patch.object(
                self.converter,
                "encode_depth_gray_rgb",
                return_value=encoded,
            ) as encoder,
        ):
            images = dataset._parse_depth_derived_images(
                "/raw/episode_0000",
                episode_data,
                include_depth=True,
                include_surface_normals=False,
            )

        encoder.assert_called_once_with(
            depth,
            scale_m_per_unit=0.001,
            near_m=0.25,
            far_m=1.0,
        )
        self.assertIs(images[self.converter.DEPTH_OUTPUT_KEY][0], encoded)

    def test_converter_rejects_noncanonical_recorded_scale(self):
        with self.assertRaisesRegex(ValueError, "canonical 0.001"):
            self.converter._resolve_canonical_episode_depth_scale(
                {"scale_m_per_unit": 0.0005},
                episode_path="episode_0000",
            )

    def test_depth_read_raises_after_three_failures_without_real_sleep(self):
        dataset = object.__new__(self.converter.JsonDataset)
        dataset.camera_to_image_key = {
            self.converter.DEPTH_COLOR_SOURCE_KEY: "observation.images.ego_view"
        }
        episode_data = {
            "info": {"depth": {"scale_m_per_unit": 0.001}},
            "data": [{"idx": 7, "depths": {self.converter.DEPTH_SOURCE_KEY: "depth.png"}}],
        }

        with (
            mock.patch.object(
                self.converter.cv2,
                "imread",
                return_value=None,
                create=True,
            ) as imread,
            mock.patch.object(self.converter.time, "sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                r"Failed to read depth image: /raw/episode_0000/depth\.png",
            ),
        ):
            dataset._parse_depth_derived_images(
                "/raw/episode_0000",
                episode_data,
                include_depth=True,
                include_surface_normals=False,
            )

        self.assertEqual(imread.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(0.05), mock.call(0.05)])


if __name__ == "__main__":
    unittest.main()
