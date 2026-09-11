from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import chunk_delta_summary
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    DEPTH_ONLY_VIDEO_KEYS,
    SURFACE_NORMAL_ONLY_VIDEO_KEYS,
    TASKS,
    ActionChunk,
    make_observation,
    validate_model_contract,
)


def _modality_config(video_keys: tuple[str, ...]) -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": list(video_keys)},
        "state": {"delta_indices": [0], "modality_keys": list(ACTION_KEYS)},
        "action": {
            "delta_indices": list(range(32)),
            "modality_keys": list(ACTION_KEYS),
            "action_configs": [
                {
                    "rep": "RELATIVE" if "arm" in key else "ABSOLUTE",
                    "type": "NON_EEF",
                    "format": "DEFAULT",
                    "state_key": None,
                }
                for key in ACTION_KEYS
            ],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }


class GeometryOnlyContractTests(unittest.TestCase):
    def test_geometry_only_contracts_apply_to_dex3_and_inspire(self):
        cases = (
            (DEPTH_ONLY_VIDEO_KEYS, True, False),
            (SURFACE_NORMAL_ONLY_VIDEO_KEYS, False, True),
        )
        for end_effector in ("dex3", "inspire-dfx"):
            for video_keys, requires_depth_gray, requires_surface_normals in cases:
                with self.subTest(end_effector=end_effector, video_keys=video_keys):
                    contract = validate_model_contract(
                        _modality_config(video_keys),
                        end_effector=end_effector,
                    )
                    self.assertTrue(contract.requires_depth)
                    self.assertEqual(contract.requires_depth_gray, requires_depth_gray)
                    self.assertEqual(contract.requires_surface_normals, requires_surface_normals)
                    self.assertEqual(contract.video_keys, video_keys)
                    self.assertEqual(
                        contract.vision_input_contract["wire_video_keys"],
                        list(video_keys),
                    )

    def test_depth_only_observation_omits_rgb(self):
        depth = np.full((480, 640, 3), 42, dtype=np.uint8)
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros(14),
            np.zeros(6),
            np.ones(6),
            TASKS["pick-red-cup"],
            video_keys=DEPTH_ONLY_VIDEO_KEYS,
            depth_gray=depth,
            end_effector="inspire-dfx",
        )

        self.assertEqual(tuple(observation["video"]), DEPTH_ONLY_VIDEO_KEYS)
        np.testing.assert_array_equal(observation["video"]["depth_gray_view"][0, 0], depth)

    def test_surface_normals_only_observation_omits_rgb(self):
        normals = np.full((480, 640, 3), 128, dtype=np.uint8)
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros(14),
            np.zeros(7),
            np.ones(7),
            TASKS["pick-red-cup"],
            video_keys=SURFACE_NORMAL_ONLY_VIDEO_KEYS,
            surface_normals=normals,
            end_effector="dex3",
        )

        self.assertEqual(tuple(observation["video"]), SURFACE_NORMAL_ONLY_VIDEO_KEYS)
        np.testing.assert_array_equal(
            observation["video"]["surface_normals_view"][0, 0],
            normals,
        )

    def test_first_target_gap_reports_each_hand_value_unit(self):
        state_reader = SimpleNamespace(
            read=lambda timeout_s: SimpleNamespace(
                arm=np.zeros(14),
                left_hand=np.zeros(6),
                right_hand=np.zeros(6),
            )
        )
        inspire_chunk = ActionChunk(
            arm=np.zeros((1, 14)),
            left_hand=np.full((1, 6), 0.1),
            right_hand=np.full((1, 6), 0.2),
            end_effector="inspire-dfx",
        )

        self.assertEqual(
            chunk_delta_summary(inspire_chunk, state_reader),
            "first-target pose gap arm=0.0000 rad, "
            "hand=0.2000 normalized_open_fraction",
        )

        state_reader.read = lambda timeout_s: SimpleNamespace(
            arm=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        dex3_chunk = ActionChunk(
            arm=np.zeros((1, 14)),
            left_hand=np.full((1, 7), 0.1),
            right_hand=np.full((1, 7), 0.2),
            end_effector="dex3",
        )
        self.assertEqual(
            chunk_delta_summary(dex3_chunk, state_reader),
            "first-target pose gap arm=0.0000 rad, hand=0.2000 rad",
        )


if __name__ == "__main__":
    unittest.main()
