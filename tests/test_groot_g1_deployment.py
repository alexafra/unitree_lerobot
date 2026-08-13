from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import pty
import queue
import select
import sys
import tempfile
import termios
from types import SimpleNamespace
from types import ModuleType
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _active_command_action,
    _confirm_before_authority,
    _confirm_while_armed,
    _confirm_goal_transition,
    _OperatorTerminal,
    _readline_before_authority,
    _readline_while_armed,
    _run_blocking_motion_with_immediate_release,
    _select_next_goal_while_holding,
    OperatorRelease,
    build_parser,
    confirm_custom_goal,
    confirm_actuation,
    run as run_groot,
    select_instruction,
    show_camera_preview,
    validate_args,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError, MsgSerializer
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    ActionChunk,
    COLOUR_VIDEO_KEYS,
    DepthEncodingContract,
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_DEPTH_VIEW_SHAPE,
    EXPECTED_EGO_VIEW_SHAPE,
    EXPECTED_JOINT_NAMES,
    EXPECTED_ROBOT_TYPE,
    INITIAL_POSE_SCHEMA_VERSION,
    JOINT_LIMIT_MARGIN_RAD,
    HAND_LIMIT_TOLERANCE_RAD,
    InitializationSpec,
    MAX_ARM_STEP_RAD,
    MAX_HAND_STEP_RAD,
    MEASURED_LIMIT_TOLERANCE_RAD,
    RGBD_VIDEO_KEYS,
    SURFACE_NORMAL_VIDEO_KEYS,
    SurfaceNormalEncodingContract,
    TASKS,
    load_initialization_spec,
    make_observation,
    parse_action_chunk,
    validate_model_contract,
    validate_policy_metadata,
    validate_measured_state,
)
from unitree_lerobot.eval_robot.image_server.rgbd_protocol import (
    RGBD_PROTOCOL,
    TeleRgbdFrame,
    pack_rgbd_packet,
    unpack_rgbd_packet,
)

try:
    from unitree_lerobot.eval_robot.image_server import image_client as image_client_module
except ModuleNotFoundError as exc:
    image_client_module = None
    image_client_import_error_message = str(exc)
else:
    image_client_import_error_message = ""
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    ARM_RELEASE_RAMP_S,
    CameraImages,
    G1Dex3StateReader,
    ImmediateControlEvent,
    INITIALIZATION_MAX_ARM_STEP_RAD,
    INITIALIZATION_MAX_HAND_STEP_RAD,
    INITIALIZATION_MIN_MOVE_S,
    MAX_ARM_DQ_RAD_S,
    MAX_ARM_TRACKING_ERROR_RAD,
    MAX_CONDITIONED_ARM_STEP_RAD,
    MAX_CONDITIONED_HAND_STEP_RAD,
    MAX_HAND_TRACKING_ERROR_RAD,
    PUBLISH_HZ,
    QUALIFIED_REAL_MODE_MACHINE,
    RobotState,
    SafeG1Dex3Actuator,
    SIM_RIGHT_HAND_PERMUTATION,
    TeleimagerCamera,
    TeleimagerColourCamera,
    _G1Dex3CommandBackend,
    _actuator_main,
    _execute_initialization,
    _ramp_real_arm_authority,
    _wait_for_initialization_start,
    build_initialization_chunk,
    decode_color_0_rgb,
    request_live_camera_config,
)
from unitree_lerobot.utils.depth_encoding import encode_depth_gray_rgb
from unitree_lerobot.utils.surface_normal_encoding import (
    DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    encode_surface_normals_rgb,
    surface_normals_encoding_metadata,
)


def modality_config(
    action_horizon: int = 16,
    *,
    rgbd: bool = False,
    surface_normals: bool = False,
):
    if rgbd and surface_normals:
        raise ValueError("test config cannot request both geometry views")
    video_keys = (
        SURFACE_NORMAL_VIDEO_KEYS
        if surface_normals
        else RGBD_VIDEO_KEYS if rgbd else COLOUR_VIDEO_KEYS
    )
    return {
        "video": {
            "delta_indices": [0],
            "modality_keys": list(video_keys),
        },
        "state": {"delta_indices": [0], "modality_keys": list(ACTION_KEYS)},
        "action": {
            "delta_indices": list(range(action_horizon)),
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


def valid_action(horizon: int = 16):
    return {key: np.zeros((1, horizon, 7), dtype=np.float32) for key in ACTION_KEYS}


class FakeHeartbeat:
    def __init__(self, value: float):
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


class FakeBackend:
    instance = None

    def __init__(self, _simulation, _network_interface):
        type(self).instance = self
        self.simulation = _simulation
        self.publishes = 0
        self.released = False
        self.closed = False
        self._arm_target = np.zeros(14)
        self._left_target = np.zeros(7)
        self._right_target = np.zeros(7)

    def state(self):
        return RobotState(
            captured_at=time.monotonic(),
            mode_machine=0,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )

    def set_weight(self, _weight):
        pass

    def prepare_measured_hold(self):
        return self.state()

    def set_target(self, arm, left, right):
        self._arm_target = arm
        self._left_target = left
        self._right_target = right

    def publish(self):
        self.publishes += 1

    def release(self):
        self.released = True

    def close(self):
        self.closed = True


class GrootG1DeploymentTests(unittest.TestCase):
    def test_policy_warm_start_is_default_with_explicit_opt_out(self):
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertTrue(defaults.policy_warm_start)
        self.assertFalse(defaults.show_camera)
        validate_args(defaults)
        self.assertFalse(parser.parse_args(["--no-policy-warm-start"]).policy_warm_start)
        self.assertTrue(parser.parse_args(["--show-camera"]).show_camera)

    def test_custom_goal_is_validated_and_mutually_exclusive_with_trained_task(self):
        parser = build_parser()
        args = parser.parse_args(["--custom-goal", "  move the cup beside the cylinder.  "])
        self.assertEqual(
            select_instruction(args.task, args.custom_goal),
            ("custom-goal", "move the cup beside the cylinder."),
        )
        with self.assertRaisesRegex(DeploymentError, "printable"):
            select_instruction(None, "bad\ngoal")
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--task", "pick-red-cup", "--custom-goal", "another goal"])

        with mock.patch.object(sys, "stdin", io.StringIO("YES\n")):
            confirm_custom_goal("move the cup beside the cylinder")
        for response in ("yes", "NO", ""):
            with (
                self.subTest(response=response),
                mock.patch.object(sys, "stdin", io.StringIO(response + "\n")),
                self.assertRaisesRegex(DeploymentError, "not confirmed"),
            ):
                confirm_custom_goal("move the cup beside the cylinder")

        measured = load_initialization_spec("measured", task_name="custom-goal")
        self.assertEqual(measured.mode, "measured")
        with self.assertRaisesRegex(DeploymentError, "exact trained tasks"):
            load_initialization_spec(
                "pose-file",
                task_name="custom-goal",
                pose_file="/does/not/matter.json",
            )

    def test_camera_preview_shows_exact_policy_rgb_and_depth_views(self):
        rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        depth = np.array([[[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch(f"{module}.cv2.imshow") as imshow,
            mock.patch(f"{module}.cv2.waitKey", return_value=-1),
        ):
            show_camera_preview(rgb, depth)

        self.assertEqual(
            [call.args[0] for call in imshow.call_args_list],
            ["GR00T input: ego_view", "GR00T input: depth_gray_view"],
        )
        np.testing.assert_array_equal(imshow.call_args_list[0].args[1], rgb[..., ::-1])
        np.testing.assert_array_equal(imshow.call_args_list[1].args[1], depth[..., ::-1])

    def test_camera_preview_labels_surface_normal_policy_view(self):
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        normals = np.full((1, 2, 3), 128, dtype=np.uint8)
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch(f"{module}.cv2.imshow") as imshow,
            mock.patch(f"{module}.cv2.waitKey", return_value=-1),
        ):
            show_camera_preview(rgb, normals, "surface_normals_view")

        self.assertEqual(
            [call.args[0] for call in imshow.call_args_list],
            ["GR00T input: ego_view", "GR00T input: surface_normals_view"],
        )
        np.testing.assert_array_equal(imshow.call_args_list[1].args[1], normals[..., ::-1])

    def test_camera_preview_q_requests_normal_runner_cleanup(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch(f"{module}.cv2.imshow"),
            mock.patch(f"{module}.cv2.waitKey", return_value=ord("q")),
            self.assertRaisesRegex(DeploymentError, "closed by user"),
        ):
            show_camera_preview(np.zeros((1, 1, 3), dtype=np.uint8), None)

    def test_deployment_safety_values_match_reviewed_local_configuration(self):
        # Relaxed local values remain pinned deliberately. CHANGEDSAFETY comments beside
        # their definitions preserve the original adapter defaults; this test is not a
        # hardware-safety qualification.
        self.assertEqual(MAX_ARM_STEP_RAD, 0.10)
        self.assertEqual(MAX_HAND_STEP_RAD, 0.60)
        self.assertEqual(JOINT_LIMIT_MARGIN_RAD, 0.03)
        self.assertEqual(HAND_LIMIT_TOLERANCE_RAD, 0.002)
        self.assertEqual(MEASURED_LIMIT_TOLERANCE_RAD, 0.01)
        self.assertEqual(ARM_RELEASE_RAMP_S, 1.5)
        self.assertEqual(MAX_ARM_DQ_RAD_S, 6.0)
        self.assertEqual(MAX_ARM_TRACKING_ERROR_RAD, 0.35)
        self.assertEqual(MAX_HAND_TRACKING_ERROR_RAD, 1.50)
        self.assertEqual(QUALIFIED_REAL_MODE_MACHINE, 6)
        self.assertAlmostEqual(MAX_CONDITIONED_ARM_STEP_RAD, 0.03)
        np.testing.assert_allclose(
            MAX_CONDITIONED_HAND_STEP_RAD,
            np.array([0.06857, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]),
        )

    def _write_initial_pose(self, directory: str, **overrides):
        payload = {
            "schema_version": INITIAL_POSE_SCHEMA_VERSION,
            "name": "reviewed pick start",
            "robot_type": EXPECTED_ROBOT_TYPE,
            "task": "pick-red-cup",
            "instruction": TASKS["pick-red-cup"],
            "joint_names": EXPECTED_JOINT_NAMES,
            "arm": [0.0] * 14,
            "hands": {"policy": "measured"},
            "source": {
                "dataset_path": "/datasets/combined_atomic_only/train",
                "episode_index": 7,
                "frame_index": 0,
            },
        }
        payload.update(overrides)
        path = Path(directory) / "initial_pose.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_initialization_modes_have_explicit_target_semantics(self):
        measured = load_initialization_spec("measured", task_name="pick-red-cup")
        self.assertIsInstance(measured, InitializationSpec)
        self.assertEqual(measured.mode, "measured")
        self.assertFalse(measured.moves)
        self.assertIsNone(measured.arm)
        self.assertIsNone(measured.left_hand)
        self.assertIsNone(measured.right_hand)

        xr_home = load_initialization_spec("xr-home", task_name="pick-red-cup")
        self.assertEqual(xr_home.mode, "xr-home")
        self.assertTrue(xr_home.moves)
        self.assertTrue(xr_home.moves_hands)
        np.testing.assert_array_equal(xr_home.arm, np.zeros(14))
        np.testing.assert_array_equal(xr_home.left_hand, np.zeros(7))
        np.testing.assert_array_equal(xr_home.right_hand, np.zeros(7))

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_initial_pose(directory)
            pose = load_initialization_spec("pose-file", task_name="pick-red-cup", pose_file=path)

        self.assertEqual(pose.mode, "pose-file")
        np.testing.assert_array_equal(pose.arm, np.zeros(14))
        self.assertIsNone(pose.left_hand)
        self.assertIsNone(pose.right_hand)

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_initial_pose(
                directory,
                hands={"policy": "explicit", "left": [0.0] * 7, "right": [0.0] * 7},
            )
            explicit_hands = load_initialization_spec(
                "pose-file",
                task_name="pick-red-cup",
                pose_file=path,
            )
        np.testing.assert_array_equal(explicit_hands.left_hand, np.zeros(7))
        np.testing.assert_array_equal(explicit_hands.right_hand, np.zeros(7))

    def test_initialization_spec_arguments_are_mutually_consistent(self):
        with self.assertRaisesRegex(DeploymentError, "requires.*pose"):
            load_initialization_spec("pose-file", task_name="pick-red-cup")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_initial_pose(directory)
            with self.assertRaisesRegex(DeploymentError, "valid only.*pose-file"):
                load_initialization_spec("measured", task_name="pick-red-cup", pose_file=path)

    def test_pose_file_requires_explicit_complete_and_matching_contract(self):
        invalid_overrides = {
            "missing explicit left hand": {"hands": {"policy": "explicit", "right": [0.0] * 7}},
            "mixed measured/explicit hands": {"hands": {"policy": "measured", "left": [0.0] * 7}},
            "wrong task": {"task": "put-red-cup"},
            "wrong robot": {"robot_type": "another_robot"},
            "wrong joint order": {"joint_names": list(reversed(EXPECTED_JOINT_NAMES))},
            "wrong arm shape": {"arm": [0.0] * 13},
            "non-finite hand": {
                "hands": {
                    "policy": "explicit",
                    "left": [0.0] * 7,
                    "right": [float("nan")] + [0.0] * 6,
                }
            },
            "out-of-range arm": {"arm": [100.0] + [0.0] * 13},
            "non-absolute dataset": {
                "source": {
                    "dataset_path": "relative/train",
                    "episode_index": 7,
                    "frame_index": 0,
                }
            },
            "non-start frame": {
                "source": {
                    "dataset_path": "/datasets/train",
                    "episode_index": 7,
                    "frame_index": 1,
                }
            },
        }
        for case, overrides in invalid_overrides.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = self._write_initial_pose(directory, **overrides)
                with self.assertRaises(DeploymentError):
                    load_initialization_spec("pose-file", task_name="pick-red-cup", pose_file=path)

    def test_pose_file_rejects_duplicate_keys_at_every_json_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_initial_pose(directory)
            document = path.read_text(encoding="utf-8")
            duplicate_root = document.replace(
                '"name": "reviewed pick start"',
                '"name": "first", "name": "second"',
                1,
            )
            path.write_text(duplicate_root, encoding="utf-8")
            with self.assertRaisesRegex(DeploymentError, "duplicate field 'name'"):
                load_initialization_spec("pose-file", task_name="pick-red-cup", pose_file=path)

            path = self._write_initial_pose(directory)
            document = path.read_text(encoding="utf-8")
            duplicate_nested = document.replace(
                '"episode_index": 7',
                '"episode_index": 7, "episode_index": 8',
                1,
            )
            path.write_text(duplicate_nested, encoding="utf-8")
            with self.assertRaisesRegex(DeploymentError, "duplicate field 'episode_index'"):
                load_initialization_spec("pose-file", task_name="pick-red-cup", pose_file=path)

    def test_initialization_path_is_bounded_and_reaches_every_explicit_target(self):
        state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=5,
            arm=np.full(14, 0.2),
            arm_dq=np.zeros(14),
            left_hand=np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2]),
            right_hand=np.array([0.2, 0.2, -0.2, 0.2, 0.2, 0.2, 0.2]),
        )
        spec = load_initialization_spec("xr-home", task_name="pick-red-cup")

        path = build_initialization_chunk(state, spec)

        self.assertGreater(path.length, 1)
        self.assertGreaterEqual(path.length, round(INITIALIZATION_MIN_MOVE_S * PUBLISH_HZ))
        np.testing.assert_array_equal(path.arm[-1], spec.arm)
        np.testing.assert_array_equal(path.left_hand[-1], spec.left_hand)
        np.testing.assert_array_equal(path.right_hand[-1], spec.right_hand)
        arm_steps = np.diff(np.vstack((state.arm, path.arm)), axis=0)
        left_steps = np.diff(np.vstack((state.left_hand, path.left_hand)), axis=0)
        right_steps = np.diff(np.vstack((state.right_hand, path.right_hand)), axis=0)
        self.assertLessEqual(float(np.max(np.abs(arm_steps))), INITIALIZATION_MAX_ARM_STEP_RAD + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(left_steps))), INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(right_steps))), INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12)

    def test_measured_initialization_and_measured_hand_targets_do_not_move(self):
        state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=5,
            arm=np.linspace(-0.1, 0.1, 14),
            arm_dq=np.zeros(14),
            left_hand=np.array([0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1]),
            right_hand=np.array([0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        measured = load_initialization_spec("measured", task_name="pick-red-cup")
        measured_path = build_initialization_chunk(state, measured)
        np.testing.assert_array_equal(measured_path.arm, state.arm[None])
        np.testing.assert_array_equal(measured_path.left_hand, state.left_hand[None])
        np.testing.assert_array_equal(measured_path.right_hand, state.right_hand[None])

        with tempfile.TemporaryDirectory() as directory:
            pose_file = self._write_initial_pose(directory, arm=[0.0] * 14)
            pose = load_initialization_spec("pose-file", task_name="pick-red-cup", pose_file=pose_file)
        pose_path = build_initialization_chunk(state, pose)
        np.testing.assert_array_equal(pose_path.left_hand, np.repeat(state.left_hand[None], pose_path.length, axis=0))
        np.testing.assert_array_equal(
            pose_path.right_hand,
            np.repeat(state.right_hand[None], pose_path.length, axis=0),
        )

    def test_live_run_discards_preflight_and_infers_fresh_only_after_run_confirmation(self):
        events = []
        initial_step_checks = []
        preflight = SimpleNamespace(length=1, name="discarded-preflight")
        live = SimpleNamespace(length=1, name="fresh-live")

        class FakePolicy:
            resets = 0

            def ping(self):
                return True

            def get_modality_config(self):
                return {}

            def get_policy_metadata(self):
                return {}

            def reset(self):
                self.resets += 1
                events.append(f"policy.reset:{self.resets}")

            def close(self):
                events.append("policy.close")

        class FakeReader:
            def close(self):
                events.append("reader.close")

        class FakeCamera:
            config = {
                "head_camera": {
                    "type": "fake",
                    "image_shape": [480, 640],
                    "binocular": False,
                    "fps": 30,
                }
            }

            def close(self):
                events.append("camera.close")

        class FakeActuator:
            def start(self):
                events.append("actuator.start")

            def arm(self):
                events.append("actuator.arm")

            def initialize(self, spec):
                events.append(f"actuator.initialize:{spec.mode}")

            def submit(self, chunk):
                events.append(f"actuator.submit:{chunk.name}")
                return 1

            def wait_completed(self, sequence, timeout_s):
                events.append(f"actuator.completed:{sequence}")

            def close(self):
                events.append("actuator.close")

        inferred = iter((preflight, live))

        def fake_infer(*_args, **kwargs):
            chunk = next(inferred)
            initial_step_checks.append(kwargs.get("validate_initial_step", True))
            events.append(f"infer:{chunk.name}")
            return chunk, 0.01

        args = argparse.Namespace(
            task="pick-red-cup",
            policy_host="127.0.0.1",
            policy_port=5555,
            image_host="camera",
            network_interface=None,
            execution_horizon=1,
            max_chunks=1,
            initialization="xr-home",
            initial_pose_file=None,
            sim=True,
            actuate=True,
            allow_unqualified_real=False,
            confirm_sim_network_isolated=True,
        )
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        contract = SimpleNamespace(action_horizon=16, video_keys=COLOUR_VIDEO_KEYS, requires_depth=False)
        with (
            mock.patch(f"{module}.Gr00tClient", return_value=FakePolicy()),
            mock.patch(f"{module}.validate_model_contract", return_value=contract),
            mock.patch(f"{module}.validate_policy_metadata", return_value=None),
            mock.patch(f"{module}.initialize_dds", side_effect=lambda *_args: events.append("dds.init")),
            mock.patch(f"{module}.G1Dex3StateReader", return_value=FakeReader()),
            mock.patch(f"{module}.TeleimagerCamera", return_value=FakeCamera()),
            mock.patch(f"{module}.SafeG1Dex3Actuator", return_value=FakeActuator()),
            mock.patch(f"{module}.infer_chunk", side_effect=fake_infer),
            mock.patch(f"{module}.chunk_delta_summary", return_value="safe"),
            mock.patch(f"{module}.confirm_actuation", side_effect=lambda *_args: events.append("confirm.ACTUATE")),
            mock.patch(
                f"{module}.confirm_initialization",
                side_effect=lambda *_args: events.append("confirm.INITIALIZE"),
            ),
            mock.patch(f"{module}.confirm_policy_start", side_effect=lambda *_args: events.append("confirm.RUN")),
        ):
            run_groot(args)

        self.assertEqual(events.count("infer:discarded-preflight"), 1)
        self.assertEqual(events.count("infer:fresh-live"), 1)
        self.assertEqual(initial_step_checks, [False, True])
        self.assertNotIn("actuator.submit:discarded-preflight", events)
        ordered = [
            "infer:discarded-preflight",
            "confirm.ACTUATE",
            "actuator.start",
            "actuator.arm",
            "confirm.INITIALIZE",
            "actuator.initialize:xr-home",
            "confirm.RUN",
            "policy.reset:2",
            "infer:fresh-live",
            "actuator.submit:fresh-live",
        ]
        positions = [events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_policy_warm_start_smoothly_reaches_first_target_then_resets_and_reinfers(self):
        events = []
        initial_step_checks = []
        preflight = SimpleNamespace(length=1, name="discarded-preflight")
        warm_start = SimpleNamespace(length=1, name="discarded-warm-start")
        live = SimpleNamespace(length=1, name="fresh-live")

        class FakePolicy:
            resets = 0

            def ping(self):
                return True

            def get_modality_config(self):
                return {}

            def get_policy_metadata(self):
                return {}

            def reset(self):
                self.resets += 1
                events.append(f"policy.reset:{self.resets}")

            def close(self):
                events.append("policy.close")

        class FakeReader:
            def close(self):
                events.append("reader.close")

        class FakeCamera:
            config = {
                "head_camera": {
                    "type": "fake",
                    "image_shape": [480, 640],
                    "binocular": False,
                    "fps": 30,
                }
            }

            def close(self):
                events.append("camera.close")

        class FakeActuator:
            def start(self):
                events.append("actuator.start")

            def arm(self):
                events.append("actuator.arm")

            def initialize(self, spec):
                events.append(f"actuator.initialize:{spec.mode}")

            def warm_start(self, chunk):
                events.append(f"actuator.warm_start:{chunk.name}")

            def submit(self, chunk):
                events.append(f"actuator.submit:{chunk.name}")
                return 1

            def wait_completed(self, sequence, timeout_s):
                events.append(f"actuator.completed:{sequence}")

            def close(self):
                events.append("actuator.close")

        inferred = iter((preflight, warm_start, live))

        def fake_infer(*_args, **kwargs):
            chunk = next(inferred)
            initial_step_checks.append(kwargs.get("validate_initial_step", True))
            events.append(f"infer:{chunk.name}")
            return chunk, 0.01

        args = argparse.Namespace(
            task="pick-red-cup",
            policy_host="127.0.0.1",
            policy_port=5555,
            image_host="camera",
            network_interface=None,
            execution_horizon=1,
            max_chunks=1,
            initialization="measured",
            initial_pose_file=None,
            policy_warm_start=True,
            sim=True,
            actuate=True,
            allow_unqualified_real=False,
            confirm_sim_network_isolated=True,
        )
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        contract = SimpleNamespace(action_horizon=16, video_keys=COLOUR_VIDEO_KEYS, requires_depth=False)
        with (
            mock.patch(f"{module}.Gr00tClient", return_value=FakePolicy()),
            mock.patch(f"{module}.validate_model_contract", return_value=contract),
            mock.patch(f"{module}.validate_policy_metadata", return_value=None),
            mock.patch(f"{module}.initialize_dds", side_effect=lambda *_args: events.append("dds.init")),
            mock.patch(f"{module}.G1Dex3StateReader", return_value=FakeReader()),
            mock.patch(f"{module}.TeleimagerCamera", return_value=FakeCamera()),
            mock.patch(f"{module}.SafeG1Dex3Actuator", return_value=FakeActuator()),
            mock.patch(f"{module}.infer_chunk", side_effect=fake_infer),
            mock.patch(f"{module}.chunk_delta_summary", return_value="large first target"),
            mock.patch(f"{module}.confirm_actuation", side_effect=lambda *_args: events.append("confirm.ACTUATE")),
            mock.patch(
                f"{module}.confirm_initialization", side_effect=lambda *_args: events.append("confirm.INITIALIZE")
            ),
            mock.patch(f"{module}.confirm_policy_start", side_effect=lambda *_args: events.append("confirm.RUN")),
            mock.patch(
                f"{module}.confirm_policy_warm_start",
                side_effect=lambda *_args: events.append("confirm.WARMUP"),
            ),
            mock.patch(
                f"{module}.confirm_policy_continue", side_effect=lambda *_args: events.append("confirm.CONTINUE")
            ),
        ):
            run_groot(args)

        self.assertEqual(initial_step_checks, [False, False, True])
        self.assertNotIn("actuator.submit:discarded-preflight", events)
        self.assertNotIn("actuator.submit:discarded-warm-start", events)
        ordered = [
            "confirm.RUN",
            "policy.reset:2",
            "infer:discarded-warm-start",
            "confirm.WARMUP",
            "actuator.warm_start:discarded-warm-start",
            "confirm.CONTINUE",
            "policy.reset:3",
            "infer:fresh-live",
            "actuator.submit:fresh-live",
        ]
        positions = [events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_live_runner_holds_then_warm_starts_a_second_goal(self):
        events = []
        initial_step_checks = []
        chunks = iter(
            SimpleNamespace(length=1, name=name)
            for name in (
                "discarded-preflight",
                "pick-warm-start",
                "pick-live",
                "put-warm-start",
                "put-live",
            )
        )

        class FakePolicy:
            def __init__(self):
                self.resets = 0

            def ping(self):
                return True

            def get_modality_config(self):
                return {}

            def get_policy_metadata(self):
                return {}

            def reset(self):
                self.resets += 1
                events.append(f"policy.reset:{self.resets}")

            def close(self):
                events.append("policy.close")

        class FakeReader:
            def close(self):
                events.append("reader.close")

        class FakeCamera:
            config = {
                "head_camera": {
                    "type": "fake",
                    "image_shape": [480, 640],
                    "binocular": False,
                    "fps": 30,
                }
            }

            def close(self):
                events.append("camera.close")

        class FakeActuator:
            def __init__(self):
                self.sequence = 0

            def start(self):
                events.append("actuator.start")

            def arm(self):
                events.append("actuator.arm")

            def initialize(self, spec):
                events.append(f"actuator.initialize:{spec.mode}")

            def warm_start(self, chunk):
                events.append(f"actuator.warm_start:{chunk.name}")

            def submit(self, chunk):
                self.sequence += 1
                events.append(f"actuator.submit:{chunk.name}:{self.sequence}")
                return self.sequence

            def wait_completed(self, sequence, timeout_s):
                events.append(f"actuator.completed:{sequence}")

            def hold(self):
                events.append("actuator.hold")

            def close(self):
                events.append("actuator.close")

        def fake_infer(*_args, **kwargs):
            chunk = next(chunks)
            initial_step_checks.append(kwargs.get("validate_initial_step", True))
            events.append(f"infer:{chunk.name}")
            return chunk, 0.01

        args = argparse.Namespace(
            task="pick-red-cup",
            custom_goal=None,
            policy_host="127.0.0.1",
            policy_port=5555,
            image_host="camera",
            network_interface=None,
            execution_horizon=1,
            max_chunks=1,
            initialization="measured",
            initial_pose_file=None,
            policy_warm_start=True,
            show_camera=False,
            sim=True,
            actuate=True,
            allow_unqualified_real=False,
            confirm_sim_network_isolated=True,
        )
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        contract = SimpleNamespace(action_horizon=16, video_keys=COLOUR_VIDEO_KEYS, requires_depth=False)
        with (
            mock.patch(f"{module}.Gr00tClient", side_effect=lambda *_args: FakePolicy()),
            mock.patch(f"{module}.validate_model_contract", return_value=contract),
            mock.patch(f"{module}.validate_policy_metadata", return_value=None),
            mock.patch(f"{module}.initialize_dds"),
            mock.patch(f"{module}.G1Dex3StateReader", return_value=FakeReader()),
            mock.patch(f"{module}.TeleimagerCamera", return_value=FakeCamera()),
            mock.patch(f"{module}.SafeG1Dex3Actuator", side_effect=lambda *_args: FakeActuator()),
            mock.patch(f"{module}.infer_chunk", side_effect=fake_infer),
            mock.patch(f"{module}.chunk_delta_summary", return_value="bounded"),
            mock.patch(f"{module}.confirm_actuation"),
            mock.patch(f"{module}.confirm_initialization"),
            mock.patch(f"{module}.confirm_policy_start"),
            mock.patch(f"{module}.confirm_policy_warm_start"),
            mock.patch(f"{module}.confirm_policy_continue"),
            mock.patch(
                f"{module}._poll_active_command",
                side_effect=(None, None, "s", None, None, None),
            ),
            mock.patch(
                f"{module}._select_next_goal_while_holding",
                return_value=("put-red-cup", TASKS["put-red-cup"]),
            ),
        ):
            run_groot(args)

        self.assertEqual(initial_step_checks, [False, False, True, False, True])
        ordered = [
            "infer:discarded-preflight",
            "infer:pick-warm-start",
            "actuator.warm_start:pick-warm-start",
            "infer:pick-live",
            "actuator.submit:pick-live:1",
            "actuator.completed:1",
            "actuator.hold",
            "infer:put-warm-start",
            "actuator.warm_start:put-warm-start",
            "infer:put-live",
            "actuator.submit:put-live:2",
            "actuator.completed:2",
            "actuator.close",
        ]
        positions = [events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("actuator.submit:discarded-preflight:1", events)
        self.assertNotIn("actuator.submit:pick-warm-start:1", events)
        self.assertNotIn("actuator.submit:put-warm-start:2", events)

    def test_shadow_skips_initial_step_validation_for_every_chunk_but_measured_live_keeps_it(self):
        class StopAfterPreflight(Exception):
            pass

        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        contract = SimpleNamespace(
            action_horizon=16,
            video_keys=COLOUR_VIDEO_KEYS,
            requires_depth=False,
        )
        for name, actuate, initialization, max_chunks, expected_checks in (
            ("shadow measured hold", False, "measured", 3, [False, False, False]),
            ("live measured hold", True, "measured", 1, [True]),
        ):
            with self.subTest(name=name):
                checks = []
                policy = SimpleNamespace(
                    ping=lambda: True,
                    get_modality_config=lambda: {},
                    get_policy_metadata=lambda: {},
                    reset=lambda: None,
                    close=lambda: None,
                )
                reader = SimpleNamespace(close=lambda: None)
                camera = SimpleNamespace(
                    config={
                        "head_camera": {
                            "type": "fake",
                            "image_shape": [480, 640],
                            "binocular": False,
                            "fps": 30,
                        }
                    },
                    close=lambda: None,
                )

                def record_preflight(*_args, **kwargs):
                    checks.append(kwargs.get("validate_initial_step", True))
                    return SimpleNamespace(length=1), 0.01

                args = argparse.Namespace(
                    task="pick-red-cup",
                    policy_host="127.0.0.1",
                    policy_port=5555,
                    image_host="camera",
                    network_interface=None,
                    execution_horizon=1,
                    max_chunks=max_chunks,
                    initialization=initialization,
                    initial_pose_file=None,
                    sim=True,
                    actuate=actuate,
                    allow_unqualified_real=False,
                    confirm_sim_network_isolated=actuate,
                )
                confirmation = StopAfterPreflight() if actuate else None
                with (
                    mock.patch(f"{module}.Gr00tClient", return_value=policy),
                    mock.patch(f"{module}.validate_model_contract", return_value=contract),
                    mock.patch(f"{module}.validate_policy_metadata", return_value=None),
                    mock.patch(f"{module}.initialize_dds"),
                    mock.patch(f"{module}.G1Dex3StateReader", return_value=reader),
                    mock.patch(f"{module}.TeleimagerCamera", return_value=camera),
                    mock.patch(f"{module}.infer_chunk", side_effect=record_preflight),
                    mock.patch(f"{module}.chunk_delta_summary", return_value="safe"),
                    mock.patch(f"{module}.confirm_actuation", side_effect=confirmation),
                    mock.patch(f"{module}.time.sleep"),
                ):
                    if actuate:
                        with self.assertRaises(StopAfterPreflight):
                            run_groot(args)
                    else:
                        run_groot(args)

                self.assertEqual(checks, expected_checks)

    def test_actuation_confirmation_uses_r_and_cancels_on_s_or_q(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{module}._confirm_before_authority", return_value="continue") as confirm,
            mock.patch("builtins.print"),
        ):
            confirm_actuation(True, "pick-red-cup", TASKS["pick-red-cup"])
        confirm.assert_called_once()

        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{module}._confirm_before_authority", return_value="stop"),
            mock.patch("builtins.print"),
            self.assertRaises(OperatorRelease),
        ):
            confirm_actuation(True, "pick-red-cup", TASKS["pick-red-cup"])

        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{module}._confirm_before_authority", side_effect=OperatorRelease),
            mock.patch("builtins.print"),
            self.assertRaises(OperatorRelease),
        ):
            confirm_actuation(True, "pick-red-cup", TASKS["pick-red-cup"])

    def test_operator_cancellation_before_actuation_never_constructs_actuator(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        contract = SimpleNamespace(
            action_horizon=16,
            video_keys=COLOUR_VIDEO_KEYS,
            requires_depth=False,
        )
        policy = SimpleNamespace(
            ping=lambda: True,
            get_modality_config=lambda: {},
            get_policy_metadata=lambda: {},
            reset=lambda: None,
            close=mock.Mock(),
        )
        reader = SimpleNamespace(close=mock.Mock())
        camera = SimpleNamespace(
            config={
                "head_camera": {
                    "type": "fake",
                    "image_shape": [480, 640],
                    "binocular": False,
                    "fps": 30,
                }
            },
            close=mock.Mock(),
        )
        args = argparse.Namespace(
            task="pick-red-cup",
            custom_goal=None,
            policy_host="127.0.0.1",
            policy_port=5555,
            image_host="camera",
            network_interface=None,
            execution_horizon=1,
            max_chunks=1,
            initialization="measured",
            initial_pose_file=None,
            policy_warm_start=True,
            show_camera=False,
            sim=True,
            actuate=True,
            allow_unqualified_real=False,
            confirm_sim_network_isolated=True,
        )
        actuator_factory = mock.Mock()
        with (
            mock.patch(f"{module}.Gr00tClient", return_value=policy),
            mock.patch(f"{module}.validate_model_contract", return_value=contract),
            mock.patch(f"{module}.validate_policy_metadata", return_value=None),
            mock.patch(f"{module}.initialize_dds"),
            mock.patch(f"{module}.G1Dex3StateReader", return_value=reader),
            mock.patch(f"{module}.TeleimagerCamera", return_value=camera),
            mock.patch(f"{module}.infer_chunk", return_value=(SimpleNamespace(length=1), 0.01)),
            mock.patch(f"{module}.chunk_delta_summary", return_value="safe"),
            mock.patch(f"{module}.confirm_actuation", side_effect=OperatorRelease),
            mock.patch(f"{module}.SafeG1Dex3Actuator", actuator_factory),
            self.assertRaises(OperatorRelease),
        ):
            run_groot(args)

        actuator_factory.assert_not_called()

    def test_armed_confirmation_keeps_watchdog_alive_and_accepts_only_three_key_input(self):
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            hold=mock.Mock(),
        )
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        stdin = io.StringIO("r\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(
                f"{module}.select.select",
                side_effect=[([], [], []), ([stdin], [], [])],
            ),
            mock.patch("builtins.print"),
        ):
            _confirm_while_armed(actuator, "ready", "RUN")

        self.assertEqual(actuator.heartbeat.call_count, 2)
        self.assertEqual(actuator.assert_healthy.call_count, 2)

        stdin = io.StringIO("s\nr\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
        ):
            _confirm_while_armed(actuator, "ready", "RUN")

        stdin = io.StringIO("q\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
            self.assertRaises(OperatorRelease),
        ):
            _confirm_while_armed(actuator, "ready", "RUN")

        stdin = io.StringIO("run\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
            self.assertRaisesRegex(DeploymentError, "Expected single-key r"),
        ):
            _confirm_while_armed(actuator, "ready", "RUN")

    def test_pre_initialization_stop_stays_at_gate_until_r_without_calling_hold(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        prompt = "Press r to INITIALIZE (no Enter); s STOP; q release: "
        keys_sent = False
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            hold=mock.Mock(),
            request_immediate_hold=mock.Mock(side_effect=DeploymentError("actuator child is not started")),
            request_immediate_release=mock.Mock(),
        )

        def write_stop_then_continue(*args, **_kwargs):
            nonlocal keys_sent
            if args and args[0] == prompt and not keys_sent:
                keys_sent = True
                os.write(master_fd, b"sr")

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=write_stop_then_continue) as printed,
                mock.patch(f"{module}._finish_operator_stop") as finish_stop,
            ):
                _confirm_while_armed(
                    actuator,
                    "Initialization gate",
                    "INITIALIZE",
                    stop_is_already_held=True,
                )

            prompt_calls = [call for call in printed.call_args_list if call.args == (prompt,)]
            self.assertTrue(keys_sent)
            self.assertEqual(len(prompt_calls), 2)
            actuator.request_immediate_hold.assert_called_once_with()
            actuator.hold.assert_not_called()
            finish_stop.assert_not_called()
            actuator.request_immediate_release.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_confirmation_timeout_releases_control_flow_without_reading_stdin(self):
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            hold=mock.Mock(),
        )
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        select_mock = mock.Mock()
        with (
            mock.patch(f"{module}.OPERATOR_CONFIRMATION_TIMEOUT_S", 0.0),
            mock.patch(f"{module}.select.select", select_mock),
            mock.patch("builtins.print"),
            self.assertRaisesRegex(DeploymentError, "Timed out waiting for RUN"),
        ):
            _confirm_while_armed(actuator, "ready", "RUN")

        actuator.heartbeat.assert_called_once_with()
        actuator.assert_healthy.assert_called_once_with()
        select_mock.assert_not_called()

    def test_active_terminal_commands_distinguish_powered_hold_from_release(self):
        for command in ("stop", "STOP", "s", "S"):
            with self.subTest(command=command):
                self.assertEqual(_active_command_action(command), "hold")
        for command in ("quit", "QUIT", "q", "Q"):
            with self.subTest(command=command):
                self.assertEqual(_active_command_action(command), "release")
        for command in (None, "", "hold", "HOLD", "h", "H", "finished", "pick-red-cup"):
            with self.subTest(command=command):
                self.assertIsNone(_active_command_action(command))

    def test_active_terminal_reacts_to_single_key_without_enter_and_restores_tty(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        for key, expected, method_name in (
            (b"s", "hold", "request_immediate_hold"),
            (b"S", "hold", "request_immediate_hold"),
            (b"q", "release", "request_immediate_release"),
            (b"Q", "release", "request_immediate_release"),
            (b"\x11", "release", "request_immediate_release"),
        ):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                actuator = SimpleNamespace(
                    request_immediate_hold=mock.Mock(),
                    request_immediate_release=mock.Mock(),
                )
                try:
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        mock.patch(f"{module}.select.select", wraps=select.select),
                        _OperatorTerminal(actuator) as terminal,
                    ):
                        os.write(master_fd, key)
                        deadline = time.monotonic() + 1.0
                        outcome = None
                        while outcome is None and time.monotonic() < deadline:
                            outcome = terminal.poll_control()
                            time.sleep(0.001)

                        self.assertEqual(outcome, expected)
                        getattr(actuator, method_name).assert_called_once_with()
                        # A single key is a single event, not a sticky command.
                        self.assertIsNone(terminal.poll_control())
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_active_terminal_non_tty_fallback_does_not_consume_line_prompt_input(self):
        stdin = io.StringIO("s\n")
        actuator = SimpleNamespace(
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        with mock.patch.object(sys, "stdin", stdin), _OperatorTerminal(actuator) as terminal:
            self.assertIsNone(terminal.poll_control())

        self.assertEqual(stdin.readline(), "s\n")
        actuator.request_immediate_hold.assert_not_called()
        actuator.request_immediate_release.assert_not_called()

    def test_active_terminal_ignores_h_keys(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        actuator = SimpleNamespace(
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with mock.patch.object(sys, "stdin", stdin), _OperatorTerminal(actuator) as terminal:
                os.write(master_fd, b"hH")
                time.sleep(0.06)
                self.assertIsNone(terminal.poll_control())

            actuator.request_immediate_hold.assert_not_called()
            actuator.request_immediate_release.assert_not_called()
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_line_prompt_uses_alt_q_to_insert_literal_lowercase_q(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.print"):
                # Bare q/Q are global release keys. Alt+q (ESC q) escapes a
                # literal lowercase q while keeping the rest ordinary input.
                os.write(master_fd, b"\x1bquickly hold the object beside it\n")
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)

            self.assertEqual(response, "quickly hold the object beside it")
            actuator.request_immediate_hold.assert_not_called()
            actuator.request_immediate_release.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_line_prompt_uses_alt_shift_q_to_insert_literal_uppercase_q(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.print"):
                os.write(master_fd, b"\x1bQuick\n")
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)

            self.assertEqual(response, "Quick")
            actuator.request_immediate_hold.assert_not_called()
            actuator.request_immediate_release.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_line_prompt_expired_standalone_escape_cannot_suppress_q_release(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        prompt_seen = threading.Event()
        writer_finished = threading.Event()
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )

        def write_expired_escape_then_q():
            if prompt_seen.wait(timeout=1.0):
                os.write(master_fd, b"\x1b")
                time.sleep(0.15)  # Longer than the 0.1 s Alt-prefix window.
                os.write(master_fd, b"q")
            writer_finished.set()

        def mark_prompt(*args, **_kwargs):
            if args and args[0] == "Next goal> ":
                prompt_seen.set()

        writer = threading.Thread(target=write_expired_escape_then_q, daemon=True)
        try:
            writer.start()
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=mark_prompt),
            ):
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)

            writer.join(timeout=1.0)
            self.assertTrue(writer_finished.is_set())
            self.assertEqual(response, "q")
            actuator.request_immediate_release.assert_called_once_with()
            actuator.request_immediate_hold.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            writer.join(timeout=1.0)
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_pre_authority_release_keys_need_no_enter_and_restore_tty(self):
        for key in (b"q", b"Q"):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                try:
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        mock.patch("builtins.print"),
                        self.assertRaises(OperatorRelease),
                    ):
                        os.write(master_fd, key)
                        _readline_before_authority("Task number: ")

                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_pre_authority_confirmation_uses_three_keys_without_enter(self):
        for key, expected in (
            (b"r", "continue"),
            (b"R", "continue"),
            (b"s", "stop"),
            (b"S", "stop"),
        ):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)

                def write_key_when_prompt_is_ready(*args, **_kwargs):
                    if args and args[0] == "Confirm> ":
                        os.write(master_fd, key)

                try:
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        mock.patch(
                            "builtins.print",
                            side_effect=write_key_when_prompt_is_ready,
                        ),
                    ):
                        started_at = time.monotonic()
                        response = _confirm_before_authority("Confirm> ")
                        elapsed = time.monotonic() - started_at

                    self.assertEqual(response, expected)
                    self.assertLess(elapsed, 0.25)
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_pre_authority_confirmation_release_keys_need_no_enter(self):
        for key in (b"q", b"Q", b"\x11"):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)

                def write_key_when_prompt_is_ready(*args, **_kwargs):
                    if args and args[0] == "Confirm> ":
                        os.write(master_fd, key)

                try:
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        mock.patch(
                            "builtins.print",
                            side_effect=write_key_when_prompt_is_ready,
                        ),
                        self.assertRaises(OperatorRelease),
                    ):
                        _confirm_before_authority("Confirm> ")

                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_rapid_rq_releases_before_actuator_start_can_run(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            start=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )

        def write_rapid_rq_when_prompt_is_ready(*args, **_kwargs):
            if args and args[0] == "Confirm> ":
                os.write(master_fd, b"rq")

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch(
                    "builtins.print",
                    side_effect=write_rapid_rq_when_prompt_is_ready,
                ),
            ):
                self.assertEqual(_confirm_before_authority("Confirm> "), "continue")
                with self.assertRaises(OperatorRelease):
                    _run_blocking_motion_with_immediate_release(actuator, actuator.start)

            # The confirmation reader must preserve the queued q, and the
            # blocking-motion guard must drain it synchronously before start.
            actuator.start.assert_not_called()
            actuator.request_immediate_release.assert_called_once_with()
            actuator.request_immediate_hold.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_pre_authority_ctrl_q_releases_after_raw_prompt_without_enter(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)

        def write_ctrl_q_when_prompt_is_ready(*args, **_kwargs):
            if args and args[0] == "Task number: ":
                os.write(master_fd, b"\x11")

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=write_ctrl_q_when_prompt_is_ready),
                self.assertRaises(OperatorRelease),
            ):
                _readline_before_authority("Task number: ")

            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_blocking_motion_release_keys_need_no_enter_and_restore_tty(self):
        for key in (b"q", b"Q", b"\x11"):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                operation_started = threading.Event()
                release_requested = threading.Event()
                writer_finished = threading.Event()
                actuator = SimpleNamespace(
                    request_immediate_hold=mock.Mock(),
                    request_immediate_release=mock.Mock(side_effect=lambda: release_requested.set()),
                    immediate_control_requested=mock.Mock(
                        side_effect=lambda: "release" if release_requested.is_set() else None
                    ),
                )

                def blocking_operation():
                    operation_started.set()
                    if not release_requested.wait(timeout=1.0):
                        raise AssertionError("q did not interrupt the blocking operation")
                    raise ImmediateControlEvent("release")

                def write_during_operation():
                    if operation_started.wait(timeout=1.0):
                        os.write(master_fd, key)
                    writer_finished.set()

                writer = threading.Thread(target=write_during_operation, daemon=True)
                try:
                    writer.start()
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        self.assertRaises(OperatorRelease),
                    ):
                        _run_blocking_motion_with_immediate_release(actuator, blocking_operation)

                    writer.join(timeout=1.0)
                    self.assertTrue(writer_finished.is_set())
                    actuator.request_immediate_release.assert_called_once_with()
                    actuator.request_immediate_hold.assert_not_called()
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    writer.join(timeout=1.0)
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_armed_line_prompt_global_keys_do_not_require_enter(self):
        for key, expected, method_name in (
            (b"q", "q", "request_immediate_release"),
            (b"Q", "q", "request_immediate_release"),
            (b"S", "stop", "request_immediate_hold"),
        ):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                actuator = SimpleNamespace(
                    heartbeat=mock.Mock(),
                    assert_healthy=mock.Mock(),
                    request_immediate_hold=mock.Mock(),
                    request_immediate_release=mock.Mock(),
                )
                try:
                    with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.print"):
                        os.write(master_fd, key)
                        response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)

                    self.assertEqual(response, expected)
                    getattr(actuator, method_name).assert_called_once_with()
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_armed_confirmation_mode_uses_three_keys_without_enter(self):
        cases = (
            (b"r", "continue", None),
            (b"R", "continue", None),
            (b"s", "stop", "request_immediate_hold"),
            (b"S", "stop", "request_immediate_hold"),
            (b"q", "q", "request_immediate_release"),
            (b"Q", "q", "request_immediate_release"),
            (b"\x11", "q", "request_immediate_release"),
        )
        for key, expected, expected_method in cases:
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                actuator = SimpleNamespace(
                    heartbeat=mock.Mock(),
                    assert_healthy=mock.Mock(),
                    request_immediate_hold=mock.Mock(),
                    request_immediate_release=mock.Mock(),
                )

                def write_key_when_prompt_is_ready(*args, **_kwargs):
                    if args and args[0] == "Confirm> ":
                        os.write(master_fd, key)

                try:
                    with (
                        mock.patch.object(sys, "stdin", stdin),
                        mock.patch(
                            "builtins.print",
                            side_effect=write_key_when_prompt_is_ready,
                        ),
                    ):
                        started_at = time.monotonic()
                        response = _readline_while_armed(
                            actuator,
                            "Confirm> ",
                            timeout_s=1.0,
                            confirmation_mode=True,
                        )
                        elapsed = time.monotonic() - started_at

                    self.assertEqual(response, expected)
                    self.assertLess(elapsed, 0.25)
                    if expected_method is None:
                        actuator.request_immediate_hold.assert_not_called()
                        actuator.request_immediate_release.assert_not_called()
                    else:
                        getattr(actuator, expected_method).assert_called_once_with()
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_armed_line_prompt_ctrl_q_releases_without_enter(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )

        def write_ctrl_q_when_prompt_is_ready(*args, **_kwargs):
            if args and args[0] == "Next goal> ":
                os.write(master_fd, b"\x11")

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=write_ctrl_q_when_prompt_is_ready),
            ):
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)

            self.assertEqual(response, "q")
            actuator.request_immediate_release.assert_called_once_with()
            actuator.request_immediate_hold.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_line_prompt_rapid_stop_then_release_preserves_queued_q(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with mock.patch.object(sys, "stdin", stdin), mock.patch("builtins.print"):
                # Both keys arrive during one raw-mode prompt. S returns STOP;
                # restoration must not flush the already queued q.
                os.write(master_fd, b"Sq")
                first = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)
                second_started_at = time.monotonic()
                second = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)
                second_elapsed = time.monotonic() - second_started_at

            self.assertEqual(first, "stop")
            self.assertEqual(second, "q")
            self.assertLess(second_elapsed, 0.25)
            actuator.request_immediate_hold.assert_called_once_with()
            actuator.request_immediate_release.assert_called_once_with()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_armed_line_prompt_q_as_prompt_appears_needs_no_enter(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        self.assertTrue(original[3] & termios.ICANON)
        prompt_seen = threading.Event()
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )

        def write_q_when_prompt_is_printed(*args, **_kwargs):
            if args and args[0] == "Next goal> ":
                # Inject q as soon as the raw-mode prompt appears, without a newline.
                os.write(master_fd, b"q")
                prompt_seen.set()

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=write_q_when_prompt_is_printed),
            ):
                started_at = time.monotonic()
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=0.5)
                elapsed = time.monotonic() - started_at

            self.assertTrue(prompt_seen.is_set())
            self.assertEqual(response, "q")
            self.assertLess(elapsed, 0.25)
            actuator.request_immediate_release.assert_called_once_with()
            actuator.request_immediate_hold.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_next_goal_line_prompt_does_not_treat_r_as_immediate_continue(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        prompt_seen = threading.Event()
        newline_sent = threading.Event()
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )

        def write_r_then_delayed_enter(*args, **_kwargs):
            if args and args[0] == "Next goal> ":
                prompt_seen.set()
                os.write(master_fd, b"r")

                def finish_line():
                    time.sleep(0.15)
                    os.write(master_fd, b"\n")
                    newline_sent.set()

                threading.Thread(target=finish_line, daemon=True).start()

        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                mock.patch("builtins.print", side_effect=write_r_then_delayed_enter),
            ):
                started_at = time.monotonic()
                response = _readline_while_armed(actuator, "Next goal> ", timeout_s=1.0)
                elapsed = time.monotonic() - started_at

            self.assertTrue(prompt_seen.is_set())
            self.assertTrue(newline_sent.wait(timeout=1.0))
            self.assertEqual(response, "r")
            self.assertGreaterEqual(elapsed, 0.12)
            actuator.request_immediate_hold.assert_not_called()
            actuator.request_immediate_release.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_active_terminal_restores_tty_when_active_runner_raises(self):
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with (
                mock.patch.object(sys, "stdin", stdin),
                self.assertRaisesRegex(RuntimeError, "injected active-run failure"),
            ):
                with _OperatorTerminal(actuator):
                    raise RuntimeError("injected active-run failure")
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_goal_transition_prompts_honor_global_hold_and_release_commands(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        actuator = SimpleNamespace(hold=mock.Mock())
        for response, expected in (("continue", "continue"), ("stop", "hold"), ("q", "release")):
            with (
                self.subTest(response=response),
                mock.patch(
                    f"{module}._readline_while_armed",
                    return_value=response,
                ) as readline,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(
                    _confirm_goal_transition(actuator, "transition", "WARMUP"),
                    expected,
                )
                self.assertTrue(readline.call_args.kwargs["confirmation_mode"])
        actuator.hold.assert_called_once_with()

    def test_held_goal_selector_services_heartbeat_and_rejects_unconfirmed_custom_text(self):
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
            hold=mock.Mock(),
        )
        # Stay in HOLD for an explicit s, reject one custom instruction, then
        # choose the second trained task by its displayed number.
        stdin = io.StringIO("s\nmove it somewhere novel\nNO\n2\n")
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
        ):
            selected = _select_next_goal_while_holding(actuator)

        expected_name = list(TASKS)[1]
        self.assertEqual(selected, (expected_name, TASKS[expected_name]))
        self.assertEqual(actuator.heartbeat.call_count, 4)
        self.assertEqual(actuator.assert_healthy.call_count, 4)

        release_actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
        )
        stdin = io.StringIO("q\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
        ):
            self.assertIsNone(_select_next_goal_while_holding(release_actuator))
        release_actuator.heartbeat.assert_called_once_with()
        release_actuator.assert_healthy.assert_called_once_with()

        custom_release_actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            assert_healthy=mock.Mock(),
        )
        stdin = io.StringIO("move it somewhere novel\nq\n")
        with (
            mock.patch.object(sys, "stdin", stdin),
            mock.patch(f"{module}.select.select", return_value=([stdin], [], [])),
            mock.patch("builtins.print"),
        ):
            self.assertIsNone(_select_next_goal_while_holding(custom_release_actuator))
        self.assertEqual(custom_release_actuator.heartbeat.call_count, 2)
        self.assertEqual(custom_release_actuator.assert_healthy.call_count, 2)

    def test_serializer_is_compatible_with_gr00t_server(self):
        try:
            from gr00t.policy.server_client import MsgSerializer as ServerSerializer
        except ImportError as exc:
            self.skipTest(f"Isaac-GR00T is not installed in this environment: {exc}")
        value = {
            "video": {
                "ego_view": np.zeros((1, 1, 4, 5, 3), dtype=np.uint8),
                "depth_gray_view": np.full((1, 1, 4, 5, 3), 42, dtype=np.uint8),
            },
            "state": {"left_arm": np.arange(7, dtype=np.float32)[None, None]},
        }

        decoded_by_server = ServerSerializer.from_bytes(MsgSerializer.to_bytes(value))
        np.testing.assert_array_equal(decoded_by_server["state"]["left_arm"], value["state"]["left_arm"])
        decoded_by_client = MsgSerializer.from_bytes(ServerSerializer.to_bytes(value))
        np.testing.assert_array_equal(decoded_by_client["video"]["ego_view"], value["video"]["ego_view"])
        np.testing.assert_array_equal(
            decoded_by_client["video"]["depth_gray_view"],
            value["video"]["depth_gray_view"],
        )

    def test_server_modality_config_round_trips_without_client_gr00t_types(self):
        try:
            from gr00t.data.types import ModalityConfig
            from gr00t.policy.server_client import MsgSerializer as ServerSerializer
        except ImportError as exc:
            self.skipTest(f"Isaac-GR00T is not installed in this environment: {exc}")
        for rgbd in (False, True):
            with self.subTest(rgbd=rgbd):
                config = {name: ModalityConfig(**value) for name, value in modality_config(rgbd=rgbd).items()}
                decoded = MsgSerializer.from_bytes(ServerSerializer.to_bytes(config))
                contract = validate_model_contract(decoded)
                self.assertEqual(contract.action_horizon, 16)
                self.assertEqual(contract.requires_depth, rgbd)

    def test_model_contract_rejects_a_different_embodiment_layout(self):
        config = modality_config()
        config["state"]["modality_keys"].append("waist")

        with self.assertRaisesRegex(DeploymentError, "Unsupported state keys"):
            validate_model_contract(config)

    def test_model_contract_accepts_only_the_exact_ordered_rgbd_views(self):
        contract = validate_model_contract(modality_config(rgbd=True))

        self.assertTrue(contract.requires_depth)
        self.assertEqual(contract.video_keys, RGBD_VIDEO_KEYS)

        reversed_config = modality_config(rgbd=True)
        reversed_config["video"]["modality_keys"].reverse()
        with self.assertRaisesRegex(DeploymentError, "Unsupported video keys"):
            validate_model_contract(reversed_config)

    def test_model_contract_accepts_surface_normals_as_aligned_depth_geometry(self):
        contract = validate_model_contract(modality_config(surface_normals=True))

        self.assertTrue(contract.requires_depth)
        self.assertFalse(contract.requires_depth_gray)
        self.assertTrue(contract.requires_surface_normals)
        self.assertEqual(contract.video_keys, SURFACE_NORMAL_VIDEO_KEYS)

    def test_policy_metadata_validates_exact_surface_normal_contract(self):
        metadata = {
            "protocol_version": 1,
            "embodiment_tag": "new_embodiment",
            "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
            "dataset_contract": {
                "robot_type": EXPECTED_ROBOT_TYPE,
                "fps": 30.0,
                "observation_state_names": EXPECTED_JOINT_NAMES,
                "action_names": EXPECTED_JOINT_NAMES,
                "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
                "video_shapes": {
                    "ego_view": EXPECTED_EGO_VIEW_SHAPE,
                    "surface_normals_view": EXPECTED_DEPTH_VIEW_SHAPE,
                },
                "surface_normals_encoding": surface_normals_encoding_metadata(),
            },
        }

        contract = validate_policy_metadata(metadata, requires_surface_normals=True)
        self.assertEqual(
            contract,
            SurfaceNormalEncodingContract(
                intrinsics=DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
                max_neighbor_depth_delta_m=0.05,
            ),
        )
        malformed = {
            **metadata,
            "dataset_contract": {
                **metadata["dataset_contract"],
                "surface_normals_encoding": {
                    **metadata["dataset_contract"]["surface_normals_encoding"],
                    "orientation": "away_from_camera",
                },
            },
        }
        with self.assertRaisesRegex(DeploymentError, "surface-normal encoding for orientation"):
            validate_policy_metadata(malformed, requires_surface_normals=True)

    def test_policy_metadata_requires_the_explicit_g1_training_tag(self):
        metadata = {
            "protocol_version": 1,
            "embodiment_tag": "new_embodiment",
            "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
            "dataset_contract": {
                "robot_type": EXPECTED_ROBOT_TYPE,
                "fps": 30.0,
                "observation_state_names": EXPECTED_JOINT_NAMES,
                "action_names": EXPECTED_JOINT_NAMES,
                "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
                "video_shapes": {
                    "ego_view": EXPECTED_EGO_VIEW_SHAPE,
                    "depth_gray_view": EXPECTED_DEPTH_VIEW_SHAPE,
                },
                "depth_encoding": {
                    "source_key": "depth_0",
                    "feature_key": "observation.images.depth_gray_view",
                    "encoding": "linear_grayscale_replicated_rgb",
                    "near_m": 0.25,
                    "far_m": 1.0,
                    "invalid_value": 0,
                    "valid_value_range": [1, 255],
                },
            },
        }
        validate_policy_metadata(metadata)
        depth_contract = validate_policy_metadata(metadata, requires_depth=True)
        self.assertEqual(depth_contract, DepthEncodingContract(near_m=0.25, far_m=1.0))
        with self.assertRaisesRegex(DeploymentError, "deployment protocol"):
            validate_policy_metadata({**metadata, "protocol_version": 2})
        with self.assertRaisesRegex(DeploymentError, "no deployment dataset contract"):
            validate_policy_metadata(
                {
                    "protocol_version": 1,
                    "embodiment_tag": "new_embodiment",
                    "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
                }
            )
        with self.assertRaisesRegex(DeploymentError, "Unsupported depth encoding"):
            validate_policy_metadata(
                {
                    **metadata,
                    "dataset_contract": {
                        **metadata["dataset_contract"],
                        "depth_encoding": {
                            **metadata["dataset_contract"]["depth_encoding"],
                            "near_m": 0.25,
                            "encoding": "per_frame_normalized",
                        },
                    },
                },
                requires_depth=True,
            )
        with self.assertRaisesRegex(DeploymentError, "requires GR00T training tag"):
            validate_policy_metadata({**metadata, "embodiment_tag": "unitree_g1_sonic"})
        with self.assertRaisesRegex(DeploymentError, "contract mismatch for fps"):
            validate_policy_metadata(
                {
                    **metadata,
                    "dataset_contract": {**metadata["dataset_contract"], "fps": 50.0},
                }
            )
        with self.assertRaisesRegex(DeploymentError, "contract mismatch for observation_state_names"):
            validate_policy_metadata(
                {
                    **metadata,
                    "dataset_contract": {
                        **metadata["dataset_contract"],
                        "observation_state_names": list(reversed(EXPECTED_JOINT_NAMES)),
                    },
                }
            )
        with self.assertRaisesRegex(DeploymentError, "no checkpoint action output contract"):
            validate_policy_metadata({key: value for key, value in metadata.items() if key != "action_output_contract"})
        with self.assertRaisesRegex(DeploymentError, "Unsupported checkpoint action output contract"):
            validate_policy_metadata(
                {
                    **metadata,
                    "action_output_contract": {
                        **EXPECTED_ACTION_OUTPUT_CONTRACT,
                        "use_relative_action": False,
                    },
                }
            )

    def test_model_contract_rejects_delta_joint_actions(self):
        config = modality_config()
        config["action"]["action_configs"][0]["rep"] = "DELTA"

        with self.assertRaisesRegex(DeploymentError, "Unsupported action representation"):
            validate_model_contract(config)

    def test_measured_state_must_be_finite_and_inside_hardware_limits(self):
        safe_arm = np.zeros(14)
        with self.assertRaisesRegex(DeploymentError, "outside its physical range"):
            unsafe_arm = safe_arm.copy()
            unsafe_arm[0] = 100.0
            validate_measured_state(unsafe_arm, np.zeros(14), np.zeros(7), np.zeros(7))
        with self.assertRaisesRegex(DeploymentError, "NaN or infinity"):
            unsafe_hand = np.zeros(7)
            unsafe_hand[3] = np.nan
            validate_measured_state(safe_arm, np.zeros(14), unsafe_hand, np.zeros(7))

    def test_state_reader_callbacks_are_fresh_bounded_and_closed(self):
        class FakeSubscriber:
            instances = {}

            def __init__(self, topic, _message_type):
                self.topic = topic
                self.handler = None
                self.closed = False
                self.instances[topic] = self

            def Init(self, handler=None, queueLen=0):
                self.handler = handler
                self.queue_len = queueLen

            def Close(self):
                self.closed = True

        def fake_module(name, **attributes):
            module = ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        fake_modules = {
            "unitree_lerobot.eval_robot.robot_control.robot_arm": fake_module(
                "robot_arm", G1_29_JointArmIndex=tuple(range(14))
            ),
            "unitree_lerobot.eval_robot.robot_control.robot_hand_unitree": fake_module(
                "robot_hand_unitree",
                Dex3_1_Left_JointIndex=tuple(range(7)),
                Dex3_1_Right_JointIndex=tuple(range(7)),
            ),
            "unitree_sdk2py": fake_module("unitree_sdk2py"),
            "unitree_sdk2py.core": fake_module("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": fake_module("unitree_sdk2py.core.channel", ChannelSubscriber=FakeSubscriber),
            "unitree_sdk2py.idl": fake_module("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.unitree_hg": fake_module("unitree_sdk2py.idl.unitree_hg"),
            "unitree_sdk2py.idl.unitree_hg.msg": fake_module("unitree_sdk2py.idl.unitree_hg.msg"),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": fake_module(
                "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                HandState_=object,
                LowState_=object,
            ),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            reader = G1Dex3StateReader(max_age_s=0.1)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                reader.read(timeout_s=0.01)
            self.assertLess(time.monotonic() - started, 0.2)

            arm_message = SimpleNamespace(
                mode_machine=5,
                motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(14)],
            )
            # A real Dex3 publisher that reports all seven q values as exact
            # zero is an offline/default placeholder and is intentionally not
            # accepted as a fresh measured state.
            hand_message = SimpleNamespace(
                motor_state=[SimpleNamespace(q=0.1 if index == 0 else 0.0) for index in range(7)]
            )
            FakeSubscriber.instances["rt/lowstate"].handler(arm_message)
            FakeSubscriber.instances["rt/dex3/left/state"].handler(hand_message)
            FakeSubscriber.instances["rt/dex3/right/state"].handler(hand_message)

            state = reader.latest()
            self.assertEqual(state.mode_machine, 5)
            np.testing.assert_array_equal(state.arm, np.zeros(14))

            last_left_update = reader._updated_at["left"]
            zero_hand_message = SimpleNamespace(
                motor_state=[SimpleNamespace(q=0.0) for _ in range(7)]
            )
            FakeSubscriber.instances["rt/dex3/left/state"].handler(zero_hand_message)
            self.assertEqual(reader._updated_at["left"], last_left_update)
            self.assertEqual(reader._rejected_zero_hand_frames["left"], 1)
            np.testing.assert_array_equal(reader.latest().left_hand, state.left_hand)
            reader.close()

        self.assertTrue(all(item.closed for item in FakeSubscriber.instances.values()))

    def test_observation_matches_saved_g1_dex3_modality_shapes(self):
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.arange(14, dtype=np.float64),
            np.arange(7, dtype=np.float64),
            np.arange(7, dtype=np.float64),
            TASKS["pick-red-cup"],
        )

        self.assertEqual(observation["video"]["ego_view"].shape, (1, 1, 480, 640, 3))
        self.assertEqual(observation["video"]["ego_view"].dtype, np.uint8)
        np.testing.assert_array_equal(observation["state"]["left_arm"][0, 0], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(observation["state"]["right_arm"][0, 0], np.arange(7, 14, dtype=np.float32))
        self.assertEqual(
            observation["language"]["annotation.human.task_description"],
            [[TASKS["pick-red-cup"]]],
        )

        with self.assertRaisesRegex(DeploymentError, "allowlist"):
            make_observation(
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros(14),
                np.zeros(7),
                np.zeros(7),
                "pick up the cylinder",
            )
        custom = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros(14),
            np.zeros(7),
            np.zeros(7),
            "pick up the cylinder",
            allow_custom_instruction=True,
        )
        self.assertEqual(
            custom["language"]["annotation.human.task_description"],
            [["pick up the cylinder"]],
        )

    def test_rgbd_observation_contains_exactly_the_two_checkpoint_views(self):
        depth = np.full((480, 640, 3), 42, dtype=np.uint8)
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros(14),
            np.zeros(7),
            np.zeros(7),
            TASKS["pick-red-cup"],
            video_keys=RGBD_VIDEO_KEYS,
            depth_gray=depth,
        )

        self.assertEqual(tuple(observation["video"]), RGBD_VIDEO_KEYS)
        self.assertEqual(observation["video"]["depth_gray_view"].shape, (1, 1, 480, 640, 3))
        np.testing.assert_array_equal(observation["video"]["depth_gray_view"][0, 0], depth)

        with self.assertRaisesRegex(DeploymentError, "requires depth_gray_view"):
            make_observation(
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros(14),
                np.zeros(7),
                np.zeros(7),
                TASKS["pick-red-cup"],
                video_keys=RGBD_VIDEO_KEYS,
            )

    def test_surface_normal_observation_contains_exactly_the_selected_views(self):
        normals = np.full((480, 640, 3), 128, dtype=np.uint8)
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros(14),
            np.zeros(7),
            np.zeros(7),
            TASKS["pick-red-cup"],
            video_keys=SURFACE_NORMAL_VIDEO_KEYS,
            surface_normals=normals,
        )

        self.assertEqual(tuple(observation["video"]), SURFACE_NORMAL_VIDEO_KEYS)
        np.testing.assert_array_equal(
            observation["video"]["surface_normals_view"][0, 0],
            normals,
        )
        with self.assertRaisesRegex(DeploymentError, "requires surface_normals_view"):
            make_observation(
                np.zeros((480, 640, 3), dtype=np.uint8),
                np.zeros(14),
                np.zeros(7),
                np.zeros(7),
                TASKS["pick-red-cup"],
                video_keys=SURFACE_NORMAL_VIDEO_KEYS,
            )

    def test_action_parser_combines_arm_order_and_keeps_execution_prefix(self):
        action = valid_action()
        action["left_arm"][:] = 0.01
        action["right_arm"][:] = 0.02

        chunk = parse_action_chunk(
            action,
            model_horizon=16,
            execution_horizon=8,
            current_arm=np.zeros(14),
            current_left=np.zeros(7),
            current_right=np.zeros(7),
        )

        self.assertEqual(chunk.arm.shape, (8, 14))
        np.testing.assert_allclose(chunk.arm[0], [0.01] * 7 + [0.02] * 7)

    def test_action_parser_execution_horizon_is_limited_by_checkpoint(self):
        action = valid_action()

        chunk = parse_action_chunk(
            action,
            model_horizon=16,
            execution_horizon=16,
            current_arm=np.zeros(14),
            current_left=np.zeros(7),
            current_right=np.zeros(7),
        )

        self.assertEqual(chunk.arm.shape, (16, 14))
        with self.assertRaisesRegex(DeploymentError, "checkpoint action horizon=16"):
            parse_action_chunk(
                action,
                model_horizon=16,
                execution_horizon=17,
                current_arm=np.zeros(14),
                current_left=np.zeros(7),
                current_right=np.zeros(7),
            )

    def test_action_parser_rejects_non_finite_values(self):
        for unsafe in (np.nan, np.inf):
            with self.subTest(unsafe=unsafe):
                action = valid_action()
                action["right_arm"][0, 0, 0] = unsafe
                with self.assertRaisesRegex(DeploymentError, "NaN or infinity"):
                    parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_action_parser_rejects_large_first_step_instead_of_clipping(self):
        action = valid_action()
        action["right_arm"][0, :, 0] = 0.20

        with self.assertRaisesRegex(DeploymentError, "jump is too large"):
            parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_discarded_preflight_may_skip_only_the_unexecuted_initial_jump(self):
        action = valid_action()
        action["right_arm"][0, :, 0] = 0.20

        chunk = parse_action_chunk(
            action,
            16,
            8,
            np.zeros(14),
            np.zeros(7),
            np.zeros(7),
            validate_initial_step=False,
        )

        np.testing.assert_array_equal(chunk.right_hand, np.zeros((8, 7)))
        self.assertAlmostEqual(chunk.arm[0, 7], 0.20)

        action["right_arm"][0, 1, 0] = 0.30
        with self.assertRaisesRegex(DeploymentError, "jump is too large"):
            parse_action_chunk(
                action,
                16,
                8,
                np.zeros(14),
                np.zeros(7),
                np.zeros(7),
                validate_initial_step=False,
            )

    def test_action_parser_rejects_invalid_unexecuted_tail(self):
        action = valid_action()
        action["left_arm"][0, 15, 0] = 100.0

        with self.assertRaisesRegex(DeploymentError, "outside"):
            parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_action_parser_accepts_calibrated_one_milliradian_hand_endpoint(self):
        action = valid_action()
        action["right_hand"][0, :, 3] = -0.001

        chunk = parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

        np.testing.assert_allclose(chunk.right_hand[:, 3], -0.001)

    def test_camera_requires_fresh_jpeg_even_if_cached_bgr_exists(self):
        stale = SimpleNamespace(jpg=None, bgr=np.zeros((2, 4, 3), dtype=np.uint8))
        config = {"head_camera": {"image_shape": [2, 4], "binocular": False}}

        with self.assertRaisesRegex(TimeoutError, "fresh JPEG"):
            decode_color_0_rgb(stale, config)

    def test_atomic_rgbd_packet_round_trips_and_decodes_both_images(self):
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[:, :] = [1, 2, 3]
        depth = np.array(
            [[0, 250, 625, 1000, 1200]] * 4,
            dtype=np.uint16,
        )
        color_ok, color_jpeg = cv2.imencode(".jpg", bgr)
        depth_ok, depth_png = cv2.imencode(".png", depth)
        self.assertTrue(color_ok and depth_ok)

        packet = pack_rgbd_packet(17, 123456, color_jpeg.tobytes(), depth_png.tobytes())
        frame = unpack_rgbd_packet(packet, received_monotonic_ns=654321)

        self.assertEqual(len(packet) - len(color_jpeg) - len(depth_png), 40)
        self.assertEqual(frame.sequence, 17)
        self.assertEqual(frame.server_capture_monotonic_ns, 123456)
        self.assertEqual(frame.received_monotonic_ns, 654321)
        decoded_bgr = cv2.imdecode(np.frombuffer(frame.color_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        decoded_depth = cv2.imdecode(
            np.frombuffer(frame.aligned_depth_png, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        self.assertEqual(decoded_bgr.shape, bgr.shape)
        np.testing.assert_array_equal(decoded_depth, depth)

    def test_atomic_rgbd_packet_rejects_corruption_and_extra_bytes(self):
        packet = pack_rgbd_packet(1, 2, b"jpeg", b"png")

        for corrupt in (packet[:20], b"BADMAGIC" + packet[8:], packet + b"extra"):
            with self.subTest(length=len(corrupt)):
                with self.assertRaises(ValueError):
                    unpack_rgbd_packet(corrupt)

    def test_image_client_rgbd_mode_subscribes_only_the_atomic_head_port(self):
        if image_client_module is None:
            self.skipTest(f"TeleImager client dependencies are unavailable: {image_client_import_error_message}")
        config = {
            "head_camera": {
                "enable_zmq": True,
                "enable_webrtc": False,
                "zmq_port": 5555,
                "rgbd_zmq_port": 5560,
            },
            "left_wrist_camera": {"enable_zmq": False},
            "right_wrist_camera": {"enable_zmq": False},
        }
        packet = pack_rgbd_packet(4, 5, b"jpeg", b"png")

        class FakeRequester:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self):
                return config

            def close(self):
                pass

        class FakeManager:
            def __init__(self):
                self.calls = []

            def subscribe(self, host, port, request_bgr=False):
                self.calls.append((host, port, request_bgr))
                return image_client_module.TeleImage(
                    fps=30.0,
                    jpg=packet,
                    received_monotonic_ns=10,
                )

            def close(self):
                pass

        manager = FakeManager()
        with (
            mock.patch.object(image_client_module, "ZMQ_Requester", FakeRequester),
            mock.patch.object(
                image_client_module.ZMQ_SubscriberManager,
                "get_instance",
                return_value=manager,
            ),
        ):
            client = image_client_module.ImageClient(host="camera", request_rgbd=True)
            with self.assertRaisesRegex(RuntimeError, "use get_head_rgbd_frame"):
                client.get_head_frame()
            frame = client.get_head_rgbd_frame()

        self.assertEqual(manager.calls, [("camera", 5560, False), ("camera", 5560, False)])
        self.assertEqual(frame.sequence, 4)
        self.assertEqual(frame.received_monotonic_ns, 10)

    def test_image_client_missing_rgbd_port_closes_requester_and_manager(self):
        if image_client_module is None:
            self.skipTest(f"TeleImager client dependencies are unavailable: {image_client_import_error_message}")
        config = {
            "head_camera": {"enable_zmq": True, "enable_webrtc": False, "zmq_port": 5555},
            "left_wrist_camera": {"enable_zmq": False},
            "right_wrist_camera": {"enable_zmq": False},
        }

        class FakeRequester:
            instance = None

            def __init__(self, *_args, **_kwargs):
                type(self).instance = self
                self.closed = False

            def request(self):
                return config

            def close(self):
                self.closed = True

        class FakeManager:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        manager = FakeManager()
        with (
            mock.patch.object(image_client_module, "ZMQ_Requester", FakeRequester),
            mock.patch.object(
                image_client_module.ZMQ_SubscriberManager,
                "get_instance",
                return_value=manager,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no rgbd_zmq_port"):
                image_client_module.ImageClient(host="camera", request_rgbd=True)

        self.assertTrue(FakeRequester.instance.closed)
        self.assertTrue(manager.closed)

    def test_image_client_subscribe_failure_closes_partial_resources(self):
        if image_client_module is None:
            self.skipTest(f"TeleImager client dependencies are unavailable: {image_client_import_error_message}")
        config = {
            "head_camera": {
                "enable_zmq": True,
                "enable_webrtc": False,
                "zmq_port": 5555,
                "rgbd_zmq_port": 5560,
            },
            "left_wrist_camera": {"enable_zmq": False},
            "right_wrist_camera": {"enable_zmq": False},
        }

        class FakeRequester:
            instance = None

            def __init__(self, *_args, **_kwargs):
                type(self).instance = self
                self.closed = False

            def request(self):
                return config

            def close(self):
                self.closed = True

        class FailingManager:
            def __init__(self):
                self.closed = False

            def subscribe(self, *_args, **_kwargs):
                raise ConnectionError("subscriber startup failed")

            def close(self):
                self.closed = True

        manager = FailingManager()
        with (
            mock.patch.object(image_client_module, "ZMQ_Requester", FakeRequester),
            mock.patch.object(
                image_client_module.ZMQ_SubscriberManager,
                "get_instance",
                return_value=manager,
            ),
        ):
            with self.assertRaisesRegex(ConnectionError, "subscriber startup failed"):
                image_client_module.ImageClient(host="camera", request_rgbd=True)

        self.assertTrue(FakeRequester.instance.closed)
        self.assertTrue(manager.closed)

    def test_rgbd_camera_encodes_atomic_aligned_depth_and_requires_a_new_sequence(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 625, dtype=np.uint16)
        color_ok, color_jpeg = cv2.imencode(".jpg", bgr)
        depth_ok, depth_png = cv2.imencode(".png", depth)
        self.assertTrue(color_ok and depth_ok)
        frame = TeleRgbdFrame(
            sequence=9,
            server_capture_monotonic_ns=1,
            received_monotonic_ns=time.monotonic_ns(),
            color_jpeg=color_jpeg.tobytes(),
            aligned_depth_png=depth_png.tobytes(),
        )

        camera = object.__new__(TeleimagerCamera)
        camera._requires_depth = True
        camera._depth_encoding = DepthEncodingContract(near_m=0.25, far_m=1.0)
        camera._depth_scale_m_per_unit = 0.001
        camera._last_rgbd_sequence = None
        camera._reported_stream_fps = False
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)

        class FakeRgbdClient:
            def __init__(self):
                self.fps_calls = 0

            def get_head_rgbd_frame(self):
                return frame

            def get_head_rgbd_fps(self):
                self.fps_calls += 1
                return 0.0 if self.fps_calls == 1 else 30.0

        camera._client = FakeRgbdClient()
        camera.config = {
            "head_camera": {
                "image_shape": [480, 640],
                "binocular": False,
            }
        }

        images = camera.read(timeout_s=0.05)

        self.assertIsInstance(images, CameraImages)
        self.assertEqual(images.sequence, 9)
        self.assertEqual(camera._client.fps_calls, 2)
        self.assertTrue(camera._reported_stream_fps)
        expected = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=0.001,
            near_m=0.25,
            far_m=1.0,
        )
        np.testing.assert_array_equal(images.depth_gray, expected)
        with self.assertRaisesRegex(TimeoutError, "not new"):
            camera.read(timeout_s=0.015)

    def test_rgbd_camera_encodes_atomic_aligned_depth_as_surface_normals(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 625, dtype=np.uint16)
        color_ok, color_jpeg = cv2.imencode(".jpg", bgr)
        depth_ok, depth_png = cv2.imencode(".png", depth)
        self.assertTrue(color_ok and depth_ok)
        frame = TeleRgbdFrame(
            sequence=12,
            server_capture_monotonic_ns=1,
            received_monotonic_ns=time.monotonic_ns(),
            color_jpeg=color_jpeg.tobytes(),
            aligned_depth_png=depth_png.tobytes(),
        )

        camera = object.__new__(TeleimagerCamera)
        camera._requires_depth = True
        camera._depth_encoding = None
        camera._surface_normal_encoding = SurfaceNormalEncodingContract(
            intrinsics=DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
            max_neighbor_depth_delta_m=0.05,
        )
        camera._depth_scale_m_per_unit = 0.001
        camera._last_rgbd_sequence = None
        camera._reported_stream_fps = False
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)
        camera._client = SimpleNamespace(
            get_head_rgbd_frame=lambda: frame,
            get_head_rgbd_fps=lambda: 30.0,
        )
        camera.config = {
            "head_camera": {
                "image_shape": [480, 640],
                "binocular": False,
            }
        }

        images = camera.read(timeout_s=0.05)

        self.assertIsNone(images.depth_gray)
        expected = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
            max_neighbor_depth_delta_m=0.05,
        )
        np.testing.assert_array_equal(images.surface_normals, expected)

    def test_rgbd_camera_fails_closed_on_server_sequence_regression(self):
        frame = TeleRgbdFrame(
            sequence=3,
            server_capture_monotonic_ns=1,
            received_monotonic_ns=time.monotonic_ns(),
            color_jpeg=b"unused",
            aligned_depth_png=b"unused",
        )
        camera = object.__new__(TeleimagerCamera)
        camera._requires_depth = True
        camera._last_rgbd_sequence = 8
        camera._reported_stream_fps = False
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)
        camera._client = SimpleNamespace(
            get_head_rgbd_frame=lambda: frame,
            get_head_rgbd_fps=lambda: 30.0,
        )

        with self.assertRaisesRegex(DeploymentError, "sequence regressed"):
            camera.read(timeout_s=0.05)

    def test_camera_config_request_refuses_local_fallback_on_live_timeout(self):
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.closed = False

            def setsockopt(self, *_args):
                pass

            def connect(self, _endpoint):
                pass

            def send(self, message):
                self.sent = message

            def poll(self, _timeout_ms, _event):
                return 0

            def close(self, linger=None):
                self.closed = True
                self.linger = linger

        class FakeContext:
            def __init__(self, socket):
                self.socket_instance = socket
                self.terminated = False

            def socket(self, _kind):
                return self.socket_instance

            def term(self):
                self.terminated = True

        socket = FakeSocket()
        context = FakeContext(socket)
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.zmq.Context",
            return_value=context,
        ):
            with self.assertRaisesRegex(DeploymentError, "refusing.*local YAML fallback"):
                request_live_camera_config("camera-host", timeout_s=0.001)

        self.assertEqual(socket.sent, b"GET_DATA")
        self.assertTrue(socket.closed)
        self.assertTrue(context.terminated)

    def test_camera_rejects_image_client_config_that_was_not_live(self):
        live_config = {
            "head_camera": {
                "enable_zmq": True,
                "fps": 30,
                "image_shape": [480, 640],
                "zmq_port": 5555,
            }
        }
        stale_config = {
            "head_camera": {
                "enable_zmq": True,
                "fps": 30,
                "image_shape": [720, 1280],
                "zmq_port": 5555,
            }
        }

        class FakeImageClient:
            instance = None

            def __init__(self, **_kwargs):
                type(self).instance = self
                self.closed = False
                self._requester = None

            def get_cam_config(self):
                return stale_config

            def close(self):
                self.closed = True

        module_name = "unitree_lerobot.eval_robot.image_server.image_client"
        fake_module = ModuleType(module_name)
        fake_module.ImageClient = FakeImageClient
        with (
            mock.patch.dict(sys.modules, {module_name: fake_module}),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.request_live_camera_config",
                return_value=live_config,
            ),
        ):
            with self.assertRaisesRegex(DeploymentError, "differs.*live server"):
                TeleimagerColourCamera("camera-host")

        self.assertTrue(FakeImageClient.instance.closed)

    def test_rgbd_camera_rejects_missing_atomic_port_before_constructing_client(self):
        live_config = {
            "head_camera": {
                "enable_zmq": True,
                "fps": 30,
                "image_shape": [480, 640],
                "type": "realsense",
                "enable_depth": True,
                "binocular": False,
                "rgbd_protocol": RGBD_PROTOCOL,
                "depth_scale_m_per_unit": 0.001,
            }
        }

        class FakeImageClient:
            calls = 0

            def __init__(self, **_kwargs):
                type(self).calls += 1

        module_name = "unitree_lerobot.eval_robot.image_server.image_client"
        fake_module = ModuleType(module_name)
        fake_module.ImageClient = FakeImageClient
        with (
            mock.patch.dict(sys.modules, {module_name: fake_module}),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.request_live_camera_config",
                return_value=live_config,
            ),
        ):
            with self.assertRaisesRegex(DeploymentError, "no valid rgbd_zmq_port"):
                TeleimagerCamera("camera-host", DepthEncodingContract(near_m=0.25, far_m=1.0))

        self.assertEqual(FakeImageClient.calls, 0)

    def test_camera_waits_for_a_live_rolling_stream_rate(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        frames = [
            SimpleNamespace(jpg=encoded.tobytes(), fps=0.0),
            SimpleNamespace(jpg=encoded.tobytes(), fps=30.0),
        ]

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def get_head_frame(self):
                frame = frames[min(self.calls, len(frames) - 1)]
                self.calls += 1
                return frame

        camera = object.__new__(TeleimagerColourCamera)
        camera._client = FakeClient()
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)
        camera._reported_stream_fps = False
        camera.config = {
            "head_camera": {
                "image_shape": [480, 640],
                "binocular": False,
            }
        }

        rgb = camera.read_rgb(timeout_s=0.1)

        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(camera._client.calls, 2)
        self.assertTrue(camera._reported_stream_fps)

    def test_camera_rejects_a_stopped_subscriber_before_using_cached_data(self):
        camera = object.__new__(TeleimagerColourCamera)
        camera._client = SimpleNamespace(get_head_frame=mock.Mock())
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: False)
        camera._reported_stream_fps = False
        camera.config = {}

        with self.assertRaisesRegex(DeploymentError, "subscriber stopped"):
            camera.read_rgb(timeout_s=0.1)

        camera._client.get_head_frame.assert_not_called()

    def test_camera_uses_configured_binocular_crop_and_converts_to_rgb(self):
        bgr = np.zeros((480, 1280, 3), dtype=np.uint8)
        bgr[:, :640] = [1, 2, 3]
        bgr[:, 640:] = [10, 20, 30]
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.assertTrue(ok)
        frame = SimpleNamespace(jpg=encoded.tobytes())
        config = {"head_camera": {"image_shape": [480, 1280], "binocular": True}}

        rgb = decode_color_0_rgb(frame, config)

        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertTrue(rgb.flags.c_contiguous)
        np.testing.assert_allclose(rgb[0, 0], [3, 2, 1], atol=3)

    def test_camera_rejects_a_self_consistent_wrong_training_shape(self):
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)

        with self.assertRaisesRegex(DeploymentError, "training contract"):
            decode_color_0_rgb(
                SimpleNamespace(jpg=encoded.tobytes()),
                {"head_camera": {"image_shape": [720, 1280], "binocular": False}},
            )

    def test_sim_right_hand_permutation_round_trips_dataset_order(self):
        dataset_order = np.array(["thumb0", "thumb1", "thumb2", "index0", "index1", "middle0", "middle1"])
        expected_simulator_order = np.array(["thumb0", "thumb1", "thumb2", "middle0", "middle1", "index0", "index1"])
        simulator_order = dataset_order[SIM_RIGHT_HAND_PERMUTATION]

        np.testing.assert_array_equal(simulator_order, expected_simulator_order)
        np.testing.assert_array_equal(simulator_order[SIM_RIGHT_HAND_PERMUTATION], dataset_order)

    def test_real_actuation_requires_interface_and_local_policy(self):
        base = dict(
            execution_horizon=8,
            max_chunks=1,
            actuate=True,
            sim=False,
            network_interface=None,
            policy_host="127.0.0.1",
            allow_unqualified_real=False,
        )
        with self.assertRaisesRegex(DeploymentError, "network-interface"):
            validate_args(argparse.Namespace(**base))

        base["network_interface"] = "eth0"
        with self.assertRaisesRegex(DeploymentError, "fail-closed"):
            validate_args(argparse.Namespace(**base))

        base["allow_unqualified_real"] = True
        base["policy_host"] = "192.168.1.5"
        with self.assertRaisesRegex(DeploymentError, "loopback"):
            validate_args(argparse.Namespace(**base))

    def test_sim_actuation_requires_an_isolated_auto_dds_network(self):
        base = dict(
            execution_horizon=8,
            max_chunks=1,
            actuate=True,
            sim=True,
            network_interface=None,
            policy_host="127.0.0.1",
            allow_unqualified_real=False,
            confirm_sim_network_isolated=False,
        )
        with self.assertRaisesRegex(DeploymentError, "not a physical safety boundary"):
            validate_args(argparse.Namespace(**base))

        base["confirm_sim_network_isolated"] = True
        validate_args(argparse.Namespace(**base))

        base["network_interface"] = "lo"
        with self.assertRaisesRegex(DeploymentError, "may not match"):
            validate_args(argparse.Namespace(**base))

    def test_actuator_backend_does_not_publish_before_arm_command(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            self.assertEqual(FakeBackend.instance.publishes, 0)
            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(FakeBackend.instance.publishes, 0)
        self.assertTrue(FakeBackend.instance.closed)

    def test_real_authority_ramp_writes_hands_once_and_acks_after_final_arm_weight(self):
        class SlowAuthorityBackend(FakeBackend):
            instance = None

            def __init__(self, simulation, network_interface):
                super().__init__(simulation, network_interface)
                type(self).instance = self
                self.hand_writes = 0
                self.arm_writes = 0
                self.weights = []

            def _publish_hands(self):
                self.hand_writes += 1

            def _publish_arm(self):
                self.arm_writes += 1
                # Deliberately slower than the patched 1 kHz schedule.  The
                # ramp must skip missed ticks instead of accumulating them.
                time.sleep(0.004)

            def set_weight(self, weight):
                self.weights.append(float(weight))

        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}._G1Dex3CommandBackend", SlowAuthorityBackend),
            mock.patch(f"{module}.ARM_AUTHORITY_RAMP_S", 0.03),
            mock.patch(f"{module}.PUBLISH_HZ", 1_000.0),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(False, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")

            backend = SlowAuthorityBackend.instance
            self.assertEqual(backend.hand_writes, 1)
            self.assertGreater(backend.arm_writes, 1)
            self.assertLess(backend.arm_writes, 20)
            self.assertEqual(backend.weights[0], 0.0)
            self.assertEqual(backend.weights[-1], 1.0)
            self.assertTrue(all(a < b for a, b in zip(backend.weights, backend.weights[1:])))

            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(backend.released)
        self.assertTrue(backend.closed)

    def test_stop_during_authority_ramp_is_orderly_cancellation_not_heartbeat_fault(self):
        class CancellingBackend(FakeBackend):
            def __init__(self, stop_event):
                super().__init__(False, None)
                self.stop_event = stop_event
                self.hand_writes = 0
                self.arm_writes = 0

            def _publish_hands(self):
                self.hand_writes += 1

            def _publish_arm(self):
                self.arm_writes += 1
                if self.arm_writes == 2:
                    self.stop_event.set()

        stop = threading.Event()
        backend = CancellingBackend(stop)

        completed = _ramp_real_arm_authority(
            backend,
            stop,
            FakeHeartbeat(time.monotonic()),
        )

        self.assertFalse(completed)
        self.assertEqual(backend.hand_writes, 1)
        self.assertEqual(backend.arm_writes, 2)

    def test_actuator_rejects_policy_chunks_until_initialization_completes(self):
        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = False
        actuator._command_queue = queue.Queue(maxsize=1)

        with self.assertRaisesRegex(DeploymentError, "complete initialization"):
            actuator.submit(
                SimpleNamespace(
                    arm=np.zeros((1, 14)),
                    left_hand=np.zeros((1, 7)),
                    right_hand=np.zeros((1, 7)),
                )
            )

        self.assertTrue(actuator._command_queue.empty())

    def test_parent_hold_is_acknowledged_idempotent_and_forbidden_during_a_chunk(self):
        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = True
        actuator._holding = False
        actuator._chunk_in_flight = False
        actuator._warm_started = True
        actuator._sequence = 7
        actuator._command_queue = queue.Queue(maxsize=1)

        with (
            mock.patch.object(actuator, "assert_healthy") as assert_healthy,
            mock.patch.object(actuator, "heartbeat") as heartbeat,
            mock.patch.object(actuator, "_wait_status") as wait_status,
        ):
            actuator.hold()
            kind, created_at = actuator._command_queue.get_nowait()
            self.assertEqual(kind, "hold")
            self.assertLessEqual(time.monotonic() - created_at, 0.1)
            assert_healthy.assert_called_once_with()
            heartbeat.assert_called_once_with()
            wait_status.assert_called_once_with("holding", timeout_s=1.0, payload=7)
            self.assertTrue(actuator._holding)
            self.assertFalse(actuator._warm_started)

            # Calling HOLD while already held must not enqueue another command.
            actuator.hold()
            self.assertTrue(actuator._command_queue.empty())
            wait_status.assert_called_once()

            actuator._holding = False
            actuator._chunk_in_flight = True
            with self.assertRaisesRegex(DeploymentError, "current action chunk completes"):
                actuator.hold()
            self.assertTrue(actuator._command_queue.empty())

    def test_parent_immediate_stop_is_nonblocking_and_release_has_priority(self):
        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = True
        actuator._closed = False
        actuator._holding = False
        actuator._warm_started = True
        actuator._chunk_in_flight = True
        actuator._pending_sequence = 7
        actuator._sequence = 7
        actuator._rtc_active = False
        actuator._urgent_hold_event = threading.Event()
        actuator._stop_event = threading.Event()
        actuator._control_lock = threading.Lock()
        actuator._command_queue = queue.Queue(maxsize=1)
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_release_requested = threading.Event()

        actuator.request_immediate_hold()
        self.assertEqual(actuator.immediate_control_requested(), "hold")
        self.assertTrue(actuator._urgent_hold_event.is_set())

        with mock.patch.object(actuator, "_wait_status", return_value=7) as wait_status:
            actuator.finish_immediate_hold()
        wait_status.assert_called_once_with("urgent_holding", timeout_s=1.0)
        self.assertEqual(actuator._command_queue.get_nowait(), ("urgent_hold_barrier",))
        self.assertIsNone(actuator.immediate_control_requested())
        self.assertTrue(actuator._holding)
        self.assertFalse(actuator._warm_started)
        self.assertFalse(actuator._chunk_in_flight)
        self.assertIsNone(actuator._pending_sequence)

        actuator.request_immediate_hold()
        actuator.request_immediate_release()
        self.assertEqual(actuator.immediate_control_requested(), "release")
        self.assertTrue(actuator._stop_event.is_set())

    def test_parent_immediate_stop_fence_ignores_obsolete_rtc_terminal_status(self):
        class _AliveProcess:
            @staticmethod
            def is_alive() -> bool:
                return True

        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = True
        actuator._closed = False
        actuator._holding = False
        actuator._warm_started = True
        actuator._chunk_in_flight = True
        actuator._pending_sequence = 7
        actuator._sequence = 7
        actuator._rtc_active = True
        actuator._rtc_terminal = ("hold", {"obsolete": True})
        actuator._urgent_hold_event = threading.Event()
        actuator._stop_event = threading.Event()
        actuator._control_lock = threading.Lock()
        actuator._command_queue = queue.Queue(maxsize=1)
        actuator._status_queue = queue.Queue(maxsize=32)
        actuator._status_queue.put(("rtc_rejected", {"sequence": 7, "obsolete": True}))
        actuator._status_queue.put(("urgent_holding", 7))
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_hold_requested.set()
        actuator._immediate_release_requested = threading.Event()
        actuator._stopped_acknowledged = False
        actuator._process = _AliveProcess()
        actuator._heartbeat = FakeHeartbeat(time.monotonic())

        self.assertEqual(actuator.finish_immediate_hold(), "hold")
        self.assertEqual(actuator._command_queue.get_nowait(), ("urgent_hold_barrier",))
        self.assertIsNone(actuator._rtc_terminal)
        self.assertTrue(actuator._holding)
        self.assertFalse(actuator._rtc_active)
        self.assertFalse(actuator._chunk_in_flight)
        self.assertIsNone(actuator._pending_sequence)

    def test_release_during_status_wait_caches_consumed_stopped_ack_for_close(self):
        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_release_requested = threading.Event()
        actuator._stopped_acknowledged = False
        actuator._heartbeat = FakeHeartbeat(time.monotonic())

        class _ReleaseThenStopQueue:
            @staticmethod
            def get(*, timeout):
                del timeout
                actuator._immediate_release_requested.set()
                return "stopped", None

        actuator._status_queue = _ReleaseThenStopQueue()

        with self.assertRaises(ImmediateControlEvent) as raised:
            actuator._wait_status("armed", timeout_s=1.0)

        self.assertEqual(raised.exception.action, "release")
        # close() reads this exact cache before draining any remaining statuses.
        self.assertTrue(actuator._stopped_acknowledged)

    def test_child_reports_initializing_then_initialized_before_accepting_actions(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
                FakeBackend,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_CONVERGENCE_DWELL_S",
                0.01,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_START_DWELL_S",
                0.0,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_MIN_DISTINCT_SAMPLES",
                1,
            ),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(
                (
                    "initialize",
                    time.monotonic(),
                    load_initialization_spec("measured", task_name="pick-red-cup"),
                )
            )
            kind, details = statuses.get(timeout=1.0)
            self.assertEqual(kind, "initializing")
            self.assertEqual(details["mode"], "measured")
            self.assertGreaterEqual(details["steps"], 1)
            self.assertEqual(statuses.get(timeout=1.0), ("initialized", "measured"))

            warm_start = InitializationSpec(
                mode="pose-file",
                label="first policy target",
                arm=np.zeros(14),
                left_hand=np.zeros(7),
                right_hand=np.zeros(7),
            )
            commands.put(("warm_start", time.monotonic(), warm_start))
            kind, details = statuses.get(timeout=1.0)
            self.assertEqual(kind, "warm_starting")
            self.assertGreaterEqual(details["steps"], 1)
            self.assertEqual(statuses.get(timeout=1.0), ("warm_started", "first policy target"))

            commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    np.zeros((1, 14)),
                    np.zeros((1, 7)),
                    np.zeros((1, 7)),
                )
            )
            self.assertEqual(statuses.get(timeout=1.0), ("completed", 1))
            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_child_hold_captures_measured_pose_and_gates_repeated_warm_start(self):
        class MeasuredPoseBackend(FakeBackend):
            instance = None

            def __init__(self, simulation, network_interface):
                super().__init__(simulation, network_interface)
                type(self).instance = self
                self.measured_arm = None
                self.measured_left = None
                self.measured_right = None

            def state(self):
                arm = self._arm_target if self.measured_arm is None else self.measured_arm
                left = self._left_target if self.measured_left is None else self.measured_left
                right = self._right_target if self.measured_right is None else self.measured_right
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=np.asarray(arm).copy(),
                    arm_dq=np.zeros(14),
                    left_hand=np.asarray(left).copy(),
                    right_hand=np.asarray(right).copy(),
                )

        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}._G1Dex3CommandBackend", MeasuredPoseBackend),
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(
                (
                    "initialize",
                    time.monotonic(),
                    load_initialization_spec("measured", task_name="pick-red-cup"),
                )
            )
            self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")
            self.assertEqual(statuses.get(timeout=1.0), ("initialized", "measured"))

            first_target = InitializationSpec(
                mode="pose-file",
                label="first goal target",
                arm=np.zeros(14),
                left_hand=np.zeros(7),
                right_hand=np.zeros(7),
            )
            commands.put(("warm_start", time.monotonic(), first_target))
            self.assertEqual(statuses.get(timeout=1.0)[0], "warm_starting")
            self.assertEqual(statuses.get(timeout=1.0), ("warm_started", "first goal target"))

            backend = MeasuredPoseBackend.instance
            backend.measured_arm = np.full(14, 0.012)
            backend.measured_left = np.array([0.021, 0.021, 0.021, -0.021, -0.021, -0.021, -0.021])
            backend.measured_right = np.array([0.032, 0.032, -0.032, 0.032, 0.032, 0.032, 0.032])
            commands.put(("hold", time.monotonic()))
            self.assertEqual(statuses.get(timeout=1.0), ("holding", 0))
            np.testing.assert_array_equal(backend._arm_target, backend.measured_arm)
            np.testing.assert_array_equal(backend._left_target, backend.measured_left)
            np.testing.assert_array_equal(backend._right_target, backend.measured_right)

            second_target = InitializationSpec(
                mode="pose-file",
                label="second goal target",
                arm=backend.measured_arm.copy(),
                left_hand=backend.measured_left.copy(),
                right_hand=backend.measured_right.copy(),
            )
            commands.put(("warm_start", time.monotonic(), second_target))
            self.assertEqual(statuses.get(timeout=1.0)[0], "warm_starting")
            self.assertEqual(statuses.get(timeout=1.0), ("warm_started", "second goal target"))

            # A third warm-start without another acknowledged HOLD is a child
            # fault, even though no policy chunk happens to be active.
            commands.put(("warm_start", time.monotonic(), second_target))
            kind, detail = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("requires an acknowledged hold", detail)
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(backend.released)
        self.assertTrue(backend.closed)

    def test_child_builds_initialization_path_from_held_command_not_offset_measurement(self):
        class TrackingErrorBackend(FakeBackend):
            instance = None

            def __init__(self, simulation, network_interface):
                super().__init__(simulation, network_interface)
                type(self).instance = self
                self.initialization_targets = []

            def prepare_measured_hold(self):
                self._arm_target = np.full(14, 0.1)
                self._left_target = np.zeros(7)
                self._right_target = np.zeros(7)
                return self.state()

            def state(self):
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    # Deliberate 0.02-rad tracking error: safely within the
                    # dwell tolerance, but far above one init command step.
                    arm=self._arm_target + 0.02,
                    arm_dq=np.zeros(14),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

            def set_target(self, arm, left, right):
                super().set_target(arm, left, right)
                self.initialization_targets.append(np.asarray(arm).copy())

        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        initialization = InitializationSpec(
            mode="pose-file",
            label="held-target origin test",
            arm=np.full(14, 0.2),
            left_hand=None,
            right_hand=None,
        )
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}._G1Dex3CommandBackend", TrackingErrorBackend),
            mock.patch(f"{module}.PUBLISH_HZ", 10_000.0),
            mock.patch(f"{module}.INITIALIZATION_MIN_MOVE_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(("initialize", time.monotonic(), initialization))
            self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")
            self.assertEqual(statuses.get(timeout=1.0), ("initialized", "pose-file"))
            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        first = TrackingErrorBackend.instance.initialization_targets[0]
        held = np.full(14, 0.1)
        self.assertLessEqual(
            float(np.max(np.abs(first - held))),
            INITIALIZATION_MAX_ARM_STEP_RAD + 1e-12,
        )
        self.assertTrue(TrackingErrorBackend.instance.released)
        self.assertTrue(TrackingErrorBackend.instance.closed)

    def test_child_faults_and_releases_if_policy_chunk_arrives_before_initialization(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(("chunk",))
            kind, message = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("before initialization", message.lower())
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_stale_initialization_command_faults_and_releases(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(
                (
                    "initialize",
                    time.monotonic() - 1.0,
                    load_initialization_spec("measured", task_name="pick-red-cup"),
                )
            )
            kind, message = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("expired", message.lower())
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_stop_during_initialization_cancels_without_jumping_to_endpoint(self):
        class TrackingBackend(FakeBackend):
            instance = None

            def __init__(self, simulation, network_interface):
                super().__init__(simulation, network_interface)
                type(self).instance = self
                self.initialization_arm_targets = []

            def state(self):
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=self._arm_target.copy(),
                    arm_dq=np.zeros(14),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

            def set_target(self, arm, left, right):
                super().set_target(arm, left, right)
                self.initialization_arm_targets.append(np.asarray(arm).copy())

        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        target = np.full(14, 0.2)
        initialization = InitializationSpec(
            mode="pose-file",
            label="nontrivial test path",
            arm=target,
            left_hand=None,
            right_hand=None,
        )
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
                TrackingBackend,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_START_DWELL_S",
                0.0,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_MIN_DISTINCT_SAMPLES",
                1,
            ),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(("initialize", time.monotonic(), initialization))
            self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                targets = TrackingBackend.instance.initialization_arm_targets
                if targets and float(np.max(np.abs(targets[-1]))) > 0.0:
                    break
                time.sleep(0.001)
            else:
                self.fail("Initialization did not begin moving")
            stop.set()
            self.assertEqual(
                statuses.get(timeout=1.0),
                ("initialization_cancelled", "pose-file"),
            )
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        emitted = np.asarray(TrackingBackend.instance.initialization_arm_targets)
        self.assertGreater(len(emitted), 0)
        self.assertLess(float(np.max(emitted[-1])), 0.2)
        steps = np.diff(np.vstack((np.zeros(14), emitted)), axis=0)
        self.assertLessEqual(float(np.max(np.abs(steps))), INITIALIZATION_MAX_ARM_STEP_RAD + 1e-12)
        self.assertTrue(TrackingBackend.instance.released)
        self.assertTrue(TrackingBackend.instance.closed)

    def test_duplicate_initialization_after_success_faults_and_releases(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        initialization = load_initialization_spec("measured", task_name="pick-red-cup")
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
                FakeBackend,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_CONVERGENCE_DWELL_S",
                0.0,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_START_DWELL_S",
                0.0,
            ),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_MIN_DISTINCT_SAMPLES",
                1,
            ),
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            commands.put(("initialize", time.monotonic(), initialization))
            self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")
            self.assertEqual(statuses.get(timeout=1.0), ("initialized", "measured"))
            commands.put(("initialize", time.monotonic(), initialization))
            kind, message = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("unexpected", message.lower())
            self.assertIn("initialize", message.lower())
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_child_rejects_malformed_gapped_and_future_action_chunk_protocol(self):
        arm = np.zeros((1, 14))
        hand = np.zeros((1, 7))
        cases = (
            ("malformed", lambda: ("chunk",), "malformed"),
            (
                "gapped sequence",
                lambda: ("chunk", 2, time.monotonic(), arm, hand, hand),
                "out-of-order",
            ),
            (
                "future timestamp",
                lambda: ("chunk", 1, time.monotonic() + 1.0, arm, hand, hand),
                "expired",
            ),
        )
        initialization = load_initialization_spec("measured", task_name="pick-red-cup")
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}._G1Dex3CommandBackend", FakeBackend),
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.0),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
        ):
            for name, make_command, expected in cases:
                with self.subTest(name=name):
                    commands = queue.Queue(maxsize=1)
                    statuses = queue.Queue(maxsize=32)
                    stop = threading.Event()
                    heartbeat = FakeHeartbeat(time.monotonic())
                    thread = threading.Thread(
                        target=_actuator_main,
                        args=(True, None, commands, statuses, stop, heartbeat),
                        daemon=True,
                    )
                    thread.start()
                    self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
                    commands.put(("arm",))
                    self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
                    commands.put(("initialize", time.monotonic(), initialization))
                    self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")
                    self.assertEqual(
                        statuses.get(timeout=1.0),
                        ("initialized", "measured"),
                    )
                    commands.put(make_command())
                    kind, message = statuses.get(timeout=1.0)
                    self.assertEqual(kind, "fault")
                    self.assertIn(expected, message.lower())
                    thread.join(timeout=1.0)
                    self.assertFalse(thread.is_alive())
                    self.assertTrue(FakeBackend.instance.released)
                    self.assertTrue(FakeBackend.instance.closed)

    def test_initialization_does_not_report_convergence_while_arm_is_moving(self):
        class MovingBackend(FakeBackend):
            def state(self):
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=self._arm_target.copy(),
                    arm_dq=np.full(14, 0.2),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

        backend = MovingBackend(False, None)
        state = backend.state()
        chunk = build_initialization_chunk(
            state,
            load_initialization_spec("measured", task_name="pick-red-cup"),
        )
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.INITIALIZATION_CONVERGENCE_TIMEOUT_S",
            0.03,
        ):
            with self.assertRaisesRegex(DeploymentError, "did not converge") as raised:
                _execute_initialization(
                    backend,
                    chunk,
                    threading.Event(),
                    FakeHeartbeat(time.monotonic()),
                    float("inf"),
                )
        message = str(raised.exception)
        self.assertIn("kLeftShoulderPitch", message)
        self.assertIn("measured minus target=+0.000 rad", message)
        self.assertIn("max arm dq=0.200 rad/s", message)

    def test_initialization_allows_real_arm_to_settle_after_three_seconds(self):
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

        class AdvancingEvent:
            def is_set(self):
                return False

            def wait(self, timeout):
                clock.now += timeout
                heartbeat.value = clock.now

        class SlowlySettlingBackend(FakeBackend):
            def __init__(self):
                super().__init__(False, None)

            def state(self):
                settling = clock.now < 3.2
                return RobotState(
                    captured_at=clock.now,
                    mode_machine=0,
                    arm=self._arm_target + (0.064 if settling else 0.0),
                    arm_dq=np.full(14, 0.02 if settling else 0.0),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

        clock = Clock()
        heartbeat = FakeHeartbeat(clock.now)
        backend = SlowlySettlingBackend()
        chunk = ActionChunk(
            arm=np.zeros((1, 14)),
            left_hand=np.zeros((1, 7)),
            right_hand=np.zeros((1, 7)),
        )
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.time.monotonic", clock.monotonic),
            mock.patch(f"{module}.PUBLISH_HZ", 10.0),
            mock.patch(f"{module}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.2),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 2),
        ):
            initialized = _execute_initialization(
                backend,
                chunk,
                AdvancingEvent(),
                heartbeat,
                float("inf"),
            )

        self.assertTrue(initialized)
        self.assertGreater(clock.now, 3.0)
        self.assertLess(clock.now, 10.0)

    def test_initialization_start_dwell_defers_until_held_state_is_stationary(self):
        class SettlingBackend:
            def __init__(self):
                self.simulation = False
                self.calls = 0
                self.publishes = 0
                self._arm_target = np.zeros(14)
                self._left_target = np.zeros(7)
                self._right_target = np.zeros(7)

            def state(self):
                self.calls += 1
                moving = self.calls <= 3
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=self._arm_target.copy(),
                    arm_dq=np.full(14, 0.2 if moving else 0.0),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

            def publish(self):
                self.publishes += 1

        backend = SettlingBackend()
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.01),
            mock.patch(f"{module}.INITIALIZATION_START_TIMEOUT_S", 0.2),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 2),
        ):
            state = _wait_for_initialization_start(
                backend,
                threading.Event(),
                FakeHeartbeat(time.monotonic()),
            )

        self.assertIsNotNone(state)
        self.assertGreaterEqual(backend.calls, 5)
        self.assertGreaterEqual(backend.publishes, 4)
        self.assertLessEqual(float(np.max(np.abs(state.arm_dq))), 0.1)

    def test_initialization_start_dwell_rejects_persistently_moving_state(self):
        class MovingBackend:
            def __init__(self):
                self.simulation = False
                self.publishes = 0
                self._arm_target = np.zeros(14)
                self._left_target = np.zeros(7)
                self._right_target = np.zeros(7)

            def state(self):
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=self._arm_target.copy(),
                    arm_dq=np.full(14, 0.2),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

            def publish(self):
                self.publishes += 1

        backend = MovingBackend()
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.005),
            mock.patch(f"{module}.INITIALIZATION_START_TIMEOUT_S", 0.03),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
            self.assertRaisesRegex(DeploymentError, "did not become stationary"),
        ):
            _wait_for_initialization_start(
                backend,
                threading.Event(),
                FakeHeartbeat(time.monotonic()),
            )

        self.assertGreater(backend.publishes, 0)

    def test_initialization_start_dwell_rejects_position_drift_inside_tracking_tolerance(self):
        class DriftingBackend:
            def __init__(self):
                self.simulation = False
                self.calls = 0
                self.publishes = 0
                self._arm_target = np.zeros(14)
                self._left_target = np.zeros(7)
                self._right_target = np.zeros(7)

            def state(self):
                self.calls += 1
                # Each sample remains inside the 0.05 rad initialization
                # tolerance, but adjacent held poses differ by more than the
                # independent 0.02 rad stability ceiling.
                offset = 0.03 if self.calls % 2 == 0 else 0.0
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=0,
                    arm=self._arm_target + offset,
                    arm_dq=np.zeros(14),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                )

            def publish(self):
                self.publishes += 1

        backend = DriftingBackend()
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.INITIALIZATION_START_DWELL_S", 0.02),
            mock.patch(f"{module}.INITIALIZATION_START_TIMEOUT_S", 0.04),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
            self.assertRaisesRegex(DeploymentError, "did not become stationary"),
        ):
            _wait_for_initialization_start(
                backend,
                threading.Event(),
                FakeHeartbeat(time.monotonic()),
            )

        self.assertGreaterEqual(backend.calls, 3)
        self.assertGreater(backend.publishes, 0)

    def test_dex3_cleanup_uses_unitree_stop_motors_command(self):
        class Publisher:
            def __init__(self):
                self.writes = []

            def Write(self, message, timeout=None):
                self.writes.append((message, timeout))
                return True

        backend = object.__new__(_G1Dex3CommandBackend)
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._left_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._right_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._left_publisher = Publisher()
        backend._right_publisher = Publisher()

        backend._stop_hands()

        for message in (backend._left_message, backend._right_message):
            for index, command in enumerate(message.motor_cmd):
                self.assertEqual(command.mode, 0x90 | index)
                self.assertEqual(
                    (command.q, command.dq, command.tau, command.kp, command.kd),
                    (0.0, 0.0, 0.0, 0.0, 0.0),
                )
        self.assertEqual(len(backend._left_publisher.writes), 1)
        self.assertEqual(len(backend._right_publisher.writes), 1)

    def test_dex3_cleanup_attempts_right_stop_when_left_stop_fails(self):
        class Publisher:
            def __init__(self, result):
                self.result = result
                self.writes = 0

            def Write(self, _message, timeout=None):
                self.writes += 1
                return self.result

        backend = object.__new__(_G1Dex3CommandBackend)
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._left_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._right_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._left_publisher = Publisher(False)
        backend._right_publisher = Publisher(True)

        with self.assertRaisesRegex(DeploymentError, "Left Dex3 stop Write failed"):
            backend._stop_hands()

        self.assertEqual(backend._left_publisher.writes, 1)
        self.assertEqual(backend._right_publisher.writes, 1)

    def test_prepare_measured_hold_rejects_full_waist_mode_for_lock_waist_policy(self):
        unsafe_state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=5,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        backend = object.__new__(_G1Dex3CommandBackend)
        backend.simulation = False
        backend.reader = SimpleNamespace(read=lambda timeout_s: unsafe_state)
        backend._arm_message = SimpleNamespace(mode_machine=None)

        with self.assertRaisesRegex(DeploymentError, "QUALIFIED_REAL_MODE_MACHINE=6"):
            backend.prepare_measured_hold()

        self.assertIsNone(backend._arm_message.mode_machine)

    def test_prepare_measured_hold_accepts_and_writes_lock_waist_mode(self):
        class FreshReader:
            @staticmethod
            def read(timeout_s):
                del timeout_s
                return RobotState(
                    captured_at=time.monotonic(),
                    mode_machine=6,
                    arm=np.zeros(14),
                    arm_dq=np.zeros(14),
                    left_hand=np.zeros(7),
                    right_hand=np.zeros(7),
                )

        backend = object.__new__(_G1Dex3CommandBackend)
        backend.simulation = False
        backend.reader = FreshReader()
        backend._arm_message = SimpleNamespace(mode_machine=None)

        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.PREARM_STATIONARY_DWELL_S", 0.01),
            mock.patch(f"{module}.PREARM_MIN_DISTINCT_SAMPLES", 1),
        ):
            state = backend.prepare_measured_hold()

        self.assertEqual(state.mode_machine, 6)
        self.assertEqual(backend._arm_message.mode_machine, 6)

    def test_prepare_measured_hold_requires_fresh_stationary_real_state(self):
        def backend_with_state(state):
            backend = object.__new__(_G1Dex3CommandBackend)
            backend.simulation = False
            backend.reader = SimpleNamespace(read=lambda timeout_s: state)
            backend._arm_message = SimpleNamespace(mode_machine=None)
            return backend

        stale = RobotState(
            captured_at=time.monotonic() - 1.0,
            mode_machine=6,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        with self.assertRaisesRegex(DeploymentError, "PREARM_STATE_MAX_AGE_S"):
            backend_with_state(stale).prepare_measured_hold()

        moving = RobotState(
            captured_at=time.monotonic(),
            mode_machine=6,
            arm=np.zeros(14),
            arm_dq=np.full(14, 0.2),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        with self.assertRaisesRegex(DeploymentError, "PREARM_MAX_ARM_DQ_RAD_S"):
            backend_with_state(moving).prepare_measured_hold()

    def test_actuator_heartbeat_expiry_faults_and_releases(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            with heartbeat.get_lock():
                heartbeat.value = time.monotonic() - 10.0
            kind, message = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("heartbeat expired", message.lower())
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_parent_surfaces_child_release_failure(self):
        class StoppedProcess:
            pid = 123
            exitcode = 0

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._closed = False
        actuator._heartbeat = FakeHeartbeat(time.monotonic())
        actuator._stop_event = threading.Event()
        actuator._process = StoppedProcess()
        actuator._status_queue = queue.Queue()
        actuator._status_queue.put(("release_failed", "right hand stop failed"))
        actuator._status_queue.put(("stopped", None))

        with self.assertRaisesRegex(DeploymentError, "release is unconfirmed.*right hand"):
            actuator.close()

        self.assertTrue(actuator._closed)

    def test_assert_healthy_surfaces_queued_fault_even_while_child_is_alive(self):
        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._status_queue = queue.Queue()
        actuator._status_queue.put(("fault", "arm DDS write failed"))
        actuator._process = SimpleNamespace(is_alive=lambda: True)
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_release_requested = threading.Event()

        with self.assertRaisesRegex(
            DeploymentError,
            "unhealthy.*fault.*arm DDS write failed",
        ):
            actuator.assert_healthy()


if __name__ == "__main__":
    unittest.main()
