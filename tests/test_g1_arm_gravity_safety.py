"""Adversarial tests for XR-compatible G1 arm gravity feed-forward.

These tests intentionally exercise the dynamics helper without constructing a
DDS publisher.  Backend integration (including arm/hand command isolation) is
covered below with message/publisher fakes.
"""

from __future__ import annotations

import builtins
from collections import deque
import hashlib
import importlib.util
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser as build_eval_parser
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ActionChunk
from unitree_lerobot.eval_robot.robot_control.g1_arm_gravity import (
    G1_ARM_GRAVITY_JOINT_NAMES,
    G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM,
    G1_ARM_GRAVITY_URDF_SHA256,
    G1ArmGravityCompensator,
    default_g1_gravity_urdf_path,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    HAND_COMMAND_HISTORY_SIZE,
    QUALIFIED_REAL_MODE_MACHINE,
    RobotState,
    SafeG1Dex3Actuator,
    _G1Dex3CommandBackend,
    _actuator_main,
    _execute_initialization,
    _ramp_real_arm_authority,
    _set_direct_target,
)
from unitree_lerobot.eval_robot.zero_state_test import build_parser as build_zero_state_parser


PINOCCHIO_AVAILABLE = importlib.util.find_spec("pinocchio") is not None


@unittest.skipUnless(PINOCCHIO_AVAILABLE, "Pinocchio is not installed in this test environment")
class G1ArmGravityDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gravity = G1ArmGravityCompensator()

    def test_reproduces_xr_rnea_at_known_poses_in_exact_arm_joint_order(self):
        self.assertEqual(
            G1_ARM_GRAVITY_JOINT_NAMES,
            tuple(
                f"{side}_{joint}_joint"
                for side in ("left", "right")
                for joint in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ),
        )
        self.assertEqual(
            tuple(str(name) for name in self.gravity._model.names[1:]),
            G1_ARM_GRAVITY_JOINT_NAMES,
        )
        cases = (
            (
                np.zeros(14),
                np.array(
                    [
                        -3.643280932755803,
                        0.20170439449602756,
                        0.0002270312986465066,
                        -3.4123275156411452,
                        -0.038808629890088,
                        -1.2190543925387718,
                        4.3575127761847856e-05,
                        -3.643280932755803,
                        -0.20170439449602756,
                        -0.0002270312986465066,
                        -3.4123275156411452,
                        0.038808629890088,
                        -1.2190543925387718,
                        -4.3575127761847856e-05,
                    ]
                ),
            ),
            (
                np.array(
                    [
                        0.2,
                        0.4,
                        -0.3,
                        0.6,
                        -0.2,
                        0.1,
                        -0.5,
                        -0.2,
                        -0.4,
                        0.3,
                        0.6,
                        0.2,
                        -0.1,
                        0.5,
                    ]
                ),
                np.array(
                    [
                        -1.0715342994949393,
                        1.8645272177298329,
                        0.8179617336434213,
                        -1.730510270300488,
                        -0.22712819227869463,
                        -0.6047060360307491,
                        -0.19465722322866924,
                        -4.151994342938352,
                        -1.8003595293706094,
                        -1.34296381318395,
                        -2.5776719011432023,
                        0.3663587690201777,
                        -0.989021170730103,
                        -0.026281654589983813,
                    ]
                ),
            ),
        )
        for q, expected in cases:
            with self.subTest(q=q.tolist()):
                np.testing.assert_allclose(
                    self.gravity.compute(q),
                    expected,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_rejects_malformed_or_nonfinite_joint_position(self):
        for bad in (
            np.zeros(13),
            np.zeros((1, 14)),
            np.r_[np.zeros(13), np.nan],
            np.r_[np.zeros(13), np.inf],
        ):
            with self.subTest(shape=bad.shape), self.assertRaisesRegex(
                DeploymentError, "14 finite joint positions"
            ):
                self.gravity.compute(bad)

    def test_rejects_nonfinite_wrong_shape_and_over_envelope_rnea_results(self):
        invalid_results = (
            (np.zeros(13), "invalid vector"),
            (np.r_[np.zeros(13), np.nan], "invalid vector"),
            (np.r_[np.zeros(13), np.inf], "invalid vector"),
            (
                np.array([10.0 + 1e-6] + [0.0] * 13),
                "left_shoulder_pitch_joint",
            ),
            (
                np.array([0.0] * 13 + [1.0 + 1e-6]),
                "right_wrist_yaw_joint",
            ),
        )
        for result, error in invalid_results:
            with (
                self.subTest(result=result),
                mock.patch.object(self.gravity._pin, "rnea", return_value=result),
                self.assertRaisesRegex(DeploymentError, error),
            ):
                self.gravity.compute(np.zeros(14))

    def test_wraps_rnea_failure_and_returns_defensive_contiguous_copy(self):
        with (
            mock.patch.object(self.gravity._pin, "rnea", side_effect=RuntimeError("dynamics broke")),
            self.assertRaisesRegex(DeploymentError, "RNEA failed: dynamics broke"),
        ):
            self.gravity.compute(np.zeros(14))

        first = self.gravity.compute(np.zeros(14))
        first[:] = 0.0
        second = self.gravity.compute(np.zeros(14))
        self.assertTrue(second.flags.c_contiguous)
        self.assertGreater(float(np.max(np.abs(second))), 1.0)

    def test_compute_latency_is_small_relative_to_100_hz_publish_period(self):
        q = np.array([0.2, 0.4, -0.3, 0.6, -0.2, 0.1, -0.5] * 2)
        self.gravity.compute(q)
        started = time.perf_counter()
        for _ in range(1_000):
            self.gravity.compute(q)
        mean_s = (time.perf_counter() - started) / 1_000
        # A broad 2 ms ceiling is one fifth of the 10 ms DDS period and catches
        # accidental model reconstruction in the hot path without relying on
        # microbenchmark-level scheduler stability.
        self.assertLess(mean_s, 0.002, f"mean gravity compute latency was {mean_s * 1e3:.3f} ms")


class G1ArmGravityFailClosedConstructionTests(unittest.TestCase):
    def test_default_asset_is_exactly_the_reviewed_urdf(self):
        path = default_g1_gravity_urdf_path()
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), G1_ARM_GRAVITY_URDF_SHA256)

    def test_rejects_modified_urdf_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "g1.urdf"
            tampered.write_bytes(default_g1_gravity_urdf_path().read_bytes() + b"\n<!-- tampered -->\n")
            with self.assertRaisesRegex(DeploymentError, "digest is not the reviewed value"):
                G1ArmGravityCompensator(tampered)

    def test_missing_pinocchio_fails_closed_with_no_fallback(self):
        real_import = builtins.__import__

        def without_pinocchio(name, *args, **kwargs):
            if name == "pinocchio":
                raise ImportError("intentionally unavailable")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=without_pinocchio),
            self.assertRaisesRegex(DeploymentError, "Pinocchio is required.*refusing"),
        ):
            G1ArmGravityCompensator()


class G1ArmGravityBackendIntegrationTests(unittest.TestCase):
    class _Publisher:
        def __init__(self):
            self.writes = []

        def Write(self, message, timeout=None):
            self.writes.append((message, timeout))
            return True

    class _Gravity:
        def __init__(self):
            self.inputs = []

        def compute(self, q):
            q = np.asarray(q, dtype=np.float64).copy()
            self.inputs.append(q)
            return q + np.arange(14, dtype=np.float64) + 0.25

    @staticmethod
    def _message(count):
        return SimpleNamespace(
            mode_pr=None,
            mode_machine=None,
            crc=None,
            motor_cmd=[
                SimpleNamespace(mode=17, q=-99.0, dq=0.0, tau=-88.0, kp=123.0, kd=45.0)
                for _ in range(count)
            ],
        )

    def _backend(self, *, simulation=True):
        backend = object.__new__(_G1Dex3CommandBackend)
        backend.simulation = simulation
        backend._arm_indices = tuple(range(14))
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._arm_message = self._message(30)
        backend._left_message = self._message(7)
        backend._right_message = self._message(7)
        backend._arm_publisher = self._Publisher()
        backend._left_publisher = self._Publisher()
        backend._right_publisher = self._Publisher()
        backend._crc = SimpleNamespace(Crc=lambda _message: 1234)
        backend._weight = 1.0
        backend._released = False
        backend._has_published = False
        backend._authority_ramp_timing_enabled = False
        backend._last_publish_timing_ms = {}
        backend._arm_target = np.zeros(14)
        backend._left_target = np.linspace(0.1, 0.7, 7)
        backend._right_target = np.linspace(-0.7, -0.1, 7)
        backend._left_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        backend._right_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        backend._arm_gravity = self._Gravity()
        state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=QUALIFIED_REAL_MODE_MACHINE,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=backend._left_target.copy(),
            right_hand=backend._right_target.copy(),
        )
        backend.reader = SimpleNamespace(latest=lambda: state)
        return backend

    def test_eval_and_zero_state_cli_default_on_with_explicit_opt_out(self):
        eval_parser = build_eval_parser()
        self.assertTrue(eval_parser.parse_args([]).gravity_feedforward)
        self.assertTrue(eval_parser.parse_args(["--gravity-feedforward"]).gravity_feedforward)
        self.assertFalse(eval_parser.parse_args(["--no-gravity-feedforward"]).gravity_feedforward)

        zero_parser = build_zero_state_parser()
        required = ["--network-interface", "eth-test"]
        self.assertTrue(zero_parser.parse_args(required).gravity_feedforward)
        self.assertTrue(
            zero_parser.parse_args([*required, "--gravity-feedforward"]).gravity_feedforward
        )
        self.assertFalse(
            zero_parser.parse_args([*required, "--no-gravity-feedforward"]).gravity_feedforward
        )

    def test_disabled_backend_rejects_nonboolean_before_dds_initialization(self):
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.G1ArmGravityCompensator") as gravity,
            mock.patch(f"{module}.initialize_dds") as initialize_dds,
            self.assertRaisesRegex(DeploymentError, "gravity_feedforward must be a bool"),
        ):
            _G1Dex3CommandBackend(False, "eth-test", gravity_feedforward=0)
        gravity.assert_not_called()
        initialize_dds.assert_not_called()

    def test_disabled_backend_skips_model_construction_before_dds_setup(self):
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.G1ArmGravityCompensator") as gravity,
            mock.patch(
                f"{module}.initialize_dds",
                side_effect=RuntimeError("stop after gravity construction boundary"),
            ) as initialize_dds,
            self.assertRaisesRegex(RuntimeError, "construction boundary"),
        ):
            _G1Dex3CommandBackend(False, "eth-test", gravity_feedforward=False)
        gravity.assert_not_called()
        initialize_dds.assert_called_once_with(False, "eth-test")

    def test_parent_process_arguments_plumb_explicit_disabled_flag_to_child(self):
        class Context:
            @staticmethod
            def Queue(maxsize):
                return queue.Queue(maxsize=maxsize)

            @staticmethod
            def Event():
                return threading.Event()

            @staticmethod
            def Value(_typecode, value):
                return SimpleNamespace(value=value, get_lock=lambda: threading.Lock())

            @staticmethod
            def Process(*, target, args, name, kwargs=None):
                return SimpleNamespace(target=target, args=args, name=name, kwargs=kwargs or {})

        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with mock.patch(f"{module}.mp.get_context", return_value=Context()):
            actuator = SafeG1Dex3Actuator(
                False,
                "eth-test",
                gravity_feedforward=False,
            )

        self.assertIs(actuator._process.target, _actuator_main)
        self.assertIs(actuator._process.args[-1], False)
        self.assertIs(
            actuator._process.kwargs["hand_pause_generation"],
            actuator._hand_pause_generation,
        )
        self.assertFalse(actuator._gravity_feedforward)

        with mock.patch(f"{module}.mp.get_context", return_value=Context()):
            default_actuator = SafeG1Dex3Actuator(False, "eth-test")
        # The compact legacy process-argument form resolves to the child's
        # default-on parameter; explicit opt-out above must append False.
        self.assertEqual(len(default_actuator._process.args), 8)
        self.assertTrue(default_actuator._gravity_feedforward)

    def test_child_plumbs_explicit_disabled_flag_to_backend(self):
        calls = []

        class Backend:
            _supports_cleanup_phases = False

            def __init__(self, simulation, network_interface, gravity_feedforward=True):
                calls.append((simulation, network_interface, gravity_feedforward))

            def close(self):
                pass

        command_queue = queue.Queue()
        status_queue = queue.Queue()
        stop = threading.Event()
        stop.set()
        heartbeat = SimpleNamespace(value=time.monotonic(), get_lock=lambda: threading.Lock())
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with mock.patch(f"{module}._G1Dex3CommandBackend", Backend):
            _actuator_main(
                False,
                "eth-test",
                command_queue,
                status_queue,
                stop,
                heartbeat,
                gravity_feedforward=False,
            )

        self.assertEqual(calls, [(False, "eth-test", False)])
        self.assertEqual(status_queue.get_nowait()[0], "ready")

    def test_child_default_enables_backend_gravity(self):
        calls = []

        class Backend:
            _supports_cleanup_phases = False

            def __init__(self, simulation, network_interface, gravity_feedforward=True):
                calls.append((simulation, network_interface, gravity_feedforward))

            def close(self):
                pass

        status_queue = queue.Queue()
        stop = threading.Event()
        stop.set()
        heartbeat = SimpleNamespace(value=time.monotonic(), get_lock=lambda: threading.Lock())
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with mock.patch(f"{module}._G1Dex3CommandBackend", Backend):
            _actuator_main(
                False,
                "eth-test",
                queue.Queue(),
                status_queue,
                stop,
                heartbeat,
            )

        self.assertEqual(calls, [(False, "eth-test", True)])
        self.assertEqual(status_queue.get_nowait()[0], "ready")

    def test_gravity_dependency_failure_precedes_dds_initialization_and_publishers(self):
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(
                f"{module}.G1ArmGravityCompensator",
                side_effect=DeploymentError("no reviewed dynamics"),
            ),
            mock.patch(f"{module}.initialize_dds") as initialize_dds,
            self.assertRaisesRegex(DeploymentError, "no reviewed dynamics"),
        ):
            _G1Dex3CommandBackend(False, "eth-test")
        initialize_dds.assert_not_called()

    def test_every_arm_write_refreshes_tau_from_final_target_in_sdk_joint_order(self):
        backend = self._backend()
        gains_before = [(cmd.kp, cmd.kd) for cmd in backend._arm_message.motor_cmd]
        hands_before = [
            (cmd.mode, cmd.q, cmd.dq, cmd.tau, cmd.kp, cmd.kd)
            for message in (backend._left_message, backend._right_message)
            for cmd in message.motor_cmd
        ]

        targets = (
            np.linspace(-0.4, 0.5, 14),
            np.linspace(0.6, -0.3, 14),
            np.full(14, 0.125),
        )
        for target in targets:
            backend.set_target(target, backend._left_target, backend._right_target)
            backend._publish_arm()
            np.testing.assert_array_equal(
                [backend._arm_message.motor_cmd[index].q for index in backend._arm_indices],
                target,
            )
            np.testing.assert_array_equal(
                [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices],
                target + np.arange(14) + 0.25,
            )

        self.assertEqual(len(backend._arm_publisher.writes), len(targets))
        self.assertEqual(len(backend._arm_gravity.inputs), len(targets))
        for actual, expected in zip(backend._arm_gravity.inputs, targets):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            [(cmd.kp, cmd.kd) for cmd in backend._arm_message.motor_cmd],
            gains_before,
        )
        self.assertEqual(
            [
                (cmd.mode, cmd.q, cmd.dq, cmd.tau, cmd.kp, cmd.kd)
                for message in (backend._left_message, backend._right_message)
                for cmd in message.motor_cmd
            ],
            hands_before,
        )
        self.assertEqual(len(backend._left_publisher.writes), 0)
        self.assertEqual(len(backend._right_publisher.writes), 0)

    @unittest.skipUnless(PINOCCHIO_AVAILABLE, "Pinocchio is not installed in this test environment")
    def test_enabled_backend_publishes_exact_xr_rnea_at_known_pose(self):
        backend = self._backend()
        backend._gravity_feedforward = True
        backend._arm_gravity = G1ArmGravityCompensator()
        q = np.zeros(14)
        expected = np.array(
            [
                -3.643280932755803,
                0.20170439449602756,
                0.0002270312986465066,
                -3.4123275156411452,
                -0.038808629890088,
                -1.2190543925387718,
                4.3575127761847856e-05,
                -3.643280932755803,
                -0.20170439449602756,
                -0.0002270312986465066,
                -3.4123275156411452,
                0.038808629890088,
                -1.2190543925387718,
                -4.3575127761847856e-05,
            ]
        )
        backend.set_target(q, backend._left_target, backend._right_target)
        backend._publish_arm()

        np.testing.assert_allclose(
            [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices],
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_disabled_mode_never_constructs_or_calls_gravity_and_writes_zero_tau(self):
        backend = self._backend(simulation=False)
        backend._gravity_feedforward = False
        backend._arm_gravity = None
        gains_before = [(cmd.kp, cmd.kd) for cmd in backend._arm_message.motor_cmd]
        hands_before = [
            (cmd.mode, cmd.q, cmd.dq, cmd.tau, cmd.kp, cmd.kd)
            for message in (backend._left_message, backend._right_message)
            for cmd in message.motor_cmd
        ]
        targets = (np.linspace(-0.4, 0.5, 14), np.linspace(0.6, -0.3, 14))
        for target in targets:
            backend.set_target(target, backend._left_target, backend._right_target)
            backend._publish_arm()
            np.testing.assert_array_equal(
                [backend._arm_message.motor_cmd[index].q for index in backend._arm_indices],
                target,
            )
            np.testing.assert_array_equal(
                [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices],
                np.zeros(14),
            )
            np.testing.assert_array_equal(backend._last_published_arm_tau, np.zeros(14))

        self.assertEqual(
            [(cmd.kp, cmd.kd) for cmd in backend._arm_message.motor_cmd], gains_before
        )
        self.assertEqual(
            [
                (cmd.mode, cmd.q, cmd.dq, cmd.tau, cmd.kp, cmd.kd)
                for message in (backend._left_message, backend._right_message)
                for cmd in message.motor_cmd
            ],
            hands_before,
        )
        self.assertEqual(len(backend._left_publisher.writes), 0)
        self.assertEqual(len(backend._right_publisher.writes), 0)

    def test_disabled_mode_release_reuses_cached_zero_tau_without_model(self):
        backend = self._backend(simulation=False)
        backend._gravity_feedforward = False
        backend._arm_gravity = None
        target = np.linspace(-0.3, 0.4, 14)
        backend.set_target(target, backend._left_target, backend._right_target)
        backend._publish_arm()
        write_count_before_release = len(backend._arm_publisher.writes)

        backend.set_target(np.linspace(0.6, -0.5, 14), backend._left_target, backend._right_target)
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.ARM_RELEASE_RAMP_S", 0.004),
            mock.patch(f"{module}.PUBLISH_HZ", 1_000.0),
        ):
            backend.release()

        self.assertEqual(backend._weight, 0.0)
        self.assertGreater(len(backend._arm_publisher.writes), write_count_before_release)
        np.testing.assert_array_equal(
            [backend._arm_message.motor_cmd[index].q for index in backend._arm_indices], target
        )
        np.testing.assert_array_equal(
            [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices],
            np.zeros(14),
        )

    def test_gravity_failure_changes_no_message_and_performs_no_dds_write(self):
        backend = self._backend()
        target = np.linspace(-0.2, 0.3, 14)
        backend.set_target(target, backend._left_target, backend._right_target)
        commands_before = [vars(command).copy() for command in backend._arm_message.motor_cmd]
        backend._arm_gravity.compute = mock.Mock(side_effect=DeploymentError("unsafe tau"))

        with self.assertRaisesRegex(DeploymentError, "unsafe tau"):
            backend._publish_arm()

        self.assertEqual([vars(command) for command in backend._arm_message.motor_cmd], commands_before)
        self.assertEqual(len(backend._arm_publisher.writes), 0)
        self.assertFalse(backend._has_published)

    def test_initialization_both_warmups_hold_and_live_all_publish_target_specific_tau(self):
        backend = self._backend()
        backend.reader.latest = lambda: RobotState(
            captured_at=time.monotonic(),
            mode_machine=QUALIFIED_REAL_MODE_MACHINE,
            arm=backend._arm_target.copy(),
            arm_dq=np.zeros(14),
            left_hand=backend._left_target.copy(),
            right_hand=backend._right_target.copy(),
        )
        heartbeat = SimpleNamespace(value=time.monotonic(), get_lock=lambda: threading.Lock())
        stages = {
            "initialization": np.full(14, 0.01),
            "warmup1": np.full(14, 0.02),
            "warmup2": np.full(14, 0.03),
        }
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.PUBLISH_HZ", 1_000.0),
            mock.patch(f"{module}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.001),
            mock.patch(f"{module}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1),
        ):
            for context, target in stages.items():
                completed = _execute_initialization(
                    backend,
                    ActionChunk(
                        arm=target[None, :],
                        left_hand=backend._left_target[None, :],
                        right_hand=backend._right_target[None, :],
                    ),
                    threading.Event(),
                    heartbeat,
                    float("inf"),
                    context=context,
                )
                self.assertTrue(completed, context)

        hold = np.full(14, 0.04)
        _set_direct_target(
            backend,
            None,
            None,
            hold,
            backend._left_target,
            backend._right_target,
        )
        backend.publish()
        live = np.full(14, 0.05)
        backend.set_target(live, backend._left_target, backend._right_target)
        backend.publish()

        for target in (*stages.values(), hold, live):
            self.assertTrue(
                any(np.array_equal(q, target) for q in backend._arm_gravity.inputs),
                f"no gravity computation observed for target {target[0]:.2f}",
            )
        self.assertEqual(len(backend._arm_gravity.inputs), len(backend._arm_publisher.writes))
        # Gravity integration must not alter the existing Dex3 gains or torques.
        for message in (backend._left_message, backend._right_message):
            for command in message.motor_cmd:
                self.assertEqual((command.tau, command.kp, command.kd), (-88.0, 123.0, 45.0))

    def test_authority_acquisition_computes_gravity_on_every_arm_write(self):
        backend = self._backend(simulation=False)
        backend._configure_messages(
            RobotState(
                captured_at=time.monotonic(),
                mode_machine=QUALIFIED_REAL_MODE_MACHINE,
                arm=np.zeros(14),
                arm_dq=np.zeros(14),
                left_hand=backend._left_target.copy(),
                right_hand=backend._right_target.copy(),
            )
        )
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        heartbeat = SimpleNamespace(value=time.monotonic(), get_lock=lambda: threading.Lock())
        with (
            mock.patch(f"{module}.ARM_AUTHORITY_RAMP_S", 0.004),
            mock.patch(f"{module}.PUBLISH_HZ", 1_000.0),
        ):
            completed = _ramp_real_arm_authority(backend, threading.Event(), heartbeat)

        self.assertTrue(completed)
        self.assertGreater(len(backend._arm_publisher.writes), 1)
        self.assertEqual(len(backend._arm_gravity.inputs), len(backend._arm_publisher.writes))
        self.assertEqual(len(backend._left_publisher.writes), 1)
        self.assertEqual(len(backend._right_publisher.writes), 1)
        self.assertEqual(backend._weight, 1.0)

    def test_authority_release_reuses_last_successful_q_tau_without_gravity_compute(self):
        backend = self._backend(simulation=False)
        published_target = np.linspace(-0.3, 0.4, 14)
        backend.set_target(published_target, backend._left_target, backend._right_target)
        backend._publish_arm()
        published_tau = np.array(
            [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices]
        )
        write_count_before_release = len(backend._arm_publisher.writes)

        # Reproduce an active-path dynamics failure after a newer target has
        # been accepted locally but before it could be published. Cleanup must
        # not retry the failed computation: it must ramp weight down while
        # preserving the exact last successfully written q/tau pair.
        unpublished_target = np.linspace(0.6, -0.5, 14)
        backend.set_target(
            unpublished_target,
            backend._left_target,
            backend._right_target,
        )
        backend._arm_gravity.compute = mock.Mock(side_effect=DeploymentError("unsafe tau"))
        with self.assertRaisesRegex(DeploymentError, "unsafe tau"):
            backend._publish_arm()
        backend._arm_gravity.compute.reset_mock()

        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}.ARM_RELEASE_RAMP_S", 0.004),
            mock.patch(f"{module}.PUBLISH_HZ", 1_000.0),
        ):
            backend.release()

        self.assertTrue(backend._released)
        self.assertEqual(backend._weight, 0.0)
        self.assertGreater(len(backend._arm_publisher.writes), write_count_before_release)
        backend._arm_gravity.compute.assert_not_called()
        np.testing.assert_array_equal(
            [backend._arm_message.motor_cmd[index].q for index in backend._arm_indices],
            published_target,
        )
        np.testing.assert_array_equal(
            [backend._arm_message.motor_cmd[index].tau for index in backend._arm_indices],
            published_tau,
        )
        self.assertEqual(len(backend._left_publisher.writes), 1)
        self.assertEqual(len(backend._right_publisher.writes), 1)

    @unittest.skipUnless(PINOCCHIO_AVAILABLE, "Pinocchio is not installed in this test environment")
    def test_reduced_contract_and_effort_envelope_are_fail_closed(self):
        import unitree_lerobot.eval_robot.robot_control.g1_arm_gravity as module

        with (
            mock.patch.object(
                module,
                "G1_ARM_GRAVITY_JOINT_NAMES",
                tuple(reversed(G1_ARM_GRAVITY_JOINT_NAMES)),
            ),
            self.assertRaisesRegex(DeploymentError, "unsafe reduced-model contract"),
        ):
            G1ArmGravityCompensator()

        unsafe_envelope = np.full(14, 1_000.0)
        with (
            mock.patch.object(module, "G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM", unsafe_envelope),
            self.assertRaisesRegex(DeploymentError, "exceeds.*URDF effort limits"),
        ):
            G1ArmGravityCompensator()

        invalid_envelopes = (
            np.ones(13),
            np.ones((1, 14)),
            np.r_[np.ones(13), np.nan],
            np.r_[np.ones(13), np.inf],
            np.r_[np.ones(13), 0.0],
            np.r_[np.ones(13), -1.0],
        )
        for envelope in invalid_envelopes:
            with (
                self.subTest(envelope=envelope),
                mock.patch.object(module, "G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM", envelope),
                self.assertRaisesRegex(DeploymentError, "14 finite positive values"),
            ):
                G1ArmGravityCompensator()


if __name__ == "__main__":
    unittest.main()
