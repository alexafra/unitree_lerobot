from __future__ import annotations

import argparse
import os
from pathlib import Path
import pty
import sys
import termios
import threading
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import OperatorRelease
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    INITIALIZATION_MAX_ARM_STEP_RAD,
    INITIALIZATION_MAX_HAND_STEP_RAD,
    ImmediateControlEvent,
    RobotState,
    build_initialization_chunk,
)
from unitree_lerobot.eval_robot.zero_state_test import (
    REQUIRED_RELEASE_RAMP_S,
    ZERO_STATE_JOINTS_RAD,
    ZERO_STATE_SOURCE,
    ZeroStateActuator,
    _run_interruptible,
    _validate_runtime,
    build_parser,
    run,
    zero_state_spec,
)


MODULE = "unitree_lerobot.eval_robot.zero_state_test"


class ZeroStateTest(unittest.TestCase):
    def test_target_is_exact_full_28_joint_measured_frame(self):
        expected = np.array(
            [
                -0.35650673508644104,
                0.17410682141780853,
                0.19246666133403778,
                1.0689928531646729,
                -0.0953824520111084,
                -0.9570353031158447,
                -0.2767454981803894,
                -0.3830517828464508,
                -0.17801368236541748,
                -0.027348002418875694,
                0.8752079606056213,
                -0.17540112137794495,
                -0.7583771347999573,
                -0.17424488067626953,
                -0.537041425704956,
                0.796114981174469,
                0.021243207156658173,
                -0.347329705953598,
                -0.033961232751607895,
                -0.2210468202829361,
                -0.022243322804570198,
                -0.3697550594806671,
                -0.8164053559303284,
                -0.02797570452094078,
                0.2404455542564392,
                0.03841176629066467,
                0.12233000993728638,
                0.018584586679935455,
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(ZERO_STATE_JOINTS_RAD, expected)
        self.assertFalse(ZERO_STATE_JOINTS_RAD.flags.writeable)
        self.assertEqual(ZERO_STATE_SOURCE["episode_index"], 0)
        self.assertEqual(ZERO_STATE_SOURCE["frame_index"], 0)

        spec = zero_state_spec()
        self.assertEqual(spec.mode, "pose-file")
        np.testing.assert_array_equal(spec.arm, expected[:14])
        np.testing.assert_array_equal(spec.left_hand, expected[14:21])
        np.testing.assert_array_equal(spec.right_hand, expected[21:])

        # Every call receives independent arrays; a caller cannot alter the
        # frozen target for a later physical run.
        assert spec.arm is not None
        spec.arm[0] = 0.0
        self.assertEqual(zero_state_spec().arm[0], expected[0])

    def test_frozen_target_matches_the_source_parquet_first_state(self):
        import pyarrow.parquet as pq

        parquet = (
            Path(ZERO_STATE_SOURCE["dataset_path"])
            / "data/chunk-000/episode_000000.parquet"
        )
        table = pq.read_table(parquet, columns=["observation.state"])
        source = np.asarray(table.column("observation.state")[0].as_py(), dtype=np.float64)
        np.testing.assert_array_equal(ZERO_STATE_JOINTS_RAD, source)

    def test_existing_slow_path_reaches_all_28_exact_targets_with_bounded_steps(self):
        state = RobotState(
            captured_at=0.0,
            mode_machine=6,
            arm=np.zeros(14, dtype=np.float64),
            arm_dq=np.zeros(14, dtype=np.float64),
            left_hand=np.zeros(7, dtype=np.float64),
            right_hand=np.zeros(7, dtype=np.float64),
        )
        path = build_initialization_chunk(state, zero_state_spec())

        np.testing.assert_array_equal(path.arm[-1], ZERO_STATE_JOINTS_RAD[:14])
        np.testing.assert_array_equal(path.left_hand[-1], ZERO_STATE_JOINTS_RAD[14:21])
        np.testing.assert_array_equal(path.right_hand[-1], ZERO_STATE_JOINTS_RAD[21:])
        arm_steps = np.diff(np.vstack((state.arm, path.arm)), axis=0)
        left_steps = np.diff(np.vstack((state.left_hand, path.left_hand)), axis=0)
        right_steps = np.diff(np.vstack((state.right_hand, path.right_hand)), axis=0)
        self.assertLessEqual(
            float(np.max(np.abs(arm_steps))),
            INITIALIZATION_MAX_ARM_STEP_RAD + 1e-12,
        )
        self.assertLessEqual(
            float(np.max(np.abs(left_steps))),
            INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12,
        )
        self.assertLessEqual(
            float(np.max(np.abs(right_steps))),
            INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12,
        )

    def test_runtime_requires_tty_nic_override_and_reviewed_release_ramp(self):
        valid = argparse.Namespace(
            network_interface="enp132s0",
            allow_unqualified_real=True,
        )
        with mock.patch.object(sys.stdin, "isatty", return_value=True):
            _validate_runtime(valid)

            missing_nic = argparse.Namespace(network_interface=None, allow_unqualified_real=True)
            with self.assertRaisesRegex(DeploymentError, "network-interface"):
                _validate_runtime(missing_nic)

            missing_override = argparse.Namespace(
                network_interface="enp132s0",
                allow_unqualified_real=False,
            )
            with self.assertRaisesRegex(DeploymentError, "allow-unqualified-real"):
                _validate_runtime(missing_override)

            with (
                mock.patch(f"{MODULE}.ARM_RELEASE_RAMP_S", REQUIRED_RELEASE_RAMP_S - 0.1),
                self.assertRaisesRegex(DeploymentError, "release ramp"),
            ):
                _validate_runtime(valid)

        with (
            mock.patch.object(sys.stdin, "isatty", return_value=False),
            self.assertRaisesRegex(DeploymentError, "interactive TTY"),
        ):
            _validate_runtime(valid)

    def test_release_interrupt_during_blocking_motion_is_operator_release(self):
        terminal = mock.Mock()
        terminal.poll_control.return_value = None

        def release() -> None:
            raise ImmediateControlEvent("release")

        with self.assertRaises(OperatorRelease):
            _run_interruptible(terminal, release)

    def test_run_commands_both_arms_and_hands_then_closes_on_q(self):
        events: list[str] = []

        class FakeActuator:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.target = None

            def start(self):
                events.append("start")

            def arm(self):
                events.append("arm")

            def assert_fixed_target_healthy(self):
                pass

            def initialize(self, target):
                events.append("initialize")
                self.target = target

            def close(self):
                events.append("close")

        actuator = FakeActuator()
        terminal = mock.MagicMock()
        terminal.__enter__.return_value = terminal
        terminal.poll_control.return_value = None
        args = argparse.Namespace(
            network_interface="enp132s0",
            allow_unqualified_real=True,
            gravity_feedforward=False,
            _run_log_dir=Path("/tmp/zero-state-test"),
        )

        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{MODULE}._confirm_before_authority", return_value="continue"),
            mock.patch(f"{MODULE}.ZeroStateActuator", return_value=actuator) as factory,
            mock.patch(f"{MODULE}._OperatorTerminal", return_value=terminal) as terminal_factory,
            mock.patch(f"{MODULE}._hold_until_release", side_effect=OperatorRelease),
            mock.patch("builtins.print"),
            self.assertRaises(OperatorRelease),
        ):
            run(args)

        factory.assert_called_once_with(
            False,
            "enp132s0",
            "none",
            gravity_feedforward=False,
            run_log_dir="/tmp/zero-state-test",
        )
        terminal_factory.assert_called_once_with(actuator, stop_enabled=False)
        terminal.__enter__.assert_called_once_with()
        terminal.__exit__.assert_called_once()
        self.assertEqual(events, ["start", "arm", "initialize", "close"])
        self.assertIsNotNone(actuator.target)
        np.testing.assert_array_equal(actuator.target.arm, ZERO_STATE_JOINTS_RAD[:14])
        np.testing.assert_array_equal(actuator.target.left_hand, ZERO_STATE_JOINTS_RAD[14:21])
        np.testing.assert_array_equal(actuator.target.right_hand, ZERO_STATE_JOINTS_RAD[21:])

    def test_q_variants_release_without_enter_during_every_authority_phase(self):
        """The real raw-key monitor remains live from child start through final HOLD."""

        for phase in ("start", "arm", "initialize", "hold"):
            for key in (b"q", b"Q", b"\x11"):
                with self.subTest(phase=phase, key=key):
                    master_fd, slave_fd = pty.openpty()
                    stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                    original_terminal = termios.tcgetattr(slave_fd)
                    phase_started = threading.Event()
                    release_requested = threading.Event()
                    writer_finished = threading.Event()
                    events: list[str] = []

                    class FakeActuator:
                        def __init__(self, *args, **kwargs):
                            self.args = args
                            self.kwargs = kwargs

                        def _run_phase(self, name: str) -> None:
                            events.append(name)
                            if phase != name:
                                return
                            phase_started.set()
                            if not release_requested.wait(timeout=1.0):
                                raise AssertionError(f"{key!r} did not interrupt {name}")
                            raise ImmediateControlEvent("release")

                        def start(self) -> None:
                            self._run_phase("start")

                        def arm(self) -> None:
                            self._run_phase("arm")

                        def assert_fixed_target_healthy(self) -> None:
                            pass

                        def initialize(self, _target) -> None:
                            self._run_phase("initialize")

                        def heartbeat(self) -> None:
                            events.append("heartbeat")
                            if phase == "hold":
                                phase_started.set()

                        def assert_healthy(self) -> None:
                            events.append("assert_healthy")
                            if phase != "hold":
                                return
                            if not release_requested.wait(timeout=1.0):
                                raise AssertionError(f"{key!r} did not interrupt hold")
                            raise ImmediateControlEvent("release")

                        def request_immediate_hold(self) -> None:
                            raise AssertionError(
                                "zero state test must not map any release key to HOLD"
                            )

                        def request_immediate_release(self) -> None:
                            events.append("request_immediate_release")
                            release_requested.set()

                        def close(self) -> None:
                            events.append("close")

                    actuator = FakeActuator()
                    args = argparse.Namespace(
                        network_interface="enp132s0",
                        allow_unqualified_real=True,
                        _run_log_dir=Path("/tmp/zero-state-test"),
                    )

                    def write_key_during_selected_phase() -> None:
                        try:
                            if phase_started.wait(timeout=1.0):
                                # Deliberately write only one byte: no Enter/newline follows it.
                                os.write(master_fd, key)
                        finally:
                            writer_finished.set()

                    writer = threading.Thread(target=write_key_during_selected_phase, daemon=True)
                    try:
                        writer.start()
                        with (
                            mock.patch.object(sys, "stdin", stdin),
                            mock.patch(
                                f"{MODULE}._confirm_before_authority",
                                return_value="continue",
                            ),
                            mock.patch(f"{MODULE}.ZeroStateActuator", return_value=actuator),
                            mock.patch("builtins.print"),
                            self.assertRaises(OperatorRelease),
                        ):
                            run(args)

                        writer.join(timeout=1.0)
                        self.assertTrue(writer_finished.is_set())
                        self.assertTrue(phase_started.is_set())
                        self.assertTrue(release_requested.is_set())
                        self.assertEqual(events.count("request_immediate_release"), 1)
                        self.assertEqual(events[-1], "close")
                        self.assertEqual(termios.tcgetattr(slave_fd), original_terminal)

                        phase_index = events.index(phase if phase != "hold" else "heartbeat")
                        release_index = events.index("request_immediate_release")
                        close_index = events.index("close")
                        self.assertLess(phase_index, release_index)
                        self.assertLess(release_index, close_index)
                    finally:
                        writer.join(timeout=1.0)
                        stdin.close()
                        os.close(master_fd)
                        os.close(slave_fd)

    def test_release_cleanup_error_is_not_hidden_by_operator_release(self):
        class FakeActuator:
            def start(self):
                pass

            def arm(self):
                pass

            def assert_fixed_target_healthy(self):
                pass

            def initialize(self, _target):
                pass

            def close(self):
                raise DeploymentError("injected unconfirmed orderly release")

        actuator = FakeActuator()
        terminal = mock.MagicMock()
        terminal.__enter__.return_value = terminal
        terminal.poll_control.return_value = None
        args = argparse.Namespace(
            network_interface="enp132s0",
            allow_unqualified_real=True,
            _run_log_dir=Path("/tmp/zero-state-test"),
        )

        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{MODULE}._confirm_before_authority", return_value="continue"),
            mock.patch(f"{MODULE}.ZeroStateActuator", return_value=actuator),
            mock.patch(f"{MODULE}._OperatorTerminal", return_value=terminal),
            mock.patch(f"{MODULE}._hold_until_release", side_effect=OperatorRelease),
            mock.patch("builtins.print"),
            self.assertRaisesRegex(DeploymentError, "unconfirmed orderly release"),
        ):
            run(args)

        terminal.__enter__.assert_called_once_with()
        terminal.__exit__.assert_called_once()

    def test_q_before_authority_never_constructs_actuator(self):
        args = argparse.Namespace(
            network_interface="enp132s0",
            allow_unqualified_real=True,
            _run_log_dir=Path("/tmp/zero-state-test"),
        )
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch(f"{MODULE}._confirm_before_authority", side_effect=OperatorRelease),
            mock.patch(f"{MODULE}.ZeroStateActuator") as factory,
            mock.patch("builtins.print"),
            self.assertRaises(OperatorRelease),
        ):
            run(args)
        factory.assert_not_called()

    def test_any_hand_feedback_pause_is_sticky_and_fail_closed(self):
        actuator = ZeroStateActuator.__new__(ZeroStateActuator)
        actuator._zero_state_hand_pause = None
        actuator.assert_healthy = mock.Mock()
        payload = {"hands": ("right",), "age_s": 0.081}

        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3."
            "SafeG1Dex3Actuator._record_auxiliary_status",
            return_value=True,
        ):
            self.assertTrue(actuator._record_auxiliary_status("hand_state_pause", payload))
            actuator._record_auxiliary_status("hand_state_recovered", {"hands": ("right",)})

        with self.assertRaisesRegex(DeploymentError, "Dex3 feedback paused"):
            actuator.assert_fixed_target_healthy()
        actuator.assert_healthy.assert_called_once_with()

    def test_cli_requires_explicit_interface(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            ["--network-interface", "enp132s0", "--allow-unqualified-real"]
        )
        self.assertEqual(args.network_interface, "enp132s0")
        self.assertTrue(args.allow_unqualified_real)


if __name__ == "__main__":
    unittest.main()
