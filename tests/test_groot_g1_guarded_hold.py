"""Software-only regression tests for cancelling guarded robot transitions."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
import io
import unittest
from unittest import mock

import numpy as np

from test_groot_g1_rtc import (
    _ChildHarness,
    _FinalIntervalRecordingBackend,
    _GatedRecordingBackend,
    _LiveHeartbeat,
    _RecordingBackend,
    _parent_handle_for_child,
    _plan,
)
from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _prepare_policy_goal,
    _readline_while_armed,
    _select_next_goal_while_holding,
    _run_return_to_start_from_hold,
    _take_held_voice_command,
    _wait_for_confirmation_input_boundary,
    _run_held_policy_io,
    confirm_policy_warm_start,
    OperatorStop,
    VOICE_STAY_HOLDING,
    _run_return_to_start_sequence_from_hold,
)
from unitree_lerobot.eval_robot.groot_contract import InitializationSpec
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    HandFreshnessResult,
    ImmediateControlEvent,
    _execute_initialization,
    _wait_for_initialization_start,
)

_SAFE = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
_RUNNER = "unitree_lerobot.eval_robot.eval_groot_g1"


def _target() -> InitializationSpec:
    arm = np.zeros(14)
    arm[0] = 0.2
    return InitializationSpec(
        mode="pose-file", label="cancelled test transition", arm=arm,
        left_hand=np.array([0.1, 0, 0, 0, 0, 0, 0]),
        right_hand=np.array([-0.1, 0, 0, 0, 0, 0, 0]),
    )


class _LaggingGuardedBackend(_FinalIntervalRecordingBackend):
    follow_commanded_target = False


class GuardedMotionHoldTests(unittest.TestCase):
    def _helper(self, operation, backend, stop, urgent, **kwargs):
        if operation == "dwell":
            return _wait_for_initialization_start(
                backend, stop, _LiveHeartbeat(), urgent_hold_event=urgent, **kwargs,
            )
        return _execute_initialization(
            backend, _plan(20), stop, _LiveHeartbeat(), float("inf"),
            urgent_hold_event=urgent, **kwargs,
        )

    def test_pending_hold_prevents_dwell_and_first_motion_target(self):
        for operation in ("dwell", "motion"):
            with self.subTest(operation=operation):
                backend = _RecordingBackend(True, None)
                urgent, stop = threading.Event(), threading.Event()
                urgent.set()
                with self.assertRaises(ImmediateControlEvent) as caught:
                    self._helper(operation, backend, stop, urgent)
                self.assertEqual(caught.exception.action, "hold")
                self.assertEqual(backend.target_snapshot(), [])
                self.assertEqual(backend.publishes, 0)
                self.assertTrue(urgent.is_set())
                self.assertFalse(stop.is_set())

    def test_hold_interrupts_stationary_start_dwell(self):
        urgent, stop = threading.Event(), threading.Event()
        backend = _RecordingBackend(True, None)
        original_publish = backend.publish

        def publish_and_stop():
            original_publish()
            urgent.set()

        with (
            mock.patch.object(backend, "publish", side_effect=publish_and_stop),
            mock.patch(f"{_SAFE}.INITIALIZATION_START_DWELL_S", 10.0),
            self.assertRaises(ImmediateControlEvent),
        ):
            self._helper("dwell", backend, stop, urgent)
        self.assertEqual(backend.publishes, 1)
        self.assertEqual(backend.target_snapshot(), [])

    def test_hold_cancels_path_before_endpoint(self):
        urgent, stop = threading.Event(), threading.Event()
        backend = _RecordingBackend(True, None)
        original_publish = backend.publish

        def publish_and_stop():
            original_publish()
            if len(backend.target_snapshot()) == 2:
                urgent.set()

        with (
            mock.patch.object(backend, "publish", side_effect=publish_and_stop),
            self.assertRaises(ImmediateControlEvent),
        ):
            self._helper("motion", backend, stop, urgent)
        self.assertEqual(len(backend.target_snapshot()), 2)
        self.assertLess(backend._arm_target[0], _plan(20).arm[-1, 0])
        self.assertTrue(urgent.is_set())

    def test_hold_cancels_convergence_without_reporting_success(self):
        urgent, stop = threading.Event(), threading.Event()
        backend = _RecordingBackend(True, None)
        original_publish = backend.publish

        def publish_and_stop():
            original_publish()
            urgent.set()

        with (
            mock.patch.object(backend, "publish", side_effect=publish_and_stop),
            mock.patch(f"{_SAFE}.INITIALIZATION_CONVERGENCE_DWELL_S", 10.0),
            self.assertRaises(ImmediateControlEvent),
        ):
            _execute_initialization(
                backend, _plan(1), stop, _LiveHeartbeat(), float("inf"),
                urgent_hold_event=urgent,
            )
        self.assertEqual(len(backend.target_snapshot()), 1)
        self.assertEqual(backend.publishes, 1)

    def test_hold_interrupts_hand_feedback_pause_in_both_helpers(self):
        for operation in ("dwell", "motion"):
            with self.subTest(operation=operation):
                backend = _RecordingBackend(True, None)
                urgent, stop = threading.Event(), threading.Event()

                def paused(*args, **kwargs):
                    urgent.set()
                    return HandFreshnessResult(ready=False)

                with (
                    mock.patch(f"{_SAFE}._observe_hand_freshness", side_effect=paused),
                    self.assertRaises(ImmediateControlEvent),
                ):
                    self._helper(operation, backend, stop, urgent,
                                 hand_freshness_gate=object(), status_queue=queue.Queue())
                self.assertEqual(backend.target_snapshot(), [])
                self.assertTrue(urgent.is_set())
                self.assertFalse(stop.is_set())

    def test_release_has_priority_over_pending_hold(self):
        for operation in ("dwell", "motion"):
            with self.subTest(operation=operation):
                backend = _RecordingBackend(True, None)
                urgent, stop = threading.Event(), threading.Event()
                urgent.set()
                stop.set()
                result = self._helper(operation, backend, stop, urgent)
                self.assertIs(result, None if operation == "dwell" else False)
                self.assertEqual(backend.target_snapshot(), [])
                self.assertTrue(urgent.is_set())

    def test_child_cancels_warmup2_and_return_then_holds_without_resuming(self):
        for kind, started, cancelled in (
            ("warm_start", "warm_starting", "warm_start_cancelled"),
            ("return_to_start_pose", "warmup_pose_started", "warmup_pose_cancelled"),
        ):
            with self.subTest(kind=kind), _ChildHarness(_LaggingGuardedBackend) as child:
                backend = child.backend
                backend.pause_after_next_target.set()
                child.commands.put((kind, time.monotonic(), _target()), timeout=0.2)
                child.assert_status(started)
                self.assertTrue(backend.publish_paused.wait(timeout=0.5))
                grip = (backend._left_target.copy(), backend._right_target.copy())
                try:
                    child.urgent_hold.set()
                finally:
                    backend.continue_publish.set()
                child.assert_status(cancelled)
                child.assert_status("holding")
                self.assertTrue(child.urgent_hold.is_set())
                self.assertFalse(child.stop.is_set())
                self.assertLess(backend._arm_target[0], _target().arm[0])
                np.testing.assert_array_equal(backend._arm_target, np.zeros(14))
                self.assertGreater(float(np.max(np.abs(grip[0]))), 0.0)
                np.testing.assert_array_equal(backend._left_target, grip[0])
                np.testing.assert_array_equal(backend._right_target, grip[1])
                child.commands.put(("urgent_hold_barrier",), timeout=0.2)
                child.assert_status("urgent_holding")
                targets = backend.target_snapshot()
                publishes = backend.publishes
                deadline = time.monotonic() + 0.3
                while backend.publishes < publishes + 4 and time.monotonic() < deadline:
                    time.sleep(0.002)
                self.assertGreaterEqual(backend.publishes, publishes + 4)
                self.assertEqual(len(backend.target_snapshot()), len(targets))
                self.assertTrue(child.statuses.empty())
                self.assertFalse(child.urgent_hold.is_set())
                self.assertTrue(child.thread.is_alive())
                self.assertFalse(backend.released)

    def test_queued_guarded_moves_before_barrier_cannot_run_after_hold(self):
        for kind in ("warm_start", "warmup_pose", "warmup1_pose",
                     "return_to_start_pose", "return_to_start_warmup1_pose"):
            with self.subTest(kind=kind), _ChildHarness(_GatedRecordingBackend) as child:
                backend = child.backend
                backend.pause_next_publish.set()
                self.assertTrue(backend.publish_paused.wait(timeout=0.5))
                actuator = _parent_handle_for_child(child)
                try:
                    child.commands.put((kind, time.monotonic(), _target()), timeout=0.2)
                    actuator.request_immediate_hold()
                finally:
                    backend.continue_publish.set()
                self.assertEqual(actuator.finish_immediate_hold(), "hold")
                self.assertTrue(actuator._holding)
                self.assertFalse(actuator._warm_started)
                self.assertFalse(child.urgent_hold.is_set())
                for arm, _, _ in backend.target_snapshot():
                    np.testing.assert_array_equal(arm, np.zeros(14))
                self.assertTrue(child.thread.is_alive())
                self.assertFalse(backend.released)


class GuardedMotionRunnerHoldTests(unittest.TestCase):
    def _actuator(self, action):
        control = {"pending": None}
        actuator = SimpleNamespace(
            immediate_control_requested=lambda: control["pending"],
            finish_immediate_hold=mock.Mock(side_effect=lambda: control.update(pending=None) or "hold"),
        )

        def interrupt(*args, **kwargs):
            control["pending"] = action
            raise ImmediateControlEvent(action)

        actuator.warm_start = mock.Mock(side_effect=interrupt)
        actuator.return_to_start_pose = mock.Mock(side_effect=interrupt)
        return actuator

    def test_interrupted_return_does_not_execute_later_stage_or_report_completion(self):
        for action in ("hold", "release"):
            with self.subTest(action=action):
                actuator = self._actuator(action)
                with (
                    mock.patch(f"{_RUNNER}._OperatorTerminal"),
                    mock.patch(f"{_RUNNER}.confirm_return_to_start", return_value="continue"),
                    mock.patch(f"{_RUNNER}._wait_for_hand_feedback"),
                    mock.patch(f"{_RUNNER}.confirm_return_to_start_complete") as confirm_complete,
                ):
                    result = _run_return_to_start_sequence_from_hold(actuator, (_target(), _target()))
                self.assertEqual(result, action)
                self.assertEqual(actuator.return_to_start_pose.call_count, 1)
                self.assertEqual(actuator.finish_immediate_hold.call_count, int(action == "hold"))
                confirm_complete.assert_not_called()

    def test_interrupted_warmup2_does_not_confirm_continue_or_restart_policy(self):
        for action in ("hold", "release"):
            with self.subTest(action=action):
                actuator = self._actuator(action)
                policy = SimpleNamespace(reset=mock.Mock())
                with (
                    mock.patch(f"{_RUNNER}._OperatorTerminal"),
                    mock.patch(f"{_RUNNER}._wait_for_hand_feedback"),
                    mock.patch(f"{_RUNNER}.infer_chunk", return_value=(_plan(1), 0.01)),
                    mock.patch(f"{_RUNNER}.chunk_delta_summary", return_value="test target"),
                    mock.patch(f"{_RUNNER}.confirm_policy_warm_start", return_value="continue"),
                    mock.patch(f"{_RUNNER}.confirm_policy_continue") as confirm_continue,
                ):
                    result = _prepare_policy_goal(
                        policy, None, None, actuator, "pick red cup", SimpleNamespace(end_effector="dex3"),
                        SimpleNamespace(execution_horizon=8, policy_warm_start=True),
                        allow_custom_instruction=False,
                    )
                self.assertEqual(result, action)
                actuator.warm_start.assert_called_once()
                policy.reset.assert_called_once()
                self.assertEqual(actuator.finish_immediate_hold.call_count, int(action == "hold"))
                confirm_continue.assert_not_called()


class PromptHoldTests(unittest.TestCase):
    def _actuator(self, pending=None):
        control = {"pending": pending}
        actuator = SimpleNamespace(
            heartbeat=mock.Mock(),
            immediate_control_requested=lambda: control["pending"],
            finish_immediate_hold=mock.Mock(side_effect=lambda: control.update(pending=None) or "hold"),
            request_immediate_release=mock.Mock(side_effect=lambda: control.update(pending="release")),
            warm_start=mock.Mock(), return_to_start_pose=mock.Mock(),
            wait_for_hand_feedback=mock.Mock(), close=mock.Mock(), hold=mock.Mock(),
            _holding=True,
        )
        def healthy():
            if control["pending"]:
                raise ImmediateControlEvent(control["pending"])
            return False
        actuator.assert_healthy = mock.Mock(side_effect=healthy)
        return actuator, control

    def test_hold_at_warmup2_confirmation_cancels_and_acknowledges(self):
        actuator, _ = self._actuator("hold")
        with (mock.patch(f"{_RUNNER}.sys.stdin", io.StringIO()),
              mock.patch(f"{_RUNNER}.select.select") as poll,
              mock.patch("builtins.print")):
            self.assertEqual(confirm_policy_warm_start(actuator, "test target"), "hold")
        actuator.finish_immediate_hold.assert_called_once()
        actuator.warm_start.assert_not_called()
        actuator.close.assert_not_called()
        actuator.request_immediate_release.assert_not_called()
        poll.assert_not_called()

    def test_hold_at_terminal_quiet_boundary_returns_stop_not_exception(self):
        actuator, _ = self._actuator("hold")
        with mock.patch(f"{_RUNNER}.select.select") as poll:
            self.assertEqual(_wait_for_confirmation_input_boundary(42, actuator), "stop")
        poll.assert_not_called()
        actuator.close.assert_not_called()

    def test_hold_racing_buffered_r_in_line_input_cannot_continue(self):
        actuator, control = self._actuator()
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        def typed_r():
            control["pending"] = "hold"
            return "r\n"
        stdin.readline.side_effect = typed_r
        with (mock.patch(f"{_RUNNER}.sys.stdin", stdin),
              mock.patch(f"{_RUNNER}.select.select", return_value=([stdin], [], [])),
              mock.patch("builtins.print")):
            result = _readline_while_armed(actuator, "gate", timeout_s=1.0, confirmation_mode=True)
        self.assertEqual(result, "stop")
        actuator.close.assert_not_called()

    def test_hold_racing_buffered_r_in_raw_prompt_cannot_continue(self):
        actuator, control = self._actuator()
        stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 42)
        def typed_r(*args):
            control["pending"] = "hold"
            return b"r"
        with (mock.patch(f"{_RUNNER}.sys.stdin", stdin),
              mock.patch(f"{_RUNNER}.termios.tcgetattr", return_value=[0, 0, 0, 0, 0, 0, [0] * 32]),
              mock.patch(f"{_RUNNER}.termios.tcsetattr"),
              mock.patch(f"{_RUNNER}._wait_for_confirmation_input_boundary", return_value=None),
              mock.patch(f"{_RUNNER}.select.select", return_value=([42], [], [])),
              mock.patch(f"{_RUNNER}.os.read", side_effect=typed_r),
              mock.patch("builtins.print")):
            result = confirm_policy_warm_start(actuator, "test target")
        self.assertEqual(result, "hold")
        actuator.finish_immediate_hold.assert_called_once()
        actuator.warm_start.assert_not_called()
        actuator.request_immediate_release.assert_not_called()

    def test_held_menu_acknowledges_hold_and_quarantines_prior_local_input(self):
        actuator, _ = self._actuator("hold")
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        stdin.fileno.return_value = 42
        with (mock.patch(f"{_RUNNER}.sys.stdin", stdin),
              mock.patch(f"{_RUNNER}._wait_for_confirmation_input_boundary", side_effect=("stop", "q")) as boundary,
              mock.patch("builtins.print")):
            result = _select_next_goal_while_holding(actuator)
        self.assertIsNone(result)  # Only the subsequent explicit q leaves the menu.
        self.assertEqual(boundary.call_args_list, [mock.call(42, actuator), mock.call(42, actuator)])
        actuator.finish_immediate_hold.assert_called_once()
        actuator.warm_start.assert_not_called()
        actuator.return_to_start_pose.assert_not_called()
        actuator.close.assert_not_called()

    def test_first_custom_menu_quarantines_r_left_by_cancelled_confirmation(self):
        actuator, _ = self._actuator()
        stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 42)
        with (mock.patch(f"{_RUNNER}.sys.stdin", stdin),
              mock.patch(f"{_RUNNER}.termios.tcgetattr", return_value=[0, 0, 0, 0, 0, 0, [0] * 32]),
              mock.patch(f"{_RUNNER}.termios.tcsetattr"),
              mock.patch(f"{_RUNNER}.select.select", return_value=([42], [], [])),
              mock.patch(f"{_RUNNER}.os.read", side_effect=(b"r", b"q")),
              mock.patch(f"{_RUNNER}.select_instruction") as resolve_custom,
              mock.patch("builtins.print")):
            self.assertIsNone(_select_next_goal_while_holding(actuator, custom_goal_mode=True))
        resolve_custom.assert_not_called()
        actuator.request_immediate_release.assert_called_once()
        actuator.warm_start.assert_not_called()

    def test_hold_at_return_final_visual_gate_cancels_without_release(self):
        actuator, control = self._actuator()
        original_health = actuator.assert_healthy.side_effect
        def hold_at_visual_gate():
            control["pending"] = "hold"
            return original_health()
        actuator.assert_healthy.side_effect = hold_at_visual_gate
        target = InitializationSpec(mode="pose-file", label="test", arm=np.zeros(14),
                                    left_hand=np.zeros(6), right_hand=np.zeros(6),
                                    end_effector="inspire-ftp")
        with (mock.patch(f"{_RUNNER}._OperatorTerminal"),
              mock.patch(f"{_RUNNER}.confirm_return_to_start", return_value="continue"),
              mock.patch(f"{_RUNNER}.sys.stdin", io.StringIO()),
              mock.patch("builtins.print")):
            self.assertEqual(_run_return_to_start_from_hold(actuator, target), "hold")
        actuator.return_to_start_pose.assert_called_once()
        actuator.finish_immediate_hold.assert_called_once()
        actuator.close.assert_not_called()
        actuator.request_immediate_release.assert_not_called()

    def test_hold_in_feedback_wait_after_warmup2_confirmation_prevents_movement(self):
        actuator, control = self._actuator()
        calls = []
        def feedback():
            calls.append(None)
            if len(calls) == 2:
                control["pending"] = "hold"
                raise ImmediateControlEvent("hold")
        actuator.wait_for_hand_feedback.side_effect = feedback
        policy = SimpleNamespace(reset=mock.Mock())
        with (mock.patch(f"{_RUNNER}._OperatorTerminal"),
              mock.patch(f"{_RUNNER}.infer_chunk", return_value=(_plan(1), 0.01)),
              mock.patch(f"{_RUNNER}.chunk_delta_summary", return_value="test target"),
              mock.patch(f"{_RUNNER}.confirm_policy_warm_start", return_value="continue"),
              mock.patch(f"{_RUNNER}.confirm_policy_continue") as continued):
            result = _prepare_policy_goal(policy, None, None, actuator, "pick red cup",
                SimpleNamespace(end_effector="dex3"),
                SimpleNamespace(execution_horizon=8, policy_warm_start=True),
                allow_custom_instruction=False)
        self.assertEqual(result, "hold")
        actuator.finish_immediate_hold.assert_called_once()
        actuator.warm_start.assert_not_called()
        policy.reset.assert_called_once()
        continued.assert_not_called()
        actuator.close.assert_not_called()

    def test_hold_during_return_pre_motion_feedback_prevents_movement(self):
        actuator, control = self._actuator()
        def feedback():
            control["pending"] = "hold"
            raise ImmediateControlEvent("hold")
        actuator.wait_for_hand_feedback.side_effect = feedback
        with (mock.patch(f"{_RUNNER}._OperatorTerminal"),
              mock.patch(f"{_RUNNER}.confirm_return_to_start", return_value="continue")):
            result = _run_return_to_start_from_hold(actuator, _target())
        self.assertEqual(result, "hold")
        actuator.return_to_start_pose.assert_not_called()
        actuator.finish_immediate_hold.assert_called_once()
        actuator.close.assert_not_called()

    def test_new_hold_cancels_phone_acceptance_wait(self):
        actuator, control = self._actuator()
        command = SimpleNamespace(goal="pick up the red cup")
        server = mock.Mock()
        server.poll_command.return_value = command
        def accepting(_command, service_callback):
            control["pending"] = "hold"
            service_callback()
            self.fail("HOLD must interrupt acceptance")
        server.accept_command.side_effect = accepting
        result = _take_held_voice_command(actuator, server, confirm_voice_text=False)
        self.assertEqual(result, VOICE_STAY_HOLDING)
        server.reject_command.assert_called_once()
        actuator.finish_immediate_hold.assert_called_once()
        actuator.warm_start.assert_not_called()
        actuator.close.assert_not_called()

    def test_hold_before_refresh_health_check_does_not_run_refresh(self):
        actuator, _ = self._actuator("hold")
        refresh = mock.Mock()
        with (mock.patch(f"{_RUNNER}._OperatorTerminal"), self.assertRaises(OperatorStop)):
            _run_held_policy_io(actuator, refresh)
        refresh.assert_not_called()
        actuator.finish_immediate_hold.assert_called_once()
        actuator.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
