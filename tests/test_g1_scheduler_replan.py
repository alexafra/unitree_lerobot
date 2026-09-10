from __future__ import annotations

import queue
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock

import numpy as np

from tests.test_groot_g1_rtc import (
    _ChildHarness,
    _GatedRecordingBackend,
    _LiveHeartbeat,
    _plan,
)
from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _run_active_goal_controlled,
    _run_active_goal_rtc,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    CONTROL_HZ,
    HandFeedbackOperatorHold,
    SafeG1Dex3Actuator,
)
from unitree_lerobot.eval_robot.groot_contract import load_initialization_spec


_EVAL_MODULE = "unitree_lerobot.eval_robot.eval_groot_g1"
_SAFE_MODULE = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"


class _StallAfterFirstActionBackend(_GatedRecordingBackend):
    """Block the publish immediately following action zero, without a race."""

    instance: _StallAfterFirstActionBackend | None = None

    def __init__(self, simulation: bool, network_interface: str | None):
        super().__init__(simulation, network_interface)
        type(self).instance = self
        self._stalled_once = False
        self._stall_armed = False

    def clear_targets(self) -> None:
        super().clear_targets()
        # _ChildHarness clears initialization history immediately before the
        # test starts; do not perturb initialization timing itself.
        self._stall_armed = True

    def publish(self) -> None:
        super().publish()
        if self._stall_armed and not self._stalled_once and len(self.target_snapshot()) == 1:
            self._stalled_once = True
            self.publish_paused.set()
            self.continue_publish.wait(timeout=1.0)


class _OffsetLaggingStallBackend(_StallAfterFirstActionBackend):
    """Report ordinary arm servo lag after the first nonzero command."""

    instance: _OffsetLaggingStallBackend | None = None

    def state(self):
        state = super().state()
        if np.any(np.abs(self._arm_target) > 1e-12):
            return type(state)(
                captured_at=state.captured_at,
                mode_machine=state.mode_machine,
                arm=self._arm_target.copy() - 0.10,
                arm_dq=state.arm_dq,
                left_hand=state.left_hand,
                right_hand=state.right_hand,
                arm_received_at=state.arm_received_at,
                left_hand_received_at=state.left_hand_received_at,
                right_hand_received_at=state.right_hand_received_at,
            )
        return state


class _ZeroMeasuredStallBackend(_StallAfterFirstActionBackend):
    """Keep feedback at zero so command-to-command checks are observable."""

    instance: _ZeroMeasuredStallBackend | None = None
    follow_commanded_target = False


class _StaleAfterStallBackend(_StallAfterFirstActionBackend):
    """Enter a soft Dex3 stale interval immediately after the injected stall."""

    instance: _StaleAfterStallBackend | None = None

    def __init__(self, simulation: bool, network_interface: str | None):
        super().__init__(simulation, network_interface)
        self.stale_hands = False

    def publish(self) -> None:
        was_stalled = self._stall_armed and not self._stalled_once and len(self.target_snapshot()) == 1
        super().publish()
        if was_stalled:
            self.stale_hands = True

    def state(self):
        state = super().state()
        if not self.stale_hands:
            return state
        stale_at = time.monotonic() - 0.10
        return type(state)(
            captured_at=stale_at,
            mode_machine=state.mode_machine,
            arm=state.arm,
            arm_dq=state.arm_dq,
            left_hand=state.left_hand,
            right_hand=state.right_hand,
            arm_received_at=time.monotonic(),
            left_hand_received_at=stale_at,
            right_hand_received_at=stale_at,
        )


def _replan_payload(*, rtc: bool, sequence: int = 1, action_index: int = 1, length: int = 8) -> dict:
    return {
        "sequence": sequence,
        "action_index": action_index,
        "discarded_actions": length - action_index,
        "lateness_s": 1.5 / CONTROL_HZ,
        "rtc": rtc,
    }


class ChildSchedulerReplanTest(unittest.TestCase):
    @staticmethod
    def _stall_one_publish_after_first_action(child: _ChildHarness) -> None:
        backend = child.backend
        assert isinstance(backend, _StallAfterFirstActionBackend)
        if not backend.publish_paused.wait(timeout=0.5):
            raise AssertionError("actuator publisher did not reach the injected stall")
        # The pause starts on a 100 Hz publish between 30 Hz action slots. Leave
        # enough margin that the next scheduler check is at least one full
        # action period late, while remaining well below the heartbeat timeout.
        time.sleep(2.25 / CONTROL_HZ)
        backend.continue_publish.set()

    def test_sync_lateness_discards_tail_freezes_last_target_and_never_catches_up(self) -> None:
        with _ChildHarness(_StallAfterFirstActionBackend) as child, mock.patch(
            f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S",
            1.0 / CONTROL_HZ,
        ):
            plan = _plan(8)
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            self._stall_one_publish_after_first_action(child)
            detail = child.assert_status("replan_required")

            self.assertEqual(detail["sequence"], 1)
            self.assertEqual(detail["action_index"], 1)
            self.assertEqual(detail["discarded_actions"], plan.length - 1)
            self.assertGreaterEqual(detail["lateness_s"], 1.0 / CONTROL_HZ)
            self.assertIs(detail["rtc"], False)

            targets_at_replan = child.backend.target_snapshot()
            publishes_at_replan = child.backend.publishes
            time.sleep(0.08)
            targets_after_wait = child.backend.target_snapshot()

            # The child remains alive and keeps publishing the last target, but
            # no later element from the stale chunk may be applied as catch-up.
            self.assertTrue(child.thread is not None and child.thread.is_alive())
            self.assertFalse(child.backend.released)
            self.assertGreater(child.backend.publishes, publishes_at_replan)
            self.assertGreaterEqual(len(targets_at_replan), 1)
            self.assertTrue(all(target[0][0] == plan.arm[0, 0] for target in targets_after_wait))
            self.assertFalse(
                any(
                    np.array_equal(target[0], stale_action)
                    for target in targets_after_wait
                    for stale_action in plan.arm[1:]
                )
            )

    def test_rtc_lateness_requests_fresh_generation_without_reusing_old_tail(self) -> None:
        with _ChildHarness(_StallAfterFirstActionBackend) as child, mock.patch(
            f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S",
            1.0 / CONTROL_HZ,
        ):
            old_plan = _plan(8)
            child.commands.put(
                (
                    "rtc_start",
                    1,
                    time.monotonic(),
                    20,
                    old_plan.arm,
                    old_plan.left_hand,
                    old_plan.right_hand,
                    old_plan.length,
                ),
                timeout=0.2,
            )
            self.assertEqual(child.assert_status("rtc_started"), 1)
            self._stall_one_publish_after_first_action(child)
            detail = child.assert_status("replan_required")

            self.assertEqual(detail["sequence"], 1)
            self.assertEqual(detail["action_index"], 1)
            self.assertEqual(detail["discarded_actions"], old_plan.length - 1)
            self.assertGreaterEqual(detail["lateness_s"], 1.0 / CONTROL_HZ)
            self.assertIs(detail["rtc"], True)

            # Replanning consumes no phantom generation. The next fresh plan is
            # sequence 2 and starts from index zero, never as rtc_replace of 1.
            fresh_plan = _plan(2)
            fresh_plan.arm[:, 1] = (0.02, 0.03)
            child.commands.put(
                (
                    "rtc_start",
                    2,
                    time.monotonic(),
                    1,
                    fresh_plan.arm,
                    fresh_plan.left_hand,
                    fresh_plan.right_hand,
                    fresh_plan.length,
                ),
                timeout=0.2,
            )
            self.assertEqual(child.assert_status("rtc_started"), 2)
            self.assertEqual(child.assert_status("rtc_completed"), 1)
            self.assertTrue(child.thread is not None and child.thread.is_alive())
            self.assertFalse(child.backend.released)

    def test_xr_replan_keeps_exact_frozen_target_until_fresh_plan_is_accepted(self) -> None:
        with _ChildHarness(
            _OffsetLaggingStallBackend,
            command_conditioning="xr",
        ) as child, mock.patch(
            f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S",
            1.0 / CONTROL_HZ,
        ):
            plan = _plan(8)
            plan.arm[:, 0] = 0.50
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            self._stall_one_publish_after_first_action(child)
            child.assert_status("replan_required")

            frozen_arm = child.backend._arm_target.copy()
            frozen_left = child.backend._left_target.copy()
            frozen_right = child.backend._right_target.copy()
            target_count = len(child.backend.target_snapshot())
            time.sleep(0.08)

            # Feedback lags the frozen command by 0.1 rad. XR conditioning
            # must remain fenced until a fresh plan is actually accepted.
            self.assertEqual(len(child.backend.target_snapshot()), target_count)
            np.testing.assert_array_equal(child.backend._arm_target, frozen_arm)
            np.testing.assert_array_equal(child.backend._left_target, frozen_left)
            np.testing.assert_array_equal(child.backend._right_target, frozen_right)
            self.assertTrue(child.thread is not None and child.thread.is_alive())

    def test_unconditioned_replan_validates_fresh_target_from_frozen_command(self) -> None:
        with _ChildHarness(_ZeroMeasuredStallBackend) as child, mock.patch(
            f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S",
            1.0 / CONTROL_HZ,
        ):
            old_plan = _plan(8)
            old_plan.arm[:, 0] = 0.09
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    old_plan.arm,
                    old_plan.left_hand,
                    old_plan.right_hand,
                ),
                timeout=0.2,
            )
            self._stall_one_publish_after_first_action(child)
            child.assert_status("replan_required")
            self.assertAlmostEqual(child.backend._arm_target[0], 0.09)

            # -0.02 is safe relative to measured zero, but it is an unsafe
            # 0.11-rad jump from the exact +0.09 command being held.
            fresh_plan = _plan(1)
            fresh_plan.arm[0, 0] = -0.02
            child.commands.put(
                (
                    "chunk",
                    2,
                    time.monotonic(),
                    fresh_plan.arm,
                    fresh_plan.left_hand,
                    fresh_plan.right_hand,
                ),
                timeout=0.2,
            )
            fault = child.assert_status("fault")
            self.assertIn("target jump is too large", fault)
            self.assertFalse(
                any(np.array_equal(target[0], fresh_plan.arm[0]) for target in child.backend.target_snapshot())
            )

    def test_operator_stop_during_stale_replan_cancels_replan_and_recovers_in_hold(self) -> None:
        with _ChildHarness(_StaleAfterStallBackend) as child, mock.patch(
            f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S",
            1.0 / CONTROL_HZ,
        ):
            plan = _plan(8)
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            self._stall_one_publish_after_first_action(child)
            child.assert_status("hand_state_pause")

            child.urgent_hold.set()
            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 1)
            child.backend.stale_hands = False
            progress = []
            while True:
                kind, value = child.statuses.get(timeout=1.0)
                if kind == "hand_state_recovered":
                    break
                self.assertEqual(kind, "hand_state_recovery_progress")
                progress.append(value["fresh_samples"])
            self.assertEqual(progress, [1, 2, 3])

            # The superseded automatic replan must not reappear. A normal
            # warm-start command proves the child is logically in HOLD.
            child.commands.put(
                (
                    "warm_start",
                    time.monotonic(),
                    load_initialization_spec("measured", task_name="pick-red-cup"),
                ),
                timeout=0.2,
            )
            child.assert_status("warm_starting")
            child.assert_status("warm_started")
            with self.assertRaises(queue.Empty):
                child.statuses.get(timeout=0.08)


class ParentReplanTest(unittest.TestCase):
    @staticmethod
    def _parent_handle_with_status(
        kind: str,
        payload: dict,
        *,
        rtc: bool,
        sequence: int | None = None,
    ) -> SafeG1Dex3Actuator:
        class _AliveProcess:
            @staticmethod
            def is_alive() -> bool:
                return True

        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = True
        actuator._holding = False
        actuator._chunk_in_flight = True
        pending_sequence = int(payload["sequence"] if sequence is None else sequence)
        actuator._pending_sequence = pending_sequence
        actuator._sequence = pending_sequence
        actuator._rtc_active = rtc
        actuator._rtc_terminal = None
        actuator._control_lock = threading.Lock()
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_release_requested = threading.Event()
        actuator._stopped_acknowledged = False
        actuator._hand_state_paused = False
        actuator._last_hand_state_event = None
        actuator._release_completed = False
        actuator._last_cleanup_phase = None
        actuator._status_queue = queue.Queue(maxsize=32)
        actuator._status_queue.put((kind, payload))
        actuator._command_queue = queue.Queue(maxsize=1)
        actuator._process = _AliveProcess()
        actuator._heartbeat = _LiveHeartbeat()
        return actuator

    def test_parent_translates_replan_status_for_sync_and_rtc_without_operator_hold(self) -> None:
        sync_payload = _replan_payload(rtc=False, sequence=7)
        sync = self._parent_handle_with_status("replan_required", sync_payload, rtc=False)
        self.assertEqual(sync.wait_completed(7, timeout_s=0.2), "replan")
        self.assertFalse(sync._chunk_in_flight)
        self.assertIsNone(sync._pending_sequence)
        self.assertFalse(sync._holding)

        rtc_payload = _replan_payload(rtc=True, sequence=11)
        rtc = self._parent_handle_with_status("replan_required", rtc_payload, rtc=True)
        self.assertEqual(rtc.poll_rtc_event(), ("replan", rtc_payload))
        self.assertFalse(rtc._chunk_in_flight)
        self.assertFalse(rtc._rtc_active)
        self.assertIsNone(rtc._pending_sequence)
        self.assertFalse(rtc._holding)

    def test_feedback_operator_hold_fences_old_rtc_and_accepts_fresh_goal(self) -> None:
        sequence = 741
        payload = {
            "context": f"active sequence={sequence} next_action_index=0 rtc=False holding=False",
            "hands": ("right",),
            "age_s": 1.01,
        }
        actuator = self._parent_handle_with_status(
            "hand_state_operator_hold",
            payload,
            rtc=True,
            sequence=sequence,
        )
        actuator._warm_started = True

        with self.assertRaises(HandFeedbackOperatorHold):
            actuator._wait_status("rtc_snapshot", timeout_s=0.2)

        self.assertTrue(actuator._holding)
        self.assertFalse(actuator._warm_started)
        self.assertFalse(actuator._chunk_in_flight)
        self.assertIsNone(actuator._pending_sequence)
        self.assertFalse(actuator._rtc_active)
        self.assertTrue(actuator._hand_state_paused)
        self.assertTrue(actuator._hand_operator_hold_pending)
        self.assertEqual(actuator._rtc_fenced_through_sequence, sequence)

        # This is the exact FIFO tail left behind in the real failure: the
        # child reports terminal RTC rejection after the parent has already
        # surfaced operator HOLD. It must not resurrect/corrupt plan state.
        actuator._status_queue.put(
            (
                "rtc_rejected",
                {"reason": "hand_state_operator_hold", "sequence": sequence, "rtc": True},
            )
        )
        actuator._status_queue.put(
            (
                "hand_state_recovered",
                {"fresh_samples": 3, "required_samples": 3, "pause_s": 0.92},
            )
        )
        actuator.assert_healthy()
        self.assertFalse(actuator._hand_state_paused)
        self.assertTrue(actuator.acknowledge_hand_operator_hold())
        self.assertFalse(actuator.acknowledge_hand_operator_hold())

        with mock.patch.object(actuator, "_wait_status", return_value=sequence + 1):
            next_sequence = actuator.start_rtc(_plan(32), action_budget=16)
        self.assertEqual(next_sequence, sequence + 1)
        self.assertTrue(actuator._rtc_active)
        self.assertTrue(actuator._chunk_in_flight)
        self.assertEqual(actuator._pending_sequence, sequence + 1)

        # Even if the old rejection is delayed until after the new plan starts,
        # its sequence fence prevents it from terminating the replacement.
        actuator._status_queue.put(
            (
                "rtc_rejected",
                {"reason": "hand_state_operator_hold", "sequence": sequence, "rtc": True},
            )
        )
        self.assertIsNone(actuator.poll_rtc_event())
        self.assertTrue(actuator._rtc_active)
        self.assertTrue(actuator._chunk_in_flight)
        self.assertEqual(actuator._pending_sequence, sequence + 1)

    def test_polling_operator_hold_returns_hold_and_discards_paired_rtc_rejection(self) -> None:
        sequence = 17
        payload = {
            "context": f"active sequence={sequence} next_action_index=0 rtc=False holding=False",
            "hands": ("right",),
            "age_s": 1.01,
        }
        actuator = self._parent_handle_with_status(
            "hand_state_operator_hold",
            payload,
            rtc=True,
            sequence=sequence,
        )
        actuator._status_queue.put(
            (
                "rtc_rejected",
                {"reason": "hand_state_operator_hold", "sequence": sequence, "rtc": True},
            )
        )

        self.assertEqual(actuator.poll_rtc_event(), ("hold", payload))
        self.assertTrue(actuator._holding)
        self.assertFalse(actuator._rtc_active)
        self.assertFalse(actuator._chunk_in_flight)
        self.assertIsNone(actuator._pending_sequence)
        self.assertIsNone(actuator.poll_rtc_event())

    def test_sync_parent_refetches_immediately_and_logs_opt_in_yellow_warning(self) -> None:
        first_plan = _plan(2)
        second_plan = _plan(2)
        second_plan.arm[:, 1] = (0.01, 0.02)
        events: list[str] = []
        actuator = mock.Mock()
        actuator.submit.side_effect = lambda _plan: events.append("submit") or len(events)

        def wait_completed(_sequence: int, *, timeout_s: float) -> str:
            del timeout_s
            outcome = "replan" if events.count("wait") == 0 else "complete"
            events.append("wait")
            return outcome

        actuator.wait_completed.side_effect = wait_completed
        terminal = SimpleNamespace(poll_control=mock.Mock(return_value=None))
        args = SimpleNamespace(
            execution_horizon=2,
            max_chunks=1,
            command_conditioning="xr",
            show_camera=False,
        )

        def infer(*_args, **_kwargs):
            events.append("infer")
            return (first_plan, 0.01) if events.count("infer") == 1 else (second_plan, 0.02)

        with (
            mock.patch(f"{_EVAL_MODULE}.infer_chunk", side_effect=infer) as infer_chunk,
            self.assertLogs("eval_groot_g1", level="WARNING") as captured,
        ):
            outcome = _run_active_goal_controlled(
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                actuator,
                "pick-red-cup",
                "pick up the red cup.",
                SimpleNamespace(),
                args,
                terminal,
                allow_custom_instruction=False,
            )

        self.assertEqual(outcome, "complete")
        self.assertEqual(infer_chunk.call_count, 2)
        self.assertEqual(actuator.submit.call_count, 2)
        self.assertEqual(actuator.wait_completed.call_count, 2)
        actuator.hold.assert_not_called()
        self.assertEqual(events, ["infer", "submit", "wait", "infer", "submit", "wait"])
        replan_records = [record for record in captured.records if "replan" in record.getMessage().lower()]
        self.assertTrue(replan_records)
        self.assertTrue(any(getattr(record, "terminal_yellow", False) for record in replan_records))

    def test_rtc_parent_discards_worker_and_starts_a_fresh_plan_after_replan(self) -> None:
        class _IdleWorker:
            instances: list[_IdleWorker] = []

            def __init__(self, _host: str, _port: int):
                self.busy = False
                self.closed: list[bool] = []
                type(self).instances.append(self)

            def poll(self):
                return None

            def close(self, *, wait: bool):
                self.closed.append(wait)

        class _TerminalContext:
            def __init__(self, _actuator):
                self.terminal = SimpleNamespace(poll_control=mock.Mock(return_value=None))

            def __enter__(self):
                return self.terminal

            def __exit__(self, *_args):
                return None

        first_plan = _plan(32)
        second_plan = _plan(32)
        second_plan.arm[:, 1] = 0.02
        replan = _replan_payload(rtc=True, sequence=1, action_index=1, length=32)
        actuator = mock.Mock()
        actuator.start_rtc.side_effect = (1, 2)
        actuator.poll_rtc_event.side_effect = (("replan", replan), ("complete", 8))
        actuator.immediate_control_requested.return_value = None
        args = SimpleNamespace(
            execution_horizon=8,
            max_chunks=1,
            policy_host="127.0.0.1",
            policy_port=5555,
            command_conditioning="xr",
            show_camera=False,
            rtc_frozen_steps=None,
            rtc_ramp_rate=None,
        )

        with (
            mock.patch(f"{_EVAL_MODULE}._OperatorTerminal", _TerminalContext),
            mock.patch(f"{_EVAL_MODULE}._RtcInferenceWorker", _IdleWorker),
            mock.patch(
                f"{_EVAL_MODULE}.infer_plan",
                side_effect=((first_plan, 0.01), (second_plan, 0.02)),
            ) as infer_plan,
            self.assertLogs("eval_groot_g1", level="WARNING") as captured,
        ):
            outcome = _run_active_goal_rtc(
                mock.Mock(),
                mock.Mock(),
                mock.Mock(),
                actuator,
                "pick-red-cup",
                "pick up the red cup.",
                SimpleNamespace(action_horizon=32),
                args,
                allow_custom_instruction=False,
            )

        self.assertEqual(outcome, "complete")
        self.assertEqual(infer_plan.call_count, 2)
        self.assertEqual(actuator.start_rtc.call_count, 2)
        np.testing.assert_array_equal(actuator.start_rtc.call_args_list[0].args[0].arm, first_plan.arm)
        np.testing.assert_array_equal(actuator.start_rtc.call_args_list[1].args[0].arm, second_plan.arm)
        self.assertEqual(len(_IdleWorker.instances), 2)
        self.assertTrue(all(worker.closed for worker in _IdleWorker.instances))
        actuator.hold.assert_not_called()
        replan_records = [record for record in captured.records if "replan" in record.getMessage().lower()]
        self.assertTrue(replan_records)
        self.assertTrue(any(getattr(record, "terminal_yellow", False) for record in replan_records))


if __name__ == "__main__":
    unittest.main()
