from __future__ import annotations

import queue
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    ACTUATOR_ARM_STATE_MAX_AGE_S,
    ACTUATOR_HAND_RECOVERY_SAMPLES,
    ACTUATOR_HAND_STATE_MAX_AGE_S,
    ACTUATOR_HAND_STATE_WARNING_AGE_S,
    G1Dex3StateReader,
    HandStateFreshnessGate,
    RobotState,
    _G1Dex3CommandBackend,
    _actuator_main,
)
from unitree_lerobot.eval_robot.groot_contract import InitializationSpec


def _state(*, captured_at: float, left_at: float, right_at: float) -> RobotState:
    return RobotState(
        captured_at=captured_at,
        mode_machine=6,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.zeros(7),
        right_hand=np.zeros(7),
        left_hand_received_at=left_at,
        right_hand_received_at=right_at,
        arm_received_at=captured_at,
    )


class HandFreshnessGateTests(unittest.TestCase):
    def test_short_gap_pauses_and_requires_distinct_paired_samples(self):
        gate = HandStateFreshnessGate()
        entered = gate.check(
            _state(captured_at=10.0, left_at=9.90, right_at=10.0),
            now=10.0,
        )
        self.assertFalse(entered.ready)
        self.assertTrue(entered.entered)
        self.assertEqual(entered.stale_hands, ("left",))

        first_pair = _state(captured_at=10.01, left_at=10.01, right_at=10.01)
        self.assertFalse(gate.check(first_pair, now=10.01).ready)
        # Re-reading one cached pair cannot satisfy the recovery dwell.
        for _ in range(10):
            self.assertFalse(gate.check(first_pair, now=10.011).ready)

        result = None
        for index in range(2, ACTUATOR_HAND_RECOVERY_SAMPLES + 1):
            timestamp = 10.0 + index * 0.01
            result = gate.check(
                _state(captured_at=timestamp, left_at=timestamp, right_at=timestamp),
                now=timestamp,
            )
        assert result is not None
        self.assertTrue(result.ready)
        self.assertTrue(result.recovered)

    def test_warning_boundary_is_soft_not_inclusive(self):
        gate = HandStateFreshnessGate()
        result = gate.check(
            _state(
                captured_at=20.0,
                left_at=20.0 - ACTUATOR_HAND_STATE_WARNING_AGE_S,
                right_at=20.0,
            ),
            now=20.0,
        )
        self.assertTrue(result.ready)


class SplitReaderDeadlineTests(unittest.TestCase):
    @staticmethod
    def _reader(now: float) -> G1Dex3StateReader:
        reader = object.__new__(G1Dex3StateReader)
        reader._simulation = False
        reader._arm_max_age_s = ACTUATOR_ARM_STATE_MAX_AGE_S
        reader._hand_max_age_s = ACTUATOR_HAND_STATE_MAX_AGE_S
        reader._arm_max_age_constant = "ACTUATOR_ARM_STATE_MAX_AGE_S"
        reader._hand_max_age_constant = "ACTUATOR_HAND_STATE_MAX_AGE_S"
        reader._arm_indices = tuple(range(14))
        reader._left_indices = tuple(range(7))
        reader._right_indices = tuple(range(7))
        reader._lock = threading.Lock()
        arm = SimpleNamespace(
            mode_machine=6,
            motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(14)],
        )
        hand = SimpleNamespace(
            motor_state=[SimpleNamespace(q=0.1 if index == 0 else 0.0) for index in range(7)]
        )
        reader._messages = {"arm": arm, "left": hand, "right": hand}
        reader._updated_at = {"arm": now, "left": now, "right": now}
        reader._rejected_zero_hand_frames = {"left": 0, "right": 0}
        return reader

    def test_arm_remains_hard_at_75ms_while_hand_uses_configured_deadline(self):
        now = 100.0
        reader = self._reader(now)
        reader._updated_at["left"] = now - 0.10
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
            return_value=now,
        ):
            state = reader.latest()
        self.assertAlmostEqual(state.left_hand_received_at, now - 0.10)

        reader._updated_at["arm"] = now - 0.076
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
                return_value=now,
            ),
            self.assertRaisesRegex(
                TimeoutError,
                r"arm .*ACTUATOR_ARM_STATE_MAX_AGE_S=0\.075s",
            ),
        ):
            reader.latest()

        reader._updated_at["arm"] = now
        reader._updated_at["left"] = now - (ACTUATOR_HAND_STATE_MAX_AGE_S + 0.001)
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
                return_value=now,
            ),
            self.assertRaisesRegex(
                TimeoutError,
                rf"left .*ACTUATOR_HAND_STATE_MAX_AGE_S="
                rf"{ACTUATOR_HAND_STATE_MAX_AGE_S:.3f}s",
            ),
        ):
            reader.latest()

    def test_missing_state_names_the_searchable_deadline_constant(self):
        now = 100.0
        reader = self._reader(now)
        reader._messages["right"] = None
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
                return_value=now,
            ),
            self.assertRaisesRegex(
                TimeoutError,
                rf"right .*ACTUATOR_HAND_STATE_MAX_AGE_S="
                rf"{ACTUATOR_HAND_STATE_MAX_AGE_S:.3f}s",
            ),
        ):
            reader.latest()


class ReleaseSchedulerTests(unittest.TestCase):
    def test_zero_weight_release_is_one_arm_write_then_both_hand_stops(self):
        backend = object.__new__(_G1Dex3CommandBackend)
        backend._released = False
        backend._has_published = True
        backend.simulation = False
        backend._weight = 0.0
        arm_weights: list[float] = []
        phases: list[str] = []
        # Release deliberately bypasses live gravity computation and reuses the
        # last successfully published q/tau pair.  This scheduler test stubs
        # that release-only write rather than the normal gravity-aware writer.
        backend._publish_last_arm_for_release = lambda: arm_weights.append(backend._weight)
        backend._stop_hands = lambda callback=None: (
            callback("left_hand_stop_begin", {}) if callback is not None else None,
            callback("left_hand_stop_end", {}) if callback is not None else None,
            callback("right_hand_stop_begin", {}) if callback is not None else None,
            callback("right_hand_stop_end", {}) if callback is not None else None,
        )

        backend.release(lambda event, _payload: phases.append(event))

        self.assertEqual(arm_weights, [0.0])
        self.assertIn("arm_release_end", phases)
        self.assertIn("left_hand_stop_begin", phases)
        self.assertIn("right_hand_stop_begin", phases)


class ActivePauseIntegrationTests(unittest.TestCase):
    def test_sync_plan_is_discarded_before_a_second_target_advances(self):
        class FakeHeartbeat:
            def __init__(self) -> None:
                self.value = time.monotonic()
                self._lock = threading.Lock()

            def get_lock(self):
                return self._lock

        class PausingBackend:
            instance = None

            def __init__(self, simulation, _network_interface):
                type(self).instance = self
                self.simulation = simulation
                self._arm_target = np.zeros(14)
                self._left_target = np.zeros(7)
                self._right_target = np.zeros(7)
                self.stale_hands = False
                self.first_policy_target = threading.Event()
                self.policy_targets: list[float] = []
                self.released = False
                self.closed = False

            def state(self):
                now = time.monotonic()
                hand_at = now - 0.10 if self.stale_hands else now
                return RobotState(
                    captured_at=min(now, hand_at),
                    mode_machine=0,
                    arm=self._arm_target.copy(),
                    arm_dq=np.zeros(14),
                    left_hand=self._left_target.copy(),
                    right_hand=self._right_target.copy(),
                    left_hand_received_at=hand_at,
                    right_hand_received_at=hand_at,
                    arm_received_at=now,
                )

            def prepare_measured_hold(self):
                return self.state()

            def set_weight(self, _weight):
                pass

            def set_target(self, arm, left, right):
                self._arm_target = np.asarray(arm).copy()
                self._left_target = np.asarray(left).copy()
                self._right_target = np.asarray(right).copy()
                value = float(self._arm_target[0])
                if value > 0.0:
                    self.policy_targets.append(value)
                    self.first_policy_target.set()

            def publish(self):
                pass

            def release(self):
                self.released = True

            def close(self):
                self.closed = True

        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat()
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
        with (
            mock.patch(f"{module}._G1Dex3CommandBackend", PausingBackend),
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
                    InitializationSpec(
                        mode="measured",
                        label="measured hold",
                        arm=None,
                        left_hand=None,
                        right_hand=None,
                    ),
                )
            )
            self.assertEqual(statuses.get(timeout=1.0)[0], "initializing")
            self.assertEqual(statuses.get(timeout=1.0), ("initialized", "measured"))

            arm = np.zeros((4, 14))
            arm[:, 0] = np.array([0.01, 0.02, 0.03, 0.04])
            commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    arm,
                    np.zeros((4, 7)),
                    np.zeros((4, 7)),
                )
            )
            backend = PausingBackend.instance
            self.assertTrue(backend.first_policy_target.wait(timeout=1.0))
            backend.stale_hands = True

            pause_seen = False
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not pause_seen:
                kind, _value = statuses.get(timeout=0.2)
                pause_seen = kind == "hand_state_pause"
            self.assertTrue(pause_seen)
            backend.stale_hands = False

            terminal_seen = False
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not terminal_seen:
                kind, value = statuses.get(timeout=0.2)
                terminal_seen = kind == "holding" and value == 1
            self.assertTrue(terminal_seen)
            self.assertLessEqual(max(backend.policy_targets), 0.01)

            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(backend.released)
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
