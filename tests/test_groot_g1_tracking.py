from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    HAND_COMMAND_HISTORY_SIZE,
    HandTrackingWatchdog,
    PublishedHandTarget,
    RobotState,
    _G1Dex3CommandBackend,
    _enforce_tracking,
)


class _TrackingBackend:
    def __init__(self) -> None:
        self._arm_target = np.zeros(14)
        self._left_target = np.zeros(7)
        self._right_target = np.zeros(7)
        self._left_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        self._right_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)

    def publish_at(
        self,
        completed_at: float,
        *,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        if left is not None:
            self._left_target = np.asarray(left, dtype=np.float64).copy()
        if right is not None:
            self._right_target = np.asarray(right, dtype=np.float64).copy()
        self._left_hand_publish_history.append(PublishedHandTarget(completed_at, self._left_target.copy()))
        self._right_hand_publish_history.append(PublishedHandTarget(completed_at, self._right_target.copy()))

    def reset_hand_publish_history(self) -> None:
        self._left_hand_publish_history.clear()
        self._right_hand_publish_history.clear()


def _state(
    sample_at: float,
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
) -> RobotState:
    return RobotState(
        captured_at=sample_at,
        mode_machine=0,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.zeros(7) if left is None else np.asarray(left, dtype=np.float64),
        right_hand=np.zeros(7) if right is None else np.asarray(right, dtype=np.float64),
        left_hand_received_at=sample_at,
        right_hand_received_at=sample_at,
    )


class HandTrackingWatchdogTest(unittest.TestCase):
    def test_state_is_compared_with_command_that_existed_at_receipt_time(self):
        backend = _TrackingBackend()
        backend.publish_at(1.0)
        newest = np.zeros(7)
        newest[0] = -2.0
        backend.publish_at(2.0, left=newest)

        # The latest target differs by 2 rad, but it was published after this
        # feedback sample. The aligned 1.0-s target is exactly zero.
        _enforce_tracking(
            backend,
            _state(1.5),
            HandTrackingWatchdog(),
            now=2.1,
            context="alignment test",
        )

    def test_hard_fault_has_aligned_and_latest_telemetry(self):
        backend = _TrackingBackend()
        backend.publish_at(1.0)
        measured = np.zeros(7)
        measured[4] = 1.6

        with self.assertRaisesRegex(
            DeploymentError,
            r"time-aligned tracking error.*measured_q=.*aligned_target_q=.*context=rtc test",
        ):
            _enforce_tracking(
                backend,
                _state(1.1, right=measured),
                HandTrackingWatchdog(),
                now=1.12,
                context="rtc test",
                desired_right=np.full(7, 0.25),
            )

    def test_warning_requires_distinct_samples_and_dwell(self):
        backend = _TrackingBackend()
        backend.publish_at(1.0)
        measured = np.zeros(7)
        measured[2] = 0.6
        watchdog = HandTrackingWatchdog()

        with mock.patch("unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.LOGGER.warning") as warning:
            _enforce_tracking(backend, _state(1.05, left=measured), watchdog, now=1.05)
            # Polling the same 50 Hz sample later must not advance dwell time.
            _enforce_tracking(backend, _state(1.05, left=measured), watchdog, now=1.40)
            _enforce_tracking(backend, _state(1.20, left=measured), watchdog, now=1.20)
            warning.assert_not_called()
            _enforce_tracking(backend, _state(1.26, left=measured), watchdog, now=1.26)
            warning.assert_called_once()

    def test_warning_uses_clear_hysteresis_and_can_restart(self):
        backend = _TrackingBackend()
        backend.publish_at(1.0)
        watchdog = HandTrackingWatchdog()

        def measured(value: float) -> np.ndarray:
            result = np.zeros(7)
            result[1] = value
            return result

        with (
            mock.patch("unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.LOGGER.warning") as warning,
            mock.patch("unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.LOGGER.info") as info,
        ):
            _enforce_tracking(backend, _state(1.01, left=measured(0.6)), watchdog)
            _enforce_tracking(backend, _state(1.22, left=measured(0.6)), watchdog)
            self.assertEqual(warning.call_count, 1)

            # Between 0.4 and 0.5 rad, the active warning remains latched.
            _enforce_tracking(backend, _state(1.24, left=measured(0.45)), watchdog)
            info.assert_not_called()
            _enforce_tracking(backend, _state(1.26, left=measured(0.39)), watchdog)
            info.assert_called_once()

            _enforce_tracking(backend, _state(1.30, left=measured(0.6)), watchdog)
            _enforce_tracking(backend, _state(1.51, left=measured(0.6)), watchdog)
            self.assertEqual(warning.call_count, 2)

    def test_reset_clears_timers_and_publish_history(self):
        backend = _TrackingBackend()
        backend.publish_at(1.0)
        watchdog = HandTrackingWatchdog()
        measured = np.zeros(7)
        measured[0] = 0.6
        _enforce_tracking(backend, _state(1.1, left=measured), watchdog)

        watchdog.reset(backend)

        self.assertEqual(len(backend._left_hand_publish_history), 0)
        self.assertEqual(len(backend._right_hand_publish_history), 0)
        self.assertTrue(np.all(np.isnan(watchdog._violation_since)))
        self.assertTrue(np.all(np.isneginf(watchdog._last_sample_at)))


class SuccessfulPublishHistoryTest(unittest.TestCase):
    @staticmethod
    def _backend_without_sdk() -> _G1Dex3CommandBackend:
        backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
        backend.simulation = False
        backend._left_target = np.arange(7, dtype=np.float64)
        backend._right_target = np.arange(7, dtype=np.float64) + 10.0
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._left_message = SimpleNamespace(motor_cmd=[SimpleNamespace(q=0.0) for _ in range(7)])
        backend._right_message = SimpleNamespace(motor_cmd=[SimpleNamespace(q=0.0) for _ in range(7)])
        backend._left_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        backend._right_hand_publish_history = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        return backend

    def test_each_hand_is_recorded_only_after_its_write_succeeds(self):
        backend = self._backend_without_sdk()
        backend._left_publisher = SimpleNamespace(Write=mock.Mock(return_value=True))
        backend._right_publisher = SimpleNamespace(Write=mock.Mock(return_value=False))

        with self.assertRaisesRegex(DeploymentError, "Right Dex3 DDS Write failed"):
            backend._publish_hands()

        self.assertEqual(len(backend._left_hand_publish_history), 1)
        self.assertEqual(len(backend._right_hand_publish_history), 0)
        np.testing.assert_array_equal(
            backend._left_hand_publish_history[-1].target,
            backend._left_target,
        )


if __name__ == "__main__":
    unittest.main()
