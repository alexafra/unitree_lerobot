from __future__ import annotations

import os
import pty
import sys
import termios
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _OperatorTerminal,
    _run_blocking_motion_with_immediate_release,
    OperatorRelease,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import ImmediateControlEvent


class BlockingTransitionKeyTests(unittest.TestCase):
    def test_s_cancels_blocking_transition_through_orderly_release(self) -> None:
        for key in (b"s", b"S"):
            with self.subTest(key=key):
                master_fd, slave_fd = pty.openpty()
                stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
                original = termios.tcgetattr(slave_fd)
                operation_started = threading.Event()
                release_requested = threading.Event()
                writer_finished = threading.Event()
                actuator = SimpleNamespace(
                    request_immediate_hold=mock.Mock(),
                    request_immediate_release=mock.Mock(
                        side_effect=lambda: release_requested.set()
                    ),
                    immediate_control_requested=mock.Mock(
                        side_effect=lambda: "release" if release_requested.is_set() else None
                    ),
                )

                def blocking_operation() -> None:
                    operation_started.set()
                    if not release_requested.wait(timeout=1.0):
                        raise AssertionError("s did not cancel the blocking transition")
                    raise ImmediateControlEvent("release")

                def write_during_operation() -> None:
                    try:
                        if operation_started.wait(timeout=1.0):
                            os.write(master_fd, key)
                    finally:
                        writer_finished.set()

                writer = threading.Thread(target=write_during_operation, daemon=True)
                try:
                    writer.start()
                    with self.assertLogs("eval_groot_g1", level="WARNING") as captured:
                        with (
                            mock.patch.object(sys, "stdin", stdin),
                            self.assertRaises(OperatorRelease),
                        ):
                            _run_blocking_motion_with_immediate_release(
                                actuator,
                                blocking_operation,
                                stage="TEST TRANSITION",
                            )

                    writer.join(timeout=1.0)
                    self.assertTrue(writer_finished.is_set())
                    self.assertTrue(
                        any(
                            "s or q" in line and "powered HOLD is unavailable" in line
                            for line in captured.output
                        )
                    )
                    actuator.request_immediate_release.assert_called_once_with()
                    actuator.request_immediate_hold.assert_not_called()
                    self.assertEqual(termios.tcgetattr(slave_fd), original)
                finally:
                    writer.join(timeout=1.0)
                    stdin.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_buffered_s_releases_before_blocking_operation_starts(self) -> None:
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        original = termios.tcgetattr(slave_fd)
        actuator = SimpleNamespace(
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
            immediate_control_requested=mock.Mock(return_value="release"),
        )
        operation = mock.Mock()
        try:
            os.write(master_fd, b"s")
            with (
                mock.patch.object(sys, "stdin", stdin),
                self.assertRaises(OperatorRelease),
            ):
                _run_blocking_motion_with_immediate_release(actuator, operation)

            operation.assert_not_called()
            actuator.request_immediate_release.assert_called_once_with()
            actuator.request_immediate_hold.assert_not_called()
            self.assertEqual(termios.tcgetattr(slave_fd), original)
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_active_terminal_keeps_s_as_powered_hold(self) -> None:
        master_fd, slave_fd = pty.openpty()
        stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
        actuator = SimpleNamespace(
            request_immediate_hold=mock.Mock(),
            request_immediate_release=mock.Mock(),
        )
        try:
            with mock.patch.object(sys, "stdin", stdin), _OperatorTerminal(actuator) as terminal:
                os.write(master_fd, b"s")
                deadline = time.monotonic() + 1.0
                response = None
                while response is None and time.monotonic() < deadline:
                    response = terminal.poll_control()
                    time.sleep(0.01)

            self.assertEqual(response, "hold")
            actuator.request_immediate_hold.assert_called_once_with()
            actuator.request_immediate_release.assert_not_called()
        finally:
            stdin.close()
            os.close(master_fd)
            os.close(slave_fd)

    def test_rejects_unknown_stop_action(self) -> None:
        actuator = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "Unsupported operator stop action"):
            _OperatorTerminal(actuator, stop_action="unsafe")


if __name__ == "__main__":
    unittest.main()
