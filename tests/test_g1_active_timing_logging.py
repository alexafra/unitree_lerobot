from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot import eval_groot_g1
from unitree_lerobot.eval_robot import run_logging
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import ActiveTimingRing, RobotState
from unitree_lerobot.eval_robot.groot_contract import load_initialization_spec
from unitree_lerobot.eval_robot.robot_control import safe_g1_dex3
from tests.test_groot_g1_rtc import _GatedRecordingBackend, _LiveHeartbeat, _plan


class ActiveTimingRingTest(unittest.TestCase):
    def test_snapshot_is_bounded_chronological_and_strict_json(self) -> None:
        ring = ActiveTimingRing(capacity=2)
        started = time.monotonic()
        backend = SimpleNamespace(
            _arm_target=np.arange(14, dtype=np.float64),
            _left_target=np.arange(7, dtype=np.float64),
            _right_target=-np.arange(7, dtype=np.float64),
        )

        for index in range(3):
            loop_started = started + index * 0.010
            ring.begin(
                loop_started=loop_started,
                heartbeat_age_s=0.001 * index,
                sequence=7,
                action_index=index,
                chunk_length=8,
                rtc=False,
                rtc_total_actions=0,
                rtc_action_budget=0,
                holding=False,
            )
            state = RobotState(
                captured_at=loop_started - 0.001,
                mode_machine=6,
                arm=np.full(14, index, dtype=np.float64),
                arm_dq=np.zeros(14, dtype=np.float64),
                left_hand=np.full(7, index, dtype=np.float64),
                right_hand=np.full(7, -index, dtype=np.float64),
                arm_received_at=loop_started - 0.001,
                left_hand_received_at=loop_started - 0.002,
                right_hand_received_at=loop_started - 0.003,
            )
            ring.note_state(state, completed_at=loop_started, lookup_ms=0.25 + index)
            ring.update(command_kind=f"command-{index}")
            ring.note_publish(
                {
                    "arm_crc": float("nan") if index == 2 else 0.10,
                    "arm_write": float("inf") if index == 2 else 0.20,
                    "left_write": 0.30,
                    "right_write": 0.40,
                    "publish_total": 1.00,
                }
            )
            ring.finish(completed_at=loop_started + 0.001)

        payload = ring.snapshot(
            trigger="action_scheduler_late",
            error="DeploymentError: Action scheduler is 0.250s late",
            backend=backend,
            command_conditioning="xr",
        )

        self.assertEqual(payload["capacity"], 2)
        self.assertEqual(payload["total_records"], 3)
        self.assertEqual(payload["retained_records"], 2)
        self.assertEqual(payload["dropped_records"], 1)
        self.assertEqual([record["sample"] for record in payload["records"]], [2, 3])
        self.assertEqual(
            [record["command_kind"] for record in payload["records"]],
            ["command-1", "command-2"],
        )
        self.assertFalse(any(key.startswith("_") for record in payload["records"] for key in record))
        self.assertIsNone(payload["records"][-1]["arm_crc_ms"])
        self.assertIsNone(payload["records"][-1]["arm_write_ms"])
        self.assertEqual(payload["last_state"]["mode_machine"], 6)
        self.assertEqual(payload["last_state"]["arm"][0], 2.0)
        self.assertEqual(payload["last_targets"]["arm"], backend._arm_target.tolist())

        # This is deliberately stricter than Python's default JSON encoder.
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded)["records"][-1]["arm_crc_ms"], None)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "active_timing.json"
            self.assertEqual(run_logging.write_json(destination, payload), destination)
            decoded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(decoded["total_records"], 3)
            self.assertEqual(list(Path(temporary).glob(".*.tmp-*")), [])

    def test_scheduler_gap_replans_and_dumps_timing_after_orderly_cleanup(self) -> None:
        commands: queue.Queue = queue.Queue(maxsize=1)
        statuses: queue.Queue = queue.Queue(maxsize=32)
        stop = threading.Event()
        urgent = threading.Event()
        heartbeat = _LiveHeartbeat()

        def await_status(expected: str, timeout_s: float = 2.0):
            deadline = time.monotonic() + timeout_s
            seen = []
            while time.monotonic() < deadline:
                kind, payload = statuses.get(timeout=max(0.01, deadline - time.monotonic()))
                seen.append((kind, payload))
                if kind == expected:
                    return payload
            raise AssertionError(f"missing {expected!r}; saw {seen!r}")

        def write_after_cleanup(path, payload):
            backend = _GatedRecordingBackend.instance
            assert backend is not None
            self.assertTrue(backend.released)
            self.assertTrue(backend.closed)
            self.assertIn(payload["trigger"], {"orderly_shutdown", "action_scheduler_replan"})
            return run_logging.write_json(path, payload)

        with tempfile.TemporaryDirectory() as temporary, (
            mock.patch.object(safe_g1_dex3, "_G1Dex3CommandBackend", _GatedRecordingBackend)
        ), mock.patch.object(safe_g1_dex3, "configure_process_logging"), mock.patch.object(
            safe_g1_dex3, "INITIALIZATION_START_DWELL_S", 0.0
        ), mock.patch.object(
            safe_g1_dex3, "INITIALIZATION_CONVERGENCE_DWELL_S", 0.0
        ), mock.patch.object(
            safe_g1_dex3, "INITIALIZATION_MIN_DISTINCT_SAMPLES", 1
        ), mock.patch.object(
            safe_g1_dex3, "INITIALIZATION_MIN_MOVE_S", 0.0
        ), mock.patch.object(
            safe_g1_dex3, "MAX_ACTION_LATENESS_S", 0.050
        ), mock.patch.object(
            safe_g1_dex3, "write_json", side_effect=write_after_cleanup
        ):
            worker = threading.Thread(
                target=safe_g1_dex3._actuator_main,
                args=(
                    True,
                    None,
                    commands,
                    statuses,
                    stop,
                    heartbeat,
                    urgent,
                    "none",
                    False,
                    False,
                    temporary,
                ),
                daemon=True,
            )
            worker.start()
            await_status("ready")
            commands.put(("arm",), timeout=0.2)
            await_status("armed")
            commands.put(
                (
                    "initialize",
                    time.monotonic(),
                    load_initialization_spec("measured", task_name="pick-red-cup"),
                ),
                timeout=0.2,
            )
            await_status("initializing")
            await_status("initialized")

            backend = _GatedRecordingBackend.instance
            assert backend is not None
            backend.clear_targets()
            plan = _plan(8)
            commands.put(
                ("chunk", 1, time.monotonic(), plan.arm, plan.left_hand, plan.right_hand),
                timeout=0.2,
            )
            deadline = time.monotonic() + 1.0
            while len(backend.target_snapshot()) < 1 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertGreaterEqual(len(backend.target_snapshot()), 1)
            backend.pause_next_publish.set()
            self.assertTrue(backend.publish_paused.wait(timeout=1.0))
            time.sleep(0.080)
            backend.continue_publish.set()

            replan = await_status("replan_required")
            self.assertEqual(replan["sequence"], 1)
            self.assertGreater(replan["lateness_s"], 0.050)
            self.assertTrue(worker.is_alive())
            self.assertFalse(backend.released)
            self.assertFalse(backend.closed)

            stop.set()
            await_status("stopped")
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
            self.assertTrue(backend.released)
            self.assertTrue(backend.closed)

            dumps = list(Path(temporary).glob("active_timing_action_scheduler_replan_*.json"))
            self.assertEqual(len(dumps), 1)
            payload = json.loads(dumps[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["trigger"], "action_scheduler_replan")
            self.assertIsNone(payload["error"])
            fault_record = next(
                record for record in payload["records"] if record["event"] == "action_scheduler_replan"
            )
            self.assertGreater(fault_record["scheduler_lateness_ms"], 50.0)
            self.assertFalse(fault_record["publish_called"])
            self.assertIsNone(fault_record["publish_total_ms"])


class RunLoggingTest(unittest.TestCase):
    @staticmethod
    def _fixed_utc_now() -> datetime:
        return datetime(2026, 8, 13, 5, 4, 3, 123456, tzinfo=timezone.utc)

    def test_run_directory_and_diagnostic_paths_are_collision_safe(self) -> None:
        fixed_datetime = mock.Mock()
        fixed_datetime.now.return_value = self._fixed_utc_now()
        with tempfile.TemporaryDirectory() as temporary, (
            mock.patch.object(run_logging, "datetime", fixed_datetime)
        ), mock.patch.object(run_logging.os, "getpid", return_value=4321):
            first_run = run_logging.create_run_directory(Path(temporary) / "nested", prefix="test")
            second_run = run_logging.create_run_directory(Path(temporary) / "nested", prefix="test")

            self.assertTrue(first_run.is_absolute())
            self.assertTrue(first_run.is_dir())
            self.assertTrue(second_run.is_dir())
            self.assertEqual(second_run.name, f"{first_run.name}_01")

            first_dump = run_logging.diagnostic_json_path(first_run, "Action scheduler LATE! / unsafe")
            first_dump.touch()
            second_dump = run_logging.diagnostic_json_path(first_run, "Action scheduler LATE! / unsafe")
            self.assertRegex(first_dump.name, r"^active_timing_action_scheduler_late_unsafe_.*_pid4321\.json$")
            self.assertEqual(second_dump.stem, f"{first_dump.stem}_01")

    def test_write_json_rejects_non_finite_values_without_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                run_logging.write_json(destination, {"bad": float("nan")})
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_process_logging_writes_one_process_owned_file_handler(self) -> None:
        isolated_root = logging.Logger("isolated-root")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "actuator.log"
            with mock.patch.object(run_logging.logging, "getLogger", return_value=isolated_root):
                run_logging.configure_process_logging(destination)
                isolated_root.info("timing marker")

            try:
                self.assertEqual(len(isolated_root.handlers), 2)
                self.assertEqual(
                    sum(isinstance(handler, logging.FileHandler) for handler in isolated_root.handlers),
                    1,
                )
                contents = destination.read_text(encoding="utf-8")
                self.assertIn("timing marker", contents)
                self.assertIn("process=", contents)
                self.assertIn("thread=", contents)
            finally:
                for handler in list(isolated_root.handlers):
                    isolated_root.removeHandler(handler)
                    handler.close()

    def test_terminal_yellow_is_opt_in_and_file_log_remains_plain(self) -> None:
        isolated_root = logging.Logger("isolated-root")
        terminal = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "parent.log"
            with (
                mock.patch.object(run_logging.logging, "getLogger", return_value=isolated_root),
                mock.patch.object(run_logging.sys, "stderr", terminal),
            ):
                run_logging.configure_process_logging(destination)
                isolated_root.warning("plain warning")
                isolated_root.warning("highlighted warning", extra={"terminal_yellow": True})

            try:
                terminal_lines = terminal.getvalue().splitlines()
                plain_line = next(line for line in terminal_lines if "plain warning" in line)
                highlighted_line = next(line for line in terminal_lines if "highlighted warning" in line)
                self.assertNotIn("\x1b", plain_line)
                self.assertTrue(highlighted_line.startswith(run_logging.TERMINAL_YELLOW))
                self.assertTrue(highlighted_line.endswith(run_logging.TERMINAL_RESET))

                contents = destination.read_text(encoding="utf-8")
                self.assertIn("plain warning", contents)
                self.assertIn("highlighted warning", contents)
                self.assertNotIn("\x1b", contents)
            finally:
                for handler in list(isolated_root.handlers):
                    isolated_root.removeHandler(handler)
                    handler.close()

    def test_normal_runner_main_creates_manifest_and_passes_absolute_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            run_directory.mkdir()
            argv = [
                "eval_groot_g1",
                "--task",
                "pick-red-cup",
                "--log-dir",
                str(Path(temporary) / "root"),
            ]
            with (
                mock.patch.object(eval_groot_g1.sys, "argv", argv),
                mock.patch.object(eval_groot_g1, "create_run_directory", return_value=run_directory),
                mock.patch.object(eval_groot_g1, "configure_process_logging") as configure,
                mock.patch.object(eval_groot_g1, "run") as run,
                mock.patch.object(eval_groot_g1.signal, "signal"),
            ):
                eval_groot_g1.main()

            configure.assert_called_once_with(run_directory / "parent.log")
            args = run.call_args.args[0]
            self.assertEqual(args._run_log_dir, str(run_directory))
            manifest = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_log_dir"], str(run_directory))
            self.assertEqual(manifest["arguments"]["task"], "pick-red-cup")
            self.assertEqual(manifest["arguments"]["log_dir"], str(Path(temporary) / "root"))


if __name__ == "__main__":
    unittest.main()
