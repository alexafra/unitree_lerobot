from __future__ import annotations

import argparse
import ast
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot import diagnose_g1_dds_hold
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    DdsHoldTimingAccumulator,
    RobotState,
)


class DdsHoldTimingAccumulatorTest(unittest.TestCase):
    def test_attributes_individual_writes_state_ages_and_cycle_jitter(self) -> None:
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
            side_effect=(10.0, 10.030),
        ):
            accumulator = DdsHoldTimingAccumulator(period_s=0.010)
            state = RobotState(
                captured_at=10.008,
                mode_machine=6,
                arm=np.zeros(14),
                arm_dq=np.zeros(14),
                left_hand=np.zeros(7),
                right_hand=np.zeros(7),
                arm_received_at=10.008,
                left_hand_received_at=10.006,
                right_hand_received_at=10.004,
            )
            accumulator.record(
                loop_started=10.010,
                state_lookup_ms=0.2,
                state=state,
                publish_timing_ms={
                    "arm_state_check": 0.1,
                    "arm_crc": 0.2,
                    "arm_write": 1.0,
                    "left_write": 2.0,
                    "right_write": 3.0,
                    "publish_total": 6.3,
                },
                completed_at=10.018,
            )
            accumulator.record(
                loop_started=10.030,
                state_lookup_ms=0.4,
                state=state,
                publish_timing_ms={
                    "arm_state_check": 0.2,
                    "arm_crc": 0.3,
                    "arm_write": 4.0,
                    "left_write": 5.0,
                    "right_write": 100.0,
                    "publish_total": 109.5,
                },
                completed_at=10.140,
            )
            summary = accumulator.summary()

        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["metrics"]["cycle_ms"]["max"], 20.0)
        self.assertAlmostEqual(summary["metrics"]["right_write_ms"]["max"], 100.0)
        self.assertEqual(summary["slowest"][0]["worst_stage"], "right_write_ms")
        self.assertEqual(summary["overruns"]["work_over_100ms"], 1)
        self.assertGreater(summary["metrics"]["right_age_ms"]["max"], 100.0)


class DdsHoldDiagnosticCliTest(unittest.TestCase):
    def test_cli_only_starts_arms_monitors_and_closes(self) -> None:
        calls: list[str] = []

        class FakeActuator:
            def __init__(self, **kwargs):
                calls.append(f"construct:{kwargs}")
                self.last_dds_hold_timing = {"samples": 1}

            def start(self):
                calls.append("start")

            def arm(self, timeout_s):
                calls.append(f"arm:{timeout_s}")

            def heartbeat(self):
                calls.append("heartbeat")

            def assert_healthy(self):
                calls.append("assert_healthy")

            def close(self):
                calls.append("close")

        args = argparse.Namespace(
            network_interface="enp0s1",
            duration=0.0,
            allow_unqualified_real=True,
        )
        with (
            mock.patch.object(diagnose_g1_dds_hold, "_parse_args", return_value=args),
            mock.patch.object(diagnose_g1_dds_hold, "_confirm"),
            mock.patch.object(diagnose_g1_dds_hold, "SafeG1Dex3Actuator", FakeActuator),
            mock.patch.object(diagnose_g1_dds_hold, "_print_summary"),
        ):
            result = diagnose_g1_dds_hold.main()

        self.assertEqual(result, 0)
        self.assertIn("start", calls)
        self.assertIn(f"arm:{diagnose_g1_dds_hold.DIAGNOSTIC_ARM_TIMEOUT_S}", calls)
        self.assertEqual(calls[-1], "close")
        self.assertFalse(any("initialize" in call or "warm_start" in call or "submit" in call for call in calls))

    def test_arm_failure_still_closes(self) -> None:
        calls: list[str] = []

        class FakeActuator:
            last_dds_hold_timing = None

            def __init__(self, **_kwargs):
                pass

            def start(self):
                calls.append("start")

            def arm(self, timeout_s):
                calls.append("arm")
                raise RuntimeError("injected")

            def close(self):
                calls.append("close")

        args = argparse.Namespace(network_interface="enp0s1", duration=0.0, allow_unqualified_real=True)
        with (
            mock.patch.object(diagnose_g1_dds_hold, "_parse_args", return_value=args),
            mock.patch.object(diagnose_g1_dds_hold, "_confirm"),
            mock.patch.object(diagnose_g1_dds_hold, "SafeG1Dex3Actuator", FakeActuator),
        ):
            result = diagnose_g1_dds_hold.main()

        self.assertEqual(result, 1)
        self.assertEqual(calls, ["start", "arm", "close"])

    def test_entrypoint_has_no_policy_or_camera_imports(self) -> None:
        source = Path(diagnose_g1_dds_hold.__file__).read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(alias.name for alias in node.names)
        forbidden = ("eval_groot_g1", "Gr00tClient", "TeleimagerCamera", "groot_contract", "image_server")
        for name in forbidden:
            self.assertFalse(any(name in imported_name for imported_name in imported), name)


if __name__ == "__main__":
    unittest.main()
