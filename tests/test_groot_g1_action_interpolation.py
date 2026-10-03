"""Offline reference-timing checks using a mocked actuator, never DDS."""

from contextlib import contextmanager
from functools import partial
import threading
import time
import unittest
from unittest import mock

import numpy as np

import test_groot_g1_rtc as rtc_tests
from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser, validate_args
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.g1_end_effectors import INSPIRE_FTP_PROFILE
from unitree_lerobot.eval_robot.robot_control.policy_reference import PolicyReferenceSampler
from unitree_lerobot.eval_robot.robot_control import safe_g1_dex3 as safe


@contextmanager
def linear_child():
    with mock.patch.object(rtc_tests, "_actuator_main", partial(safe._actuator_main, action_interpolation="linear")):
        with rtc_tests._ChildHarness(
            command_conditioning="xr", conditioner_type=rtc_tests._RecordingConditioner
        ) as child:
            yield child


class ActionInterpolationTests(unittest.TestCase):
    def wait_until(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        self.fail("mock actuator did not reach the expected condition")

    def test_cli_keeps_legacy_default_and_linear_requires_conditioning(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).action_interpolation, "legacy")
        args = parser.parse_args([
            "--no-actuate", "--end-effector", "dex3",
            "--action-interpolation", "linear", "--command-conditioning", "none",
        ])
        with self.assertRaisesRegex(DeploymentError, "requires --command-conditioning xr"):
            validate_args(args)

    def test_actuator_rejects_unsupported_mode_before_spawning(self):
        with self.assertRaisesRegex(DeploymentError, "must be legacy or linear"):
            safe.SafeG1Dex3Actuator(True, None, action_interpolation="cubic")
        with self.assertRaisesRegex(DeploymentError, "requires XR"):
            safe.SafeG1Dex3Actuator(True, None, action_interpolation="linear")

    def test_linear_samples_intermediate_targets_and_preserves_final_budget(self):
        plan = rtc_tests._plan(8)
        with linear_child() as child:
            child.commands.put(rtc_tests._rtc_start_command(1, plan, action_budget=3))
            child.assert_status("rtc_started")
            self.assertEqual(child.assert_status("rtc_completed"), 3)
            spy = rtc_tests._RecordingConditioner.instance
            values = np.array([target[0][0] for target in spy.desired])
            self.assertTrue(np.any((values > 0) & (values < 0.01)))
            self.assertLessEqual(float(values.max()), 0.02 + 1e-12)
            self.assertTrue(np.any(np.isclose(values, 0.02)))
            commands = [target[0] for target in child.backend.target_snapshot()]
            self.assertTrue(all(np.max(np.abs(b - a)) <= safe.MAX_CONDITIONED_ARM_STEP_RAD + 1e-12
                                for a, b in zip(commands, commands[1:])))

    def test_one_authorized_action_never_approaches_the_next_row(self):
        with linear_child() as child:
            child.commands.put(rtc_tests._rtc_start_command(1, rtc_tests._plan(8), action_budget=1))
            child.assert_status("rtc_started")
            child.assert_status("rtc_completed")
            for arm, _, _ in rtc_tests._RecordingConditioner.instance.desired:
                np.testing.assert_array_equal(arm, np.zeros(14))

    def test_urgent_hold_stops_sampling(self):
        with linear_child() as child:
            child.commands.put(rtc_tests._rtc_start_command(1, rtc_tests._plan(32), action_budget=32))
            child.assert_status("rtc_started")
            child.wait_for_target_count(2)
            child.urgent_hold.set()
            child.assert_status("holding")
            count = len(rtc_tests._RecordingConditioner.instance.desired)
            time.sleep(0.04)
            self.assertEqual(len(rtc_tests._RecordingConditioner.instance.desired), count)

    def test_legacy_only_sets_original_knots(self):
        with rtc_tests._ChildHarness(command_conditioning="xr", conditioner_type=rtc_tests._RecordingConditioner) as child:
            child.commands.put(rtc_tests._rtc_start_command(1, rtc_tests._plan(8), action_budget=3))
            child.assert_status("rtc_started")
            child.assert_status("rtc_completed")
            values = [target[0][0] for target in rtc_tests._RecordingConditioner.instance.desired]
            np.testing.assert_allclose(values, [0.0, 0.01, 0.02])

    def test_midperiod_rtc_replacement_preserves_active_interval_and_clock(self):
        class ControlledClock:
            def __init__(self):
                self.frozen_at = None

            def monotonic(self):
                return time.monotonic() if self.frozen_at is None else self.frozen_at

            def monotonic_ns(self):
                return int(self.monotonic() * 1e9)

            def __getattr__(self, name):
                return getattr(time, name)

        clock = ControlledClock()

        class RecordingSampler(PolicyReferenceSampler):
            instance = None

            def __init__(self):
                super().__init__()
                type(self).instance = self
                self.segments = []
                self.samples = []
                self.first_segment_started = threading.Event()

            def start_segment(self, start, end, *, starts_at, period_s):
                super().start_segment(start, end, starts_at=starts_at, period_s=period_s)
                self.segments.append((start.copy(), end.copy(), starts_at, period_s))
                if len(self.segments) == 1:
                    # Hold the child clock inside its very first interval.
                    # Queue processing continues, but no policy knot can become
                    # due until this test explicitly advances the clock.
                    clock.frozen_at = starts_at + 0.01
                    self.first_segment_started.set()

            def sample(self, now):
                reference = super().sample(now)
                self.samples.append((now, reference.copy()))
                return reference

        plan = rtc_tests._plan(32)
        replacement = rtc_tests._plan(32)
        # Keep the old prefix but dramatically change the unfrozen future.
        # The in-progress old 0 -> .01 interval must not start chasing .17.
        replacement.arm[2:, 0] += 0.15
        with mock.patch.object(safe, "time", clock), mock.patch.object(
            safe, "PolicyReferenceSampler", RecordingSampler
        ), linear_child() as child:
            child.commands.put(rtc_tests._rtc_start_command(1, plan, action_budget=20))
            child.assert_status("rtc_started")
            sampler = RecordingSampler.instance
            self.assertTrue(sampler.first_segment_started.wait(timeout=1.0))
            self.wait_until(lambda: len(sampler.samples) >= 2)
            _, _, starts_at, period_s = sampler.segments[0]
            reference_before = sampler.samples[-1][1].copy()
            child.commands.put((
                "rtc_replace", 2, clock.monotonic(), 1, 0, plan.length,
                replacement.arm, replacement.left_hand, replacement.right_hand,
                replacement.length,
            ))
            acknowledgement = child.assert_status("rtc_replaced")
            self.assertEqual(acknowledgement["action_index"], 1)
            sample_count = len(sampler.samples)
            self.wait_until(lambda: len(sampler.samples) > sample_count)
            self.assertEqual(len(sampler.segments), 1)
            np.testing.assert_array_equal(sampler.samples[-1][1], reference_before)
            self.assertAlmostEqual(reference_before[0], 0.003, places=7)

            # At the *old* clock's next boundary, use B[1] -> B[2]. A reset
            # action clock, skipped row, or premature replacement fails here.
            clock.frozen_at = starts_at + period_s
            self.wait_until(lambda: len(sampler.segments) >= 2)
            new_start, new_end, new_time, new_period = sampler.segments[1]
            self.assertEqual(new_time, starts_at + period_s)
            self.assertEqual(new_period, period_s)
            self.assertAlmostEqual(new_start[0], 0.01)
            self.assertAlmostEqual(new_end[0], 0.17)
            self.assertEqual(rtc_tests._RecordingConditioner.instance.reset_count, 1)

    def test_inspire_26d_reference_preserves_hand_slices_and_conditioner_bounds(self):
        arm_start = np.zeros(14)
        arm_end = arm_start.copy()
        arm_end[:2] = (0.06, 0.03)
        left_start, left_end = np.linspace(0.1, 0.2, 6), np.linspace(0.9, 0.8, 6)
        right_start, right_end = np.linspace(0.8, 0.9, 6), np.linspace(0.2, 0.1, 6)
        start = np.concatenate((arm_start, left_start, right_start))
        end = np.concatenate((arm_end, left_end, right_end))
        sampler = PolicyReferenceSampler()
        sampler.start_segment(start, end, starts_at=0.0, period_s=1.0 / 30.0)
        reference = sampler.sample(0.02)
        self.assertEqual(reference.shape, (26,))
        np.testing.assert_allclose(reference, 0.4 * start + 0.6 * end)

        conditioner = safe.XrPolicyOutputConditioner(INSPIRE_FTP_PROFILE)
        conditioner.reset(arm_start, left_start, right_start)
        conditioner.set_desired(reference[:14], reference[14:20], reference[20:], now=0.02)
        command = conditioner.next_command(
            arm_start, arm_start, left_start, right_start, now=0.02
        )
        self.assertEqual(command.arm.shape, (1, 14))
        self.assertEqual(command.left_hand.shape, (1, 6))
        self.assertEqual(command.right_hand.shape, (1, 6))
        np.testing.assert_array_equal(conditioner._desired_left, reference[14:20])
        np.testing.assert_array_equal(conditioner._desired_right, reference[20:26])
        self.assertTrue(np.all(np.abs(command.arm[0] - arm_start) <= safe.MAX_CONDITIONED_ARM_STEP_RAD + 1e-12))
        for values, origin, desired in (
            (command.left_hand[0], left_start, reference[14:20]),
            (command.right_hand[0], right_start, reference[20:]),
        ):
            self.assertTrue(np.all(values >= 0.0))
            self.assertTrue(np.all(values <= 1.0))
            self.assertTrue(np.all(np.abs(values - origin) <= INSPIRE_FTP_PROFILE.conditioned_step + 1e-12))
            self.assertTrue(np.all(values >= np.minimum(origin, desired)))
            self.assertTrue(np.all(values <= np.maximum(origin, desired)))


if __name__ == "__main__":
    unittest.main()
