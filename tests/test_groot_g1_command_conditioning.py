import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser, validate_args
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ARM_UPPER,
    ActionChunk,
    HAND_LIMIT_TOLERANCE_RAD,
    LEFT_HAND_LOWER,
    validate_action_chunk,
    validate_action_chunk_limits,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    MAX_CONDITIONED_ARM_STEP_RAD,
    MAX_CONDITIONED_HAND_STEP_RAD,
    SafeG1Dex3Actuator,
    XR_ARM_FINAL_COMMAND_LEAD_RAD,
    XR_ARM_INITIAL_COMMAND_LEAD_RAD,
    XrPolicyOutputConditioner,
)


class GrootG1CommandConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = np.zeros(14, dtype=np.float64)
        self.left = np.zeros(7, dtype=np.float64)
        self.right = np.zeros(7, dtype=np.float64)

    def test_raw_jump_may_be_conditioned_but_raw_limits_remain_hard(self):
        arm = np.zeros((2, 14), dtype=np.float64)
        arm[1, 0] = 0.5
        chunk = ActionChunk(arm=arm, left_hand=np.zeros((2, 7)), right_hand=np.zeros((2, 7)))

        validate_action_chunk_limits(chunk)
        with self.assertRaisesRegex(DeploymentError, "target jump"):
            validate_action_chunk(chunk, self.arm, self.left, self.right)

        arm[1, 0] = ARM_UPPER[0] + 1.0
        with self.assertRaisesRegex(DeploymentError, "outside its safety-margined joint range"):
            validate_action_chunk_limits(chunk)

    def test_xr_arm_limiter_uses_one_global_vector_scale_and_five_second_ramp(self):
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, self.left, self.right)
        desired_arm = self.arm.copy()
        desired_arm[:2] = (1.0, 0.5)
        conditioner.set_desired(desired_arm, self.left, self.right, now=10.0)

        current_arm = self.arm
        for _ in range(3):
            initial = conditioner.next_command(
                self.arm,
                current_arm,
                self.left,
                self.right,
                now=10.0,
            )
            current_arm = initial.arm[0]
        np.testing.assert_allclose(
            initial.arm[0, :2],
            (XR_ARM_INITIAL_COMMAND_LEAD_RAD, XR_ARM_INITIAL_COMMAND_LEAD_RAD / 2.0),
        )

        for _ in range(2):
            final = conditioner.next_command(
                self.arm,
                current_arm,
                self.left,
                self.right,
                now=15.0,
            )
            current_arm = final.arm[0]
        np.testing.assert_allclose(
            final.arm[0, :2],
            (XR_ARM_FINAL_COMMAND_LEAD_RAD, XR_ARM_FINAL_COMMAND_LEAD_RAD / 2.0),
        )

    def test_dex3_is_not_lowpass_filtered_twice_and_uses_only_final_slew(self):
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, self.left, self.right)
        desired_left = self.left.copy()
        desired_right = self.right.copy()
        desired_left[2] = 1.0
        desired_right[4] = 1.0
        conditioner.set_desired(self.arm, desired_left, desired_right, now=0.0)

        first = conditioner.next_command(
            self.arm,
            self.arm,
            self.left,
            self.right,
            now=0.0,
        )
        self.assertAlmostEqual(first.left_hand[0, 2], MAX_CONDITIONED_HAND_STEP_RAD[2])
        self.assertAlmostEqual(first.right_hand[0, 4], MAX_CONDITIONED_HAND_STEP_RAD[4])

        second = conditioner.next_command(
            self.arm,
            first.arm[0],
            first.left_hand[0],
            first.right_hand[0],
            now=0.01,
        )
        self.assertAlmostEqual(second.left_hand[0, 2], 2.0 * MAX_CONDITIONED_HAND_STEP_RAD[2])
        self.assertAlmostEqual(second.right_hand[0, 4], 2.0 * MAX_CONDITIONED_HAND_STEP_RAD[4])

        conditioner.reset(self.arm, self.left, self.right)
        held = conditioner.next_command(
            self.arm,
            self.arm,
            self.left,
            self.right,
            now=1.0,
        )
        np.testing.assert_array_equal(held.left_hand[0], self.left)
        np.testing.assert_array_equal(held.right_hand[0], self.right)

    def test_dex3_slew_uses_urdf_per_joint_limits_and_one_hand_wide_scale(self):
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, self.left, self.right)
        desired_left = self.left.copy()
        desired_left[:2] = 1.0
        conditioner.set_desired(self.arm, desired_left, self.right, now=0.0)

        command = conditioner.next_command(
            self.arm,
            self.arm,
            self.left,
            self.right,
            now=0.0,
        )

        # Thumb0's 6.857-rad/s URDF maximum is the active constraint. A single
        # hand-wide scale preserves the requested equal-joint motion instead of
        # independently clipping thumb0 and thumb1 to different values.
        self.assertAlmostEqual(command.left_hand[0, 0], MAX_CONDITIONED_HAND_STEP_RAD[0])
        self.assertAlmostEqual(command.left_hand[0, 1], command.left_hand[0, 0])
        self.assertLess(command.left_hand[0, 1], MAX_CONDITIONED_HAND_STEP_RAD[1])

        conditioner.reset(self.arm, self.left, self.right)
        desired_left = self.left.copy()
        desired_left[1] = 1.0
        conditioner.set_desired(self.arm, desired_left, self.right, now=1.0)
        non_thumb = conditioner.next_command(
            self.arm,
            self.arm,
            self.left,
            self.right,
            now=1.0,
        )
        self.assertAlmostEqual(non_thumb.left_hand[0, 1], MAX_CONDITIONED_HAND_STEP_RAD[1])

    def test_conditioned_hand_recovers_inward_from_measured_tolerance_without_illegal_target(self):
        current_left = self.left.copy()
        current_left[0] = LEFT_HAND_LOWER[0] - 0.01
        desired_left = self.left.copy()
        desired_left[0] = LEFT_HAND_LOWER[0]
        # Engage the hand-wide slew scale on another joint. The off-limit seed
        # must still make the minimum inward recovery on this write.
        desired_left[1] = 1.0
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, current_left, self.right)
        conditioner.set_desired(self.arm, desired_left, self.right, now=0.0)

        command = conditioner.next_command(
            self.arm,
            self.arm,
            current_left,
            self.right,
            now=0.0,
        )

        self.assertGreaterEqual(
            command.left_hand[0, 0],
            LEFT_HAND_LOWER[0] - HAND_LIMIT_TOLERANCE_RAD,
        )
        validate_action_chunk(command, self.arm, current_left, self.right)

    def test_conditioner_never_hides_nonfinite_or_out_of_range_desired_targets(self):
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, self.left, self.right)
        bad = self.arm.copy()
        bad[0] = np.nan
        with self.assertRaisesRegex(DeploymentError, "NaN or infinity"):
            conditioner.set_desired(bad, self.left, self.right, now=0.0)

        bad[0] = ARM_UPPER[0] + 1.0
        with self.assertRaisesRegex(DeploymentError, "outside its safety-margined joint range"):
            conditioner.set_desired(bad, self.left, self.right, now=0.0)

    def test_arm_reversal_is_conditioned_before_final_step_validation(self):
        conditioner = XrPolicyOutputConditioner()
        conditioner.reset(self.arm, self.left, self.right)
        positive = self.arm.copy()
        positive[0] = 1.0
        conditioner.set_desired(positive, self.left, self.right, now=0.0)
        first = conditioner.next_command(
            self.arm,
            self.arm,
            self.left,
            self.right,
            now=0.0,
        )

        negative = self.arm.copy()
        negative[0] = -1.0
        conditioner.set_desired(negative, self.left, self.right, now=0.01)
        reversed_command = conditioner.next_command(
            self.arm,
            first.arm[0],
            first.left_hand[0],
            first.right_hand[0],
            now=0.01,
        )

        self.assertLessEqual(
            float(np.max(np.abs(reversed_command.arm[0] - first.arm[0]))),
            MAX_CONDITIONED_ARM_STEP_RAD,
        )
        self.assertLess(reversed_command.arm[0, 0], first.arm[0, 0])

    def test_cli_defaults_to_xr_and_allows_explicit_unconditioned_comparison(self):
        default_args = build_parser().parse_args(["--task", "pick-red-cup"])
        self.assertEqual(default_args.command_conditioning, "xr")
        validate_args(default_args)

        unconditioned_args = build_parser().parse_args(["--task", "pick-red-cup", "--command-conditioning", "none"])
        self.assertEqual(unconditioned_args.command_conditioning, "none")
        validate_args(unconditioned_args)

    def test_actuator_process_receives_selected_conditioning_mode(self):
        context = mock.Mock()
        process = mock.Mock()
        context.Process.return_value = process
        context.Queue.side_effect = [mock.Mock(), mock.Mock()]
        context.Event.side_effect = [mock.Mock(), mock.Mock()]
        context.Value.return_value = mock.Mock()
        module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"

        with mock.patch(f"{module}.mp.get_context", return_value=context):
            actuator = SafeG1Dex3Actuator(True, None, "xr")

        self.assertEqual(actuator._command_conditioning, "xr")
        process_args = context.Process.call_args.kwargs["args"]
        self.assertEqual(process_args[-1], "xr")

    def test_unknown_actuator_conditioning_mode_fails_before_process_creation(self):
        with self.assertRaisesRegex(DeploymentError, "Unknown command conditioning mode"):
            SafeG1Dex3Actuator(True, None, "not-a-mode")


if __name__ == "__main__":
    unittest.main()
