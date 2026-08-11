import time
import unittest

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ActionChunk,
    HAND_LIMIT_TOLERANCE_RAD,
    InitializationSpec,
    LEFT_HAND_LOWER,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_UPPER,
    MEASURED_LIMIT_TOLERANCE_RAD,
    RIGHT_HAND_UPPER,
    load_initialization_spec,
    validate_action_chunk_limits,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    INITIALIZATION_MAX_HAND_STEP_RAD,
    RobotState,
    _validate_initialization_hand_recovery,
    build_initialization_chunk,
)


class GrootG1MeasuredHandRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = np.zeros(7, dtype=np.float64)
        self.right = np.zeros(7, dtype=np.float64)
        offset = (HAND_LIMIT_TOLERANCE_RAD + MEASURED_LIMIT_TOLERANCE_RAD) / 2.0
        self.left[0] = LEFT_HAND_LOWER[0] - offset
        self.right[1] = RIGHT_HAND_UPPER[1] + offset
        self.state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=6,
            arm=np.zeros(14, dtype=np.float64),
            arm_dq=np.zeros(14, dtype=np.float64),
            left_hand=self.left,
            right_hand=self.right,
        )

    def test_measured_no_motion_hold_preserves_accepted_q_exactly(self):
        spec = load_initialization_spec("measured", task_name="pick-red-cup")

        hold = build_initialization_chunk(self.state, spec)

        self.assertEqual(hold.length, 1)
        np.testing.assert_array_equal(hold.arm[0], self.state.arm)
        np.testing.assert_array_equal(hold.left_hand[0], self.left)
        np.testing.assert_array_equal(hold.right_hand[0], self.right)

        # The same values remain illegal as raw policy targets.  The exception
        # is scoped to the exact measured hold built above.
        with self.assertRaisesRegex(DeploymentError, "outside its safety-margined joint range"):
            validate_action_chunk_limits(hold)

    def test_explicit_initialization_recovers_monotonically_then_ends_strict(self):
        spec = InitializationSpec(
            mode="pose-file",
            label="strict warm-start endpoint",
            arm=np.zeros(14, dtype=np.float64),
            left_hand=np.zeros(7, dtype=np.float64),
            right_hand=np.zeros(7, dtype=np.float64),
        )

        path = build_initialization_chunk(self.state, spec)

        left_delta = np.diff(np.concatenate(([self.left[0]], path.left_hand[:, 0])))
        right_delta = np.diff(np.concatenate(([self.right[1]], path.right_hand[:, 1])))
        self.assertTrue(np.all(left_delta >= -1e-12))
        self.assertTrue(np.all(right_delta <= 1e-12))
        self.assertLessEqual(float(np.max(np.abs(left_delta))), INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(right_delta))), INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12)
        np.testing.assert_array_equal(path.left_hand[-1], spec.left_hand)
        np.testing.assert_array_equal(path.right_hand[-1], spec.right_hand)
        validate_action_chunk_limits(
            ActionChunk(
                arm=path.arm[-1:],
                left_hand=path.left_hand[-1:],
                right_hand=path.right_hand[-1:],
            )
        )

    def test_preserved_hands_move_only_to_nearest_strict_boundary_during_moving_init(self):
        spec = InitializationSpec(
            mode="pose-file",
            label="move arm and preserve hands",
            arm=np.full(14, 0.1, dtype=np.float64),
            left_hand=None,
            right_hand=None,
        )

        path = build_initialization_chunk(self.state, spec)

        self.assertAlmostEqual(path.left_hand[-1, 0], LEFT_HAND_LOWER[0] - HAND_LIMIT_TOLERANCE_RAD)
        self.assertAlmostEqual(path.right_hand[-1, 1], RIGHT_HAND_UPPER[1] + HAND_LIMIT_TOLERANCE_RAD)
        np.testing.assert_array_equal(path.left_hand[-1, 1:], self.left[1:])
        np.testing.assert_array_equal(path.right_hand[-1, [0, 2, 3, 4, 5, 6]], self.right[[0, 2, 3, 4, 5, 6]])

    def test_transitional_exception_rejects_an_outward_step(self):
        outward = self.left[None].copy()
        outward[0, 0] -= 0.001

        with self.assertRaisesRegex(DeploymentError, "not monotonic inward"):
            _validate_initialization_hand_recovery(
                "left hand",
                outward,
                self.left,
                LEFT_HAND_LOWER,
                LEFT_HAND_UPPER,
                LEFT_HAND_JOINT_NAMES,
            )


if __name__ == "__main__":
    unittest.main()
