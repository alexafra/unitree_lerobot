"""The mounted Inspire wiring requires asymmetric deployment yaw bounds."""

import numpy as np
import pytest

from unitree_lerobot.eval_robot.g1_end_effectors import INSPIRE_FTP_PROFILE
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ARM_UPPER,
    ActionChunk,
    InitializationSpec,
    INSPIRE_ARM_COMMAND_LOWER,
    INSPIRE_ARM_COMMAND_UPPER,
    arm_command_bounds,
    parse_action_plan,
    validate_action_chunk_limits,
    validate_initialization_spec,
    validate_measured_state,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import XrPolicyOutputConditioner


def _chunk(left_yaw: float, right_yaw: float) -> ActionChunk:
    arm = np.zeros((1, 14), dtype=np.float64)
    arm[0, 6] = left_yaw
    arm[0, 13] = right_yaw
    hands = np.full((1, 6), 0.5, dtype=np.float64)
    return ActionChunk(arm=arm, left_hand=hands, right_hand=hands.copy(), end_effector="inspire-ftp")


def test_outward_60_inward_80_on_both_yaws() -> None:
    outward = np.deg2rad(60)
    inward = np.deg2rad(80)
    assert INSPIRE_ARM_COMMAND_LOWER[6] == pytest.approx(-inward)
    assert INSPIRE_ARM_COMMAND_UPPER[6] == pytest.approx(outward)
    assert INSPIRE_ARM_COMMAND_LOWER[13] == pytest.approx(-outward)
    assert INSPIRE_ARM_COMMAND_UPPER[13] == pytest.approx(inward)
    validate_action_chunk_limits(_chunk(outward, -outward))
    validate_action_chunk_limits(_chunk(-inward, inward))

    for left_yaw, right_yaw in (
        (outward + 0.01, 0.0),
        (0.0, -outward - 0.01),
        (-inward - 0.01, 0.0),
        (0.0, inward + 0.01),
    ):
        with pytest.raises(DeploymentError, match="INSPIRE_ARM_COMMAND"):
            validate_action_chunk_limits(_chunk(left_yaw, right_yaw))


def test_policy_parser_projects_yaw_without_changing_dex3_or_physical_limits() -> None:
    arms = np.zeros((1, 14), dtype=np.float64)
    arms[0, 6], arms[0, 13] = 1.5, -1.5
    action = {
        "left_arm": arms[:, None, :7],
        "right_arm": arms[:, None, 7:],
        "left_hand": np.full((1, 1, 6), 0.5),
        "right_hand": np.full((1, 1, 6), 0.5),
    }
    current_arm = np.zeros(14)
    current_hand = np.full(6, 0.5)
    plan = parse_action_plan(
        action, 1, current_arm, current_hand, current_hand,
        validate_target_steps=False, end_effector="inspire-ftp",
    )
    assert plan.arm[0, 6] == pytest.approx(np.deg2rad(60))
    assert plan.arm[0, 13] == pytest.approx(-np.deg2rad(60))
    assert ARM_UPPER[6] == pytest.approx(1.614429558)
    dex3_lower, dex3_upper = arm_command_bounds("dex3")
    assert dex3_upper[6] > plan.arm[0, 6]
    assert dex3_lower[13] < plan.arm[0, 13]


def test_inspire_moving_start_and_xr_conditioner_respect_command_bounds() -> None:
    arm = np.zeros(14)
    arm[6] = np.deg2rad(70)
    hands = np.full(6, 0.5)
    spec = InitializationSpec(
        mode="training-start", label="yaw-limit-test", arm=arm,
        left_hand=hands, right_hand=hands, end_effector="inspire-ftp",
    )
    with pytest.raises(DeploymentError, match="INSPIRE_ARM_COMMAND_UPPER"):
        validate_initialization_spec(spec, allow_training_start=True)

    conditioner = XrPolicyOutputConditioner(INSPIRE_FTP_PROFILE)
    conditioner.reset(np.zeros(14), hands, hands)
    with pytest.raises(DeploymentError, match="INSPIRE_ARM_COMMAND_UPPER"):
        conditioner.set_desired(arm, hands, hands, now=0.0)

    arm[6] = INSPIRE_ARM_COMMAND_UPPER[6]
    arm[13] = INSPIRE_ARM_COMMAND_LOWER[13]
    conditioner.set_desired(arm, hands, hands, now=0.0)
    next_command = conditioner.next_command(
        np.zeros(14), np.zeros(14), hands, hands, now=0.0,
    )
    validate_action_chunk_limits(next_command)
    assert next_command.arm[0, 6] <= INSPIRE_ARM_COMMAND_UPPER[6]
    assert next_command.arm[0, 13] >= INSPIRE_ARM_COMMAND_LOWER[13]


def test_measured_state_still_uses_physical_motor_bounds() -> None:
    arm = np.zeros(14)
    arm[6], arm[13] = 1.5, -1.5
    hands = np.full(6, 0.5)
    validate_measured_state(arm, np.zeros(14), hands, hands, end_effector="inspire-ftp")
