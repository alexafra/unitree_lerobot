import numpy as np
import pytest

from unitree_lerobot.eval_robot.groot_contract import InitializationSpec
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    INITIALIZATION_MAX_ARM_STEP_RAD,
    WARMUP2_SPEED_SCALE,
    RobotState,
    _build_policy_warm_start_chunk,
    build_initialization_chunk,
)


def _state() -> RobotState:
    return RobotState(
        captured_at=1.0,
        mode_machine=5,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.ones(6),
        right_hand=np.ones(6),
    )


def _target() -> InitializationSpec:
    arm = np.zeros(14)
    arm[3] = -0.3
    return InitializationSpec(
        mode="pose-file",
        label="first policy target",
        arm=arm,
        left_hand=np.ones(6),
        right_hand=np.ones(6),
        end_effector="inspire-ftp",
    )


def test_warmup2_is_slower_without_changing_ordinary_initialization() -> None:
    state = _state()
    target = _target()

    ordinary = build_initialization_chunk(
        state,
        target,
        allow_policy_warm_start=True,
    )
    explicit_default = build_initialization_chunk(
        state,
        target,
        allow_policy_warm_start=True,
        speed_scale=1.0,
    )
    warmup2 = _build_policy_warm_start_chunk(
        state,
        target,
        allow_policy_warm_start=True,
    )

    assert WARMUP2_SPEED_SCALE == 0.75
    np.testing.assert_array_equal(ordinary.arm, explicit_default.arm)
    assert ordinary.length == 180
    assert warmup2.length == 240
    np.testing.assert_allclose(warmup2.arm[-1], target.arm)
    warmup2_step = np.max(
        np.abs(np.diff(np.vstack((state.arm, warmup2.arm)), axis=0))
    )
    assert warmup2_step <= INITIALIZATION_MAX_ARM_STEP_RAD * 0.75 + 1e-12


@pytest.mark.parametrize("speed_scale", [0.0, -0.1, 1.01, np.nan, np.inf])
def test_initialization_speed_scale_rejects_invalid_values(speed_scale: float) -> None:
    with pytest.raises(ValueError, match="speed_scale"):
        build_initialization_chunk(
            _state(),
            _target(),
            allow_policy_warm_start=True,
            speed_scale=speed_scale,
        )
