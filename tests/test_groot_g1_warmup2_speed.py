import numpy as np
import pytest

from unitree_lerobot.eval_robot.groot_contract import (
    InitializationSpec,
    inspire_startup_elbow_settle_spec,
    load_initialization_spec,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    INITIALIZATION_MAX_ARM_STEP_RAD,
    INITIALIZATION_MAX_HAND_STEP_RAD,
    WARMUP1_SPEED_SCALE,
    WARMUP2_SPEED_SCALE,
    RobotState,
    _build_policy_warm_start_chunk,
    _build_warmup1_chunk,
    build_initialization_chunk,
)
from unitree_lerobot.eval_robot.training_start_pose import training_start_spec


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


def test_warmups_are_another_20_percent_faster_without_changing_ordinary_initialization() -> None:
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

    training_start = InitializationSpec(
        mode="training-start",
        label="Warmup1: reviewed training start",
        arm=target.arm,
        left_hand=target.left_hand,
        right_hand=target.right_hand,
        end_effector=target.end_effector,
    )
    warmup1 = _build_warmup1_chunk(state, training_start)

    assert WARMUP1_SPEED_SCALE == pytest.approx(1.40 * 1.20)
    assert WARMUP2_SPEED_SCALE == pytest.approx(1.05 * 1.20)
    np.testing.assert_array_equal(ordinary.arm, explicit_default.arm)
    assert ordinary.length == 180
    assert warmup1.length == 108
    assert warmup2.length == 143
    assert warmup1.length / 129 == pytest.approx(1 / 1.20, abs=0.005)
    assert warmup2.length / 172 == pytest.approx(1 / 1.20, abs=0.005)
    assert warmup1.length < ordinary.length
    assert warmup2.length < 240
    np.testing.assert_allclose(warmup1.arm[-1], target.arm)
    np.testing.assert_allclose(warmup2.arm[-1], target.arm)
    warmup1_step = np.max(np.abs(np.diff(np.vstack((state.arm, warmup1.arm)), axis=0)))
    warmup2_step = np.max(np.abs(np.diff(np.vstack((state.arm, warmup2.arm)), axis=0)))
    assert warmup1_step <= INITIALIZATION_MAX_ARM_STEP_RAD * WARMUP1_SPEED_SCALE + 1e-12
    assert warmup2_step <= INITIALIZATION_MAX_ARM_STEP_RAD * WARMUP2_SPEED_SCALE + 1e-12


def test_dex3_warmup1_and_hand_dominated_warmup2_use_scaled_private_limits() -> None:
    state = RobotState(
        captured_at=1.0,
        mode_machine=5,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.zeros(7),
        right_hand=np.zeros(7),
    )
    warmup1_target = training_start_spec("dex3")
    ordinary = build_initialization_chunk(state, warmup1_target)
    warmup1 = _build_warmup1_chunk(state, warmup1_target)

    hand_target = InitializationSpec(
        mode="pose-file",
        label="first policy target",
        arm=np.zeros(14),
        left_hand=np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]),
        right_hand=np.array([0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5]),
        end_effector="dex3",
    )
    warmup2 = _build_policy_warm_start_chunk(
        state,
        hand_target,
        allow_policy_warm_start=False,
    )

    assert ordinary.length == 642
    assert warmup1.length == 382
    warmup1_hand_step = max(
        np.max(np.abs(np.diff(np.vstack((state.left_hand, warmup1.left_hand)), axis=0))),
        np.max(np.abs(np.diff(np.vstack((state.right_hand, warmup1.right_hand)), axis=0))),
    )
    warmup2_hand_step = max(
        np.max(np.abs(np.diff(np.vstack((state.left_hand, warmup2.left_hand)), axis=0))),
        np.max(np.abs(np.diff(np.vstack((state.right_hand, warmup2.right_hand)), axis=0))),
    )
    assert warmup1_hand_step <= INITIALIZATION_MAX_HAND_STEP_RAD * WARMUP1_SPEED_SCALE + 1e-12
    assert warmup2_hand_step <= INITIALIZATION_MAX_HAND_STEP_RAD * WARMUP2_SPEED_SCALE + 1e-12


def test_canonical_inspire_warmup1_is_faster_but_hand_writer_cap_is_unchanged() -> None:
    target = training_start_spec("inspire-ftp")
    starts = (
        (inspire_startup_elbow_settle_spec("inspire-ftp"), 616, 367),
        (
            load_initialization_spec(
                "xr-home",
                task_name="pick-red-cup",
                end_effector="inspire-ftp",
            ),
            676,
            403,
        ),
    )

    for start, ordinary_steps, warmup1_steps in starts:
        state = RobotState(
            captured_at=1.0,
            mode_machine=5,
            arm=start.arm.copy(),
            arm_dq=np.zeros(14),
            left_hand=start.left_hand.copy(),
            right_hand=start.right_hand.copy(),
        )
        ordinary = build_initialization_chunk(
            state,
            target,
            allow_training_start=True,
        )
        warmup1 = _build_warmup1_chunk(state, target)
        hand_step = max(
            np.max(np.abs(np.diff(np.vstack((state.left_hand, warmup1.left_hand)), axis=0))),
            np.max(np.abs(np.diff(np.vstack((state.right_hand, warmup1.right_hand)), axis=0))),
        )

        assert ordinary.length == ordinary_steps
        assert warmup1.length == warmup1_steps
        assert hand_step <= 0.2 + 1e-12


@pytest.mark.parametrize("speed_scale", [0.0, -0.1, 1.01, 1.41, np.nan, np.inf])
def test_initialization_speed_scale_rejects_invalid_values(speed_scale: float) -> None:
    with pytest.raises(ValueError, match="speed_scale"):
        build_initialization_chunk(
            _state(),
            _target(),
            allow_policy_warm_start=True,
            speed_scale=speed_scale,
        )
