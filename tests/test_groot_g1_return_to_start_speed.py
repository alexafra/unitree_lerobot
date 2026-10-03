"""Publisher-free checks for the dedicated, bounded Return-to-Start rates."""

from dataclasses import replace
import queue
import time
from unittest import mock

import numpy as np
import pytest

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    MAX_ARM_STEP_RAD,
    inspire_startup_elbow_settle_spec,
    load_initialization_spec,
)
from unitree_lerobot.eval_robot.robot_control import safe_g1_dex3 as runtime
from unitree_lerobot.eval_robot.training_start_pose import training_start_spec


def _home(profile):
    if profile == "dex3":
        return load_initialization_spec("xr-home", task_name="pick-red-cup", end_effector=profile)
    return inspire_startup_elbow_settle_spec(profile)


def _state(spec):
    return runtime.RobotState(
        captured_at=1.0,
        mode_machine=5,
        arm=spec.arm.copy(),
        arm_dq=np.zeros(14),
        left_hand=spec.left_hand.copy(),
        right_hand=spec.right_hand.copy(),
    )


def _max_step(current, path):
    return np.max(np.abs(np.diff(np.vstack((current, path)), axis=0)))


@pytest.mark.parametrize("profile", ["dex3", "inspire-ftp", "inspire-dfx"])
@pytest.mark.parametrize("warmup1", [False, True])
def test_return_is_another_20_percent_faster_with_unchanged_endpoint_and_caps(profile, warmup1):
    home = _home(profile)
    target = training_start_spec(profile) if warmup1 else home
    state = _state(home)
    if not warmup1:
        state.arm[[3, 10]] = 0.6
    ordinary = (
        runtime._build_warmup1_chunk(state, target)
        if warmup1
        else runtime.build_initialization_chunk(state, target, allow_startup_settle=True)
    )
    returned = runtime._build_return_to_start_chunk(state, target, warmup1=warmup1)

    assert runtime.RETURN_TO_START_SPEED_SCALE == pytest.approx(1.33 * 1.20)
    assert returned.length < ordinary.length
    assert returned.length / ordinary.length == pytest.approx(1 / 1.596, abs=0.004)
    scale = 1.596 * (runtime.WARMUP1_SPEED_SCALE if warmup1 else 1.0)
    assert _max_step(state.arm, returned.arm) <= runtime.INITIALIZATION_MAX_ARM_STEP_RAD * scale + 1e-12
    assert _max_step(state.arm, returned.arm) <= MAX_ARM_STEP_RAD + 1e-12
    hand_cap = runtime.INITIALIZATION_MAX_HAND_STEP_RAD * scale if profile == "dex3" else 0.2
    assert _max_step(state.left_hand, returned.left_hand) <= hand_cap + 1e-12
    assert _max_step(state.right_hand, returned.right_hand) <= hand_cap + 1e-12
    for field in ("arm", "left_hand", "right_hand"):
        np.testing.assert_allclose(getattr(returned, field)[-1], getattr(target, field), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(getattr(returned, field)[-1], getattr(ordinary, field)[-1])
    assert returned.end_effector == profile


@pytest.mark.parametrize("profile", ["inspire-ftp", "inspire-dfx"])
@pytest.mark.parametrize("warmup1", [False, True])
def test_inspire_writer_cap_can_limit_return_acceleration(profile, warmup1):
    target = training_start_spec(profile) if warmup1 else _home(profile)
    state = _state(target)
    state.left_hand[:] = 1.0 - target.left_hand
    state.right_hand[:] = 1.0 - target.right_hand
    # Remove only the test's minimum move duration so hand slew is binding.
    with mock.patch.object(runtime, "INITIALIZATION_MIN_MOVE_S", 0.0):
        ordinary = (
            runtime._build_warmup1_chunk(state, target)
            if warmup1
            else runtime.build_initialization_chunk(state, target, allow_startup_settle=True)
        )
        returned = runtime._build_return_to_start_chunk(state, target, warmup1=warmup1)
    assert returned.length == ordinary.length
    assert _max_step(state.left_hand, returned.left_hand) <= 0.2 + 1e-12
    assert _max_step(state.right_hand, returned.right_hand) <= 0.2 + 1e-12


@pytest.mark.parametrize(
    "command_kind,warmup1,expected_builder,expected_kwargs",
    [
        ("warmup_pose", False, "build_initialization_chunk", {"allow_training_start": True, "allow_startup_settle": True}),
        ("warmup1_pose", True, "_build_warmup1_chunk", {}),
        ("return_to_start_pose", False, "_build_return_to_start_chunk", {"warmup1": False}),
        ("return_to_start_warmup1_pose", True, "_build_return_to_start_chunk", {"warmup1": True}),
    ],
)
def test_child_dispatch_selects_only_the_explicit_stage_rate(command_kind, warmup1, expected_builder, expected_kwargs):
    home = _home("inspire-ftp")
    state = _state(home)
    target = training_start_spec("inspire-ftp") if warmup1 else home
    with (
        mock.patch.object(runtime, "build_initialization_chunk") as ordinary,
        mock.patch.object(runtime, "_build_warmup1_chunk") as startup_warmup1,
        mock.patch.object(runtime, "_build_return_to_start_chunk") as returned,
    ):
        builders = {
            "build_initialization_chunk": ordinary,
            "_build_warmup1_chunk": startup_warmup1,
            "_build_return_to_start_chunk": returned,
        }
        result = runtime._build_guarded_warmup_pose_chunk(state, target, command_kind=command_kind)
        selected = builders[expected_builder]
        assert result is selected.return_value
        selected.assert_called_once_with(state, target, **expected_kwargs)
        for name, builder in builders.items():
            if name != expected_builder:
                builder.assert_not_called()


def _parent(profile="inspire-ftp"):
    actuator = object.__new__(runtime.SafeG1Dex3Actuator)
    actuator.end_effector = profile
    actuator._initialized = True
    actuator._holding = True
    actuator._chunk_in_flight = False
    actuator._command_queue = queue.Queue(maxsize=1)
    actuator._wait_for_hand_feedback = mock.Mock()
    actuator.heartbeat = mock.Mock()
    actuator._wait_status = mock.Mock()
    return actuator


@pytest.mark.parametrize("warmup1", [False, True])
def test_parent_dispatch_uses_dedicated_return_kind_and_keeps_feedback_and_hold(warmup1):
    actuator = _parent()
    target = training_start_spec("inspire-ftp") if warmup1 else _home("inspire-ftp")
    actuator.return_to_start_pose(target, warmup1=warmup1)
    kind, created_at, queued = actuator._command_queue.get_nowait()
    assert kind == ("return_to_start_warmup1_pose" if warmup1 else "return_to_start_pose")
    assert time.monotonic() - created_at < 0.5
    assert queued is target
    actuator._wait_for_hand_feedback.assert_called_once_with()
    actuator.heartbeat.assert_called_once_with()
    actuator._wait_status.assert_called_once_with(
        "warmup_pose_completed",
        timeout_s=(
            runtime.INITIALIZATION_START_TIMEOUT_S
            + runtime.INITIALIZATION_MAX_DURATION_S
            + runtime.INITIALIZATION_CONVERGENCE_TIMEOUT_S
            + 3.0
        ),
        payload=target.label,
    )
    assert actuator._holding


@pytest.mark.parametrize("initialized,holding,in_flight", [(False, True, False), (True, False, False), (True, True, True)])
def test_parent_return_requires_initialized_hold_with_no_chunk(initialized, holding, in_flight):
    actuator = _parent()
    actuator._initialized, actuator._holding, actuator._chunk_in_flight = initialized, holding, in_flight
    with pytest.raises(DeploymentError, match="initialized HOLD"):
        actuator.return_to_start_pose(_home("inspire-ftp"))
    assert actuator._command_queue.empty()
    actuator._wait_for_hand_feedback.assert_not_called()


def test_parent_return_preserves_feedback_failure_without_queueing():
    actuator = _parent()
    actuator._wait_for_hand_feedback.side_effect = DeploymentError("feedback unavailable")
    with pytest.raises(DeploymentError, match="feedback unavailable"):
        actuator.return_to_start_pose(_home("inspire-ftp"))
    assert actuator._command_queue.empty()
    actuator.heartbeat.assert_not_called()


@pytest.mark.parametrize("warmup1", [False, True])
def test_return_validation_rejects_measured_wrong_profile_and_wrong_stage(warmup1):
    actuator = _parent()
    measured = load_initialization_spec("measured", task_name="pick-red-cup", end_effector="inspire-ftp")
    with pytest.raises(DeploymentError, match="fixed moving pose"):
        actuator.return_to_start_pose(measured, warmup1=warmup1)
    with pytest.raises(DeploymentError, match="payload is tagged"):
        actuator.return_to_start_pose(_home("dex3"), warmup1=warmup1)
    target = _home("inspire-ftp") if warmup1 else training_start_spec("inspire-ftp")
    with pytest.raises(DeploymentError, match="Warmup1|Unsupported initialization mode"):
        actuator.return_to_start_pose(target, warmup1=warmup1)
    assert actuator._command_queue.empty()


def test_return_does_not_accept_policy_pose_or_relax_exact_settle_validation():
    home = _home("inspire-ftp")
    state = _state(home)
    wrong_arm = home.arm.copy()
    wrong_arm[3] = -0.15
    for target in (replace(home, arm=wrong_arm), replace(home, mode="pose-file")):
        with pytest.raises(DeploymentError):
            runtime._build_return_to_start_chunk(state, target)
        actuator = _parent()
        with pytest.raises(DeploymentError):
            actuator.return_to_start_pose(target)
        assert actuator._command_queue.empty()
    with pytest.raises(ValueError, match="speed_scale"):
        runtime.build_initialization_chunk(state, home, allow_startup_settle=True, speed_scale=1.596)
    with pytest.raises(DeploymentError, match="Unsupported guarded"):
        runtime._build_guarded_warmup_pose_chunk(state, home, command_kind="unknown")
