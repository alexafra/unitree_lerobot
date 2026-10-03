"""CPU-only opt-in and protocol checks; no cameras, DDS, server or robot."""

from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from unitree_lerobot.eval_robot import eval_groot_g1 as runner
from unitree_lerobot.eval_robot.groot_client import DeploymentError, Gr00tClient
from unitree_lerobot.eval_robot.groot_contract import ActionChunk


def _plan():
    return ActionChunk(
        arm=np.arange(32 * 14, dtype=np.float32).reshape(32, 14) / 1000,
        left_hand=np.ones((32, 6), dtype=np.float32),
        right_hand=np.ones((32, 6), dtype=np.float32),
        end_effector="inspire-ftp",
    )


def _args(*argv):
    return runner.build_parser().parse_args(["--no-actuate", "--task", "pick-red-cup", *argv])


def _pure_ack(**updates):
    return {
        "rtc_applied": True,
        "rtc_previous_action_horizon": 24,
        "rtc_overlap_steps": 24,
        "rtc_frozen_steps": 6,
        "rtc_pure_applied": True,
        **updates,
    }


def test_no_flag_preserves_synchronous_default_and_legacy_options():
    args = _args()
    runner.validate_args(args)
    assert args.inference_mode == "synchronous"
    assert args.rtc_pure is False
    options = runner._rtc_options(_plan(), 8, 6, 7.0)
    assert set(options) == {
        "inference_mode", "rtc_previous_action", "rtc_overlap_steps",
        "rtc_frozen_steps", "rtc_ramp_rate",
    }
    assert options["rtc_ramp_rate"] == 7.0


@pytest.mark.parametrize("extra", [[], ["--inference-mode", "rtc"], ["--inference-mode", "synchronous"]])
def test_pure_flag_selects_async_rtc_without_other_default_changes(extra):
    args = _args("--rtc-pure", *extra)
    runner.validate_args(args)
    assert args.rtc_pure is True
    assert args.inference_mode == "rtc"
    assert args.execution_horizon == 8
    assert args.action_interpolation == "legacy"
    assert args.command_conditioning == "xr"


def test_pure_rejects_irrelevant_velocity_ramp_override():
    with pytest.raises(DeploymentError, match="not applicable with --rtc-pure"):
        runner.validate_args(_args("--rtc-pure", "--rtc-ramp-rate", "3"))
    with pytest.raises(DeploymentError, match="not applicable with --rtc-pure"):
        runner._rtc_options(_plan(), 8, 6, 3.0, rtc_pure=True)


def test_pure_wire_only_adds_opt_in_to_existing_physical_tail():
    plan = _plan()
    legacy = runner._rtc_options(plan, 8, 6, None)
    pure = runner._rtc_options(plan, 8, 6, None, rtc_pure=True)
    assert set(pure) == set(legacy) | {"rtc_pure"}
    assert pure["rtc_pure"] is True
    for key in ("inference_mode", "rtc_overlap_steps", "rtc_frozen_steps"):
        assert pure[key] == legacy[key]
    for key, tail in legacy["rtc_previous_action"].items():
        np.testing.assert_array_equal(pure["rtc_previous_action"][key], tail)
    np.testing.assert_array_equal(pure["rtc_previous_action"]["left_arm"][0], plan.arm[8:, :7])


@pytest.mark.parametrize("advertised", [None, False, "true", 1])
def test_missing_strict_pure_capability_fails_before_dds_or_publishers(advertised):
    args = _args("--rtc-pure")
    policy = mock.Mock()
    policy.ping.return_value = True
    capability = {"protocol_version": 1, "physical_action_tail": True, "backend": "pytorch"}
    if advertised is not None:
        capability["pure_inference_guidance"] = advertised
    policy.get_policy_metadata.return_value = {"rtc": capability}
    contract = SimpleNamespace(video_keys=("ego_view",), action_horizon=32, end_effector="inspire-ftp")
    with (
        mock.patch.object(runner, "Gr00tClient", return_value=policy),
        mock.patch.object(runner, "validate_model_contract", return_value=contract),
        mock.patch.object(runner, "validate_policy_metadata", return_value=None),
        mock.patch.object(runner, "initialize_dds") as dds,
        mock.patch.object(runner, "SafeG1Dex3Actuator") as actuator,
        pytest.raises(DeploymentError, match="pure_inference_guidance=true"),
    ):
        runner.run(args)
    dds.assert_not_called()
    actuator.assert_not_called()
    policy.close.assert_called_once()


@pytest.mark.parametrize("pure", [False, True])
def test_matching_server_capability_reaches_existing_dds_gate(pure):
    args = _args(*(["--rtc-pure"] if pure else ["--inference-mode", "rtc"]))
    policy = mock.Mock()
    capability = {"protocol_version": 1, "physical_action_tail": True, "backend": "pytorch"}
    if pure:
        capability["pure_inference_guidance"] = True
    policy.get_policy_metadata.return_value = {"rtc": capability}
    contract = SimpleNamespace(video_keys=("ego_view",), action_horizon=32, end_effector="inspire-ftp")
    with (
        mock.patch.object(runner, "Gr00tClient", return_value=policy),
        mock.patch.object(runner, "validate_model_contract", return_value=contract),
        mock.patch.object(runner, "validate_policy_metadata", return_value=None),
        mock.patch.object(runner, "require_inspire_ftp_sdk"),
        mock.patch.object(runner, "initialize_dds", side_effect=RuntimeError("DDS sentinel")) as dds,
        mock.patch.object(runner, "SafeG1Dex3Actuator") as actuator,
        pytest.raises(RuntimeError, match="DDS sentinel"),
    ):
        runner.run(args)
    dds.assert_called_once()
    actuator.assert_not_called()


@pytest.mark.parametrize("ack", [None, False, "true", 1])
def test_pure_response_cannot_silently_fall_back_to_legacy(ack):
    client = object.__new__(Gr00tClient)
    client.call = mock.Mock(return_value=({}, _pure_ack(rtc_pure_applied=ack, rtc_ramp_rate=6.0)))
    with pytest.raises(DeploymentError, match="refusing legacy fallback"):
        client.get_action({}, runner._rtc_options(_plan(), 8, 6, None, rtc_pure=True))


def test_pure_response_accepts_guidance_without_legacy_ramp():
    client = object.__new__(Gr00tClient)
    action = {"left_arm": np.zeros((1, 32, 7), dtype=np.float32)}
    client.call = mock.Mock(return_value=(action, _pure_ack()))
    assert client.get_action({}, runner._rtc_options(_plan(), 8, 6, None, rtc_pure=True)) is action


@pytest.mark.parametrize("field,value", [
    ("rtc_applied", False), ("rtc_previous_action_horizon", 23),
    ("rtc_overlap_steps", 23), ("rtc_frozen_steps", 5),
])
def test_pure_keeps_existing_tail_ack_checks(field, value):
    client = object.__new__(Gr00tClient)
    client.call = mock.Mock(return_value=({}, _pure_ack(**{field: value})))
    with pytest.raises(DeploymentError, match="did not acknowledge the requested RTC conditioning"):
        client.get_action({}, runner._rtc_options(_plan(), 8, 6, None, rtc_pure=True))


def test_ordinary_first_chunk_does_not_require_rtc_ack():
    client = object.__new__(Gr00tClient)
    action = {"left_arm": np.zeros((1, 32, 7), dtype=np.float32)}
    client.call = mock.Mock(return_value=(action, {}))
    assert client.get_action({}) is action


def test_calibration_measures_guided_roundtrip_discards_actions_and_preserves_first_plan():
    plan = _plan()
    before = plan.arm.copy()
    policy = mock.Mock()
    args = _args("--rtc-pure")
    observation = {"calibration": True}
    with (
        mock.patch.object(runner, "capture_policy_observation", return_value=(observation, mock.Mock())),
        mock.patch.object(runner.time, "monotonic", side_effect=[100.0, 100.25]),
        mock.patch.object(runner, "parse_action_plan") as parse,
    ):
        measured = runner._calibrate_rtc_pure_delay(
            policy, plan, 0.1, mock.Mock(), mock.Mock(), "pick up the red cup.",
            mock.Mock(), args, allow_custom_instruction=False,
        )
    assert measured == 0.25
    assert policy.get_action.call_args.args[0] is observation
    options = policy.get_action.call_args.args[1]
    assert options["rtc_pure"] is True
    assert options["rtc_overlap_steps"] == 24
    assert options["rtc_frozen_steps"] == 4
    np.testing.assert_array_equal(plan.arm, before)
    parse.assert_not_called()


@pytest.mark.parametrize("extra", [[], ["--rtc-frozen-steps", "3"]])
def test_calibration_refuses_known_insufficient_overlap_before_starting_actions(extra):
    with (
        mock.patch.object(runner, "capture_policy_observation", return_value=({}, mock.Mock())),
        mock.patch.object(runner.time, "monotonic", side_effect=[100.0, 101.0]),
        pytest.raises(DeploymentError, match="action execution was not started"),
    ):
        runner._calibrate_rtc_pure_delay(
            mock.Mock(), _plan(), 0.1, mock.Mock(), mock.Mock(), "pick up the red cup.",
            mock.Mock(), _args("--rtc-pure", *extra), allow_custom_instruction=False,
        )


def test_live_calibration_services_heartbeat_and_uses_worker_owned_socket():
    actuator = mock.Mock(hand_pause_generation=0, assert_healthy=mock.Mock())
    actuator.immediate_control_requested.return_value = None
    worker = mock.Mock()
    worker.poll.side_effect = [None, runner._RtcResponse(0, {}, 0.25, None)]
    policy = mock.Mock()
    with (
        mock.patch.object(runner, "capture_policy_observation", return_value=({}, mock.Mock())),
        mock.patch.object(runner, "_RtcInferenceWorker", return_value=worker),
        mock.patch.object(runner.time, "monotonic", side_effect=[100.0, 100.25]),
        mock.patch.object(runner.time, "sleep"),
    ):
        duration = runner._calibrate_rtc_pure_delay(
            policy, _plan(), 0.1, mock.Mock(), mock.Mock(), "pick up the red cup.",
            mock.Mock(), _args("--rtc-pure"), actuator=actuator, allow_custom_instruction=False,
        )
    assert duration == 0.25
    assert actuator.heartbeat.call_count == 2
    assert actuator.assert_healthy.call_count == 2
    policy.get_action.assert_not_called()
    worker.close.assert_called_once_with(wait=False)
    assert worker.submit.call_args.args[0].options["rtc_pure"] is True


def test_live_calibration_releases_immediately_while_guidance_is_in_flight():
    actuator = mock.Mock(hand_pause_generation=0, assert_healthy=mock.Mock())
    actuator.immediate_control_requested.side_effect = [None, "release", "release"]
    worker = mock.Mock()
    with (
        mock.patch.object(runner, "capture_policy_observation", return_value=({}, mock.Mock())),
        mock.patch.object(runner, "_RtcInferenceWorker", return_value=worker),
        pytest.raises(runner.OperatorRelease),
    ):
        runner._calibrate_rtc_pure_delay(
            mock.Mock(), _plan(), 0.1, mock.Mock(), mock.Mock(), "pick up the red cup.",
            mock.Mock(), _args("--rtc-pure"), actuator=actuator, allow_custom_instruction=False,
        )
    worker.close.assert_called_once_with(wait=False)
    actuator.start_rtc.assert_not_called()


@pytest.mark.parametrize("pure", [False, True])
def test_live_bootstrap_uses_guided_calibration_only_when_opted_in(pure):
    args = _args(*(["--rtc-pure"] if pure else ["--inference-mode", "rtc"]))
    plan = _plan()
    actuator = mock.Mock()
    actuator.poll_rtc_event.return_value = ("complete", {})
    histories = []
    ordering = []

    def history_factory(*a, **kw):
        history = deque(*a, **kw)
        histories.append(history)
        return history

    def calibration(*a, **kw):
        ordering.append("calibrate")
        return 0.5

    def start(*a, **kw):
        ordering.append("start")
        return 1

    actuator.start_rtc.side_effect = start
    with (
        mock.patch.object(runner, "infer_plan", return_value=(plan, 0.1)),
        mock.patch.object(runner, "_calibrate_rtc_pure_delay", side_effect=calibration) as calibrate,
        mock.patch.object(runner, "_poll_active_command", return_value=None),
        mock.patch.object(runner, "_RtcInferenceWorker"),
        mock.patch.object(runner, "deque", side_effect=history_factory),
    ):
        result = runner._run_active_goal_rtc_controlled(
            mock.Mock(), mock.Mock(), mock.Mock(), actuator, "pick-red-cup",
            "pick up the red cup.", mock.Mock(), args, mock.Mock(),
            allow_custom_instruction=False,
        )
    assert result == "complete"
    assert ordering == (["calibrate", "start"] if pure else ["start"])
    assert calibrate.call_count == int(pure)
    assert list(histories[0]) == ([16] if pure else [4])
    assert actuator.start_rtc.call_args.args[0] is plan


def test_calibration_failure_does_not_start_rtc_execution():
    actuator = mock.Mock()
    with (
        mock.patch.object(runner, "infer_plan", return_value=(_plan(), 0.1)),
        mock.patch.object(runner, "_calibrate_rtc_pure_delay", side_effect=DeploymentError("bad calibration")),
        pytest.raises(DeploymentError, match="bad calibration"),
    ):
        runner._run_active_goal_rtc_controlled(
            mock.Mock(), mock.Mock(), mock.Mock(), actuator, "pick-red-cup",
            "pick up the red cup.", mock.Mock(), _args("--rtc-pure"), mock.Mock(),
            allow_custom_instruction=False,
        )
    actuator.start_rtc.assert_not_called()


def test_pure_shadow_runs_guided_calibration_before_starting_virtual_clock():
    args = _args("--rtc-pure")
    args.max_chunks = 0  # Exercise setup/cleanup only, with no wall-clock wait.
    policy = mock.Mock()
    with (
        mock.patch.object(runner, "_calibrate_rtc_pure_delay", return_value=0.25) as calibrate,
        mock.patch.object(runner, "_RtcInferenceWorker"),
    ):
        runner._run_shadow_rtc(
            _plan(), 0.1, mock.Mock(), mock.Mock(), "pick up the red cup.", mock.Mock(), args,
            allow_custom_instruction=False, policy=policy,
        )
    assert calibrate.call_args.args[0] is policy
