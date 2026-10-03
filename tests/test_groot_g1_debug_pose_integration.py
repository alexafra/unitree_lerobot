"""CPU-only Warmup1/debug CLI integration; no sockets, publishers or workers."""

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import test_groot_g1_full_demo_integration as full_demo_tests
import unitree_lerobot.eval_robot.eval_groot_g1 as runner
from unitree_lerobot.eval_robot.g1_end_effectors import get_end_effector_profile
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ARM_JOINT_NAMES, ActionChunk
from unitree_lerobot.eval_robot.training_start_pose import training_start_spec


@pytest.fixture(autouse=True)
def clear_debug_global(monkeypatch):
    monkeypatch.setattr(runner, "_ACTION_DEBUG_RECORDER", None)


def _pose_file(tmp_path, profile="dex3"):
    spec = training_start_spec(profile)
    names = get_end_effector_profile(profile)
    payload = {
        "schema_version": 1,
        "name": "another measured start",
        "robot_type": "g1",
        "end_effector": profile,
        "arm_unit": "rad",
        "hand_unit": names.value_unit,
        "joint_names": {
            "arm": list(ARM_JOINT_NAMES),
            "left_hand": list(names.left_joint_names),
            "right_hand": list(names.right_joint_names),
        },
        "arm": spec.arm.tolist(),
        "left_hand": spec.left_hand.tolist(),
        "right_hand": spec.right_hand.tolist(),
        "source": {
            "dataset_path": "/offline/measured/train",
            "episode_index": 123,
            "frame_index": 0,
            "quantity": "observation.state",
            "task": "pick up the red cup.",
        },
    }
    payload["arm"][0] += 0.01
    path = tmp_path / "selected-pose.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_new_flags_are_opt_in_and_preserve_control_defaults():
    parser = runner.build_parser()
    baseline = parser.parse_args([])
    assert baseline.debug_actions is False
    assert baseline.warmup1_pose_file is None
    enabled = parser.parse_args(["--debug-actions", "--warmup1-pose-file", "/offline/pose.json"])
    assert enabled.debug_actions is True
    assert str(enabled.warmup1_pose_file) == "/offline/pose.json"
    for field in (
        "actuate", "warmup1", "policy_warm_start", "future_goal_warmup2",
        "return_to_start", "execution_horizon", "inference_mode", "rtc_pure",
        "command_conditioning", "action_interpolation", "initialization", "end_effector",
    ):
        assert getattr(enabled, field) == getattr(baseline, field)


def test_custom_pose_rejected_when_warmup1_disabled_before_external_work(monkeypatch):
    args = runner.build_parser().parse_args([
        "--no-actuate", "--task", "pick-red-cup", "--no-warmup1",
        "--warmup1-pose-file", "/offline/missing.json",
    ])
    client = mock.Mock(side_effect=AssertionError("no policy connection permitted"))
    dds = mock.Mock(side_effect=AssertionError("no DDS permitted"))
    loader = mock.Mock(side_effect=AssertionError("disabled pose must not load"))
    monkeypatch.setattr(runner, "Gr00tClient", client)
    monkeypatch.setattr(runner, "initialize_dds", dds)
    monkeypatch.setattr(runner, "load_warmup1_pose", loader)
    with pytest.raises(DeploymentError, match="warmup1-pose-file requires"):
        runner.run(args)
    client.assert_not_called()
    dds.assert_not_called()
    loader.assert_not_called()


def test_invalid_pose_fails_before_policy_dds_or_debug_writer(monkeypatch, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    args = runner.build_parser().parse_args([
        "--no-actuate", "--task", "pick-red-cup", "--warmup1",
        "--warmup1-pose-file", str(path), "--debug-actions",
    ])
    args._run_log_dir = str(tmp_path)
    boundaries = {}
    for name in ("Gr00tClient", "initialize_dds", "SafeG1Dex3Actuator", "ActionDebugRecorder", "TeleimagerCamera"):
        boundaries[name] = mock.Mock(side_effect=AssertionError(f"must not call {name}"))
        monkeypatch.setattr(runner, name, boundaries[name])
    monkeypatch.setattr(runner.os, "chdir", mock.Mock())
    with pytest.raises(DeploymentError, match="fields do not match"):
        runner.run(args)
    for boundary in boundaries.values():
        boundary.assert_not_called()
    assert runner._ACTION_DEBUG_RECORDER is None


def test_disabled_debug_has_no_writer_or_actuator_sink_and_disabled_warmup_has_no_load(monkeypatch, tmp_path):
    offline = full_demo_tests._OfflineRun(monkeypatch, tmp_path)
    debug_type = mock.Mock(side_effect=AssertionError("disabled debug must not construct writer"))
    pose_loader = mock.Mock(side_effect=AssertionError("disabled Warmup1 must not load pose"))
    monkeypatch.setattr(runner, "ActionDebugRecorder", debug_type)
    monkeypatch.setattr(runner, "load_warmup1_pose", pose_loader)
    runner.run(offline.args(
        "--actuate", "--sim", "--confirm-sim-network-isolated", "--no-warmup1", "--no-warmup2",
    ))
    debug_type.assert_not_called()
    pose_loader.assert_not_called()
    assert "action_debug_sink" not in offline.actuator_type.call_args.kwargs
    assert runner._ACTION_DEBUG_RECORDER is None


def test_custom_warmup_object_is_reused_for_return_and_saved_in_all_metadata(monkeypatch, tmp_path):
    offline = full_demo_tests._OfflineRun(monkeypatch, tmp_path)
    path, payload = _pose_file(tmp_path)
    offline.actuator.warmup1_pose = mock.Mock()
    returns = mock.Mock(return_value="continue")
    monkeypatch.setattr(runner, "_prepare_policy_goal", mock.Mock(return_value="hold"))
    monkeypatch.setattr(runner, "_select_next_goal_while_holding", mock.Mock(side_effect=[runner.RETURN_TO_START, None]))
    monkeypatch.setattr(runner, "_run_return_to_start_sequence_from_hold", returns)
    sink = object()
    debug = SimpleNamespace(
        record=mock.Mock(),
        make_sink=mock.Mock(return_value=sink),
        close=mock.Mock(side_effect=lambda: offline.events.append("debug.close")),
    )
    debug_type = mock.Mock(return_value=debug)
    monkeypatch.setattr(runner, "ActionDebugRecorder", debug_type)
    runner.run(offline.args(
        "--actuate", "--sim", "--confirm-sim-network-isolated", "--return-to-start",
        "--warmup1", "--warmup1-pose-file", str(path), "--no-warmup2",
        "--debug-actions", "--record-full", "--record-vision",
    ))
    selected = offline.actuator.warmup1_pose.call_args.args[0]
    assert selected.label == "Warmup1: another measured start"
    np.testing.assert_array_equal(selected.arm, payload["arm"])
    returns.assert_called_once()
    assert returns.call_args.args[1] == (selected,)
    assert returns.call_args.args[1][0] is selected
    assert returns.call_args.kwargs["warmup1_spec"] is selected
    assert not selected.arm.flags.writeable
    saved = json.loads((offline.run_dir / "warmup1_pose.json").read_text(encoding="utf-8"))
    assert saved["arm"] == payload["arm"]
    assert saved["source"]["pose_file"] == str(path.resolve())
    assert saved["source"]["pose_name"] == payload["name"]
    assert saved["source"]["quantity"] == "observation.state"
    assert len(saved["source"]["sha256"]) == 64
    assert debug_type.call_args.kwargs["metadata"]["warmup1"] == saved
    assert offline.stream_type.call_args.kwargs["metadata"]["warmup1"] == saved
    assert offline.recorder_type.call_args.kwargs["metadata"]["warmup1"] == saved
    debug.make_sink.assert_called_once_with("actuator")
    assert offline.actuator_type.call_args.kwargs["action_debug_sink"] is sink
    assert offline.events.index("actuator.close") < offline.events.index("debug.close")
    debug.close.assert_called_once()
    assert runner._ACTION_DEBUG_RECORDER is None


def test_debug_request_records_state_language_and_rtc_options_but_not_video(monkeypatch):
    recorder = SimpleNamespace(record=mock.Mock())
    monkeypatch.setattr(runner, "_ACTION_DEBUG_RECORDER", recorder)
    state = {"left_arm": np.zeros((1, 1, 7), dtype=np.float32)}
    language = {"annotation.human.action.task_description": [["pick up the red cup."]]}
    options = {"inference_mode": "rtc", "rtc_pure": True, "rtc_frozen_steps": 5}
    observation = {"state": state, "language": language, "video": object()}
    request_id = runner._debug_policy_request(observation, options)
    assert isinstance(request_id, str) and request_id
    recorder.record.assert_called_once()
    assert recorder.record.call_args.args == ("policy_request",)
    fields = recorder.record.call_args.kwargs
    assert fields["request_id"] == request_id
    assert fields["state"] is state
    assert fields["language"] is language
    assert fields["options"] is options
    assert "video" not in fields and "observation" not in fields


def test_debug_plan_records_exact_vectors_and_extra_context(monkeypatch):
    recorder = SimpleNamespace(record=mock.Mock())
    monkeypatch.setattr(runner, "_ACTION_DEBUG_RECORDER", recorder)
    plan = ActionChunk(arm=np.zeros((32, 14)), left_hand=np.ones((32, 6)), right_hand=np.zeros((32, 6)))
    runner._debug_plan("rtc_replacement", plan, request_id="request-1", sequence=5)
    recorder.record.assert_called_once()
    assert recorder.record.call_args.args == ("rtc_replacement",)
    fields = recorder.record.call_args.kwargs
    assert fields["arm"] is plan.arm
    assert fields["left_hand"] is plan.left_hand
    assert fields["right_hand"] is plan.right_hand
    assert fields["request_id"] == "request-1" and fields["sequence"] == 5


def test_debug_helpers_do_nothing_when_disabled(monkeypatch):
    clock = mock.Mock(side_effect=AssertionError("disabled debug must not generate IDs"))
    monkeypatch.setattr(runner.time, "monotonic_ns", clock)
    assert runner._debug_policy_request(object(), object()) is None
    runner._debug_plan("unused", object())
    runner._debug_action_event("unused", arbitrary=object())
    clock.assert_not_called()


def test_debug_record_failures_cannot_interrupt_policy_helpers(monkeypatch):
    recorder = SimpleNamespace(record=mock.Mock(side_effect=RuntimeError("writer broken")))
    monkeypatch.setattr(runner, "_ACTION_DEBUG_RECORDER", recorder)
    assert runner._debug_policy_request({"state": {}, "language": {}}, {"rtc_pure": True})
    runner._debug_action_event("event", sample=1)
    runner._debug_plan("plan", SimpleNamespace(arm=[], left_hand=[], right_hand=[]))
    assert recorder.record.call_count == 3
