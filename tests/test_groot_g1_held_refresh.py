"""Publisher-free integration check for a held RGB -> depth policy switch."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

from unitree_lerobot.eval_robot import eval_groot_g1 as runner
from unitree_lerobot.eval_robot.groot_contract import ModelContract, DepthEncodingContract


def test_ctrl_r_is_instant_only_in_held_prompt(monkeypatch):
    terminal = SimpleNamespace(fileno=lambda: 7)
    monkeypatch.setattr(sys, "stdin", terminal)
    monkeypatch.setattr(runner.termios, "tcgetattr", lambda _fd: [0, 0, 0, 0, 0, 0, [0] * 32])
    monkeypatch.setattr(runner.termios, "tcsetattr", lambda *_args: None)
    monkeypatch.setattr(runner.select, "select", lambda *_args: ([7], [], []))
    monkeypatch.setattr(runner, "_pump_camera_preview_events", lambda: None)
    actuator = SimpleNamespace(heartbeat=lambda: None, assert_healthy=lambda: False)
    keys = iter((b"\x12",))
    monkeypatch.setattr(runner.os, "read", lambda *_args: next(keys))
    assert runner._readline_with_immediate_prompt_controls(
        actuator, "HOLD> ", None, goal_mode_toggle=True, policy_refresh=True,
    ) == runner.REFRESH_POLICY
    keys = iter((b"\x12", b"\n"))
    assert runner._readline_with_immediate_prompt_controls(
        actuator, "Task> ", None, goal_mode_toggle=True, policy_refresh=False,
    ) == ""


def test_held_refresh_reopens_camera_recording_and_requires_explicit_goal(monkeypatch, tmp_path):
    events = []
    old_contract = ModelContract(32)
    new_contract = ModelContract(32, video_keys=("ego_view", "depth_gray_view"))
    old_metadata = {"server_instance_id": "old", "model_identity": {
        "model_name": "old-model", "checkpoint": "checkpoint-30000"}}
    new_metadata = {"server_instance_id": "new", "model_identity": {
        "model_name": "new-model", "checkpoint": "checkpoint-30000"}}

    def policy(name, metadata):
        return SimpleNamespace(
            ping=lambda: True, get_modality_config=lambda: {},
            get_policy_metadata=lambda: metadata,
            reset=lambda: events.append(f"{name}.reset"),
            close=lambda: events.append(f"{name}.close"),
        )

    policies = iter((policy("old", old_metadata), policy("new", new_metadata)))
    contracts = iter((old_contract, new_contract))
    cameras = []

    def camera(_host, *, depth_encoding=None, **_kwargs):
        number = len(cameras)
        events.append(f"camera{number}.open:{depth_encoding is not None}")
        instance = SimpleNamespace(
            config={"head_camera": {}},
            close=lambda: events.append(f"camera{number}.close"),
        )
        cameras.append(instance)
        return instance

    def recorder(path, *_args, **_kwargs):
        Path(path).mkdir(parents=True, exist_ok=False)
        number = len(recorders)
        instance = SimpleNamespace(close=lambda: events.append(f"recorder{number}.close"), stats={})
        recorders.append(instance)
        return instance

    recorders = []
    actuator = SimpleNamespace(
        _holding=True,
        start=lambda: events.append("actuator.start"),
        arm=lambda: events.append("actuator.arm"),
        initialize=lambda _spec: events.append("actuator.initialize"),
        hold=lambda: events.append("actuator.hold"),
        heartbeat=lambda: None,
        assert_healthy=lambda: True,
        immediate_control_requested=lambda: None,
        close=lambda: events.append("actuator.close"),
    )
    args = runner.build_parser().parse_args([
        "--task", "pick-red-cup", "--end-effector", "dex3", "--actuate", "--sim",
        "--confirm-sim-network-isolated", "--no-return-to-start", "--no-warmup1",
        "--no-warmup2", "--record-vision", "--vision-recordings-dir", str(tmp_path / "recordings"),
    ])
    args._run_log_dir = str(tmp_path)
    selections = iter((runner.REFRESH_POLICY, ("down-red-cup", runner.TASKS["down-red-cup"]), None))
    active_contracts = []

    def active(_policy, _reader, _camera, _actuator, _task, _instruction, contract, *_args, **_kwargs):
        active_contracts.append(contract.video_keys)
        return "hold"

    def infer(*_args, **_kwargs):
        events.append("preflight")
        return SimpleNamespace(length=1), 0.01

    with mock.patch.object(runner, "Gr00tClient", side_effect=lambda *_args: next(policies)), \
         mock.patch.object(runner, "validate_model_contract", side_effect=lambda *_args, **_kwargs: next(contracts)), \
         mock.patch.object(runner, "validate_policy_metadata", side_effect=lambda metadata, **_kwargs:
                           DepthEncodingContract(.25, 1.0) if metadata is new_metadata else None), \
         mock.patch.object(runner, "initialize_dds"), \
         mock.patch.object(runner, "G1Dex3StateReader", return_value=SimpleNamespace(close=lambda: events.append("reader.close"))), \
         mock.patch.object(runner, "TeleimagerCamera", side_effect=camera), \
         mock.patch.object(runner, "NonBlockingVisionRecorder", side_effect=recorder), \
         mock.patch.object(runner, "SafeG1Dex3Actuator", return_value=actuator), \
         mock.patch.object(runner, "infer_chunk", side_effect=infer), \
         mock.patch.object(runner, "chunk_delta_summary", return_value="bounded"), \
         mock.patch.object(runner, "confirm_actuation"), \
         mock.patch.object(runner, "confirm_initialization"), \
         mock.patch.object(runner, "confirm_policy_start"), \
         mock.patch.object(runner, "_prepare_policy_goal", return_value="ready"), \
         mock.patch.object(runner, "_run_active_goal", side_effect=active), \
         mock.patch.object(runner, "_run_with_immediate_operator_keys", side_effect=lambda _a, fn: fn()), \
         mock.patch.object(runner, "_select_next_goal_while_holding", side_effect=lambda *_a, **_kw: next(selections)):
        runner.run(args)

    assert active_contracts == [("ego_view",), ("ego_view", "depth_gray_view")]
    assert events.index("camera0.close") < events.index("camera1.open:True")
    assert events.index("recorder0.close") < events.index("camera1.open:True")
    assert events.count("preflight") == 2  # one startup, one discarded refresh action
    assert events.count("actuator.initialize") == 1
    assert events.count("actuator.start") == 1
    assert events.count("actuator.close") == 1
    assert (tmp_path / "recordings" / "old-model").is_dir()
    assert (tmp_path / "recordings" / "new-model").is_dir()
