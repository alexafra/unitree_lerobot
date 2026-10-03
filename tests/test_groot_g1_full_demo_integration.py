"""Offline CLI wiring tests: never open sockets, spawn workers, or actuate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import unitree_lerobot.eval_robot.eval_groot_g1 as runner
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import COLOUR_VIDEO_KEYS, ModelContract


class _OfflineRun:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.events: list[str] = []
        self.tmp_path = tmp_path
        self.run_dir = tmp_path / "client-log"
        self.run_dir.mkdir()
        self.recordings_dir = tmp_path / "Recordings_Data"
        self.observations: list[dict[str, object]] = []
        self.camera_reads = 0
        self.stream = SimpleNamespace(
            operator_exit_requested=False,
            stats={},
            close=mock.Mock(side_effect=lambda: self.events.append("demo.close")),
            mark_event=mock.Mock(),
        )
        self.recorder = SimpleNamespace(
            stats={},
            submit_observation=mock.Mock(),
            close=mock.Mock(side_effect=lambda: self.events.append("vision.close")),
        )
        self.policy = SimpleNamespace(
            ping=lambda: True,
            get_modality_config=lambda: {},
            get_policy_metadata=lambda: {
                "model_identity": {
                    "model_name": "offline-model",
                    "checkpoint": "checkpoint-120",
                    "model_path": "/offline/checkpoint-120",
                }
            },
            reset=lambda: self.events.append("policy.reset"),
            close=lambda: self.events.append("policy.close"),
        )
        self.reader = SimpleNamespace(
            read=mock.Mock(
                return_value=SimpleNamespace(
                    arm=np.zeros(14, dtype=np.float64),
                    left_hand=np.zeros(7, dtype=np.float64),
                    right_hand=np.zeros(7, dtype=np.float64),
                )
            ),
            close=lambda: self.events.append("reader.close"),
        )
        self.camera = SimpleNamespace(
            config={
                "head_camera": {
                    "type": "fake",
                    "image_shape": [480, 640],
                    "binocular": False,
                    "fps": 30,
                }
            },
            read=mock.Mock(side_effect=self._camera_read),
            close=lambda: self.events.append("camera.close"),
        )
        self.actuator = SimpleNamespace(
            start=lambda: self.events.append("actuator.start"),
            arm=lambda: self.events.append("actuator.arm"),
            initialize=lambda _spec: self.events.append("actuator.initialize"),
            close=lambda: self.events.append("actuator.close"),
        )
        self.contract = ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS)
        self.stream_type = mock.Mock(side_effect=self._stream_start)
        self.recorder_type = mock.Mock(side_effect=self._recorder_start)
        self.actuator_type = mock.Mock(return_value=self.actuator)
        self.infer = mock.Mock(side_effect=self._infer)
        self.preview = mock.Mock(side_effect=AssertionError("parent must not render demo frames"))

        # Every external boundary is replaced. Infer uses the actual observation
        # builder so any extra parent camera read/preview or recorder swap is visible.
        monkeypatch.setattr(runner, "Gr00tClient", mock.Mock(return_value=self.policy))
        monkeypatch.setattr(runner, "validate_model_contract", mock.Mock(return_value=self.contract))
        monkeypatch.setattr(runner, "validate_policy_metadata", mock.Mock(return_value=None))
        monkeypatch.setattr(runner, "initialize_dds", mock.Mock())
        monkeypatch.setattr(runner, "G1Dex3StateReader", mock.Mock(return_value=self.reader))
        monkeypatch.setattr(runner, "TeleimagerCamera", mock.Mock(side_effect=self._camera_start))
        monkeypatch.setattr(runner, "ContinuousDemoStream", self.stream_type)
        monkeypatch.setattr(runner, "NonBlockingVisionRecorder", self.recorder_type)
        monkeypatch.setattr(runner, "SafeG1Dex3Actuator", self.actuator_type)
        monkeypatch.setattr(runner, "infer_chunk", self.infer)
        monkeypatch.setattr(runner, "chunk_delta_summary", lambda *_args: "bounded")
        monkeypatch.setattr(runner, "show_camera_preview", self.preview)
        monkeypatch.setattr(runner, "close_camera_preview", mock.Mock())
        monkeypatch.setattr(runner, "confirm_actuation", mock.Mock())
        monkeypatch.setattr(runner, "confirm_initialization", mock.Mock())
        monkeypatch.setattr(runner, "confirm_policy_start", mock.Mock())
        monkeypatch.setattr(runner, "_prepare_policy_goal", mock.Mock(return_value="release"))
        monkeypatch.setattr(
            runner,
            "_run_blocking_motion_with_immediate_release",
            lambda _actuator, operation, **_kwargs: operation(),
        )
        monkeypatch.setattr(runner, "_CONTINUOUS_DEMO_STREAM", None)
        monkeypatch.setattr(runner.os, "chdir", mock.Mock())

    def args(self, *flags: str):
        args = runner.build_parser().parse_args(
            [
                "--end-effector", "dex3", "--no-actuate", "--no-return-to-start",
                "--task",
                "pick-red-cup",
                "--max-chunks",
                "1",
                "--vision-recordings-dir",
                str(self.recordings_dir),
                *flags,
            ]
        )
        args._run_log_dir = str(self.run_dir)
        return args

    def _camera_start(self, *_args, **_kwargs):
        self.events.append("camera.start")
        return self.camera

    def _stream_start(self, *_args, **_kwargs):
        self.events.append("demo.start")
        return self.stream

    def _recorder_start(self, output_dir, *_args, **_kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=False)
        return self.recorder

    def _camera_read(self, *, timeout_s):
        assert timeout_s > 0
        self.camera_reads += 1
        self.events.append("camera.read")
        return SimpleNamespace(
            rgb=np.full((480, 640, 3), self.camera_reads, dtype=np.uint8),
            depth_gray=None,
            surface_normals=None,
        )

    def _infer(self, policy, reader, camera, instruction, contract, *_args, **kwargs):
        assert policy is self.policy
        assert reader is self.reader
        assert camera is self.camera
        self.events.append("infer")
        observation, _state = runner.capture_policy_observation(
            reader,
            camera,
            instruction,
            contract,
            camera_timeout_s=kwargs.get("camera_timeout_s", 0.5),
            show_camera=kwargs.get("show_camera", False),
            vision_recorder=kwargs.get("vision_recorder"),
        )
        self.observations.append(observation)
        return SimpleNamespace(length=1), 0.01


@pytest.fixture
def offline_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _OfflineRun:
    return _OfflineRun(monkeypatch, tmp_path)


def test_full_demo_flags_are_opt_in_and_aliases_preserve_control_defaults() -> None:
    parser = runner.build_parser()
    defaults = parser.parse_args([])
    assert defaults.record_full is False
    assert defaults.demo_fps == 30.0
    for flag in ("--record-full", "--record_full"):
        args = parser.parse_args([flag])
        assert args.record_full is True
        assert args.record_vision is False
        assert args.actuate == defaults.actuate
        assert args.execution_horizon == defaults.execution_horizon
        assert args.command_conditioning == defaults.command_conditioning
        assert args.action_interpolation == defaults.action_interpolation
    assert runner.CONTROL_HZ == 30.0


@pytest.mark.parametrize("fps", ["0", "-1", "31", "nan", "inf"])
def test_invalid_demo_fps_is_rejected_without_starting_anything(fps: str) -> None:
    args = runner.build_parser().parse_args([f"--demo-fps={fps}"])
    with pytest.raises(DeploymentError, match="demo-fps"):
        runner.validate_args(args)


def test_disabled_demo_does_not_construct_worker_or_read_extra_images(offline_run: _OfflineRun) -> None:
    runner.run(offline_run.args())
    offline_run.stream_type.assert_not_called()
    offline_run.recorder_type.assert_not_called()
    offline_run.actuator_type.assert_not_called()
    assert offline_run.camera_reads == offline_run.infer.call_count == 1
    offline_run.preview.assert_not_called()


def test_preview_only_starts_before_preflight_without_recording_writer(offline_run: _OfflineRun) -> None:
    runner.run(offline_run.args("--show-camera"))
    offline_run.stream_type.assert_called_once()
    kwargs = offline_run.stream_type.call_args.kwargs
    assert kwargs["output_dir"] is None
    assert kwargs["show_camera"] is True
    assert kwargs["target_fps"] == 30.0
    assert tuple(kwargs["video_keys"]) == COLOUR_VIDEO_KEYS
    assert kwargs["depth_encoding"] is None
    assert kwargs["surface_normal_encoding"] is None
    assert kwargs["prefer_atomic_rgbd"] is False
    assert offline_run.events.index("camera.start") < offline_run.events.index("demo.start")
    assert offline_run.events.index("demo.start") < offline_run.events.index("infer")
    offline_run.recorder_type.assert_not_called()
    offline_run.actuator_type.assert_not_called()
    offline_run.preview.assert_not_called()
    assert offline_run.camera_reads == offline_run.infer.call_count == 1
    assert not offline_run.recordings_dir.exists()
    offline_run.stream.close.assert_called_once_with()
    assert runner._CONTINUOUS_DEMO_STREAM is None


@pytest.mark.parametrize("with_policy_recording", [False, True])
def test_full_demo_and_policy_capture_recordings_remain_separate(
    offline_run: _OfflineRun,
    with_policy_recording: bool,
) -> None:
    flags = ["--record-full", "--demo-fps", "24"]
    if with_policy_recording:
        flags.append("--record-vision")
    args = offline_run.args(*flags)
    runner.run(args)
    kwargs = offline_run.stream_type.call_args.kwargs
    demo_dir = Path(kwargs["output_dir"])
    assert demo_dir.name == "full_demo"
    assert demo_dir.is_relative_to(offline_run.recordings_dir)
    assert "offline-model" in demo_dir.parts
    assert kwargs["metadata"]["client_log_directory"] == str(offline_run.run_dir)
    assert kwargs["show_camera"] is False
    assert kwargs["target_fps"] == 24.0
    assert runner.CONTROL_HZ == 30.0
    assert args.execution_horizon == 8
    assert offline_run.camera_reads == offline_run.infer.call_count == 1
    offline_run.actuator_type.assert_not_called()
    if with_policy_recording:
        offline_run.recorder_type.assert_called_once()
        call = offline_run.recorder_type.call_args
        policy_dir = Path(call.kwargs.get("output_dir", call.args[0] if call.args else None))
        assert demo_dir.parent == policy_dir
        assert offline_run.infer.call_args.kwargs["vision_recorder"] is offline_run.recorder
        submitted = offline_run.recorder.submit_observation.call_args.args[0]
        assert submitted is offline_run.observations[0]
        assert tuple(submitted["video"]) == COLOUR_VIDEO_KEYS
        assert np.all(submitted["video"]["ego_view"] == 1)
        offline_run.recorder.close.assert_called_once_with()
    else:
        offline_run.recorder_type.assert_not_called()
        assert args._vision_recorder is None
    offline_run.stream.close.assert_called_once_with()
    assert runner._CONTINUOUS_DEMO_STREAM is None


def test_actuator_release_precedes_demo_close_and_policy_camera_close(offline_run: _OfflineRun) -> None:
    args = offline_run.args(
        "--record-full",
        "--record-vision",
        "--show-camera",
        "--actuate",
        "--sim",
        "--confirm-sim-network-isolated",
        "--no-warmup1",
        "--no-warmup2",
    )
    runner.run(args)
    events = offline_run.events
    assert events.index("demo.start") < events.index("infer") < events.index("actuator.start")
    assert events.index("demo.start") < events.index("actuator.initialize")
    assert events.count("actuator.close") == 1
    assert events.index("actuator.close") < events.index("demo.close") < events.index("camera.close")
    assert events.index("actuator.close") < events.index("vision.close")
    assert offline_run.camera_reads == offline_run.infer.call_count == 1
    assert runner._CONTINUOUS_DEMO_STREAM is None


def test_demo_cleanup_failure_does_not_mask_policy_error_or_skip_camera_close(offline_run: _OfflineRun) -> None:
    offline_run.infer.side_effect = RuntimeError("offline inference failure")
    offline_run.stream.close.side_effect = RuntimeError("offline recorder cleanup failure")
    with pytest.raises(RuntimeError, match="offline inference failure"):
        runner.run(offline_run.args("--record-full"))
    offline_run.stream.close.assert_called_once_with()
    assert "camera.close" in offline_run.events
    assert "policy.close" in offline_run.events
    offline_run.actuator_type.assert_not_called()
    assert runner._CONTINUOUS_DEMO_STREAM is None


def test_worker_preview_quit_requests_release_without_parent_highgui(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = SimpleNamespace(operator_exit_requested=False)
    monkeypatch.setattr(runner, "_CONTINUOUS_DEMO_STREAM", stream)
    monkeypatch.setattr(runner, "_CAMERA_PREVIEW_ACTIVE", False)
    poll = mock.Mock(side_effect=AssertionError("parent HighGUI must remain unused"))
    monkeypatch.setattr(runner.cv2, "pollKey", poll)
    runner._pump_camera_preview_events()
    stream.operator_exit_requested = True
    with pytest.raises(runner.OperatorRelease):
        runner._pump_camera_preview_events()
    poll.assert_not_called()
