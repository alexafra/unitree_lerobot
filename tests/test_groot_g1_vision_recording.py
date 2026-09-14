from __future__ import annotations

import json
from pathlib import Path
import queue
import time
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

import unitree_lerobot.eval_robot.eval_groot_g1 as eval_groot_g1
import unitree_lerobot.eval_robot.vision_recorder as vision_recorder_module
from unitree_lerobot.eval_robot.eval_groot_g1 import (
    build_parser,
    capture_policy_observation,
    infer_chunk,
    run as run_groot,
)
from unitree_lerobot.eval_robot.groot_contract import (
    ActionChunk,
    COLOUR_VIDEO_KEYS,
    DEPTH_ONLY_VIDEO_KEYS,
    RGBD_VIDEO_KEYS,
    SURFACE_NORMAL_ONLY_VIDEO_KEYS,
    SURFACE_NORMAL_VIDEO_KEYS,
    TASKS,
    ModelContract,
)
from unitree_lerobot.eval_robot.vision_recorder import NonBlockingVisionRecorder


class _FakeStateReader:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            arm=np.linspace(-0.1, 0.1, 14, dtype=np.float64),
            left_hand=np.linspace(0.0, 0.6, 7, dtype=np.float64),
            right_hand=np.linspace(0.6, 0.0, 7, dtype=np.float64),
        )

    def read(self, timeout_s: float) -> object:
        assert timeout_s == 0.5
        return self.state


class _FakeCamera:
    def __init__(
        self,
        *,
        rgb: np.ndarray,
        depth_gray: np.ndarray | None,
        surface_normals: np.ndarray | None,
    ) -> None:
        self.images = SimpleNamespace(
            rgb=rgb,
            depth_gray=depth_gray,
            surface_normals=surface_normals,
        )

    def read(self, timeout_s: float) -> object:
        assert timeout_s == 0.5
        return self.images


class _CollectingRecorder:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    def submit_observation(self, observation: dict[str, object]) -> None:
        self.submissions.append(observation)


def test_record_vision_cli_is_strictly_opt_in() -> None:
    parser = build_parser()

    assert parser.parse_args([]).record_vision is False
    assert parser.parse_args(["--record-vision"]).record_vision is True


@pytest.mark.parametrize(
    "video_keys",
    [
        COLOUR_VIDEO_KEYS,
        RGBD_VIDEO_KEYS,
        SURFACE_NORMAL_VIDEO_KEYS,
        DEPTH_ONLY_VIDEO_KEYS,
        SURFACE_NORMAL_ONLY_VIDEO_KEYS,
    ],
)
def test_capture_submits_the_exact_selected_policy_video_observation(
    video_keys: tuple[str, ...],
) -> None:
    rgb = np.full((480, 640, 3), 17, dtype=np.uint8)
    depth = np.full((480, 640, 3), 83, dtype=np.uint8) if "depth_gray_view" in video_keys else None
    normals = np.full((480, 640, 3), 149, dtype=np.uint8) if "surface_normals_view" in video_keys else None
    camera = _FakeCamera(rgb=rgb, depth_gray=depth, surface_normals=normals)
    recorder = _CollectingRecorder()
    contract = ModelContract(action_horizon=32, video_keys=video_keys)

    observation, _state = capture_policy_observation(
        _FakeStateReader(),
        camera,
        TASKS["pick-red-cup"],
        contract,
        vision_recorder=recorder,
    )

    assert len(recorder.submissions) == 1
    assert recorder.submissions[0] is observation
    submitted_video = recorder.submissions[0]["video"]
    assert isinstance(submitted_video, dict)
    assert tuple(submitted_video) == video_keys
    expected_by_key = {
        "ego_view": rgb,
        "depth_gray_view": depth,
        "surface_normals_view": normals,
    }
    for key in video_keys:
        submitted = submitted_video[key]
        assert isinstance(submitted, np.ndarray)
        assert submitted.shape == (1, 1, 480, 640, 3)
        np.testing.assert_array_equal(submitted[0, 0], expected_by_key[key])


def test_capture_with_recording_disabled_does_not_change_the_observation() -> None:
    rgb = np.full((480, 640, 3), 23, dtype=np.uint8)
    camera = _FakeCamera(rgb=rgb, depth_gray=None, surface_normals=None)
    contract = ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS)

    observation, _state = capture_policy_observation(
        _FakeStateReader(),
        camera,
        TASKS["pick-red-cup"],
        contract,
        vision_recorder=None,
    )

    assert tuple(observation["video"]) == COLOUR_VIDEO_KEYS
    np.testing.assert_array_equal(observation["video"]["ego_view"][0, 0], rgb)


def test_recorder_failure_never_rejects_a_valid_policy_observation() -> None:
    class FailingRecorder:
        def submit_observation(self, _observation: dict[str, object]) -> None:
            raise RuntimeError("unexpected recorder implementation failure")

    rgb = np.full((480, 640, 3), 31, dtype=np.uint8)
    with mock.patch.object(eval_groot_g1, "LOGGER") as logger:
        observation, _state = capture_policy_observation(
            _FakeStateReader(),
            _FakeCamera(rgb=rgb, depth_gray=None, surface_normals=None),
            TASKS["pick-red-cup"],
            ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS),
            vision_recorder=FailingRecorder(),
        )

    logger.assert_not_called()
    np.testing.assert_array_equal(observation["video"]["ego_view"][0, 0], rgb)


def test_recorder_keeps_each_policy_capture_even_when_safety_recaptures() -> None:
    """Recording cadence follows captures, including a subsequently discarded one."""

    actuator = SimpleNamespace(hand_pause_generation=0)
    actuator.immediate_control_requested = lambda: None
    actuator.wait_for_hand_feedback = lambda: None
    actuator.heartbeat = lambda: None

    class ChangingCamera:
        calls = 0

        def read(self, timeout_s: float) -> object:
            assert timeout_s == 0.5
            self.calls += 1
            if self.calls == 1:
                actuator.hand_pause_generation += 1
            return SimpleNamespace(
                rgb=np.full((480, 640, 3), self.calls, dtype=np.uint8),
                depth_gray=None,
                surface_normals=None,
            )

    class FakePolicy:
        calls = 0

        def get_action(self, _observation: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            return {}

        def reset(self) -> None:
            raise AssertionError("the policy is not called for the discarded capture")

    recorder = _CollectingRecorder()
    policy = FakePolicy()
    parsed = ActionChunk(
        arm=np.zeros((1, 14), dtype=np.float64),
        left_hand=np.zeros((1, 7), dtype=np.float64),
        right_hand=np.zeros((1, 7), dtype=np.float64),
    )
    with mock.patch.object(eval_groot_g1, "parse_action_chunk", return_value=parsed):
        infer_chunk(
            policy,
            _FakeStateReader(),
            ChangingCamera(),
            TASKS["pick-red-cup"],
            ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS),
            execution_horizon=1,
            actuator=actuator,
            vision_recorder=recorder,
        )

    assert policy.calls == 1
    assert len(recorder.submissions) == 2
    assert np.all(recorder.submissions[0]["video"]["ego_view"] == 1)
    assert np.all(recorder.submissions[1]["video"]["ego_view"] == 2)


@pytest.mark.parametrize("enabled", [False, True])
def test_run_constructs_and_passes_only_an_opted_in_recorder(
    tmp_path: Path,
    enabled: bool,
) -> None:
    argv = ["--task", "pick-red-cup", "--max-chunks", "1"]
    if enabled:
        argv.append("--record-vision")
    args = build_parser().parse_args(argv)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args._run_log_dir = str(run_dir)
    contract = ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS)
    fake_chunk = SimpleNamespace(length=1)

    class FakePolicy:
        def ping(self) -> bool:
            return True

        def get_modality_config(self) -> dict[str, object]:
            return {}

        def get_policy_metadata(self) -> dict[str, object]:
            return {}

        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeReader:
        def close(self) -> None:
            pass

    class FakeCamera:
        config = {
            "head_camera": {
                "type": "fake",
                "image_shape": [480, 640],
                "binocular": False,
                "fps": 30,
            }
        }

        def close(self) -> None:
            pass

    fake_recorder = mock.Mock()
    recorder_type = mock.Mock(return_value=fake_recorder)
    infer = mock.Mock(return_value=(fake_chunk, 0.01))
    with (
        mock.patch.object(eval_groot_g1, "Gr00tClient", return_value=FakePolicy()),
        mock.patch.object(eval_groot_g1, "validate_model_contract", return_value=contract),
        mock.patch.object(eval_groot_g1, "validate_policy_metadata", return_value=None),
        mock.patch.object(eval_groot_g1, "initialize_dds"),
        mock.patch.object(eval_groot_g1, "G1Dex3StateReader", return_value=FakeReader()),
        mock.patch.object(eval_groot_g1, "TeleimagerCamera", return_value=FakeCamera()),
        mock.patch.object(eval_groot_g1, "NonBlockingVisionRecorder", recorder_type),
        mock.patch.object(eval_groot_g1, "infer_chunk", infer),
        mock.patch.object(eval_groot_g1, "chunk_delta_summary", return_value="bounded"),
    ):
        run_groot(args)

    if not enabled:
        recorder_type.assert_not_called()
        fake_recorder.close.assert_not_called()
        assert getattr(args, "_vision_recorder", None) is None
        return

    recorder_type.assert_called_once()
    constructor_call = recorder_type.call_args
    output_dir = constructor_call.kwargs.get(
        "output_dir",
        constructor_call.args[0] if constructor_call.args else None,
    )
    video_keys = constructor_call.kwargs.get(
        "video_keys",
        constructor_call.args[1] if len(constructor_call.args) > 1 else None,
    )
    assert Path(output_dir) == run_dir / "vision_recording"
    assert tuple(video_keys) == COLOUR_VIDEO_KEYS
    assert infer.call_args.kwargs["vision_recorder"] is fake_recorder
    fake_recorder.close.assert_called_once_with()


def test_actuator_release_precedes_vision_recorder_close(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--task",
            "pick-red-cup",
            "--record-vision",
            "--actuate",
            "--sim",
            "--confirm-sim-network-isolated",
            "--no-warmup1",
            "--no-warmup2",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args._run_log_dir = str(run_dir)
    events: list[str] = []
    contract = ModelContract(action_horizon=32, video_keys=COLOUR_VIDEO_KEYS)
    fake_chunk = SimpleNamespace(length=1)

    class FakePolicy:
        def ping(self) -> bool:
            return True

        def get_modality_config(self) -> dict[str, object]:
            return {}

        def get_policy_metadata(self) -> dict[str, object]:
            return {}

        def reset(self) -> None:
            pass

        def close(self) -> None:
            events.append("policy.close")

    class FakeReader:
        def close(self) -> None:
            events.append("reader.close")

    class FakeCamera:
        config = {
            "head_camera": {
                "type": "fake",
                "image_shape": [480, 640],
                "binocular": False,
                "fps": 30,
            }
        }

        def close(self) -> None:
            events.append("camera.close")

    class FakeActuator:
        def start(self) -> None:
            events.append("actuator.start")

        def arm(self) -> None:
            events.append("actuator.arm")

        def initialize(self, _spec: object) -> None:
            events.append("actuator.initialize")

        def close(self) -> None:
            events.append("actuator.close")

    class FakeRecorder:
        def close(self) -> None:
            events.append("recorder.close")

    recorder = FakeRecorder()
    with (
        mock.patch.object(eval_groot_g1, "Gr00tClient", return_value=FakePolicy()),
        mock.patch.object(eval_groot_g1, "validate_model_contract", return_value=contract),
        mock.patch.object(eval_groot_g1, "validate_policy_metadata", return_value=None),
        mock.patch.object(eval_groot_g1, "initialize_dds"),
        mock.patch.object(eval_groot_g1, "G1Dex3StateReader", return_value=FakeReader()),
        mock.patch.object(eval_groot_g1, "TeleimagerCamera", return_value=FakeCamera()),
        mock.patch.object(eval_groot_g1, "NonBlockingVisionRecorder", return_value=recorder),
        mock.patch.object(eval_groot_g1, "SafeG1Dex3Actuator", return_value=FakeActuator()),
        mock.patch.object(eval_groot_g1, "infer_chunk", return_value=(fake_chunk, 0.01)),
        mock.patch.object(eval_groot_g1, "chunk_delta_summary", return_value="bounded"),
        mock.patch.object(eval_groot_g1, "confirm_actuation"),
        mock.patch.object(eval_groot_g1, "confirm_initialization"),
        mock.patch.object(eval_groot_g1, "confirm_policy_start"),
        mock.patch.object(eval_groot_g1, "_prepare_policy_goal", return_value="release"),
        mock.patch.object(
            eval_groot_g1,
            "_run_blocking_motion_with_immediate_release",
            side_effect=lambda _actuator, operation, **_kwargs: operation(),
        ),
    ):
        run_groot(args)

    assert events.count("actuator.close") == 1
    assert events.count("recorder.close") == 1
    assert events.index("actuator.close") < events.index("recorder.close")


def _wait_for_path(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"timed out waiting for recorder artifact {path}"


def _tiny_colour_observation(value: int = 19) -> dict[str, object]:
    rgb = np.full((12, 16, 3), value, dtype=np.uint8)
    return {"video": {"ego_view": rgb[None, None]}}


class _MockHungProcess:
    def __init__(self, events: list[object], *, dies_after_kill: bool) -> None:
        self.events = events
        self.alive = True
        self.dies_after_kill = dies_after_kill

    def is_alive(self) -> bool:
        self.events.append("is_alive")
        return self.alive

    def join(self, timeout: float) -> None:
        self.events.append(("join", timeout))

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        if self.dies_after_kill:
            self.alive = False


def _recorder_with_mock_process(
    output_dir: Path,
    process: _MockHungProcess,
) -> NonBlockingVisionRecorder:
    output_dir.mkdir()
    (output_dir / "ego_view").mkdir()
    (output_dir / "frames.jsonl").write_text("", encoding="utf-8")
    recorder = object.__new__(NonBlockingVisionRecorder)
    recorder.output_dir = output_dir
    recorder.video_keys = COLOUR_VIDEO_KEYS
    recorder._closed = False
    recorder._first_drop_reason = None
    recorder._final_stats = None
    recorder._queue = mock.Mock()
    recorder._stopping = mock.Mock()
    recorder._submitted = SimpleNamespace(value=1)
    recorder._queue_dropped = SimpleNamespace(value=0)
    recorder._written = SimpleNamespace(value=0)
    recorder._write_dropped = SimpleNamespace(value=0)
    recorder._process = process
    return recorder


def test_immediate_submit_then_close_accounts_for_every_sample(tmp_path: Path) -> None:
    """close() must drain or explicitly drop an item accepted immediately before it."""

    output_dir = tmp_path / "vision_recording"
    recorder = NonBlockingVisionRecorder(output_dir, COLOUR_VIDEO_KEYS)

    assert recorder.submit_observation(_tiny_colour_observation()) is True
    recorder.close()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 1
    assert summary["written"] + summary["dropped"] == summary["submitted"]
    assert recorder.stats == summary


def test_queue_full_drop_never_logs_from_submit_hot_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "vision_recording"
    recorder = NonBlockingVisionRecorder(output_dir, COLOUR_VIDEO_KEYS)

    try:
        with (
            mock.patch(
                "unitree_lerobot.eval_robot.vision_recorder.LOGGER",
            ) as logger,
            mock.patch.object(recorder._queue, "put_nowait", side_effect=queue.Full),
        ):
            assert recorder.submit_observation(_tiny_colour_observation()) is False
            logger.assert_not_called()

        assert recorder.stats["submitted"] == 1
        assert recorder.stats["dropped"] == 1
        assert recorder.stats["first_drop_reason"] == "Full"
    finally:
        # Cleanup/reporting occurs after leaving the logger assertion because it
        # is explicitly outside the timing-critical submit path.
        recorder.close()


def test_forced_worker_termination_writes_reconciled_summary_and_cleans_orphans(
    tmp_path: Path,
) -> None:
    """A killed writer must leave truthful accounting and no unindexed image files."""

    output_dir = tmp_path / "vision_recording"
    recorder = NonBlockingVisionRecorder(output_dir, COLOUR_VIDEO_KEYS)
    recorder._process.terminate()
    recorder._process.join(timeout=1.0)
    assert not recorder._process.is_alive()

    # Model one item rejected at enqueue and one item accepted by the queue but
    # lost with the killed worker.  The fallback summary must reconcile the latter
    # as dropped rather than reporting fewer outcomes than submissions.
    recorder._submitted.value = 2
    recorder._queue_dropped.value = 1
    recorder._written.value = 0
    recorder._write_dropped.value = 0

    orphan = output_dir / "ego_view" / "frame-000000.png"
    orphan.write_bytes(b"worker died before indexing this image")
    partial = output_dir / "ego_view" / ".frame-000001.png.partial"
    partial.write_bytes(b"incomplete")

    recorder.close()

    summary_path = output_dir / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["submitted"] == 2
    assert summary["written"] == 0
    assert summary["dropped"] == 2
    assert summary["written"] + summary["dropped"] == summary["submitted"]
    assert summary["worker_error"]
    assert summary["clean_shutdown"] is False
    assert recorder.stats == summary
    assert not orphan.exists()
    assert not partial.exists()
    assert not list(output_dir.rglob("*.partial"))


def test_close_timeout_terminates_then_kills_before_repairing(tmp_path: Path) -> None:
    """Exercise the actual bounded-close escalation without waiting on a process."""

    events: list[object] = []
    process = _MockHungProcess(events, dies_after_kill=True)
    output_dir = tmp_path / "vision_recording"
    recorder = _recorder_with_mock_process(output_dir, process)
    orphan = output_dir / "ego_view" / "frame-000000.png"
    orphan.write_bytes(b"unindexed")
    actual_repair = vision_recorder_module._repair_interrupted_recording

    def repair_only_after_stop(path: Path, video_keys: tuple[str, ...]) -> int:
        assert process.alive is False
        events.append("repair")
        return actual_repair(path, video_keys)

    with mock.patch.object(
        vision_recorder_module,
        "_repair_interrupted_recording",
        side_effect=repair_only_after_stop,
    ) as repair:
        recorder.close()

    repair.assert_called_once_with(output_dir, COLOUR_VIDEO_KEYS)
    assert events.index("terminate") < events.index("kill") < events.index("repair")
    assert [event for event in events if isinstance(event, tuple)] == [
        ("join", vision_recorder_module.VISION_RECORDING_CLOSE_TIMEOUT_S),
        ("join", 0.5),
        ("join", 0.5),
    ]
    assert not orphan.exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 1
    assert summary["written"] == 0
    assert summary["dropped"] == 1
    assert summary["clean_shutdown"] is False
    assert "bounded close deadline" in summary["worker_error"]


def test_close_does_not_repair_while_worker_survives_kill(tmp_path: Path) -> None:
    events: list[object] = []
    process = _MockHungProcess(events, dies_after_kill=False)
    output_dir = tmp_path / "vision_recording"
    recorder = _recorder_with_mock_process(output_dir, process)
    orphan = output_dir / "ego_view" / "frame-000000.png"
    orphan.write_bytes(b"possibly still owned by worker")

    with mock.patch.object(vision_recorder_module, "_repair_interrupted_recording") as repair:
        recorder.close()

    repair.assert_not_called()
    assert events.index("terminate") < events.index("kill")
    assert orphan.exists()
    assert not (output_dir / "summary.json").exists()
    assert recorder.stats["clean_shutdown"] is False
    assert "could not be stopped" in recorder.stats["worker_error"]


def test_final_output_path_symlink_is_rejected_without_writing_target(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    output_link = run_dir / "vision_recording"
    output_link.symlink_to(outside, target_is_directory=True)
    recorder: NonBlockingVisionRecorder | None = None

    try:
        with pytest.raises((ValueError, RuntimeError), match="(?i)symbolic link|symlink"):
            recorder = NonBlockingVisionRecorder(output_link, COLOUR_VIDEO_KEYS)
    finally:
        if recorder is not None:
            recorder.close()

    assert output_link.is_symlink()
    assert not outside.exists()


def test_process_recorder_round_trip_is_exact_bounded_and_drop_on_overload(
    tmp_path: Path,
) -> None:
    """One focused process test covers losslessness and overload semantics.

    The arbitrary normal image deliberately cannot be derived from another input
    in the observation.  A byte-exact result therefore proves the recorder reused
    the already-computed policy view instead of running geometry processing.
    """

    height, width = 480, 640
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = x % 251
    rgb[..., 1] = y % 239
    rgb[..., 2] = (x + y) % 233
    normals = np.empty_like(rgb)
    normals[..., 0] = (3 * x + 17) % 256
    normals[..., 1] = (5 * y + 29) % 256
    normals[..., 2] = (7 * x + 11 * y + 43) % 256
    observation = {
        "video": {
            "ego_view": rgb[None, None],
            "surface_normals_view": normals[None, None],
        },
        # Non-video values must not be serialized into the vision artifacts.
        "state": {"private_marker": np.array([123.0])},
        "annotation": {"human.task_description": "private task text"},
    }
    output_dir = tmp_path / "vision_recording"
    recorder = NonBlockingVisionRecorder(
        output_dir,
        SURFACE_NORMAL_VIDEO_KEYS,
        metadata={"run_id": "unit-test", "transport": "policy-observation"},
    )

    first_rgb_path = output_dir / "ego_view" / "frame-000000.png"
    first_normals_path = output_dir / "surface_normals_view" / "frame-000000.png"
    try:
        recorder.submit_observation(observation)
        _wait_for_path(first_rgb_path)
        _wait_for_path(first_normals_path)

        # The child is deliberately much slower at lossless PNG output than this
        # burst. Capacity-one DROP-NEW must bound both caller time and memory rather
        # than letting policy capture wait for the writer.
        started = time.monotonic()
        for _ in range(64):
            recorder.submit_observation(observation)
        submit_elapsed_s = time.monotonic() - started
    finally:
        recorder.close()

    assert submit_elapsed_s < 0.5
    decoded_rgb = cv2.cvtColor(cv2.imread(str(first_rgb_path)), cv2.COLOR_BGR2RGB)
    decoded_normals = cv2.cvtColor(
        cv2.imread(str(first_normals_path)),
        cv2.COLOR_BGR2RGB,
    )
    np.testing.assert_array_equal(decoded_rgb, rgb)
    np.testing.assert_array_equal(decoded_normals, normals)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_keys"] == list(SURFACE_NORMAL_VIDEO_KEYS)
    assert manifest["metadata"] == {
        "run_id": "unit-test",
        "transport": "policy-observation",
    }
    for artifact in ("manifest.json", "frames.jsonl", "summary.json"):
        contents = (output_dir / artifact).read_text(encoding="utf-8")
        assert "private task text" not in contents
        assert "private_marker" not in contents

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["submitted"] == 65
    assert summary["written"] >= 1
    assert summary["dropped"] >= 1
    assert summary["written"] + summary["dropped"] == summary["submitted"]
    assert summary["worker_error"] is None
    assert summary["clean_shutdown"] is True
    for key in ("submitted", "written", "dropped", "worker_error", "clean_shutdown"):
        assert recorder.stats[key] == summary[key]

    frame_records = [
        json.loads(line) for line in (output_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(frame_records) == summary["written"]
    assert len(list((output_dir / "ego_view").glob("frame-*.png"))) == summary["written"]
    assert len(list((output_dir / "surface_normals_view").glob("frame-*.png"))) == summary["written"]

    # close() is non-throwing and idempotent; it must not resurrect a worker or
    # alter the completed summary.
    recorder.close()
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == summary


def test_recorder_refuses_an_existing_directory_without_overwriting_it(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "vision_recording"
    output_dir.mkdir()
    existing_summary = output_dir / "summary.json"
    existing_summary.write_text('{"belongs_to":"older-run"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        NonBlockingVisionRecorder(output_dir, COLOUR_VIDEO_KEYS)

    assert existing_summary.read_text(encoding="utf-8") == '{"belongs_to":"older-run"}\n'
    assert list(output_dir.iterdir()) == [existing_summary]
