"""Offline tests: no real camera connection, GUI, robot or DDS participant."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from unitree_lerobot.eval_robot import demo_stream as demo
from unitree_lerobot.eval_robot.robot_control import safe_g1_dex3 as camera_module


def _counters():
    return {key: SimpleNamespace(value=0) for key in demo._COUNTERS}


def test_rgb_timestamp_proxy_retains_exact_consumed_frame():
    frame = SimpleNamespace(received_monotonic_ns=123)
    calls = []

    def get_frame():
        calls.append(True)
        return frame

    proxy = demo._TimestampedColorClient(SimpleNamespace(get_head_frame=get_frame, other="value"))
    assert proxy.get_head_frame() is frame
    assert proxy.received_monotonic_ns == 123
    assert len(calls) == 1
    assert proxy.other == "value"


@pytest.mark.parametrize("stamp", [None, -1, 1_100_000_000, 700_000_000])
def test_rgb_missing_future_or_stale_receive_stamp_is_rejected(stamp):
    camera = SimpleNamespace(_requires_depth=False, _client=SimpleNamespace(received_monotonic_ns=stamp))
    with pytest.raises(TimeoutError):
        demo._source_identity(camera, SimpleNamespace(sequence=None), ("ego_view",), 1_000_000_000)


def test_legacy_composite_is_rgb_led_and_preserves_component_timestamps():
    camera = SimpleNamespace(
        _requires_depth=True, _geometry_transport="legacy",
        _last_color_received_ns=990_000_000, _last_depth_received_ns=980_000_000,
    )
    identity, stamps = demo._source_identity(camera, SimpleNamespace(sequence=None), ("ego_view", "depth_gray_view"), 1_000_000_000)
    camera._last_depth_received_ns = 995_000_000
    next_identity, next_stamps = demo._source_identity(camera, SimpleNamespace(sequence=None), ("ego_view", "depth_gray_view"), 1_000_000_000)
    assert identity == next_identity
    assert stamps["color_received_monotonic_ns"] == 990_000_000
    assert stamps["depth_received_monotonic_ns"] == 980_000_000
    assert next_stamps["depth_received_monotonic_ns"] == 995_000_000
    geometry_identity, _ = demo._source_identity(camera, SimpleNamespace(sequence=None), ("depth_gray_view",), 1_000_000_000)
    assert geometry_identity[-1] == 995_000_000
    assert stamps["server_capture_monotonic_ns"] is None


def test_atomic_identity_uses_sequence_not_repeated_receive_time():
    camera = SimpleNamespace(
        _geometry_transport="atomic", _last_rgbd_received_ns=990_000_000,
        _last_rgbd_server_capture_ns=888,
    )
    first, stamps = demo._source_identity(camera, SimpleNamespace(sequence=7), ("ego_view",), 1_000_000_000)
    camera._last_rgbd_received_ns = 999_000_000
    second, _ = demo._source_identity(camera, SimpleNamespace(sequence=7), ("ego_view",), 1_000_000_000)
    assert first == second == ("atomic_rgbd", 7)
    assert stamps["server_capture_monotonic_ns"] == 888


@pytest.mark.parametrize("configured, expected", [(15, 15), (60, 30), (0, 30), (None, 30), (float("nan"), 30)])
def test_native_frame_rate_caps_requested_rate(configured, expected):
    assert demo._effective_fps(30, {"head_camera": {"fps": configured}}) == expected


def test_demo_writer_preserves_pixels_timestamps_and_phase_markers(tmp_path):
    output = tmp_path / "full-demo"
    frames = queue.Queue()
    events = queue.Queue()
    stopping = threading.Event()
    stopping.set()
    counters = _counters()
    counters["captured"].value = 1
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    source = {"transport": "legacy_rgb", "color_received_monotonic_ns": 10}
    record = {"sample_index": 0, "client_monotonic_ns": 20, "client_utc_ns": 30, "source": source}
    frames.put((record, {"ego_view": rgb}))
    events.put({"label": "initialization_started", "client_monotonic_ns": 15})
    writer_state = {}
    demo._demo_writer(output, ("ego_view",), {"checkpoint": "test"}, None, 30, frames, events, stopping, counters, queue.Queue(), writer_state)
    written = json.loads((output / "frames.jsonl").read_text())
    assert written["source"] == source
    assert written["client_monotonic_ns"] == 20
    assert written["client_utc_ns"] == 30
    decoded = cv2.cvtColor(cv2.imread(str(output / written["files"]["ego_view"])), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded, rgb)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["not_policy_observations"] is True
    assert manifest["metadata"]["checkpoint"] == "test"
    assert json.loads((output / "events.jsonl").read_text())["label"] == "initialization_started"
    assert json.loads((output / "summary.json").read_text())["clean_shutdown"] is True
    assert counters["written"].value == 1
    assert writer_state["ready"] is True


def test_demo_writer_failure_is_reported_without_raising(tmp_path, monkeypatch):
    frames = queue.Queue()
    frames.put(({"sample_index": 0}, {"ego_view": np.zeros((2, 2, 3), np.uint8)}))
    stopping = threading.Event()
    stopping.set()
    state = {}
    statuses = queue.Queue()

    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(demo, "_write_png", fail_write)
    demo._demo_writer(tmp_path / "recording", ("ego_view",), {}, None, 30, frames, queue.Queue(), stopping, _counters(), statuses, state)
    assert "disk full" in state["recording_error"]
    assert not json.loads((tmp_path / "recording" / "summary.json").read_text())["clean_shutdown"]


class _StatusQueue(queue.Queue):
    def cancel_join_thread(self):
        pass


class _ReadyPipe:
    def __init__(self, counters, writer_state=None):
        self.sent = []
        self.counters = counters
        self.writer_state = writer_state

    def send(self, payload):
        if payload.get("ok"):
            assert self.counters["captured"].value > 0
            if self.writer_state is not None:
                assert self.writer_state["ready"]
        self.sent.append(payload)

    def close(self):
        pass


def _run_fake_worker(monkeypatch, *, writer=None, stamps=None, preview=True):
    stopping = threading.Event()
    operator_exit = threading.Event()
    counters = _counters()
    ready = _ReadyPipe(counters)
    previewed = []
    statuses = _StatusQueue()
    total_frames = len(stamps) if stamps is not None else 8
    epoch = time.monotonic_ns()

    class FakeCamera:
        config = {"head_camera": {"fps": 10000}}
        _requires_depth = False
        closed = False

        def __init__(self, *args, **kwargs):
            self.index = 0
            self._client = SimpleNamespace(get_head_frame=self._frame)

        def _frame(self):
            stamp = stamps[self.index] if stamps is not None else self.index * 1000
            return SimpleNamespace(received_monotonic_ns=epoch + stamp)

        def read(self, timeout_s):
            assert timeout_s == 0.02
            self._client.get_head_frame()
            self.index += 1
            if self.index == total_frames:
                stopping.set()
            return SimpleNamespace(rgb=np.full((2, 2, 3), self.index, np.uint8), sequence=None)

        def close(self):
            FakeCamera.closed = True

    monkeypatch.setattr(camera_module, "TeleimagerCamera", FakeCamera)
    monkeypatch.setattr(demo.os, "nice", lambda value: None)
    monkeypatch.setattr(demo.os, "umask", lambda value: None)
    monkeypatch.setenv("DISPLAY", ":offline-test")
    monkeypatch.setattr(demo.cv2, "setNumThreads", lambda value: None)
    monkeypatch.setattr(demo.cv2, "pollKey", lambda: -1)
    monkeypatch.setattr(demo.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(demo.cv2, "destroyWindow", lambda *args: None)
    monkeypatch.setattr(demo.cv2, "imshow", lambda name, image: previewed.append(image.copy()))
    if writer is not None:
        monkeypatch.setattr(demo, "_demo_writer", writer)
    options = {
        "image_host": "offline", "video_keys": ("ego_view",),
        "output_dir": "/not-used-by-fake-writer" if writer else None,
        "metadata": {}, "depth_encoding": None, "surface_normal_encoding": None,
        "depth_colormap": None, "prefer_atomic_rgbd": False,
        "show_camera": preview, "target_fps": 10000,
    }
    demo._demo_worker(options, stopping, ready, operator_exit, counters, queue.Queue(), statuses)
    assert FakeCamera.closed
    return counters, ready.sent, previewed, statuses


def test_camera_cache_duplicates_are_not_recorded_or_redisplayed(monkeypatch):
    counters, ready, shown, _ = _run_fake_worker(monkeypatch, stamps=[0, 0, 1000, 1000, 2000, 2000])
    assert counters["captured"].value == 3
    assert counters["duplicate_polls"].value == 3
    assert len(shown) == 3
    assert ready[0]["ok"] is True


def test_regressed_rgb_timestamp_is_not_accepted(monkeypatch):
    counters, _, shown, _ = _run_fake_worker(monkeypatch, stamps=[0, 2000, 1000, 3000])
    assert counters["captured"].value == 3
    assert counters["read_errors"].value == 1
    assert len(shown) == 3


def test_slow_recording_cannot_backpressure_camera_or_preview(monkeypatch):
    def slow_writer(**kwargs):
        kwargs["writer_state"]["ready"] = True
        kwargs["stopping"].wait(1)

    counters, ready, shown, _ = _run_fake_worker(monkeypatch, writer=slow_writer)
    assert counters["captured"].value == 8
    assert len(shown) == 8
    assert counters["recording_dropped"].value == 8 - demo.DEMO_QUEUE_CAPACITY
    assert ready[0]["ok"]


def test_recording_startup_failure_never_reports_ready(monkeypatch):
    def failed_writer(**kwargs):
        kwargs["writer_state"]["recording_error"] = "disk full"

    _, ready, _, _ = _run_fake_worker(monkeypatch, writer=failed_writer)
    assert ready and ready[0]["ok"] is False
    assert "disk full" in ready[0]["error"]


def test_preview_q_is_an_explicit_operator_exit_request(monkeypatch):
    exit_requested = threading.Event()
    monkeypatch.setattr(demo.cv2, "pollKey", lambda: ord("q"))
    demo._pump_preview(set(), exit_requested)
    assert exit_requested.is_set()


def test_close_is_bounded_idempotent_and_never_reads_disk():
    stream = demo.ContinuousDemoStream.__new__(demo.ContinuousDemoStream)
    stream._closed = False
    stream.output_dir = None
    stream._status = {}
    stream._stopping = threading.Event()
    stream._operator_exit = threading.Event()
    stream._counters = _counters()
    stream._events = _StatusQueue()
    stream._statuses = _StatusQueue()
    joins = []

    class StuckProcess:
        alive = True
        exitcode = -9

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            joins.append(timeout)

        def terminate(self):
            pass

        def kill(self):
            self.alive = False

    stream._process = StuckProcess()
    # Queue close is not needed for this in-process stand-in.
    stream._events.close = lambda: None
    stream._statuses.close = lambda: None
    stream.close()
    stream.close()
    assert joins == [demo.DEMO_CLOSE_TIMEOUT_S, 0.5, 0.5]
    assert stream.stats["forced_shutdown"] is True
    assert not stream.stats["clean_shutdown"]
    assert stream._stopping.is_set()


def test_event_overflow_is_nonblocking_and_does_not_touch_control():
    stream = demo.ContinuousDemoStream.__new__(demo.ContinuousDemoStream)
    stream._closed = False
    stream.output_dir = Path("unused")
    stream._events = queue.Queue(maxsize=1)
    stream._counters = _counters()
    assert stream.mark_event("initialization_started")
    assert not stream.mark_event("task_started")
    assert stream._counters["events_dropped"].value == 1
