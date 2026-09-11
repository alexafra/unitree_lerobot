from __future__ import annotations

import csv
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unitree_lerobot.eval_robot.diagnose_dex3_dropouts import (
    DEFAULT_HAND_GAP_S,
    DEFAULT_LF_HAND_GAP_S,
    DEFAULT_LOWSTATE_GAP_S,
    Dex3DropoutDiagnostic,
    GapInterval,
    StreamStats,
    TopicSpec,
    TraceWriter,
    _inspect_lowstate,
    _write_json_exclusive,
    _write_summary_with_close_status,
    build_summary,
    build_topic_specs,
)


def _info(source: int, handle: int = 101):
    values = dict(
        sample_state=2,
        view_state=4,
        instance_state=16,
        valid_data=True,
        source_timestamp=source,
        instance_handle=11,
        publication_handle=handle,
        disposed_generation_count=0,
        no_writers_generation_count=0,
        sample_rank=0,
        generation_rank=0,
        absolute_generation_rank=0,
    )
    return SimpleNamespace(**values)


def _hand(source: int, handle: int = 101, dof: int = 7):
    motors = [
        SimpleNamespace(
            mode=1,
            q=float(i) / 10,
            motorstate=0,
            temperature=(30, 31),
            vol=24.0,
            sensor=(0, 0),
            reserve=(0, 0, 0, 0),
        )
        for i in range(dof)
    ]
    return SimpleNamespace(
        motor_state=motors,
        error=(0, 0),
        reserve=(0, 0),
        power_v=24.0,
        power_a=0.5,
        system_v=12.0,
        device_v=5.0,
        sample_info=_info(source, handle),
    )


def test_default_topic_set_and_thresholds() -> None:
    specs = {item.key: item for item in build_topic_specs()}
    assert {item.topic for item in specs.values()} == {
        "rt/dex3/left/state",
        "rt/dex3/right/state",
        "rt/lf/dex3/left/state",
        "rt/lf/dex3/right/state",
        "rt/lowstate",
    }
    assert specs["hf_left"].gap_threshold_s == DEFAULT_HAND_GAP_S == 0.075
    assert specs["lf_left"].gap_threshold_s == DEFAULT_LF_HAND_GAP_S == 0.350
    assert specs["lowstate"].gap_threshold_s == DEFAULT_LOWSTATE_GAP_S == 0.075


def test_raw_callback_and_valid_receipt_gaps_are_separate() -> None:
    stats = StreamStats(TopicSpec("hf_left", "rt/dex3/left/state", "hand", 0.075))
    source = 1_700_000_000_000_000_000
    stats.observe(_hand(source), 1_000_000_000, source + 1_000_000, capture_details=False)
    rejected = stats.observe(
        _hand(source + 100_000_000, dof=6),
        1_100_000_000,
        source + 101_000_000,
        capture_details=False,
    )
    recovered = stats.observe(
        _hand(source + 1_000_000_000),
        2_000_000_000,
        source + 1_001_000_000,
        capture_details=False,
    )
    assert not rejected.accepted
    assert recovered.callback_gap_s == pytest.approx(0.9)
    assert recovered.valid_gap_s == pytest.approx(1.0)
    assert stats.maximum_callback_gap_s == pytest.approx(0.9)
    assert stats.maximum_valid_gap_s == pytest.approx(1.0)
    assert stats.gap_count == 1
    assert stats.gap_intervals == [GapInterval(1_000_000_000, 2_000_000_000, True)]


def test_rejected_callback_can_record_largest_callback_gap() -> None:
    stats = StreamStats(TopicSpec("hf_left", "rt/dex3/left/state", "hand", 0.075))
    source = 1_700_000_000_000_000_000
    stats.observe(_hand(source), 1_000_000_000, source + 1_000_000, capture_details=False)
    stats.observe(
        _hand(source + 1_000_000_000, dof=6),
        2_000_000_000,
        source + 1_001_000_000,
        capture_details=False,
    )
    stats.observe(
        _hand(source + 1_010_000_000),
        2_010_000_000,
        source + 1_011_000_000,
        capture_details=False,
    )
    assert stats.maximum_callback_gap_s == pytest.approx(1.0)
    assert stats.maximum_valid_gap_s == pytest.approx(1.01)


def test_lowstate_uses_only_fields_in_local_hg_schema() -> None:
    message = SimpleNamespace(
        mode_pr=2,
        mode_machine=5,
        version=(1, 2),
        tick=123,
        reserve=(0, 0, 0, 0),
        crc=456,
    )
    health = _inspect_lowstate(message, capture_details=True).health
    assert health == {
        "raw_mode_pr": 2,
        "raw_mode_machine": 5,
        "raw_version": (1, 2),
        "raw_tick": 123,
        "raw_reserve": (0, 0, 0, 0),
        "raw_crc": 456,
    }


def test_gap_open_poll_does_not_reacquire_stream_lock() -> None:
    spec = TopicSpec("hf_left", "rt/dex3/left/state", "hand", 0.075)
    stats = StreamStats(spec, callback_count=1, accepted_count=1)
    stats.last_callback_monotonic_ns = 1_000_000_000
    stats.last_valid_monotonic_ns = 1_000_000_000
    diagnostic = object.__new__(Dex3DropoutDiagnostic)
    diagnostic.specs = {spec.key: spec}
    diagnostic.stats = {spec.key: stats}
    diagnostic._locks = {spec.key: threading.Lock()}
    diagnostic._stale_reported = {spec.key: False}
    diagnostic._missing_reported = {spec.key: False}
    diagnostic._events = queue.SimpleQueue()
    diagnostic.poll_staleness(
        started_monotonic_ns=0,
        now_monotonic_ns=2_000_000_000,
        now_wall_ns=1_700_000_000_000_000_000,
    )
    events = diagnostic.drain_events()
    assert [event.kind for event in events] == ["gap_open"]
    assert events[0].details["valid_age_s"] == 1.0


def test_trace_writer_and_exclusive_outputs(tmp_path: Path) -> None:
    stats = StreamStats(TopicSpec("hf_left", "topic", "hand", 0.075))
    source = 1_700_000_000_000_000_000
    record = stats.observe(
        _hand(source),
        1_000_000_000,
        source + 1_000_000,
        capture_details=True,
    )
    trace_path = tmp_path / "trace.csv"
    writer = TraceWriter(trace_path, queue_size=2)
    writer.submit(record)
    writer.close()
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["stream"] == "hf_left"
    assert rows[0]["publication_handle"] == "101"
    summary_path = tmp_path / "summary.json"
    _write_json_exclusive(summary_path, {"ok": True})
    with pytest.raises(FileExistsError):
        _write_json_exclusive(summary_path, {"ok": False})


def test_close_failure_is_saved_and_returns_nonzero(tmp_path: Path) -> None:
    summary_path = tmp_path / "failed_close_summary.json"
    status = _write_summary_with_close_status(
        summary_path,
        {"schema_version": 1},
        "RuntimeError('trace writer failed')",
    )
    assert status != 0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["close_error"] == (
        "RuntimeError('trace writer failed')"
    )


def test_close_waits_for_inflight_callback_before_closing_trace() -> None:
    class Subscriber:
        def __init__(self) -> None:
            self.closed = threading.Event()

        def Close(self) -> None:
            self.closed.set()

    class Trace:
        def __init__(self) -> None:
            self.dropped = 0
            self.records = []
            self.closed = False

        def submit(self, record) -> None:
            assert not self.closed
            self.records.append(record)

        def close(self) -> None:
            self.closed = True

    spec = TopicSpec("hf_left", "rt/dex3/left/state", "hand", 0.075)
    diagnostic = object.__new__(Dex3DropoutDiagnostic)
    diagnostic.specs = {spec.key: spec}
    diagnostic.stats = {spec.key: StreamStats(spec)}
    diagnostic._locks = {spec.key: threading.Lock()}
    diagnostic._stale_reported = {spec.key: False}
    diagnostic._missing_reported = {spec.key: False}
    diagnostic._events = queue.SimpleQueue()
    diagnostic._closing = threading.Event()
    diagnostic._callback_gate = threading.Condition()
    diagnostic._active_callbacks = 0
    diagnostic._close_lock = threading.Lock()
    diagnostic._closed = False
    diagnostic._final_trace_dropped = 0
    diagnostic.trace_every = 1
    trace = Trace()
    diagnostic._trace_writer = trace
    subscriber = Subscriber()
    diagnostic._subscribers = {spec.key: subscriber}

    # Hold the per-stream lock so the callback is registered as active but cannot
    # reach the trace writer until shutdown has begun.
    diagnostic._locks[spec.key].acquire()
    callback = diagnostic._handler(spec.key)
    callback_thread = threading.Thread(target=callback, args=(_hand(1_700_000_000_000_000_000),))
    callback_thread.start()
    with diagnostic._callback_gate:
        assert diagnostic._callback_gate.wait_for(
            lambda: diagnostic._active_callbacks == 1,
            timeout=1.0,
        )

    close_thread = threading.Thread(target=diagnostic.close)
    close_thread.start()
    assert subscriber.closed.wait(timeout=1.0)
    assert close_thread.is_alive()
    assert not trace.closed

    diagnostic._locks[spec.key].release()
    callback_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)
    assert not callback_thread.is_alive()
    assert not close_thread.is_alive()
    assert trace.closed
    assert len(trace.records) == 1

    callback(_hand(1_700_000_000_100_000_000))
    assert diagnostic.stats[spec.key].callback_count == 1


def test_summary_reports_same_side_and_lowstate_overlap(tmp_path: Path) -> None:
    specs = build_topic_specs()
    snapshot = {spec.key: StreamStats(spec, accepted_count=10) for spec in specs}
    snapshot["hf_left"].gap_intervals = [GapInterval(1_000_000_000, 2_000_000_000, True)]
    snapshot["lf_left"].gap_intervals = [GapInterval(1_500_000_000, 2_500_000_000, True)]
    snapshot["lowstate"].gap_intervals = [GapInterval(1_750_000_000, 2_250_000_000, True)]
    summary = build_summary(
        snapshot,
        started_monotonic_ns=0,
        ended_monotonic_ns=3_000_000_000,
        anchor_wall_ns=1_700_000_000_000_000_000,
        dds_status_enabled=False,
        trace_csv=None,
        trace_every=1,
        trace_dropped=0,
        events_jsonl=tmp_path / "events.jsonl",
    )
    same_side = summary["correlations"]["same_side_hf_lf"]["left"]
    lowstate = summary["correlations"]["hand_lowstate"]["hf_left"]
    assert same_side["overlap_duration_s"] == pytest.approx(0.5)
    assert lowstate["overlap_duration_s"] == pytest.approx(0.25)
    assert not summary["dds_status_instrumentation"]["enabled"]
    assert "synchronized" in summary["clock_caveat"]


def test_source_is_subscriber_only() -> None:
    source_path = Path(__file__).parents[1] / "unitree_lerobot/eval_robot/diagnose_dex3_dropouts.py"
    source = source_path.read_text(encoding="utf-8")
    assert "ChannelPublisher" not in source
    assert ".Write(" not in source
    assert "HandCmd_" not in source
    assert "LowCmd_" not in source
    assert "safe_g1_dex3" not in source
    assert "ChannelFactoryInitialize(0, networkInterface=network_interface)" in source
