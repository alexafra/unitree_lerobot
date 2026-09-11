#!/usr/bin/env python3
"""Subscriber-only Dex3 DDS dropout forensics.

The diagnostic creates state subscribers only. It never constructs a command
publisher, command message, camera client, or policy client. Its normal mode is
event-only and lightweight. ``--trace-csv`` is an explicit higher-overhead mode
that records every accepted/rejected callback for offline timing analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HAND_DOF = 7
DEFAULT_HAND_GAP_S = 0.075
DEFAULT_LF_HAND_GAP_S = 0.350
DEFAULT_LOWSTATE_GAP_S = 0.075
DEFAULT_MISSING_STREAM_S = 2.0
TRACE_QUEUE_SIZE = 10_000
TRACE_CLOSE_TIMEOUT_S = 10.0

SAMPLE_INFO_FIELDS = (
    "sample_state",
    "view_state",
    "instance_state",
    "valid_data",
    "source_timestamp",
    "instance_handle",
    "publication_handle",
    "disposed_generation_count",
    "no_writers_generation_count",
    "sample_rank",
    "generation_rank",
    "absolute_generation_rank",
)


@dataclass(frozen=True)
class TopicSpec:
    key: str
    topic: str
    message_kind: str
    gap_threshold_s: float


@dataclass(frozen=True)
class MessageInspection:
    accepted: bool
    reject_reason: str
    health_fingerprint: tuple[Any, ...]
    health: dict[str, Any]


@dataclass(frozen=True)
class GapInterval:
    start_monotonic_ns: int
    end_monotonic_ns: int
    recovered: bool

    @property
    def duration_s(self) -> float:
        return (self.end_monotonic_ns - self.start_monotonic_ns) / 1e9


@dataclass(frozen=True)
class SampleRecord:
    stream: str
    topic: str
    message_kind: str
    callback_thread_native_id: int
    callback_index: int
    accepted_index: int
    accepted: bool
    reject_reason: str
    receive_monotonic_ns: int
    receive_wall_ns: int
    callback_gap_s: float | None
    valid_gap_s: float | None
    source_timestamp_ns: int | None
    source_gap_s: float | None
    source_to_receive_s: float | None
    source_to_receive_change_s: float | None
    publication_handle: int | None
    publication_handle_changed: bool
    sample_info: dict[str, Any]
    health: dict[str, Any]
    health_initialized: bool
    health_changed: bool

    @classmethod
    def header(cls) -> tuple[str, ...]:
        return (
            "stream",
            "topic",
            "message_kind",
            "callback_thread_native_id",
            "callback_index",
            "accepted_index",
            "accepted",
            "reject_reason",
            "receive_monotonic_ns",
            "receive_wall_ns",
            "receive_utc",
            "callback_gap_s",
            "valid_gap_s",
            "source_timestamp_ns",
            "source_gap_s",
            "source_to_receive_s",
            "source_to_receive_change_s",
            "publication_handle",
            "publication_handle_changed",
            "sample_info_json",
            "health_json",
        )

    def row(self) -> tuple[Any, ...]:
        return (
            self.stream,
            self.topic,
            self.message_kind,
            self.callback_thread_native_id,
            self.callback_index,
            self.accepted_index,
            self.accepted,
            self.reject_reason,
            self.receive_monotonic_ns,
            self.receive_wall_ns,
            _utc_from_ns(self.receive_wall_ns),
            self.callback_gap_s,
            self.valid_gap_s,
            self.source_timestamp_ns,
            self.source_gap_s,
            self.source_to_receive_s,
            self.source_to_receive_change_s,
            self.publication_handle,
            self.publication_handle_changed,
            _json_compact(self.sample_info),
            _json_compact(self.health),
        )


@dataclass(frozen=True)
class ProbeEvent:
    stream: str
    kind: str
    receive_monotonic_ns: int
    receive_wall_ns: int
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "receive_utc": _utc_from_ns(self.receive_wall_ns),
            "receive_wall_ns": self.receive_wall_ns,
            "receive_monotonic_ns": self.receive_monotonic_ns,
            "stream": self.stream,
            "kind": self.kind,
            "details": self.details,
        }

    def format(self) -> str:
        return _json_compact(self.as_dict())


@dataclass
class StreamStats:
    spec: TopicSpec
    callback_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    last_reject_reason: str = ""
    last_callback_monotonic_ns: int | None = None
    last_valid_monotonic_ns: int | None = None
    maximum_callback_gap_s: float = 0.0
    maximum_valid_gap_s: float = 0.0
    gap_count: int = 0
    gap_intervals: list[GapInterval] = field(default_factory=list)
    metadata_count: int = 0
    source_timestamp_ns: int | None = None
    maximum_source_gap_s: float = 0.0
    source_to_receive_s: float | None = None
    maximum_source_to_receive_change_s: float = 0.0
    publication_handle: int | None = None
    publication_handles: set[int] = field(default_factory=set)
    publication_handle_changes: int = 0
    callback_thread_ids: set[int] = field(default_factory=set)
    latest_health_fingerprint: tuple[Any, ...] | None = None
    latest_health: dict[str, Any] = field(default_factory=dict)
    health_change_count: int = 0
    nonzero_hand_error_samples: int = 0
    nonzero_motor_state_samples: int = 0

    def observe(
        self,
        message: Any,
        receive_monotonic_ns: int,
        receive_wall_ns: int,
        *,
        capture_details: bool,
    ) -> SampleRecord:
        self.callback_count += 1
        thread_id = threading.get_native_id()
        self.callback_thread_ids.add(thread_id)
        previous_callback_ns = self.last_callback_monotonic_ns
        previous_valid_ns = self.last_valid_monotonic_ns
        self.last_callback_monotonic_ns = receive_monotonic_ns
        inspection = _inspect_message(message, self.spec.message_kind, capture_details)
        raw_info = getattr(message, "sample_info", None)
        source_timestamp_ns = _positive_int(getattr(raw_info, "source_timestamp", None))
        publication_handle = _nonnegative_int(getattr(raw_info, "publication_handle", None))
        sample_info = (
            _sample_info(message)
            if capture_details
            else {
                "source_timestamp": source_timestamp_ns,
                "publication_handle": publication_handle,
            }
        )

        callback_gap_s = None if previous_callback_ns is None else (receive_monotonic_ns - previous_callback_ns) / 1e9
        valid_gap_s = None if previous_valid_ns is None else (receive_monotonic_ns - previous_valid_ns) / 1e9
        source_gap_s = (
            None
            if source_timestamp_ns is None or self.source_timestamp_ns is None
            else (source_timestamp_ns - self.source_timestamp_ns) / 1e9
        )
        source_to_receive_s = None if source_timestamp_ns is None else (receive_wall_ns - source_timestamp_ns) / 1e9
        source_to_receive_change_s = (
            None
            if source_to_receive_s is None or self.source_to_receive_s is None
            else source_to_receive_s - self.source_to_receive_s
        )
        publication_handle_changed = (
            publication_handle is not None
            and self.publication_handle is not None
            and publication_handle != self.publication_handle
        )
        health_initialized = inspection.accepted and self.latest_health_fingerprint is None
        health_changed = (
            inspection.accepted
            and self.latest_health_fingerprint is not None
            and inspection.health_fingerprint != self.latest_health_fingerprint
        )

        if callback_gap_s is not None:
            self.maximum_callback_gap_s = max(self.maximum_callback_gap_s, callback_gap_s)

        if not inspection.accepted:
            self.rejected_count += 1
            self.last_reject_reason = inspection.reject_reason
        else:
            self.accepted_count += 1
            if valid_gap_s is not None:
                self.maximum_valid_gap_s = max(self.maximum_valid_gap_s, valid_gap_s)
                if valid_gap_s > self.spec.gap_threshold_s:
                    self.gap_count += 1
                    self.gap_intervals.append(GapInterval(previous_valid_ns, receive_monotonic_ns, recovered=True))
            self.last_valid_monotonic_ns = receive_monotonic_ns
            if raw_info is not None:
                self.metadata_count += 1
            if source_gap_s is not None and source_gap_s >= 0.0:
                self.maximum_source_gap_s = max(self.maximum_source_gap_s, source_gap_s)
            if source_to_receive_change_s is not None:
                self.maximum_source_to_receive_change_s = max(
                    self.maximum_source_to_receive_change_s,
                    abs(source_to_receive_change_s),
                )
            self.source_timestamp_ns = source_timestamp_ns
            self.source_to_receive_s = source_to_receive_s
            if publication_handle is not None:
                self.publication_handles.add(publication_handle)
                if publication_handle_changed:
                    self.publication_handle_changes += 1
                self.publication_handle = publication_handle
            if health_changed:
                self.health_change_count += 1
            self.latest_health_fingerprint = inspection.health_fingerprint
            self.latest_health = inspection.health
            if any(value not in (None, 0) for value in inspection.health.get("raw_hand_error", ())):
                self.nonzero_hand_error_samples += 1
            if any(value not in (None, 0) for value in inspection.health.get("raw_motor_state", ())):
                self.nonzero_motor_state_samples += 1

        return SampleRecord(
            stream=self.spec.key,
            topic=self.spec.topic,
            message_kind=self.spec.message_kind,
            callback_thread_native_id=thread_id,
            callback_index=self.callback_count,
            accepted_index=self.accepted_count,
            accepted=inspection.accepted,
            reject_reason=inspection.reject_reason,
            receive_monotonic_ns=receive_monotonic_ns,
            receive_wall_ns=receive_wall_ns,
            callback_gap_s=callback_gap_s,
            valid_gap_s=valid_gap_s,
            source_timestamp_ns=source_timestamp_ns,
            source_gap_s=source_gap_s,
            source_to_receive_s=source_to_receive_s,
            source_to_receive_change_s=source_to_receive_change_s,
            publication_handle=publication_handle,
            publication_handle_changed=publication_handle_changed,
            sample_info=sample_info,
            health=inspection.health,
            health_initialized=health_initialized,
            health_changed=health_changed,
        )


class TraceWriter:
    """Write optional per-callback traces away from Cyclone receive threads."""

    def __init__(self, path: Path, queue_size: int = TRACE_QUEUE_SIZE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(SampleRecord.header())
        self._queue: queue.Queue[SampleRecord | None] = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._error: BaseException | None = None
        self._abort = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="dex3-trace-writer", daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def submit(self, record: SampleRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _run(self) -> None:
        last_flush = time.monotonic()
        try:
            while not self._abort.is_set():
                try:
                    record = self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if record is None:
                    break
                self._writer.writerow(record.row())
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    self._file.flush()
                    last_flush = now
        except BaseException as exc:
            self._error = exc
        finally:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + TRACE_CLOSE_TIMEOUT_S
        try:
            self._queue.put(None, timeout=min(2.0, TRACE_CLOSE_TIMEOUT_S))
        except queue.Full:
            self._dropped += self._queue.qsize()
            self._abort.set()
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            self._abort.set()
            self._thread.join(timeout=0.5)
        if self._thread.is_alive():
            raise RuntimeError("Dex3 trace writer did not stop within its bounded deadline")
        self._file.close()
        if self._error is not None:
            raise RuntimeError(f"Dex3 trace writer failed: {self._error}") from self._error


class JsonlEventWriter:
    """Main-thread JSONL sink; callback threads only enqueue ProbeEvent objects."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8")

    def write(self, event: ProbeEvent) -> None:
        self._file.write(event.format() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _attribute_sequence(items: tuple[Any, ...], attribute: str) -> tuple[Any, ...]:
    return tuple(getattr(item, attribute, None) for item in items)


def _nested_sequence(items: tuple[Any, ...], attribute: str) -> tuple[tuple[Any, ...], ...]:
    return tuple(_sequence(getattr(item, attribute, None)) for item in items)


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _sample_info(message: Any) -> dict[str, Any]:
    info = getattr(message, "sample_info", None)
    if info is None:
        return {}
    return {name: getattr(info, name, None) for name in SAMPLE_INFO_FIELDS}


def _status_fields(status: Any) -> dict[str, int]:
    fields = getattr(type(status), "_fields_", ())
    return {name: int(getattr(status, name)) for name, *_ in fields}


def _json_default(value: Any) -> Any:
    scalar = getattr(value, "value", None)
    if isinstance(scalar, (bool, int, float, str)):
        return scalar
    return repr(value)


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=_json_default)


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=_json_default)
        stream.write("\n")


def _write_summary_with_close_status(
    path: Path,
    summary: dict[str, Any],
    close_error: str | None,
) -> int:
    """Persist shutdown evidence before returning the process exit status."""

    summary["close_error"] = close_error
    _write_json_exclusive(path, summary)
    return 1 if close_error is not None else 0


def _utc_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds")


def _inspect_hand(message: Any, capture_details: bool) -> MessageInspection:
    motors = _sequence(getattr(message, "motor_state", None))
    if len(motors) < HAND_DOF:
        return MessageInspection(False, f"short_motor_state:{len(motors)}", (), {})
    try:
        positions = tuple(float(motors[index].q) for index in range(HAND_DOF))
    except (AttributeError, TypeError, ValueError) as exc:
        return MessageInspection(False, f"invalid_joint_position:{type(exc).__name__}", (), {})
    if not all(math.isfinite(value) for value in positions):
        return MessageInspection(False, "nonfinite_joint_position", (), {})

    errors = _sequence(getattr(message, "error", None))
    motor_modes = _attribute_sequence(motors, "mode")
    motor_states = _attribute_sequence(motors, "motorstate")
    health: dict[str, Any] = {
        "raw_hand_error": errors,
        "raw_motor_mode": motor_modes,
        "raw_motor_state": motor_states,
    }
    if capture_details:
        health.update(
            {
                "raw_hand_reserve": _sequence(getattr(message, "reserve", None)),
                "raw_power_v": getattr(message, "power_v", None),
                "raw_power_a": getattr(message, "power_a", None),
                "raw_system_v": getattr(message, "system_v", None),
                "raw_device_v": getattr(message, "device_v", None),
                "raw_motor_temperature": _nested_sequence(motors, "temperature"),
                "raw_motor_voltage": _attribute_sequence(motors, "vol"),
                "raw_motor_sensor": _nested_sequence(motors, "sensor"),
                "raw_motor_reserve": _nested_sequence(motors, "reserve"),
            }
        )
    return MessageInspection(
        accepted=True,
        reject_reason="",
        health_fingerprint=(errors, motor_modes, motor_states),
        health=health,
    )


def _inspect_lowstate(message: Any, capture_details: bool) -> MessageInspection:
    mode_pr = getattr(message, "mode_pr", None)
    mode_machine = getattr(message, "mode_machine", None)
    health: dict[str, Any] = {
        "raw_mode_pr": mode_pr,
        "raw_mode_machine": mode_machine,
    }
    if capture_details:
        health.update(
            {
                "raw_version": _sequence(getattr(message, "version", None)),
                "raw_tick": getattr(message, "tick", None),
                "raw_reserve": _sequence(getattr(message, "reserve", None)),
                "raw_crc": getattr(message, "crc", None),
            }
        )
    return MessageInspection(
        accepted=True,
        reject_reason="",
        health_fingerprint=(mode_pr, mode_machine),
        health=health,
    )


def _inspect_message(message: Any, message_kind: str, capture_details: bool) -> MessageInspection:
    if message_kind == "hand":
        return _inspect_hand(message, capture_details)
    if message_kind == "lowstate":
        return _inspect_lowstate(message, capture_details)
    raise ValueError(f"unsupported message kind: {message_kind}")


def build_topic_specs(
    *,
    hand_gap_s: float = DEFAULT_HAND_GAP_S,
    lf_hand_gap_s: float = DEFAULT_LF_HAND_GAP_S,
    lowstate_gap_s: float = DEFAULT_LOWSTATE_GAP_S,
    include_lf_hands: bool = True,
    lowstate_topic: str | None = "rt/lowstate",
) -> tuple[TopicSpec, ...]:
    specs = [
        TopicSpec("hf_left", "rt/dex3/left/state", "hand", hand_gap_s),
        TopicSpec("hf_right", "rt/dex3/right/state", "hand", hand_gap_s),
    ]
    if include_lf_hands:
        # These are the state topics used by Unitree's current SDK2 Dex3 example.
        specs.extend(
            (
                TopicSpec("lf_left", "rt/lf/dex3/left/state", "hand", lf_hand_gap_s),
                TopicSpec("lf_right", "rt/lf/dex3/right/state", "hand", lf_hand_gap_s),
            )
        )
    if lowstate_topic is not None:
        specs.append(TopicSpec("lowstate", lowstate_topic, "lowstate", lowstate_gap_s))
    return tuple(specs)


def _initialize_subscriber_dds(network_interface: str) -> None:
    """Create only the domain-0 DDS factory used by the state subscribers."""

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, networkInterface=network_interface)


class Dex3DropoutDiagnostic:
    def __init__(
        self,
        network_interface: str,
        specs: tuple[TopicSpec, ...],
        trace_csv: Path | None,
        *,
        dds_status: bool = False,
        trace_every: int = 1,
    ):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        _initialize_subscriber_dds(network_interface)
        self.specs = {spec.key: spec for spec in specs}
        self.stats = {spec.key: StreamStats(spec) for spec in specs}
        self._locks = {spec.key: threading.Lock() for spec in specs}
        self._stale_reported = {spec.key: False for spec in specs}
        self._missing_reported = {spec.key: False for spec in specs}
        self._events: queue.SimpleQueue[ProbeEvent] = queue.SimpleQueue()
        self._trace_writer = TraceWriter(trace_csv) if trace_csv is not None else None
        self._final_trace_dropped = 0
        self._closing = threading.Event()
        self._callback_gate = threading.Condition()
        self._active_callbacks = 0
        self._close_lock = threading.Lock()
        self._closed = False
        self.dds_status_enabled = dds_status
        self.trace_every = trace_every
        message_types = {"hand": HandState_, "lowstate": LowState_}
        self._subscribers = {
            spec.key: ChannelSubscriber(spec.topic, message_types[spec.message_kind]) for spec in specs
        }
        try:
            for key, subscriber in self._subscribers.items():
                subscriber.Init(handler=self._handler(key))
                if dds_status:
                    self._install_dds_status_listener(key, subscriber)
        except BaseException:
            self.close()
            raise

    def _enter_callback(self) -> bool:
        """Register an in-flight callback unless shutdown has started."""

        with self._callback_gate:
            if self._closing.is_set():
                return False
            self._active_callbacks += 1
            return True

    def _leave_callback(self) -> None:
        with self._callback_gate:
            self._active_callbacks -= 1
            if self._active_callbacks == 0:
                self._callback_gate.notify_all()

    def _handler(self, key: str):
        def receive(message: Any) -> None:
            if message is None or not self._enter_callback():
                return
            try:
                receive_monotonic_ns = time.monotonic_ns()
                receive_wall_ns = time.time_ns()
                with self._locks[key]:
                    previous_reject_reason = self.stats[key].last_reject_reason
                    trace_writer = self._trace_writer
                    trace_this_callback = trace_writer is not None and (
                        (self.stats[key].callback_count + 1) % self.trace_every == 0
                    )
                    record = self.stats[key].observe(
                        message,
                        receive_monotonic_ns,
                        receive_wall_ns,
                        capture_details=trace_this_callback,
                    )
                if trace_this_callback:
                    trace_writer.submit(record)
                if not record.accepted:
                    if record.reject_reason != previous_reject_reason:
                        self._emit(
                            key,
                            "sample_rejected_by_probe",
                            receive_monotonic_ns,
                            receive_wall_ns,
                            {"reason": record.reject_reason},
                        )
                    return
                recovered_gap = record.valid_gap_s is not None and record.valid_gap_s > self.specs[key].gap_threshold_s
                detailed_health = None
                if recovered_gap or record.health_initialized or record.health_changed:
                    detailed_inspection = _inspect_message(message, self.specs[key].message_kind, True)
                    if detailed_inspection.accepted:
                        detailed_health = detailed_inspection.health
                if recovered_gap:
                    gap_start_monotonic_ns = receive_monotonic_ns - int(record.valid_gap_s * 1e9)
                    gap_start_wall_ns = receive_wall_ns - (receive_monotonic_ns - gap_start_monotonic_ns)
                    self._emit(
                        key,
                        "gap_recovery",
                        receive_monotonic_ns,
                        receive_wall_ns,
                        {
                            "callback_gap_s": record.callback_gap_s,
                            "valid_gap_s": record.valid_gap_s,
                            "gap_start_monotonic_ns": gap_start_monotonic_ns,
                            "gap_start_wall_ns": gap_start_wall_ns,
                            "gap_start_utc": _utc_from_ns(gap_start_wall_ns),
                            "recovery_monotonic_ns": receive_monotonic_ns,
                            "recovery_wall_ns": receive_wall_ns,
                            "recovery_utc": _utc_from_ns(receive_wall_ns),
                            "source_gap_s": record.source_gap_s,
                            "source_to_receive_s": record.source_to_receive_s,
                            "source_to_receive_change_s": record.source_to_receive_change_s,
                            "publication_handle": record.publication_handle,
                            "publication_handle_changed": record.publication_handle_changed,
                            "callback_thread_native_id": record.callback_thread_native_id,
                            "health": detailed_health or record.health,
                            "correlation": self.correlation_snapshot(receive_monotonic_ns),
                        },
                    )
                if record.publication_handle_changed:
                    self._emit(
                        key,
                        "publication_handle_changed",
                        receive_monotonic_ns,
                        receive_wall_ns,
                        {"publication_handle": record.publication_handle},
                    )
                if record.health_initialized or record.health_changed:
                    self._emit(
                        key,
                        "health_initial" if record.health_initialized else "health_changed",
                        receive_monotonic_ns,
                        receive_wall_ns,
                        detailed_health or record.health,
                    )
            finally:
                self._leave_callback()

        return receive

    def _emit(
        self,
        stream: str,
        kind: str,
        receive_monotonic_ns: int,
        receive_wall_ns: int,
        details: dict[str, Any],
    ) -> None:
        self._events.put(
            ProbeEvent(
                stream=stream,
                kind=kind,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_ns=receive_wall_ns,
                details=details,
            )
        )

    def _status_handler(self, key: str, kind: str):
        def receive(_reader: Any, status: Any) -> None:
            if not self._enter_callback():
                return
            try:
                self._emit(key, kind, time.monotonic_ns(), time.time_ns(), _status_fields(status))
            finally:
                self._leave_callback()

        return receive

    def _install_dds_status_listener(self, key: str, subscriber: Any) -> None:
        """Extend Unitree's data listener with read-only Cyclone status callbacks."""

        try:
            channel = getattr(subscriber, "_ChannelSubscriber__channel")
            channel_reader = getattr(channel, "_Channel__reader")
            data_reader = getattr(channel_reader, "_Reader__reader")
            listener = data_reader.get_listener()
            listener.set_on_liveliness_changed(self._status_handler(key, "dds_liveliness_changed"))
            listener.set_on_sample_lost(self._status_handler(key, "dds_sample_lost"))
            listener.set_on_sample_rejected(self._status_handler(key, "dds_sample_rejected"))
            listener.set_on_requested_deadline_missed(self._status_handler(key, "dds_deadline_missed"))
            listener.set_on_requested_incompatible_qos(self._status_handler(key, "dds_incompatible_qos"))
            listener.set_on_subscription_matched(self._status_handler(key, "dds_subscription_matched"))
            data_reader.set_listener(listener)
            self._emit(
                key,
                "dds_reader_instrumented",
                time.monotonic_ns(),
                time.time_ns(),
                {"reader_guid": str(data_reader.guid), "reader_qos": repr(data_reader.get_qos())},
            )
        except Exception as exc:
            self._emit(
                key,
                "dds_status_instrumentation_unavailable",
                time.monotonic_ns(),
                time.time_ns(),
                {"exception": repr(exc)},
            )

    def correlation_snapshot(self, now_monotonic_ns: int) -> dict[str, dict[str, Any]]:
        result = {}
        for key, stats in self.stats.items():
            with self._locks[key]:
                callback_ns = stats.last_callback_monotonic_ns
                valid_ns = stats.last_valid_monotonic_ns
                result[key] = {
                    "callback_age_s": (None if callback_ns is None else (now_monotonic_ns - callback_ns) / 1e9),
                    "valid_age_s": (None if valid_ns is None else (now_monotonic_ns - valid_ns) / 1e9),
                    "callbacks": stats.callback_count,
                    "accepted": stats.accepted_count,
                    "publication_handle": stats.publication_handle,
                }
        return result

    def poll_staleness(
        self,
        started_monotonic_ns: int,
        now_monotonic_ns: int | None = None,
        now_wall_ns: int | None = None,
    ) -> None:
        if now_monotonic_ns is None:
            now_monotonic_ns = time.monotonic_ns()
        if now_wall_ns is None:
            now_wall_ns = time.time_ns()
        for key, stats in self.stats.items():
            with self._locks[key]:
                callback_ns = stats.last_callback_monotonic_ns
                valid_ns = stats.last_valid_monotonic_ns
                source_timestamp_ns = stats.source_timestamp_ns
                publication_handle = stats.publication_handle
            if valid_ns is None:
                missing_s = (now_monotonic_ns - started_monotonic_ns) / 1e9
                if missing_s >= DEFAULT_MISSING_STREAM_S and not self._missing_reported[key]:
                    self._missing_reported[key] = True
                    self._emit(
                        key,
                        "stream_missing",
                        now_monotonic_ns,
                        now_wall_ns,
                        {"seconds_without_first_sample": missing_s, "topic": stats.spec.topic},
                    )
                continue
            callback_age_s = None if callback_ns is None else (now_monotonic_ns - callback_ns) / 1e9
            valid_age_s = (now_monotonic_ns - valid_ns) / 1e9
            stale = valid_age_s > stats.spec.gap_threshold_s
            if stale and not self._stale_reported[key]:
                self._stale_reported[key] = True
                self._emit(
                    key,
                    "gap_open",
                    now_monotonic_ns,
                    now_wall_ns,
                    {
                        "callback_age_s": callback_age_s,
                        "valid_age_s": valid_age_s,
                        "last_valid_monotonic_ns": valid_ns,
                        "inferred_last_valid_wall_ns": now_wall_ns - (now_monotonic_ns - valid_ns),
                        "last_source_timestamp_ns": source_timestamp_ns,
                        "publication_handle": publication_handle,
                        "correlation": self.correlation_snapshot(now_monotonic_ns),
                    },
                )
            elif not stale:
                self._stale_reported[key] = False

    def heartbeat(self, now_monotonic_ns: int) -> str:
        return "DEX3_HEARTBEAT " + _json_compact(
            {"utc": _utc_from_ns(time.time_ns()), "streams": self.correlation_snapshot(now_monotonic_ns)}
        )

    def drain_events(self) -> list[ProbeEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def snapshot(self) -> dict[str, StreamStats]:
        result = {}
        for key, stats in self.stats.items():
            with self._locks[key]:
                result[key] = replace(
                    stats,
                    gap_intervals=list(stats.gap_intervals),
                    publication_handles=set(stats.publication_handles),
                    callback_thread_ids=set(stats.callback_thread_ids),
                    latest_health=dict(stats.latest_health),
                )
        return result

    @property
    def trace_dropped(self) -> int:
        if self._trace_writer is not None:
            return self._trace_writer.dropped
        return self._final_trace_dropped

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._callback_gate:
                self._closing.set()
            for subscriber in getattr(self, "_subscribers", {}).values():
                with contextlib.suppress(Exception):
                    subscriber.Close()
            with self._callback_gate:
                while self._active_callbacks:
                    self._callback_gate.wait()
            trace_writer = getattr(self, "_trace_writer", None)
            if trace_writer is not None:
                self._final_trace_dropped = trace_writer.dropped
                self._trace_writer = None
                trace_writer.close()
            self._closed = True


def _effective_gap_intervals(stats: StreamStats, ended_monotonic_ns: int) -> list[GapInterval]:
    intervals = list(stats.gap_intervals)
    last_valid_ns = stats.last_valid_monotonic_ns
    if last_valid_ns is not None and (ended_monotonic_ns - last_valid_ns) / 1e9 > stats.spec.gap_threshold_s:
        intervals.append(GapInterval(last_valid_ns, ended_monotonic_ns, recovered=False))
    return intervals


def _interval_summary(
    interval: GapInterval,
    anchor_monotonic_ns: int,
    anchor_wall_ns: int,
) -> dict[str, Any]:
    start_wall_ns = anchor_wall_ns + interval.start_monotonic_ns - anchor_monotonic_ns
    end_wall_ns = anchor_wall_ns + interval.end_monotonic_ns - anchor_monotonic_ns
    return {
        "start_monotonic_ns": interval.start_monotonic_ns,
        "end_monotonic_ns": interval.end_monotonic_ns,
        "start_wall_ns": start_wall_ns,
        "end_wall_ns": end_wall_ns,
        "start_utc": _utc_from_ns(start_wall_ns),
        "end_utc": _utc_from_ns(end_wall_ns),
        "duration_s": interval.duration_s,
        "recovered": interval.recovered,
    }


def _correlate_intervals(
    stream_a: str,
    intervals_a: list[GapInterval],
    stream_b: str,
    intervals_b: list[GapInterval],
) -> dict[str, Any]:
    overlap_s = 0.0
    overlap_pairs = 0
    overlapping_a: set[int] = set()
    overlapping_b: set[int] = set()
    for index_a, interval_a in enumerate(intervals_a):
        for index_b, interval_b in enumerate(intervals_b):
            overlap_ns = max(
                0,
                min(interval_a.end_monotonic_ns, interval_b.end_monotonic_ns)
                - max(interval_a.start_monotonic_ns, interval_b.start_monotonic_ns),
            )
            if overlap_ns:
                overlap_s += overlap_ns / 1e9
                overlap_pairs += 1
                overlapping_a.add(index_a)
                overlapping_b.add(index_b)
    duration_a_s = sum(interval.duration_s for interval in intervals_a)
    duration_b_s = sum(interval.duration_s for interval in intervals_b)
    return {
        "stream_a": stream_a,
        "stream_b": stream_b,
        "gap_count_a": len(intervals_a),
        "gap_count_b": len(intervals_b),
        "overlapping_gap_count_a": len(overlapping_a),
        "overlapping_gap_count_b": len(overlapping_b),
        "overlap_pair_count": overlap_pairs,
        "overlap_duration_s": overlap_s,
        "gap_duration_a_s": duration_a_s,
        "gap_duration_b_s": duration_b_s,
        "a_duration_overlap_fraction": (None if duration_a_s == 0.0 else overlap_s / duration_a_s),
        "b_duration_overlap_fraction": (None if duration_b_s == 0.0 else overlap_s / duration_b_s),
    }


def build_summary(
    snapshot: dict[str, StreamStats],
    *,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    anchor_wall_ns: int,
    dds_status_enabled: bool,
    trace_csv: Path | None,
    trace_every: int,
    trace_dropped: int,
    events_jsonl: Path,
) -> dict[str, Any]:
    elapsed_s = (ended_monotonic_ns - started_monotonic_ns) / 1e9
    intervals = {key: _effective_gap_intervals(stats, ended_monotonic_ns) for key, stats in snapshot.items()}
    streams: dict[str, Any] = {}
    for key, stats in snapshot.items():
        stream_intervals = intervals[key]
        streams[key] = {
            "topic": stats.spec.topic,
            "message_kind": stats.spec.message_kind,
            "gap_threshold_s": stats.spec.gap_threshold_s,
            "callback_count": stats.callback_count,
            "accepted_count": stats.accepted_count,
            "rejected_count": stats.rejected_count,
            "last_reject_reason": stats.last_reject_reason,
            "accepted_rate_hz": stats.accepted_count / elapsed_s if elapsed_s > 0.0 else 0.0,
            "maximum_callback_gap_s": stats.maximum_callback_gap_s,
            "maximum_valid_gap_s": stats.maximum_valid_gap_s,
            "maximum_source_gap_s": stats.maximum_source_gap_s,
            "maximum_abs_source_to_receive_change_s": (stats.maximum_source_to_receive_change_s),
            "metadata_count": stats.metadata_count,
            "publication_handles": sorted(stats.publication_handles),
            "publication_handle_changes": stats.publication_handle_changes,
            "callback_thread_native_ids": sorted(stats.callback_thread_ids),
            "health_change_count": stats.health_change_count,
            "nonzero_raw_hand_error_samples": stats.nonzero_hand_error_samples,
            "nonzero_raw_motor_state_samples": stats.nonzero_motor_state_samples,
            "latest_raw_health": stats.latest_health,
            "gap_count": len(stream_intervals),
            "gap_duration_s": sum(interval.duration_s for interval in stream_intervals),
            "gaps": [
                _interval_summary(interval, started_monotonic_ns, anchor_wall_ns) for interval in stream_intervals
            ],
        }

    def correlate(a: str, b: str) -> dict[str, Any] | None:
        if a not in intervals or b not in intervals:
            return None
        if snapshot[a].accepted_count == 0 or snapshot[b].accepted_count == 0:
            return {
                "stream_a": a,
                "stream_b": b,
                "available": False,
                "reason": "one or both streams had no accepted samples",
            }
        result = _correlate_intervals(a, intervals[a], b, intervals[b])
        result["available"] = True
        return result

    correlations = {
        "bilateral_hf": correlate("hf_left", "hf_right"),
        "same_side_hf_lf": {
            "left": correlate("hf_left", "lf_left"),
            "right": correlate("hf_right", "lf_right"),
        },
        "hand_lowstate": {
            key: correlate(key, "lowstate") for key in ("hf_left", "hf_right", "lf_left", "lf_right") if key in snapshot
        },
    }
    return {
        "schema_version": 1,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "started_wall_ns": anchor_wall_ns,
        "started_utc": _utc_from_ns(anchor_wall_ns),
        "elapsed_s": elapsed_s,
        "event_log": str(events_jsonl),
        "dds_status_instrumentation": {
            "enabled": dds_status_enabled,
            "note": "Private Cyclone listener extension; off by default to avoid perturbing delivery.",
        },
        "trace": {
            "enabled": trace_csv is not None,
            "path": None if trace_csv is None else str(trace_csv),
            "every_nth_callback": trace_every,
            "dropped_queue_rows": trace_dropped,
        },
        "gap_interval_definition": (
            "Full interval from the previous accepted receipt to the next accepted receipt "
            "when that interval exceeds the stream threshold; unrecovered final gaps are censored at run end."
        ),
        "clock_caveat": (
            "source_to_receive uses writer and local wall clocks; its absolute value is meaningful only "
            "when those clocks are synchronized. Monotonic receive gaps are authoritative locally."
        ),
        "source_gap_caveat": (
            "A large source timestamp gap alone cannot distinguish publisher silence from intermediate "
            "samples lost before this reader (including KeepLast behavior)."
        ),
        "publication_handle_caveat": (
            "The publication handle is a local opaque DDS identifier. A change is evidence of endpoint "
            "replacement; a stable value only rules out that obvious form of recreation."
        ),
        "dds_status_caveat": (
            "DDS reader status callbacks are best-effort diagnostics. No sample_lost, liveliness, or "
            "deadline event does not prove that delivery was healthy under BestEffort/no-Deadline QoS."
        ),
        "overlap_caveat": (
            "Gap overlaps use the complete previous-accepted-to-next-accepted interval, so each interval "
            "includes up to one nominal sample period before the actual silence began."
        ),
        "raw_health_caveat": (
            "Hand error/motorstate/mode fields are preserved as raw integers; this diagnostic does not "
            "assign undocumented bit meanings."
        ),
        "streams": streams,
        "correlations": correlations,
    }


def _print_summary(snapshot: dict[str, StreamStats], elapsed_s: float) -> None:
    print("\nSTREAM SUMMARY")
    print(
        " stream       accepted rejected   rate_hz gaps  max_valid_gap "
        "max_source_gap handles handle_changes health_changes"
    )
    for key, stats in snapshot.items():
        handles = ",".join(str(value) for value in sorted(stats.publication_handles)) or "-"
        rate_hz = stats.accepted_count / elapsed_s if elapsed_s > 0.0 else 0.0
        print(
            f" {key:<12} {stats.accepted_count:8d} {stats.rejected_count:8d} "
            f"{rate_hz:9.2f} {stats.gap_count:4d} {stats.maximum_valid_gap_s:13.6f} "
            f"{stats.maximum_source_gap_s:14.6f} {handles:<12} "
            f"{stats.publication_handle_changes:14d} {stats.health_change_count:14d}"
        )
        if stats.spec.message_kind == "hand":
            print(
                f"   raw health: nonzero_hand_error_samples={stats.nonzero_hand_error_samples} "
                f"nonzero_motor_state_samples={stats.nonzero_motor_state_samples} "
                f"latest={_json_compact(stats.latest_health)}"
            )
    print(
        "Note: source_gap alone cannot distinguish a publisher pause from packets/samples lost "
        "between writer and reader. If --dds-status is enabled, combine its best-effort DDS "
        "status events with packet capture."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True, help="Robot DDS interface, e.g. enp132s0")
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds; 0 runs until Ctrl-C")
    parser.add_argument("--status-hz", type=float, default=1.0, help="Heartbeat rate (default: 1)")
    parser.add_argument("--hand-gap-s", type=float, default=DEFAULT_HAND_GAP_S)
    parser.add_argument("--lf-hand-gap-s", type=float, default=DEFAULT_LF_HAND_GAP_S)
    parser.add_argument("--lowstate-gap-s", type=float, default=DEFAULT_LOWSTATE_GAP_S)
    parser.add_argument(
        "--lf-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also correlate Unitree's rt/lf/dex3 left/right state topics (default: enabled)",
    )
    parser.add_argument(
        "--lowstate-topic",
        choices=("rt/lowstate", "rt/lf/lowstate", "none"),
        default="rt/lowstate",
        help="Robot-wide state correlation topic (default: rt/lowstate)",
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        help="HIGH OVERHEAD/SHORT RUNS ONLY: sampled callback trace; path must not exist",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="With --trace-csv, record every Nth callback per stream (default: 1)",
    )
    parser.add_argument(
        "--dds-status",
        action="store_true",
        help="Opt in to version-brittle private Cyclone listener status instrumentation",
    )
    parser.add_argument("--events-jsonl", required=True, type=Path, help="New event-log path")
    parser.add_argument("--summary-json", required=True, type=Path, help="New summary path")
    return parser


def _validate_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    valid = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        operator = ">= 0" if allow_zero else "> 0"
        raise SystemExit(f"{name} must be finite and {operator}")


def main() -> int:
    args = build_parser().parse_args()
    _validate_positive("--duration", args.duration, allow_zero=True)
    _validate_positive("--status-hz", args.status_hz)
    _validate_positive("--hand-gap-s", args.hand_gap_s)
    _validate_positive("--lf-hand-gap-s", args.lf_hand_gap_s)
    _validate_positive("--lowstate-gap-s", args.lowstate_gap_s)
    if args.trace_every <= 0:
        raise SystemExit("--trace-every must be > 0")
    lowstate_topic = None if args.lowstate_topic == "none" else args.lowstate_topic
    specs = build_topic_specs(
        hand_gap_s=args.hand_gap_s,
        lf_hand_gap_s=args.lf_hand_gap_s,
        lowstate_gap_s=args.lowstate_gap_s,
        include_lf_hands=args.lf_hands,
        lowstate_topic=lowstate_topic,
    )
    trace_csv = None if args.trace_csv is None else args.trace_csv.expanduser().resolve()
    events_jsonl = args.events_jsonl.expanduser().resolve()
    summary_json = args.summary_json.expanduser().resolve()
    output_paths = [events_jsonl, summary_json, *(() if trace_csv is None else (trace_csv,))]
    if len(set(output_paths)) != len(output_paths):
        raise SystemExit("--events-jsonl, --summary-json, and --trace-csv must be distinct")
    for path in output_paths:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    print("SUBSCRIBER ONLY: no DDS command publishers or command messages are created.")
    print("Topics: " + ", ".join(f"{spec.key}={spec.topic}" for spec in specs))
    print(
        "Interpret valid/callback gaps with source timestamps, publication handles, "
        "and correlated LF/lowstate ages. Source gaps alone are not discriminating."
    )
    print(f"Events JSONL: {events_jsonl}")
    print(f"Summary JSON: {summary_json}")
    if args.dds_status:
        print("DDS status listener instrumentation: ENABLED (opt-in/private Cyclone API)")
    if trace_csv is not None:
        print(
            "WARNING: --trace-csv is for short dedicated runs and can produce hundreds of MB. "
            f"Recording every {args.trace_every} callback(s) per stream."
        )
        print(f"Per-callback trace: {trace_csv}")

    started_monotonic_ns = time.monotonic_ns()
    started_wall_ns = time.time_ns()
    events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    event_stream = events_jsonl.open("x", encoding="utf-8")
    try:
        diagnostic = Dex3DropoutDiagnostic(
            args.network_interface,
            specs,
            trace_csv,
            dds_status=args.dds_status,
            trace_every=args.trace_every,
        )
    except BaseException:
        event_stream.close()
        raise
    next_status_ns = started_monotonic_ns
    close_error: str | None = None

    def emit(event: ProbeEvent) -> None:
        line = event.format()
        event_stream.write(line + "\n")
        event_stream.flush()
        print(line, flush=True)

    try:
        while args.duration == 0.0 or (time.monotonic_ns() - started_monotonic_ns) / 1e9 < args.duration:
            now_monotonic_ns = time.monotonic_ns()
            diagnostic.poll_staleness(
                started_monotonic_ns,
                now_monotonic_ns=now_monotonic_ns,
            )
            for event in diagnostic.drain_events():
                emit(event)
            if now_monotonic_ns >= next_status_ns:
                emit(
                    ProbeEvent(
                        stream="all",
                        kind="heartbeat",
                        receive_monotonic_ns=now_monotonic_ns,
                        receive_wall_ns=time.time_ns(),
                        details={"streams": diagnostic.correlation_snapshot(now_monotonic_ns)},
                    )
                )
                next_status_ns = now_monotonic_ns + int(1e9 / args.status_hz)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        ended_monotonic_ns = time.monotonic_ns()
        elapsed_s = (ended_monotonic_ns - started_monotonic_ns) / 1e9
        try:
            diagnostic.close()
        except Exception as exc:
            close_error = repr(exc)
        for event in diagnostic.drain_events():
            emit(event)
        snapshot = diagnostic.snapshot()
        event_stream.close()

    _print_summary(snapshot, elapsed_s)
    summary = build_summary(
        snapshot,
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        anchor_wall_ns=started_wall_ns,
        dds_status_enabled=args.dds_status,
        trace_csv=trace_csv,
        trace_every=args.trace_every,
        trace_dropped=diagnostic.trace_dropped,
        events_jsonl=events_jsonl,
    )
    exit_status = _write_summary_with_close_status(summary_json, summary, close_error)
    print(f"Events JSONL saved: {events_jsonl}")
    print(f"Summary JSON saved: {summary_json}")
    if trace_csv is not None:
        print(f"Trace CSV: {trace_csv} (dropped_queue_rows={diagnostic.trace_dropped})")
    if close_error is not None:
        print(
            f"ERROR: diagnostic shutdown failed after the summary was saved: {close_error}",
            file=sys.stderr,
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
