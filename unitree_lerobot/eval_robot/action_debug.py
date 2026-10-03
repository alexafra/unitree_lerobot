"""Opt-in, fail-isolated action diagnostics; no JSON or disk I/O in control ticks.

The parent owns a low-priority spawned writer. Each producer has independent
lock-free counters and uses a bounded, drop-new multiprocessing queue. Array
snapshots are copied before enqueue; JSON conversion happens only in the writer.
Successful DDS writes are transport acceptance, not device acknowledgement.
"""

from __future__ import annotations

import json
from collections import deque
from itertools import count
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any


LOGGER = logging.getLogger(__name__)
ACTION_DEBUG_QUEUE_CAPACITY = 256
ACTION_DEBUG_SAMPLE_HZ = 30.0
_POLICY_THREAD_SOURCES = tuple(f"policy-thread-{index}" for index in range(1, 17))
_SOURCES = ("policy", "actuator", *_POLICY_THREAD_SOURCES)


def _snapshot(value: Any) -> Any:
    """Copy numeric arrays without serializing them in the producer."""
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    if hasattr(value, "shape") and hasattr(value, "copy"):
        return value.copy()
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported action-debug value: {type(value).__name__}")


class ActionDebugSink:
    """Pickleable single-producer endpoint; record() never waits on queue space."""

    def __init__(self, source: str, work_queue: Any, counters: Any, accepting: Any, writer_alive: Any):
        self.source = source
        self._queue = work_queue
        self._counters = counters
        self._accepting = accepting
        self._writer_alive = writer_alive

    def record(self, kind: str, **fields: Any) -> bool:
        try:
            self._counters[0] += 1
            if not self._accepting.value or not self._writer_alive.value:
                self._counters[1] += 1
                return False
            item = {
                "schema_version": 1,
                "source": self.source,
                "source_index": int(self._counters[0]),
                "kind": kind,
                "monotonic_ns": time.monotonic_ns(),
                "utc_ns": time.time_ns(),
                "fields": _snapshot(fields),
            }
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._counters[1] += 1
        except BaseException:
            # Debugging must not turn allocation/IPC/serialization failure into
            # a control exception. No terminal logging in this producer path.
            self._counters[1] += 1
            self._counters[2] += 1
        return False

    def detach_at_process_exit(self) -> None:
        """A dead writer must never hold up the independent actuator's exit."""
        try:
            self._queue.cancel_join_thread()
        except BaseException:
            pass

    def note_sampling(self, missed_slots: int, sample_errors: int) -> None:
        try:
            self._counters[3] = missed_slots
            self._counters[4] = sample_errors
        except BaseException:
            pass


def _counter_summary(counters: dict[str, Any]) -> dict[str, Any]:
    return {
        source: {"submitted": int(values[0]), "queue_dropped": int(values[1]), "producer_errors": int(values[2]),
                 "missed_sample_slots": int(values[3]), "sample_errors": int(values[4])}
        for source, values in counters.items()
        if source in {"policy", "actuator"} or values[0]
    }


def _action_debug_writer(
    output_dir: str, metadata: dict[str, Any], capacity: int, work_queue: Any,
    counters: dict[str, Any], accepting: Any, stopping: Any, writer_alive: Any,
    written: Any, write_dropped: Any, writer_failed: Any,
) -> None:
    """The only owner of action-debug JSON encoding and filesystem writes."""
    directory = Path(output_dir)
    owns_directory = False
    handle = None
    error = None
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        os.umask(0o077)
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        owns_directory = True
        manifest = {
            "schema_version": 1, "kind": "groot_action_debug",
            "sample_hz": ACTION_DEBUG_SAMPLE_HZ,
            "queue_capacity": capacity, "overflow": "drop-new",
            "clock": "monotonic_ns is shared host clock; utc_ns is wall clock",
            "semantics": "Sampled joint feedback and DDS-accepted targets; not device acknowledgements. "
                         "Joint vectors follow profile metadata. Hands are not necessarily radians. "
                         "Scheduled actions are separately recorded; dropped events create source_index gaps.",
            "metadata": metadata,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, default=_json_default, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        handle = (directory / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        stopped_empty_at = None
        while True:
            try:
                item = work_queue.get(timeout=0.05)
            except queue.Empty:
                if stopping.is_set():
                    if stopped_empty_at is None:
                        stopped_empty_at = time.monotonic()
                    # Producer feeder tails can arrive after stop. Bound the
                    # drain even if a producer was killed before flushing.
                    accepted = sum(int(v[0] - v[1]) for v in counters.values())
                    if written.value + write_dropped.value >= accepted or time.monotonic() - stopped_empty_at >= 0.5:
                        break
                continue
            stopped_empty_at = None
            try:
                encoded = json.dumps(item, default=_json_default, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError, OverflowError):
                write_dropped.value += 1
                continue
            handle.write(encoded + "\n")
            written.value += 1
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        writer_failed.value = 1
    finally:
        writer_alive.value = 0
        try:
            if handle is not None:
                handle.close()
            if owns_directory:
                sources = _counter_summary(counters)
                accepted = sum(v["submitted"] - v["queue_dropped"] for v in sources.values())
                summary = {
                    "sources": sources, "written": int(written.value),
                    "serialization_dropped": int(write_dropped.value),
                    "unwritten_accepted": max(0, accepted - int(written.value) - int(write_dropped.value)),
                    "writer_error": error, "clean_shutdown": bool(stopping.is_set() and error is None),
                }
                (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except BaseException:
            writer_failed.value = 1


class ActionDebugRecorder:
    """Parent-owned writer. Close only after the actuator has released authority.

    Constructor/start failures disable diagnostics, never robot operation. With
    enabled=False there is no process, queue, filesystem access or allocation of
    shared memory. make_sink() returns None for that disabled case.
    """

    def __init__(self, output_dir: str | Path, *, metadata: dict[str, Any] | None = None,
                 enabled: bool = True, queue_capacity: int = ACTION_DEBUG_QUEUE_CAPACITY):
        self.output_dir = Path(output_dir)
        self._process = None
        self._queue = None
        self._closed = False
        self._error = None
        self._counters: dict[str, Any] = {}
        self._sinks: dict[str, ActionDebugSink] = {}
        self._thread_sinks = deque()
        self._thread_local = threading.local()
        self._untracked_thread_drops = count()
        self._untracked_thread_dropped = 0
        if not enabled:
            return
        try:
            if queue_capacity < 1:
                raise ValueError("queue_capacity must be positive")
            context = mp.get_context("spawn")
            self._queue = context.Queue(maxsize=queue_capacity)
            self._accepting = context.RawValue("b", 1)
            self._writer_alive = context.RawValue("b", 1)
            self._written = context.RawValue("Q", 0)
            self._write_dropped = context.RawValue("Q", 0)
            self._writer_failed = context.RawValue("b", 0)
            self._stopping = context.Event()
            self._counters = {source: context.RawArray("Q", 5) for source in _SOURCES}
            self._sinks = {
                source: ActionDebugSink(source, self._queue, values, self._accepting, self._writer_alive)
                for source, values in self._counters.items()
            }
            self._thread_local.sink = self._sinks["policy"]
            self._thread_sinks.extend(self._sinks[source] for source in _POLICY_THREAD_SOURCES)
            self._process = context.Process(
                target=_action_debug_writer,
                args=(str(self.output_dir), metadata or {}, queue_capacity, self._queue, self._counters,
                      self._accepting, self._stopping, self._writer_alive,
                      self._written, self._write_dropped, self._writer_failed),
                name="groot-action-debug-writer", daemon=True,
            )
            self._process.start()
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            if hasattr(self, "_accepting"):
                self._accepting.value = 0
            self._sinks = {}
            self._thread_sinks.clear()
            self._thread_local.sink = None

    def make_sink(self, source: str = "actuator") -> ActionDebugSink | None:
        return self._sinks.get(source)

    def record(self, kind: str, **fields: Any) -> bool:
        if not self._sinks:
            return False
        sink = getattr(self._thread_local, "sink", None)
        if sink is None:
            try:
                # deque.popleft is atomic in CPython; no registration mutex
                # can delay the parent's heartbeat behind an inference thread.
                sink = self._thread_sinks.popleft()
            except IndexError:
                next(self._untracked_thread_drops)
                return False
            self._thread_local.sink = sink
        return False if sink is None else sink.record(kind, **fields)

    def release_thread_sink(self) -> None:
        """Return an inference worker's slot after its final record, in finally.

        Reuse keeps cumulative source indices and supports arbitrarily many
        sequential goal workers without growing shared-memory storage.
        """
        sink = getattr(self._thread_local, "sink", None)
        if sink is not None and sink.source in _POLICY_THREAD_SOURCES:
            del self._thread_local.sink
            self._thread_sinks.append(sink)

    def close(self, timeout_s: float = 2.0) -> dict[str, Any]:
        """Bounded post-control drain; returns terminal counters for CLI logging."""
        if not self._closed:
            self._closed = True
            self._untracked_thread_dropped = next(self._untracked_thread_drops)
            try:
                if hasattr(self, "_accepting"):
                    self._accepting.value = 0
                    self._stopping.set()
                if self._process is not None and self._process.pid is not None:
                    self._process.join(timeout=max(0.0, timeout_s))
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(timeout=0.2)
                        self._error = "Writer exceeded bounded post-control drain; tail may be missing"
                    if self._process.exitcode not in (None, 0):
                        self._writer_failed.value = 1
                if self._queue is not None:
                    self._queue.cancel_join_thread()
                    self._queue.close()
            except BaseException as exc:
                self._error = f"{type(exc).__name__}: {exc}"
        sources = _counter_summary(self._counters)
        written = int(self._written.value) if hasattr(self, "_written") else 0
        dropped = int(self._write_dropped.value) if hasattr(self, "_write_dropped") else 0
        accepted = sum(v["submitted"] - v["queue_dropped"] for v in sources.values())
        return {
            "output_dir": str(self.output_dir), "enabled": bool(self._counters), "sources": sources,
            "written": written, "serialization_dropped": dropped,
            "unwritten_accepted": max(0, accepted - written - dropped),
            "untracked_thread_dropped": self._untracked_thread_dropped,
            "writer_failed": bool(self._writer_failed.value) if hasattr(self, "_writer_failed") else False,
            "error": self._error,
        }


class ActionDebugSampler:
    """Fail-isolated 30 Hz backend snapshots, plus every scheduled policy action."""

    def __init__(self, sink: ActionDebugSink, profile: Any, arm_joint_names: Any,
                 *, sample_hz: float = ACTION_DEBUG_SAMPLE_HZ):
        self.sink = sink
        self.period_s = 1.0 / sample_hz
        self.next_sample_at: float | None = None
        self.missed_samples = 0
        self.sample_errors = 0
        self.state = None
        self.context: dict[str, Any] = {"phase": "authority_acquisition"}
        self.raw_action = None
        self.desired = None
        self.sink.record("actuator_metadata", profile=profile.name,
                         arm_joint_names=tuple(arm_joint_names),
                         left_hand_joint_names=profile.left_joint_names,
                         right_hand_joint_names=profile.right_joint_names,
                         units={"arm_position": "rad", "arm_velocity": "rad/s", "arm_torque": "N*m",
                                "hand_position": profile.value_unit}, sample_hz=sample_hz)

    def scheduled(self, arm: Any, left: Any, right: Any, **timing: Any) -> None:
        try:
            self.raw_action = {"arm": arm, "left_hand": left, "right_hand": right}
            self.sink.record("scheduled_action", raw_target=self.raw_action, **timing)
        except BaseException:
            self.sample_errors += 1

    def sample(self, backend: Any, *, now: float | None = None, publish_ok: bool = True) -> None:
        try:
            now = time.monotonic() if now is None else now
            if self.next_sample_at is not None and now < self.next_sample_at:
                return
            if self.next_sample_at is None:
                self.next_sample_at = now
            skipped = max(0, int((now - self.next_sample_at) / self.period_s))
            self.missed_samples += skipped
            self.sink.note_sampling(self.missed_samples, self.sample_errors)
            self.next_sample_at += (skipped + 1) * self.period_s
            state = self.state
            measured = None if state is None else {
                "arm": state.arm, "arm_dq": state.arm_dq,
                "left_hand": state.left_hand, "right_hand": state.right_hand,
                "waist": getattr(state, "waist", None), "waist_dq": getattr(state, "waist_dq", None),
                "captured_at": state.captured_at,
                "arm_received_at": getattr(state, "arm_received_at", None),
                "left_hand_received_at": getattr(state, "left_hand_received_at", None),
                "right_hand_received_at": getattr(state, "right_hand_received_at", None),
            }
            hands = {}
            for side in ("left", "right"):
                history = getattr(backend, f"_{side}_hand_publish_history", ())
                latest = history[-1] if history else None
                hands[f"{side}_hand"] = None if latest is None else latest.target
                hands[f"{side}_hand_completed_at"] = None if latest is None else latest.completed_at
            self.sink.record(
                "low_level_sample", sample_at=now, publish_ok=publish_ok,
                context=self.context, measured=measured, raw_scheduled_target=self.raw_action,
                desired_target=self.desired,
                conditioned_target={"arm": backend._arm_target, "left_hand": backend._left_target,
                                    "right_hand": backend._right_target},
                published={"arm": getattr(backend, "_last_published_arm_q", None),
                           "arm_tau": getattr(backend, "_last_published_arm_tau", None),
                           "arm_completed_at": getattr(backend, "_action_debug_arm_completed_at", None), **hands},
                arm_authority_weight=getattr(backend, "_weight", None),
                publish_timing_ms=getattr(backend, "_last_publish_timing_ms", {}),
                missed_sample_slots=self.missed_samples, sample_errors=self.sample_errors,
            )
        except BaseException:
            self.sample_errors += 1

    def finish(self) -> None:
        try:
            self.sink.note_sampling(self.missed_samples, self.sample_errors)
            self.sink.record("actuator_debug_summary", missed_sample_slots=self.missed_samples,
                             sample_errors=self.sample_errors)
        except BaseException:
            pass
