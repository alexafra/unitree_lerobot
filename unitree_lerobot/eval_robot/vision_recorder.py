"""Fail-isolated recording of the vision arrays sampled by the policy client.

The robot process only performs a non-blocking enqueue.  Lossless PNG encoding
and filesystem I/O run in a low-priority spawned process; when the single-slot
queue is occupied, the new recording sample is dropped instead of delaying
observation capture or command handling.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
VISION_RECORDING_SCHEMA_VERSION = 1
VISION_RECORDING_QUEUE_CAPACITY = 1
VISION_RECORDING_CLOSE_TIMEOUT_S = 2.0
VISION_RECORDING_START_TIMEOUT_S = 5.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".png",
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if not ok:
        raise RuntimeError(f"OpenCV could not encode {path.name}")
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded.tobytes())
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _repair_interrupted_recording(output_dir: Path, video_keys: tuple[str, ...]) -> int:
    """Remove unfinished/orphaned images and retain complete frame records only.

    This runs only after the recorder process has exited or been terminated, and
    therefore cannot contend with robot control.  It returns the number of
    complete samples retained in ``frames.jsonl``.
    """

    for partial in output_dir.rglob("*.partial"):
        if partial.is_file():
            partial.unlink(missing_ok=True)

    frames_path = output_dir / "frames.jsonl"
    valid_records: list[dict[str, Any]] = []
    referenced: dict[str, set[Path]] = {key: set() for key in video_keys}
    if frames_path.is_file():
        for line in frames_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
                files = record["files"]
                if not isinstance(files, dict) or set(files) != set(video_keys):
                    continue
                resolved_files: dict[str, Path] = {}
                for key in video_keys:
                    relative = Path(files[key])
                    expected_parent = Path(key)
                    if (
                        relative.is_absolute()
                        or relative.parent != expected_parent
                        or relative.name.startswith(".")
                        or relative.suffix != ".png"
                    ):
                        raise ValueError("unsafe frame path")
                    destination = output_dir / relative
                    if not destination.is_file() or destination.is_symlink():
                        raise ValueError("missing frame")
                    resolved_files[key] = destination
                valid_records.append(record)
                for key, destination in resolved_files.items():
                    referenced[key].add(destination)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    for key in video_keys:
        key_dir = output_dir / key
        if not key_dir.is_dir() or key_dir.is_symlink():
            continue
        for frame_path in key_dir.glob("frame-*.png"):
            if frame_path not in referenced[key] and frame_path.is_file():
                frame_path.unlink(missing_ok=True)

    if frames_path.is_file():
        encoded = "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in valid_records
        )
        temporary = frames_path.with_name(f".{frames_path.name}.partial")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(frames_path)
    return len(valid_records)


def _recording_worker(
    output_dir_text: str,
    video_keys: tuple[str, ...],
    metadata: dict[str, Any],
    work_queue: Any,
    stopping: Any,
    ready: Any,
    submitted: Any,
    queue_dropped: Any,
    written: Any,
    write_dropped: Any,
) -> None:
    """Own every codec and filesystem operation for one recording."""

    output_dir = Path(output_dir_text)
    worker_error: str | None = None
    clean_shutdown = False
    owns_output_dir = False
    manifest_handle = None
    writes_enabled = True
    try:
        # Recording is explicitly subordinate to inference and the independent
        # actuator watchdog process.  Failure to lower priority is harmless and
        # must not prevent the publisher-free startup check.
        try:
            os.nice(10)
        except OSError:
            pass
        os.umask(0o077)
        cv2.setNumThreads(1)

        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        owns_output_dir = True
        for key in video_keys:
            (output_dir / key).mkdir(mode=0o700)
        _write_json(
            output_dir / "manifest.json",
            {
                "schema_version": VISION_RECORDING_SCHEMA_VERSION,
                "kind": "groot_policy_observation_vision",
                "video_keys": list(video_keys),
                "sampling": (
                    "one sample per successful client observation construction; includes "
                    "publisher-free preflight, Warmup2, and any safety-discarded recapture"
                ),
                "pixel_contract": (
                    "lossless uint8 RGB arrays before server-side crop, resize, and normalization; "
                    "batch/time axes removed from files"
                ),
                "queue": {
                    "capacity": VISION_RECORDING_QUEUE_CAPACITY,
                    "overflow": "drop-new",
                },
                "metadata": metadata,
            },
        )
        manifest_handle = (output_dir / "frames.jsonl").open(
            "x",
            encoding="utf-8",
            buffering=1,
        )
        (output_dir / "frames.jsonl").chmod(0o600)
        ready.send({"ok": True, "pid": os.getpid()})
        ready.close()

        while True:
            try:
                item = work_queue.get(timeout=0.1)
            except queue.Empty:
                accounted = int(written.value + write_dropped.value)
                accepted = int(submitted.value - queue_dropped.value)
                if stopping.is_set() and accounted >= accepted:
                    break
                continue
            if item is None:
                break

            sample_index, monotonic_ns, utc_ns, views = item
            if not writes_enabled:
                write_dropped.value += 1
                continue
            created: list[Path] = []
            try:
                files: dict[str, str] = {}
                for key in video_keys:
                    relative = Path(key) / f"frame-{sample_index:06d}.png"
                    destination = output_dir / relative
                    _write_png(destination, views[key])
                    created.append(destination)
                    files[key] = relative.as_posix()
                manifest_handle.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "client_monotonic_ns": monotonic_ns,
                            "client_utc_ns": utc_ns,
                            "files": files,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                written.value += 1
            except Exception as exc:  # recorder errors never reach robot logic
                for path in created:
                    path.unlink(missing_ok=True)
                write_dropped.value += 1
                worker_error = f"{type(exc).__name__}: {exc}"
                writes_enabled = False

        clean_shutdown = True
    except Exception as exc:
        worker_error = f"{type(exc).__name__}: {exc}"
        try:
            ready.send({"ok": False, "error": worker_error})
            ready.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if manifest_handle is not None:
            try:
                manifest_handle.close()
            except OSError:
                clean_shutdown = False
        if owns_output_dir and output_dir.is_dir():
            try:
                _write_json(
                    output_dir / "summary.json",
                    {
                        "schema_version": VISION_RECORDING_SCHEMA_VERSION,
                        "submitted": int(submitted.value),
                        "written": int(written.value),
                        "dropped": int(queue_dropped.value + write_dropped.value),
                        "worker_error": worker_error,
                        "clean_shutdown": bool(clean_shutdown and worker_error is None),
                    },
                )
            except Exception:
                pass


class NonBlockingVisionRecorder:
    """Record selected policy-video arrays without backpressuring the caller."""

    def __init__(
        self,
        output_dir: str | Path,
        video_keys: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        requested_output = Path(output_dir).expanduser()
        if requested_output.name in {"", ".", ".."}:
            raise ValueError("Vision recording output must name a new directory")
        try:
            output_parent = requested_output.parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"Vision recording parent directory does not exist: {requested_output.parent}"
            ) from exc
        self.output_dir = output_parent / requested_output.name
        self.video_keys = tuple(video_keys)
        self._closed = False
        self._first_drop_reason: str | None = None
        self._final_stats: dict[str, Any] | None = None

        if not self.video_keys or len(set(self.video_keys)) != len(self.video_keys):
            raise ValueError("Vision recording requires distinct video keys")
        if any(not key or key in {".", ".."} or Path(key).name != key for key in self.video_keys):
            raise ValueError("Vision recording video keys must be safe single path components")
        metadata = {} if metadata is None else dict(metadata)
        # Fail before spawning if caller metadata cannot be represented exactly.
        json.dumps(metadata, allow_nan=False)
        # Do not resolve the final component: doing so would follow a malicious
        # or accidentally pre-created symlink outside the run directory.  The
        # child repeats this no-overwrite guarantee with mkdir(exist_ok=False).
        try:
            self.output_dir.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "Vision recording output already exists or is a symbolic link: "
                f"{self.output_dir}"
            )

        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=VISION_RECORDING_QUEUE_CAPACITY)
        self._stopping = context.Event()
        self._submitted = context.Value("Q", 0, lock=False)
        self._queue_dropped = context.Value("Q", 0, lock=False)
        self._written = context.Value("Q", 0, lock=False)
        self._write_dropped = context.Value("Q", 0, lock=False)
        ready_parent, ready_child = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_recording_worker,
            args=(
                str(self.output_dir),
                self.video_keys,
                metadata,
                self._queue,
                self._stopping,
                ready_child,
                self._submitted,
                self._queue_dropped,
                self._written,
                self._write_dropped,
            ),
            name="groot-vision-recorder",
            daemon=True,
        )
        try:
            self._process.start()
            ready_child.close()
            if not ready_parent.poll(VISION_RECORDING_START_TIMEOUT_S):
                raise RuntimeError("Vision recorder did not become ready before its startup deadline")
            status = ready_parent.recv()
            if not status.get("ok"):
                raise RuntimeError(f"Vision recorder startup failed: {status.get('error')}")
        except Exception:
            self._stop_worker()
            raise
        finally:
            ready_parent.close()

    def submit_observation(self, observation: dict[str, object]) -> bool:
        """Queue one observation immediately, returning false when it is dropped."""

        if self._closed:
            return False
        self._submitted.value += 1
        try:
            if not self._process.is_alive():
                raise RuntimeError("recorder worker is no longer running")
            video = observation.get("video")
            if not isinstance(video, dict):
                raise ValueError("observation.video is not a mapping")
            views: dict[str, np.ndarray] = {}
            for key in self.video_keys:
                value = np.asarray(video[key])
                if value.dtype != np.uint8 or value.ndim != 5 or value.shape[:2] != (1, 1):
                    raise ValueError(f"observation.video[{key!r}] must be uint8 [1,1,H,W,C]")
                frame = value[0, 0]
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"observation.video[{key!r}] must contain three-channel RGB frames")
                views[key] = np.ascontiguousarray(frame)
            self._queue.put_nowait(
                (
                    int(self._submitted.value - 1),
                    time.monotonic_ns(),
                    time.time_ns(),
                    views,
                )
            )
            return True
        except Exception as exc:
            self._queue_dropped.value += 1
            # Do not log from this timing-critical path.  In particular, a
            # FileHandler on the same slow/full filesystem could turn the first
            # queue-overflow warning into robot-control backpressure.  The
            # reason is folded into the durable summary after authority release.
            if self._first_drop_reason is None:
                detail = str(exc)
                self._first_drop_reason = type(exc).__name__ + (f": {detail}" if detail else "")
            return False

    @property
    def stats(self) -> dict[str, Any]:
        if self._final_stats is not None:
            return dict(self._final_stats)
        return {
            "submitted": int(self._submitted.value),
            "written": int(self._written.value),
            "dropped": int(self._queue_dropped.value + self._write_dropped.value),
            "worker_error": None,
            "first_drop_reason": self._first_drop_reason,
            "clean_shutdown": False,
        }

    def _stop_worker(self) -> tuple[bool, bool]:
        """Request a bounded stop and return ``(forced, fully_stopped)``."""

        process = getattr(self, "_process", None)
        stopping = getattr(self, "_stopping", None)
        work_queue = getattr(self, "_queue", None)
        if stopping is not None:
            stopping.set()
        if work_queue is not None:
            try:
                work_queue.put_nowait(None)
            except (queue.Full, BrokenPipeError, OSError):
                pass
        forced = False
        if process is not None and process.is_alive():
            process.join(timeout=VISION_RECORDING_CLOSE_TIMEOUT_S)
            if process.is_alive():
                forced = True
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
        stopped = process is None or not process.is_alive()
        return forced, stopped

    def close(self) -> None:
        """Stop within a fixed deadline; never raise into deployment cleanup."""

        if self._closed:
            return
        self._closed = True
        try:
            forced, worker_stopped = self._stop_worker()
            summary_path = self.output_dir / "summary.json"
            summary: dict[str, Any] | None = None
            if summary_path.is_file():
                try:
                    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        summary = loaded
                except (OSError, json.JSONDecodeError):
                    pass

            submitted = int(self._submitted.value)
            try:
                summary_counts_match = bool(
                    summary is not None
                    and int(summary.get("submitted", -1)) == submitted
                    and int(summary.get("written", -1)) + int(summary.get("dropped", -1))
                    == submitted
                )
            except (TypeError, ValueError):
                summary_counts_match = False
            summary_unclean = bool(
                summary is not None
                and (summary.get("worker_error") is not None or not summary.get("clean_shutdown", False))
            )
            needs_repair = forced or summary_unclean or not summary_counts_match
            can_repair = (
                worker_stopped and self.output_dir.is_dir() and not self.output_dir.is_symlink()
            )
            if needs_repair and can_repair:
                retained = _repair_interrupted_recording(self.output_dir, self.video_keys)
                existing_error = summary.get("worker_error") if summary is not None else None
                reason = str(
                    existing_error
                    or (
                        "recorder worker exceeded the bounded close deadline"
                        if forced
                        else "recorder worker exited without a complete summary"
                    )
                )
                summary = {
                    "schema_version": VISION_RECORDING_SCHEMA_VERSION,
                    "submitted": submitted,
                    "written": retained,
                    "dropped": max(0, submitted - retained),
                    "worker_error": reason,
                    "first_drop_reason": self._first_drop_reason,
                    "clean_shutdown": False,
                }
                _write_json(summary_path, summary)
            elif needs_repair and not worker_stopped:
                # A process stuck in uninterruptible I/O may survive terminate
                # and kill.  Never race it by repairing files concurrently.
                summary = {
                    "schema_version": VISION_RECORDING_SCHEMA_VERSION,
                    "submitted": submitted,
                    "written": int(self._written.value),
                    "dropped": max(0, submitted - int(self._written.value)),
                    "worker_error": "recorder worker could not be stopped; output was not repaired",
                    "first_drop_reason": self._first_drop_reason,
                    "clean_shutdown": False,
                }
            if summary is None:
                summary = {
                    "schema_version": VISION_RECORDING_SCHEMA_VERSION,
                    "submitted": submitted,
                    "written": int(self._written.value),
                    "dropped": max(0, submitted - int(self._written.value)),
                    "worker_error": "recorder output directory was unavailable during cleanup",
                    "first_drop_reason": self._first_drop_reason,
                    "clean_shutdown": False,
                }
            elif self._first_drop_reason is not None:
                summary["first_drop_reason"] = self._first_drop_reason
                if worker_stopped:
                    _write_json(summary_path, summary)
            self._final_stats = summary
        except Exception as exc:
            LOGGER.warning("Vision recorder cleanup failed after robot release: %s", exc)
        finally:
            work_queue = getattr(self, "_queue", None)
            if work_queue is not None:
                try:
                    work_queue.cancel_join_thread()
                    work_queue.close()
                except (OSError, ValueError):
                    pass
