"""Continuous, best-effort camera demo capture, independent of policy inference.

The spawned process owns its own read-only TeleImager subscriber and HighGUI
loop. A bounded writer thread consumes those SAME frames; neither disk nor GUI
work occurs in the control process. This is not a policy-observation recorder.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from unitree_lerobot.eval_robot.vision_recorder import _write_json, _write_png
from unitree_lerobot.utils.depth_colormap import (
    FIXED_TURBO_DEPTH_COLORMAP,
    apply_fixed_turbo_depth_colormap_array,
    fixed_turbo_depth_colormap_contract,
)


DEMO_START_TIMEOUT_S = 8.0
DEMO_CLOSE_TIMEOUT_S = 2.0
DEMO_QUEUE_CAPACITY = 2
DEMO_SCHEMA_VERSION = 1
_VIDEO_ATTRIBUTES = {
    "ego_view": "rgb",
    "depth_gray_view": "depth_gray",
    "surface_normals_view": "surface_normals",
}
_COUNTERS = (
    "captured", "written", "recording_dropped", "duplicate_polls",
    "read_errors", "preview_frames", "events_dropped", "recording_errors", "worker_errors",
)


def _report(status_queue: Any, **status: Any) -> None:
    try:
        status_queue.put_nowait(status)
    except (queue.Full, OSError, ValueError):
        pass


class _TimestampedColorClient:
    """Retain the exact RGB frame consumed by read(), without a second fetch.

    TeleimagerCamera exposes geometry timestamps, but its colour-only read()
    currently returns no source timestamp. This child-local proxy fills that
    gap without touching the policy client's camera or subscriber singleton.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.received_monotonic_ns: int | None = None

    def get_head_frame(self) -> Any:
        frame = self.client.get_head_frame()
        self.received_monotonic_ns = getattr(frame, "received_monotonic_ns", None)
        return frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def _source_identity(
    camera: Any, images: Any, video_keys: tuple[str, ...], now_ns: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Identify an actual received frame, not the time the cache was polled.

    Legacy RGB/depth streams are independent, not atomic. RGB-led composites
    may reuse a still-fresh geometry component and report its separate stamp.
    Geometry-only runs are led by geometry arrivals instead. The legacy
    protocol has no camera capture clock or sequence, so we do not invent one.
    """

    from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import RGBD_MAX_RECEIVE_AGE_S

    transport = getattr(camera, "_geometry_transport", "legacy")
    if transport == "atomic" and getattr(images, "sequence", None) is not None:
        received = getattr(camera, "_last_rgbd_received_ns", None)
        stamps = {
            "transport": "atomic_rgbd",
            "sequence": int(images.sequence),
            "received_monotonic_ns": received,
            "server_capture_monotonic_ns": getattr(camera, "_last_rgbd_server_capture_ns", None),
        }
        identity = ("atomic_rgbd", images.sequence)
        required = (received,)
    else:
        requires_depth = getattr(camera, "_requires_depth", False)
        color = (
            getattr(camera, "_last_color_received_ns", None)
            if requires_depth
            else getattr(camera._client, "received_monotonic_ns", None)
        )
        depth = getattr(camera, "_last_depth_received_ns", None) if requires_depth else None
        stamps = {
            "transport": "legacy_independent" if requires_depth else "legacy_rgb",
            "color_received_monotonic_ns": color,
            "depth_received_monotonic_ns": depth,
            "server_capture_monotonic_ns": None,
        }
        leader = color if "ego_view" in video_keys else depth
        identity = (stamps["transport"], leader)
        required = (color, depth) if requires_depth else (color,)
    for stamp in required:
        if stamp is None:
            raise TimeoutError("Demo frame is missing its receive timestamp")
        age_s = (now_ns - int(stamp)) / 1e9
        if age_s < 0 or age_s > RGBD_MAX_RECEIVE_AGE_S:
            raise TimeoutError(f"Demo frame is stale ({age_s:.3f}s since receive)")
    return identity, stamps


def _demo_views(images: Any, video_keys: tuple[str, ...], depth_colormap: str | None) -> dict[str, np.ndarray]:
    views = {}
    for key in video_keys:
        value = np.asarray(getattr(images, _VIDEO_ATTRIBUTES[key]))
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(f"Demo view {key!r} must be uint8 HxWx3 RGB")
        if key == "depth_gray_view" and depth_colormap is not None:
            value = apply_fixed_turbo_depth_colormap_array(value)
        views[key] = np.ascontiguousarray(value)
    return views


def _effective_fps(target_fps: float, config: dict[str, Any]) -> float:
    try:
        advertised = float(config["head_camera"]["fps"])
    except (KeyError, TypeError, ValueError):
        return target_fps
    return min(target_fps, advertised) if math.isfinite(advertised) and advertised > 0 else target_fps


def _counts(counters: dict[str, Any]) -> dict[str, int]:
    return {key: int(value.value) for key, value in counters.items()}


def _demo_writer(
    output_dir: Path, video_keys: tuple[str, ...], metadata: dict[str, Any],
    depth_colormap: str | None, target_fps: float, frame_queue: Any,
    event_queue: Any, stopping: Any, counters: dict[str, Any], status_queue: Any,
    writer_state: dict[str, Any],
) -> None:
    """Disk errors disable only recording; a blocked disk cannot freeze preview."""

    owns_output = False
    frames = events = None
    error = None
    try:
        output_dir.mkdir(mode=0o700, exist_ok=False)
        owns_output = True
        writer_state["owns_output"] = True
        for key in video_keys:
            (output_dir / key).mkdir(mode=0o700)
        _write_json(output_dir / "manifest.json", {
            "schema_version": DEMO_SCHEMA_VERSION,
            "kind": "groot_continuous_demo_vision",
            "video_keys": list(video_keys),
            "target_fps": target_fps,
            "sampling": "independent camera arrivals, capped at target/configured camera FPS; no duplicated cache polls",
            "not_policy_observations": True,
            "pixel_contract": "lossless uint8 RGB camera views before server crop/resize/normalization; optional recorded depth colormap",
            "timing": "frames.jsonl is authoritative; receive clocks are local monotonic, not camera capture clocks; client_utc_ns is sampling wall clock",
            "legacy_geometry": "independent streams; RGB-led composite (depth-led for geometry-only); component receive stamps preserved",
            "queue": {"capacity": DEMO_QUEUE_CAPACITY, "overflow": "drop-new", "backpressure": False},
            "view_transforms": {"depth_gray_view": fixed_turbo_depth_colormap_contract()} if depth_colormap else {},
            "metadata": metadata,
        })
        frames = (output_dir / "frames.jsonl").open("x", encoding="utf-8", buffering=1)
        events = (output_dir / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        writer_state["ready"] = True
        _report(status_queue, recording_ready=True)
        while not stopping.is_set() or not frame_queue.empty() or not event_queue.empty():
            # Bounded event batch prevents metadata traffic starving frames.
            for _ in range(16):
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                events.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            try:
                record, views = frame_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            created = []
            try:
                files = {}
                for key in video_keys:
                    relative = Path(key) / f"frame-{record['sample_index']:08d}.png"
                    path = output_dir / relative
                    _write_png(path, views[key])
                    created.append(path)
                    files[key] = relative.as_posix()
                frames.write(json.dumps({**record, "files": files}, sort_keys=True, allow_nan=False) + "\n")
                counters["written"].value += 1
            except Exception:
                for path in created:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        counters["recording_errors"].value += 1
        writer_state["recording_error"] = error
        _report(status_queue, recording_error=error, recording_ready=False)
    finally:
        for handle in (frames, events):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        if owns_output:
            try:
                _write_json(output_dir / "summary.json", {
                    "schema_version": DEMO_SCHEMA_VERSION,
                    **_counts(counters),
                    "recording_error": error,
                    "clean_shutdown": bool(stopping.is_set() and error is None),
                })
            except Exception:
                pass


def _pump_preview(windows: set[str], operator_exit: Any) -> None:
    if cv2.pollKey() & 0xFF == ord("q"):
        operator_exit.set()
    for name in windows:
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            operator_exit.set()


def _demo_worker(
    options: dict[str, Any], stopping: Any, ready: Any, operator_exit: Any,
    counters: dict[str, Any], event_queue: Any, status_queue: Any,
) -> None:
    camera = writer = None
    writer_stopping = threading.Event()
    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=DEMO_QUEUE_CAPACITY)
    writer_state: dict[str, Any] = {}
    windows: set[str] = set()
    preview = options["show_camera"]
    ready_sent = False
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        os.umask(0o077)
        cv2.setNumThreads(1)
        if preview and os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # Qt-backed imshow may abort instead of raising with no display.
            preview = False
            _report(status_queue, preview_error="No desktop display; recording remains enabled")
        # Importing this module initializes no robot, DDS participant or GPU.
        # Spawn is essential: ImageClient owns process-global subscribers.
        from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import TeleimagerCamera

        camera = TeleimagerCamera(
            options["image_host"], depth_encoding=options["depth_encoding"],
            surface_normal_encoding=options["surface_normal_encoding"],
            prefer_atomic_rgbd=options["prefer_atomic_rgbd"],
        )
        if not getattr(camera, "_requires_depth", False):
            camera._client = _TimestampedColorClient(camera._client)
        effective_fps = _effective_fps(options["target_fps"], camera.config)
        if options["output_dir"] is not None:
            writer = threading.Thread(target=_demo_writer, kwargs={
                "output_dir": Path(options["output_dir"]),
                "video_keys": options["video_keys"],
                "metadata": {**options["metadata"], "camera_config": camera.config, "effective_target_fps": effective_fps},
                "depth_colormap": options["depth_colormap"], "target_fps": options["target_fps"],
                "frame_queue": frame_queue, "event_queue": event_queue, "stopping": writer_stopping,
                "counters": counters, "status_queue": status_queue,
                "writer_state": writer_state,
            }, name="demo-png-writer", daemon=True)
            writer.start()
        interval = 1.0 / effective_fps
        next_capture = time.monotonic()
        previous_identity = None
        reported_camera_error = None
        first_frame_seen = False
        while not stopping.is_set():
            if not ready_sent:
                if writer_state.get("recording_error"):
                    raise RuntimeError(f"Demo recording startup failed: {writer_state['recording_error']}")
                if first_frame_seen and (writer is None or writer_state.get("ready")):
                    ready.send({"ok": True, "effective_target_fps": effective_fps})
                    ready_sent = True
                    ready.close()
            if preview:
                try:
                    _pump_preview(windows, operator_exit)
                except Exception as exc:
                    preview = False
                    _report(status_queue, preview_error=f"{type(exc).__name__}: {exc}")
            remaining = next_capture - time.monotonic()
            if remaining > 0:
                stopping.wait(min(remaining, 0.005))
                continue
            next_capture += interval
            if next_capture < time.monotonic():
                next_capture = time.monotonic() + interval
            try:
                images = camera.read(timeout_s=0.02)
                monotonic_ns = time.monotonic_ns()
                identity, source = _source_identity(camera, images, options["video_keys"], monotonic_ns)
                if identity == previous_identity:
                    counters["duplicate_polls"].value += 1
                    continue
                if previous_identity is not None and identity[0] == previous_identity[0] and identity[1] < previous_identity[1]:
                    raise TimeoutError("Demo source timestamp/sequence regressed")
                views = _demo_views(images, options["video_keys"], options["depth_colormap"])
                previous_identity = identity
                record = {
                    "sample_index": int(counters["captured"].value),
                    "client_monotonic_ns": monotonic_ns, "client_utc_ns": time.time_ns(),
                    "source": source,
                }
                counters["captured"].value += 1
                first_frame_seen = True
                if reported_camera_error is not None:
                    _report(status_queue, camera_error=None)
                    reported_camera_error = None
            except Exception as exc:
                counters["read_errors"].value += 1
                detail = f"{type(exc).__name__}: {exc}"
                if reported_camera_error is None:
                    _report(status_queue, camera_error=detail)
                    reported_camera_error = detail
                continue
            if writer is not None:
                try:
                    if not writer.is_alive():
                        raise queue.Full
                    frame_queue.put_nowait((record, views))
                except queue.Full:
                    counters["recording_dropped"].value += 1
            if preview:
                try:
                    for key, view in views.items():
                        if options["metadata"].get("model_identity", {}).get("blinded") and key != "ego_view":
                            continue
                        name = f"GR00T live demo: {key} (q to exit)"
                        cv2.imshow(name, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                        windows.add(name)
                    counters["preview_frames"].value += 1
                except Exception as exc:
                    preview = False
                    _report(status_queue, preview_error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        counters["worker_errors"].value += 1
        _report(status_queue, worker_error=error)
        if not ready_sent:
            try:
                ready.send({"ok": False, "error": error})
            except (OSError, EOFError):
                pass
    finally:
        writer_stopping.set()
        if writer is not None:
            writer.join(timeout=1.0)
            if writer.is_alive():
                counters["recording_errors"].value += 1
                _report(status_queue, recording_error="PNG writer exceeded shutdown deadline")
            elif writer_state.get("owns_output"):
                try:
                    counts = _counts(counters)
                    _write_json(Path(options["output_dir"]) / "summary.json", {
                        "schema_version": DEMO_SCHEMA_VERSION, **counts,
                        "recording_dropped": counts["captured"] - counts["written"],
                        "recording_error": writer_state.get("recording_error"),
                        "clean_shutdown": bool(stopping.is_set() and not writer_state.get("recording_error")),
                    })
                except Exception:
                    pass
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        for name in windows:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        ready.close()
        # Never wait on IPC feeder threads at interpreter exit.
        status_queue.cancel_join_thread()


class ContinuousDemoStream:
    """Spawn a non-control, camera-native demo stream before robot initialization."""

    def __init__(
        self, *, image_host: str, video_keys: tuple[str, ...],
        output_dir: str | Path | None = None, metadata: dict[str, Any] | None = None,
        depth_encoding: Any = None, surface_normal_encoding: Any = None,
        depth_colormap: str | None = None, show_camera: bool = False,
        target_fps: float = 30.0, prefer_atomic_rgbd: bool = False,
    ) -> None:
        video_keys = tuple(video_keys)
        if not video_keys or len(set(video_keys)) != len(video_keys) or not set(video_keys) <= _VIDEO_ATTRIBUTES.keys():
            raise ValueError("Demo requires distinct supported video keys")
        if not math.isfinite(target_fps) or not 0 < target_fps <= 30:
            raise ValueError("Demo target_fps must be finite and in (0, 30]")
        if depth_colormap not in {None, FIXED_TURBO_DEPTH_COLORMAP}:
            raise ValueError("Unsupported demo depth colormap")
        if depth_colormap and "depth_gray_view" not in video_keys:
            raise ValueError("Demo depth colormap requires depth_gray_view")
        metadata = dict(metadata or {})
        json.dumps(metadata, allow_nan=False)
        self.output_dir = None
        if output_dir is not None:
            requested = Path(output_dir).expanduser()
            if requested.name in {"", ".", ".."}:
                raise ValueError("Demo output must name a new directory")
            self.output_dir = requested.parent.resolve(strict=True) / requested.name
            if self.output_dir.exists() or self.output_dir.is_symlink():
                raise ValueError(f"Demo output already exists: {self.output_dir}")
        self._closed = False
        self._status: dict[str, Any] = {"clean_shutdown": False}
        context = mp.get_context("spawn")
        self._stopping = context.Event()
        self._operator_exit = context.Event()
        self._counters = {key: context.Value("Q", 0, lock=False) for key in _COUNTERS}
        self._events = context.Queue(maxsize=32)
        self._statuses = context.Queue(maxsize=32)
        ready_parent, ready_child = context.Pipe(duplex=False)
        options = {
            "image_host": image_host, "video_keys": video_keys,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "metadata": metadata, "depth_encoding": depth_encoding,
            "surface_normal_encoding": surface_normal_encoding, "depth_colormap": depth_colormap,
            "show_camera": show_camera, "target_fps": float(target_fps),
            "prefer_atomic_rgbd": prefer_atomic_rgbd,
        }
        self._process = context.Process(
            target=_demo_worker,
            args=(options, self._stopping, ready_child, self._operator_exit,
                  self._counters, self._events, self._statuses),
            name="groot-continuous-demo", daemon=True,
        )
        try:
            self._process.start()
            ready_child.close()
            if not ready_parent.poll(DEMO_START_TIMEOUT_S):
                raise RuntimeError("Continuous demo camera startup exceeded its deadline")
            status = ready_parent.recv()
            if not status.get("ok"):
                raise RuntimeError(f"Continuous demo camera startup failed: {status.get('error')}")
            self._status.update(status)
        except Exception:
            self.close()
            raise
        finally:
            ready_child.close()
            ready_parent.close()

    @property
    def operator_exit_requested(self) -> bool:
        return self._operator_exit.is_set()

    def mark_event(self, label: str, details: dict[str, Any] | None = None) -> bool:
        """Best-effort phase marker, never a control-process filesystem write."""
        if self._closed or self.output_dir is None:
            return False
        try:
            event = {"label": str(label), "details": dict(details or {}),
                     "client_monotonic_ns": time.monotonic_ns(), "client_utc_ns": time.time_ns()}
            # Detach from caller-owned mutable metadata before async pickling.
            event = json.loads(json.dumps(event, allow_nan=False))
            self._events.put_nowait(event)
            return True
        except Exception:
            self._counters["events_dropped"].value += 1
            return False

    @property
    def stats(self) -> dict[str, Any]:
        try:
            for _ in range(32):
                self._status.update(self._statuses.get_nowait())
        except (queue.Empty, OSError, ValueError):
            pass
        counts = _counts(self._counters)
        if self._closed and self.output_dir is not None:
            counts["recording_dropped"] = counts["captured"] - counts["written"]
        return {**self._status, **counts,
                "running": self._process.is_alive(),
                "operator_exit_requested": self.operator_exit_requested}

    def close(self) -> None:
        """Bounded, idempotent cleanup after actuator release; no disk I/O."""
        if self._closed:
            return
        self._closed = True
        forced = False
        try:
            self._stopping.set()
            if self._process.is_alive():
                self._process.join(timeout=DEMO_CLOSE_TIMEOUT_S)
            if self._process.is_alive():
                forced = True
                self._process.terminate()
                self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
            self.stats
            self._status.update(forced_shutdown=forced, clean_shutdown=(
                not forced and self._process.exitcode == 0
                and self._counters["recording_errors"].value == 0
                and self._counters["worker_errors"].value == 0
                and not any(self._status.get(key) for key in ("worker_error", "recording_error"))
            ))
        except Exception as exc:
            self._status.update(clean_shutdown=False, close_error=f"{type(exc).__name__}: {exc}")
        finally:
            for channel in (self._events, self._statuses):
                try:
                    channel.cancel_join_thread()
                    channel.close()
                except (OSError, ValueError):
                    pass
