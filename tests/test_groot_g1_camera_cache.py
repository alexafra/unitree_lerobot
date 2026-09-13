from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import DepthEncodingContract
from unitree_lerobot.eval_robot.image_server.rgbd_protocol import TeleRgbdFrame
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    RGBD_MAX_RECEIVE_AGE_S,
    TeleimagerCamera,
)


def _encoded_frames(
    *,
    bgr_value: int = 0,
    depth_value: int = 625,
) -> tuple[bytes, bytes]:
    bgr = np.full((480, 640, 3), bgr_value, dtype=np.uint8)
    depth = np.full((480, 640), depth_value, dtype=np.uint16)
    color_ok, color_jpeg = cv2.imencode(".jpg", bgr)
    depth_ok, depth_png = cv2.imencode(".png", depth)
    assert color_ok and depth_ok
    return color_jpeg.tobytes(), depth_png.tobytes()


def _geometry_camera(*, transport: str = "legacy") -> TeleimagerCamera:
    camera = object.__new__(TeleimagerCamera)
    camera._requires_depth = True
    camera._depth_encoding = DepthEncodingContract(near_m=0.25, far_m=1.0)
    camera._surface_normal_encoding = None
    camera._depth_scale_m_per_unit = 0.001
    camera._geometry_transport = transport
    camera._atomic_ever_succeeded = transport == "atomic"
    camera._geometry_selection_logged = True
    camera._last_color_received_ns = None
    camera._last_depth_received_ns = None
    camera._last_rgbd_sequence = None
    camera._last_rgbd_server_capture_ns = None
    camera._reported_stream_fps = True
    camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)
    camera._depth_subscriber = SimpleNamespace(is_alive=lambda: True)
    camera._rgbd_subscriber = SimpleNamespace(is_alive=lambda: True)
    camera.config = {
        "head_camera": {
            "image_shape": [480, 640],
            "binocular": False,
        }
    }
    return camera


def test_legacy_geometry_reuses_latest_valid_component_caches_until_stale() -> None:
    first_jpeg, depth_png = _encoded_frames(bgr_value=0)
    second_jpeg, _ = _encoded_frames(bgr_value=80)
    received_ns = time.monotonic_ns()

    class LegacyClient:
        color_jpeg = first_jpeg
        color_received_ns = received_ns

        def get_head_frame(self):
            return SimpleNamespace(
                jpg=self.color_jpeg,
                fps=30.0,
                received_monotonic_ns=self.color_received_ns,
            )

        def get_head_depth_frame(self):
            return SimpleNamespace(
                jpg=depth_png,
                fps=30.0,
                received_monotonic_ns=received_ns,
            )

    camera = _geometry_camera()
    client = LegacyClient()
    camera._client = client

    first = camera.read(timeout_s=0.02)
    repeated = camera.read(timeout_s=0.02)
    np.testing.assert_array_equal(repeated.rgb, first.rgb)
    np.testing.assert_array_equal(repeated.depth_gray, first.depth_gray)
    assert not np.shares_memory(repeated.rgb, first.rgb)
    assert not np.shares_memory(repeated.depth_gray, first.depth_gray)

    # A new RGB receipt does not make the still-fresh cached depth unusable.
    client.color_jpeg = second_jpeg
    client.color_received_ns = received_ns + 1
    mixed_latest = camera.read(timeout_s=0.02)
    assert not np.array_equal(mixed_latest.rgb, first.rgb)
    np.testing.assert_array_equal(mixed_latest.depth_gray, first.depth_gray)

    stale_now_ns = received_ns + int((RGBD_MAX_RECEIVE_AGE_S + 0.001) * 1e9)
    with (
        mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic_ns",
            return_value=stale_now_ns,
        ),
        pytest.raises(TimeoutError, match="stale"),
    ):
        camera._read_legacy_geometry(timeout_s=0.01)


def test_legacy_geometry_survives_low_level_cache_clear_until_original_age_limit() -> None:
    color_jpeg, depth_png = _encoded_frames()
    received_ns = time.monotonic_ns()

    class LegacyClient:
        cleared = False
        color_received_ns = received_ns

        def get_head_frame(self):
            if self.cleared:
                return SimpleNamespace(jpg=None, fps=0.0, received_monotonic_ns=None)
            return SimpleNamespace(
                jpg=color_jpeg,
                fps=30.0,
                received_monotonic_ns=self.color_received_ns,
            )

        def get_head_depth_frame(self):
            if self.cleared:
                return SimpleNamespace(jpg=None, fps=0.0, received_monotonic_ns=None)
            return SimpleNamespace(
                jpg=depth_png,
                fps=30.0,
                received_monotonic_ns=received_ns,
            )

    camera = _geometry_camera()
    client = LegacyClient()
    camera._client = client
    accepted = camera._read_legacy_geometry(timeout_s=0.02)
    client.cleared = True

    # The low-level subscriber currently clears its ring/FPS after 100 ms.
    # Adapter-level validated caches must cover the remaining freshness window.
    still_fresh_ns = received_ns + 120_000_000
    with mock.patch(
        "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic_ns",
        return_value=still_fresh_ns,
    ):
        cached = camera._read_legacy_geometry(timeout_s=0.02)
    np.testing.assert_array_equal(cached.rgb, accepted.rgb)
    np.testing.assert_array_equal(cached.depth_gray, accepted.depth_gray)
    assert not np.shares_memory(cached.rgb, accepted.rgb)
    assert not np.shares_memory(cached.depth_gray, accepted.depth_gray)

    # Isolate depth expiry: a current RGB update cannot renew the older depth.
    stale_now_ns = received_ns + int((RGBD_MAX_RECEIVE_AGE_S + 0.001) * 1e9)
    client.cleared = False
    client.color_received_ns = stale_now_ns
    with (
        mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic_ns",
            return_value=stale_now_ns,
        ),
        mock.patch.object(client, "get_head_depth_frame", return_value=SimpleNamespace(
            jpg=None,
            fps=0.0,
            received_monotonic_ns=None,
        )),
        pytest.raises(TimeoutError, match="aligned-depth.*stale"),
    ):
        camera._read_legacy_geometry(timeout_s=0.01)


def test_atomic_geometry_reuses_latest_complete_packet_when_no_new_packet_arrives() -> None:
    color_jpeg, depth_png = _encoded_frames()
    received_ns = time.monotonic_ns()
    packet = TeleRgbdFrame(
        sequence=9,
        server_capture_monotonic_ns=123,
        received_monotonic_ns=received_ns,
        color_jpeg=color_jpeg,
        aligned_depth_png=depth_png,
    )

    class AtomicClient:
        def __init__(self):
            self.frames = [packet, packet, None]
            self.index = 0

        def get_head_rgbd_frame(self):
            value = self.frames[min(self.index, len(self.frames) - 1)]
            self.index += 1
            return value

        def get_head_rgbd_fps(self):
            return 30.0

    camera = _geometry_camera(transport="atomic")
    camera._client = AtomicClient()

    first = camera._read_atomic_geometry(timeout_s=0.02)
    repeated = camera._read_atomic_geometry(timeout_s=0.02)
    missing_update = camera._read_atomic_geometry(timeout_s=0.02)

    assert first.sequence == repeated.sequence == missing_update.sequence == 9
    np.testing.assert_array_equal(repeated.rgb, first.rgb)
    np.testing.assert_array_equal(missing_update.depth_gray, first.depth_gray)
    assert not np.shares_memory(repeated.rgb, first.rgb)
    assert not np.shares_memory(missing_update.depth_gray, first.depth_gray)


def test_atomic_duplicate_receipt_does_not_extend_original_cache_age() -> None:
    color_jpeg, depth_png = _encoded_frames()
    accepted_ns = time.monotonic_ns()
    first_packet = TeleRgbdFrame(
        sequence=9,
        server_capture_monotonic_ns=123,
        received_monotonic_ns=accepted_ns,
        color_jpeg=color_jpeg,
        aligned_depth_png=depth_png,
    )
    duplicate_packet = TeleRgbdFrame(
        sequence=9,
        server_capture_monotonic_ns=123,
        received_monotonic_ns=accepted_ns + 140_000_000,
        color_jpeg=color_jpeg,
        aligned_depth_png=depth_png,
    )

    class AtomicClient:
        frame = first_packet

        def get_head_rgbd_frame(self):
            return self.frame

        def get_head_rgbd_fps(self):
            return 30.0

    camera = _geometry_camera(transport="atomic")
    client = AtomicClient()
    camera._client = client
    camera._read_atomic_geometry(timeout_s=0.02)
    client.frame = duplicate_packet

    stale_now_ns = accepted_ns + int((RGBD_MAX_RECEIVE_AGE_S + 0.001) * 1e9)
    with (
        mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic_ns",
            return_value=stale_now_ns,
        ),
        pytest.raises(TimeoutError, match="atomic RGBD packet is stale"),
    ):
        camera._read_atomic_geometry(timeout_s=0.01)


def test_atomic_geometry_rejects_stale_cached_packet() -> None:
    color_jpeg, depth_png = _encoded_frames()
    received_ns = time.monotonic_ns()
    packet = TeleRgbdFrame(
        sequence=9,
        server_capture_monotonic_ns=123,
        received_monotonic_ns=received_ns,
        color_jpeg=color_jpeg,
        aligned_depth_png=depth_png,
    )
    camera = _geometry_camera(transport="atomic")
    camera._client = SimpleNamespace(
        get_head_rgbd_frame=lambda: packet,
        get_head_rgbd_fps=lambda: 30.0,
    )
    camera._read_atomic_geometry(timeout_s=0.02)

    stale_now_ns = received_ns + int((RGBD_MAX_RECEIVE_AGE_S + 0.001) * 1e9)
    with (
        mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic_ns",
            return_value=stale_now_ns,
        ),
        pytest.raises(TimeoutError, match="atomic RGBD packet is stale"),
    ):
        camera._read_atomic_geometry(timeout_s=0.01)


@pytest.mark.parametrize(
    ("failure_kind", "expected"),
    [
        ("regression", "sequence regressed"),
        ("corruption", "Invalid TeleImager atomic RGBD packet"),
    ],
)
def test_atomic_geometry_keeps_regression_and_corruption_fail_closed(
    failure_kind: str,
    expected: str,
) -> None:
    color_jpeg, depth_png = _encoded_frames()
    accepted = TeleRgbdFrame(
        sequence=9,
        server_capture_monotonic_ns=123,
        received_monotonic_ns=time.monotonic_ns(),
        color_jpeg=color_jpeg,
        aligned_depth_png=depth_png,
    )
    next_frame: TeleRgbdFrame | Exception
    if failure_kind == "regression":
        next_frame = TeleRgbdFrame(
            sequence=8,
            server_capture_monotonic_ns=124,
            received_monotonic_ns=time.monotonic_ns(),
            color_jpeg=b"unused",
            aligned_depth_png=b"unused",
        )
    else:
        next_frame = ValueError("corrupt atomic packet")

    class AtomicClient:
        def __init__(self):
            self.frames = [accepted, next_frame]

        def get_head_rgbd_frame(self):
            value = self.frames.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        def get_head_rgbd_fps(self):
            return 30.0

    camera = _geometry_camera(transport="atomic")
    camera._client = AtomicClient()
    camera._read_atomic_geometry(timeout_s=0.02)

    with pytest.raises(DeploymentError, match=expected):
        camera._read_atomic_geometry(timeout_s=0.02)
