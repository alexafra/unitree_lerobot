"""Versioned wire format for capture-synchronised colour and aligned depth.

The legacy TeleImager colour and depth ports carry bare JPEG/PNG payloads and
remain unchanged.  This module defines the opt-in, single-message RGBD stream
used when a consumer needs proof that both images came from the same camera
capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional, Union


RGBD_MAGIC = b"TELRGBD\0"
RGBD_VERSION = 1
RGBD_PROTOCOL = "teleimager-rgbd-v1"

# magic, version, padding, capture sequence, server monotonic timestamp,
# colour JPEG length, aligned-depth PNG length
_RGBD_HEADER = struct.Struct("!8sB7xQQII")
_MAX_COMPONENT_BYTES = 64 * 1024 * 1024
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class TeleRgbdFrame:
    """Encoded images and metadata from one RealSense capture."""

    sequence: int
    server_capture_monotonic_ns: int
    received_monotonic_ns: Optional[int]
    color_jpeg: bytes
    aligned_depth_png: bytes


def _validate_uint64(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")
    return value


BytesLike = Union[bytes, bytearray, memoryview]


def _coerce_component(name: str, value: BytesLike) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    encoded = bytes(value)
    if not encoded:
        raise ValueError(f"{name} must not be empty")
    if len(encoded) > _MAX_COMPONENT_BYTES:
        raise ValueError(f"{name} exceeds the {_MAX_COMPONENT_BYTES}-byte protocol limit")
    return encoded


def pack_rgbd_packet(
    sequence: int,
    server_capture_monotonic_ns: int,
    color_jpeg: BytesLike,
    aligned_depth_png: BytesLike,
) -> bytes:
    """Pack one colour JPEG and aligned uint16-depth PNG into one ZMQ message."""

    sequence = _validate_uint64("sequence", sequence)
    server_capture_monotonic_ns = _validate_uint64(
        "server_capture_monotonic_ns",
        server_capture_monotonic_ns,
    )
    color_jpeg = _coerce_component("color_jpeg", color_jpeg)
    aligned_depth_png = _coerce_component("aligned_depth_png", aligned_depth_png)
    header = _RGBD_HEADER.pack(
        RGBD_MAGIC,
        RGBD_VERSION,
        sequence,
        server_capture_monotonic_ns,
        len(color_jpeg),
        len(aligned_depth_png),
    )
    return header + color_jpeg + aligned_depth_png


def unpack_rgbd_packet(
    packet: BytesLike,
    *,
    received_monotonic_ns: Optional[int] = None,
) -> TeleRgbdFrame:
    """Validate and unpack a version-1 RGBD packet.

    ``received_monotonic_ns`` is recorded by the receiving host and is not part
    of the wire payload.  It is the timestamp consumers should use for local
    freshness checks; monotonic clocks on different hosts are not comparable.
    """

    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise TypeError("packet must be bytes-like")
    if received_monotonic_ns is not None:
        received_monotonic_ns = _validate_uint64(
            "received_monotonic_ns",
            received_monotonic_ns,
        )

    packet = bytes(packet)
    if len(packet) < _RGBD_HEADER.size:
        raise ValueError("RGBD packet is shorter than its header")

    (
        magic,
        version,
        sequence,
        server_capture_monotonic_ns,
        color_length,
        depth_length,
    ) = _RGBD_HEADER.unpack_from(packet)
    if magic != RGBD_MAGIC:
        raise ValueError("RGBD packet magic is invalid")
    if version != RGBD_VERSION:
        raise ValueError(
            f"Unsupported RGBD packet version {version}; expected {RGBD_VERSION}",
        )
    if not color_length or not depth_length:
        raise ValueError("RGBD packet contains an empty image component")
    if color_length > _MAX_COMPONENT_BYTES or depth_length > _MAX_COMPONENT_BYTES:
        raise ValueError("RGBD packet image component exceeds the protocol size limit")

    expected_length = _RGBD_HEADER.size + color_length + depth_length
    if len(packet) != expected_length:
        raise ValueError(
            f"RGBD packet length {len(packet)} does not match declared length {expected_length}",
        )

    color_start = _RGBD_HEADER.size
    depth_start = color_start + color_length
    return TeleRgbdFrame(
        sequence=sequence,
        server_capture_monotonic_ns=server_capture_monotonic_ns,
        received_monotonic_ns=received_monotonic_ns,
        color_jpeg=packet[color_start:depth_start],
        aligned_depth_png=packet[depth_start:],
    )
