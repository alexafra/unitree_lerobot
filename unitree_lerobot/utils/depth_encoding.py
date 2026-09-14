"""Shared Unitree aligned-depth encoding used by training and deployment."""

from __future__ import annotations

import numpy as np


CANONICAL_DEPTH_SCALE_M_PER_UNIT = 0.001
# Backwards-compatible public name. Processing paths canonicalize to the exact
# constant above rather than retaining the RealSense SDK's float32 expansion.
DEFAULT_DEPTH_SCALE_M_PER_UNIT = CANONICAL_DEPTH_SCALE_M_PER_UNIT
DEFAULT_DEPTH_NEAR_M = 0.25
DEFAULT_DEPTH_FAR_M = 1.0
DEPTH_SOURCE_KEY = "depth_0"
DEPTH_COLOR_SOURCE_KEY = "color_0"
DEPTH_OUTPUT_KEY = "depth_gray_view"
DEPTH_ENCODING = "linear_grayscale_replicated_rgb"


def _finite_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def canonicalize_depth_scale_m_per_unit(
    value: float,
    *,
    name: str = "scale_m_per_unit",
) -> float:
    """Validate a reported millimetre scale and return canonical ``0.001``.

    RealSense exposes its scale as a float32, which commonly arrives in JSON as
    ``0.0010000000474974513``.  That value and literal ``0.001`` describe the
    same float32.  Accept only that float32 value, then deliberately return the
    exact shared Python constant so collection, conversion, preview, and live
    deployment never retain subtly different scale spellings.
    """

    scale = _finite_float(value, name=name)
    if scale <= 0.0 or np.float32(scale) != np.float32(CANONICAL_DEPTH_SCALE_M_PER_UNIT):
        raise ValueError(
            f"{name} must equal the canonical {CANONICAL_DEPTH_SCALE_M_PER_UNIT} m/unit "
            f"at float32 precision, got {value!r}"
        )
    return CANONICAL_DEPTH_SCALE_M_PER_UNIT


def encode_depth_gray_rgb(
    depth_u16: np.ndarray,
    *,
    scale_m_per_unit: float,
    near_m: float = DEFAULT_DEPTH_NEAR_M,
    far_m: float = DEFAULT_DEPTH_FAR_M,
) -> np.ndarray:
    """Encode aligned uint16 depth as the three-channel uint8 GR00T view.

    Zero is reserved for invalid sensor pixels. Every nonzero source value maps
    to 1..255 after conversion to metres and fixed near/far normalization. The
    fixed metric mapping is intentionally independent of the contents of an
    individual frame.
    """

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected an HxW uint16 depth image; got shape={depth.shape}, dtype={depth.dtype}")

    scale = _finite_float(scale_m_per_unit, name="scale_m_per_unit")
    near = _finite_float(near_m, name="near_m")
    far = _finite_float(far_m, name="far_m")
    if scale <= 0:
        raise ValueError(f"scale_m_per_unit must be positive, got {scale}")
    if far <= near:
        raise ValueError(f"far_m ({far}) must be greater than near_m ({near})")

    depth_m = depth.astype(np.float32) * scale
    valid = depth != 0
    normalized = np.clip((depth_m - near) / (far - near), 0.0, 1.0)

    gray = np.zeros_like(depth, dtype=np.uint8)
    gray[valid] = 1 + np.round(254 * normalized[valid]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)
