"""Surface-normal encoding derived from aligned metric depth.

The same pure NumPy transform is used by offline dataset conversion and live
deployment so a surface-normal checkpoint sees identical geometry in both
paths.  Input depth must already be aligned to the color image whose pinhole
intrinsics are supplied here.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from unitree_lerobot.utils.depth_encoding import (
    DEFAULT_DEPTH_SCALE_M_PER_UNIT,
    DEPTH_COLOR_SOURCE_KEY,
    DEPTH_SOURCE_KEY,
)


SURFACE_NORMAL_SOURCE_KEY = DEPTH_SOURCE_KEY
SURFACE_NORMAL_COLOR_SOURCE_KEY = DEPTH_COLOR_SOURCE_KEY
SURFACE_NORMAL_OUTPUT_KEY = "surface_normals_view"
SURFACE_NORMAL_ENCODING = "camera_xyz_uint8"
SURFACE_NORMAL_ENCODING_VERSION = 1

SURFACE_NORMAL_COORDINATE_FRAME = "camera_optical_x_right_y_down_z_forward"
SURFACE_NORMAL_ORIENTATION = "camera_facing_dot_normal_point_lte_zero"
SURFACE_NORMAL_METHOD = "central_difference_3d"
DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M = 0.05


@dataclasses.dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole intrinsics for the image to which depth has been aligned."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


# Intel RealSense D435I 242322076480 color intrinsics at 640x480.  Depth in the
# source episodes is aligned to color_0, so the color—not raw depth—intrinsics
# apply to both offline conversion and deployment.
DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480 = PinholeIntrinsics(
    width=640,
    height=480,
    fx=605.421508789062,
    fy=605.590515136719,
    cx=321.856811523438,
    cy=242.249740600586,
)


def _finite_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def _validated_intrinsics(intrinsics: PinholeIntrinsics) -> PinholeIntrinsics:
    if not isinstance(intrinsics, PinholeIntrinsics):
        raise TypeError(f"intrinsics must be PinholeIntrinsics, got {type(intrinsics).__name__}")
    if isinstance(intrinsics.width, bool) or not isinstance(intrinsics.width, (int, np.integer)):
        raise ValueError(f"intrinsics.width must be an integer, got {intrinsics.width!r}")
    if isinstance(intrinsics.height, bool) or not isinstance(intrinsics.height, (int, np.integer)):
        raise ValueError(f"intrinsics.height must be an integer, got {intrinsics.height!r}")
    if intrinsics.width < 3 or intrinsics.height < 3:
        raise ValueError(f"intrinsics resolution must be at least 3x3, got {intrinsics.width}x{intrinsics.height}")

    fx = _finite_float(intrinsics.fx, name="intrinsics.fx")
    fy = _finite_float(intrinsics.fy, name="intrinsics.fy")
    cx = _finite_float(intrinsics.cx, name="intrinsics.cx")
    cy = _finite_float(intrinsics.cy, name="intrinsics.cy")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"intrinsics focal lengths must be positive, got fx={fx}, fy={fy}")
    if not (0 <= cx < intrinsics.width and 0 <= cy < intrinsics.height):
        raise ValueError(
            "intrinsics principal point must lie inside the image; "
            f"got cx={cx}, cy={cy}, resolution={intrinsics.width}x{intrinsics.height}"
        )
    return PinholeIntrinsics(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def encode_surface_normals_rgb(
    depth_u16: np.ndarray,
    *,
    scale_m_per_unit: float,
    intrinsics: PinholeIntrinsics = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    max_neighbor_depth_delta_m: float = DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
) -> np.ndarray:
    """Encode aligned uint16 depth as camera-frame XYZ surface normals.

    Four-neighbour central differences are computed after pinhole
    back-projection into metric 3D. Normals are oriented toward the camera and
    XYZ components in ``[-1, 1]`` map to ``[1, 255]``. ``[0, 0, 0]`` is
    reserved for invalid pixels, including the one-pixel image border, missing
    depth, degenerate geometry, and depth discontinuities larger than the
    configured metric threshold.
    """

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected an HxW uint16 depth image; got shape={depth.shape}, dtype={depth.dtype}")

    camera = _validated_intrinsics(intrinsics)
    expected_shape = (camera.height, camera.width)
    if depth.shape != expected_shape:
        raise ValueError(f"Depth shape {depth.shape} does not match intrinsics shape {expected_shape}")

    scale = _finite_float(scale_m_per_unit, name="scale_m_per_unit")
    discontinuity = _finite_float(
        max_neighbor_depth_delta_m,
        name="max_neighbor_depth_delta_m",
    )
    if scale <= 0:
        raise ValueError(f"scale_m_per_unit must be positive, got {scale}")
    if discontinuity <= 0:
        raise ValueError(f"max_neighbor_depth_delta_m must be positive, got {discontinuity}")

    depth_m = depth.astype(np.float32) * np.float32(scale)
    x_scale = (np.arange(camera.width, dtype=np.float32) - np.float32(camera.cx)) / np.float32(camera.fx)
    y_scale = (np.arange(camera.height, dtype=np.float32) - np.float32(camera.cy)) / np.float32(camera.fy)
    point_x = depth_m * x_scale[None, :]
    point_y = depth_m * y_scale[:, None]

    center_depth = depth_m[1:-1, 1:-1]
    left_depth = depth_m[1:-1, :-2]
    right_depth = depth_m[1:-1, 2:]
    up_depth = depth_m[:-2, 1:-1]
    down_depth = depth_m[2:, 1:-1]

    tangent_x_x = point_x[1:-1, 2:] - point_x[1:-1, :-2]
    tangent_x_y = point_y[1:-1, 2:] - point_y[1:-1, :-2]
    tangent_x_z = right_depth - left_depth
    tangent_y_x = point_x[2:, 1:-1] - point_x[:-2, 1:-1]
    tangent_y_y = point_y[2:, 1:-1] - point_y[:-2, 1:-1]
    tangent_y_z = down_depth - up_depth

    normal_x = tangent_x_y * tangent_y_z - tangent_x_z * tangent_y_y
    normal_y = tangent_x_z * tangent_y_x - tangent_x_x * tangent_y_z
    normal_z = tangent_x_x * tangent_y_y - tangent_x_y * tangent_y_x
    norm = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)

    source_valid = (
        (depth[1:-1, 1:-1] != 0)
        & (depth[1:-1, :-2] != 0)
        & (depth[1:-1, 2:] != 0)
        & (depth[:-2, 1:-1] != 0)
        & (depth[2:, 1:-1] != 0)
    )
    locally_continuous = (
        (np.abs(left_depth - center_depth) <= discontinuity)
        & (np.abs(right_depth - center_depth) <= discontinuity)
        & (np.abs(up_depth - center_depth) <= discontinuity)
        & (np.abs(down_depth - center_depth) <= discontinuity)
    )
    valid = source_valid & locally_continuous & np.isfinite(norm) & (norm > np.float32(1e-12))

    # The raw tangent cross product is normally away-facing (+Z for a flat
    # fronto-parallel surface). Flip it whenever it points away from the camera
    # origin, making the convention valid for sloped surfaces as well.
    center_x = point_x[1:-1, 1:-1]
    center_y = point_y[1:-1, 1:-1]
    dot_normal_point = normal_x * center_x + normal_y * center_y + normal_z * center_depth
    orientation_sign = np.where(dot_normal_point > 0, np.float32(-1.0), np.float32(1.0))
    safe_norm = np.where(valid, norm, np.float32(1.0))

    normals = np.stack(
        (
            orientation_sign * normal_x / safe_norm,
            orientation_sign * normal_y / safe_norm,
            orientation_sign * normal_z / safe_norm,
        ),
        axis=-1,
    )
    encoded_inner = 1 + np.rint(np.float32(127.0) * (np.clip(normals, -1.0, 1.0) + 1.0)).astype(np.uint8)

    encoded = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    encoded[1:-1, 1:-1][valid] = encoded_inner[valid]
    return np.ascontiguousarray(encoded)


def surface_normals_encoding_metadata(
    *,
    intrinsics: PinholeIntrinsics = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    max_neighbor_depth_delta_m: float = DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
) -> dict[str, object]:
    """Return the versioned, JSON-serializable contract for this encoding."""

    camera = _validated_intrinsics(intrinsics)
    discontinuity = _finite_float(
        max_neighbor_depth_delta_m,
        name="max_neighbor_depth_delta_m",
    )
    if discontinuity <= 0:
        raise ValueError(f"max_neighbor_depth_delta_m must be positive, got {discontinuity}")

    return {
        "source_key": SURFACE_NORMAL_SOURCE_KEY,
        "aligned_to": SURFACE_NORMAL_COLOR_SOURCE_KEY,
        "feature_key": f"observation.images.{SURFACE_NORMAL_OUTPUT_KEY}",
        "encoding": SURFACE_NORMAL_ENCODING,
        "encoding_version": SURFACE_NORMAL_ENCODING_VERSION,
        "depth_scale_source": "episode.info.depth.scale_m_per_unit",
        "default_scale_m_per_unit": DEFAULT_DEPTH_SCALE_M_PER_UNIT,
        "intrinsics": {
            "model": "pinhole",
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        },
        "axis_order": ["x", "y", "z"],
        "coordinate_frame": SURFACE_NORMAL_COORDINATE_FRAME,
        "orientation": SURFACE_NORMAL_ORIENTATION,
        "method": SURFACE_NORMAL_METHOD,
        "neighbor_offset_pixels": 1,
        "max_neighbor_depth_delta_m": discontinuity,
        "invalid_value": [0, 0, 0],
        "valid_component_range": [1, 255],
    }
