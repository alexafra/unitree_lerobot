"""Pure GR00T/Unitree observation and action contract checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.utils.depth_encoding import (
    DEPTH_ENCODING,
    DEPTH_OUTPUT_KEY,
    DEPTH_SOURCE_KEY,
)
from unitree_lerobot.utils.surface_normal_encoding import (
    DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
    PinholeIntrinsics,
    SURFACE_NORMAL_OUTPUT_KEY,
    surface_normals_encoding_metadata,
)


TASKS = {
    "pick-toothpaste": "pick up the cylinder toothpaste.",
    "put-toothpaste": "put down the cylinder toothpaste.",
    "pick-red-cup": "pick up the red cup.",
    "put-red-cup": "put down the red cup.",
}

COLOUR_VIDEO_KEYS = ("ego_view",)
RGBD_VIDEO_KEYS = ("ego_view", DEPTH_OUTPUT_KEY)
SURFACE_NORMAL_VIDEO_KEYS = ("ego_view", SURFACE_NORMAL_OUTPUT_KEY)
SUPPORTED_VIDEO_KEYS = (COLOUR_VIDEO_KEYS, RGBD_VIDEO_KEYS, SURFACE_NORMAL_VIDEO_KEYS)
# Backwards-compatible public name used by existing callers/tests.
VIDEO_KEYS = COLOUR_VIDEO_KEYS
STATE_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand")
ACTION_KEYS = STATE_KEYS
LANGUAGE_KEYS = ("annotation.human.task_description",)
EXPECTED_TRAINING_TAG = "new_embodiment"
EXPECTED_ROBOT_TYPE = "Unitree_G1_Dex3_HeadOnly"
EXPECTED_EGO_VIEW_SHAPE = [480, 640, 3]
EXPECTED_DEPTH_VIEW_SHAPE = [480, 640, 3]
EXPECTED_SURFACE_NORMAL_VIEW_SHAPE = [480, 640, 3]
EXPECTED_ACTION_OUTPUT_CONTRACT = {
    "semantics": "absolute_joint_position",
    "use_relative_action": True,
    "relative_keys_decoded_to_absolute": ["left_arm", "right_arm"],
}
EXPECTED_JOINT_NAMES = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]
ARM_JOINT_NAMES = tuple(EXPECTED_JOINT_NAMES[:14])
LEFT_HAND_JOINT_NAMES = tuple(EXPECTED_JOINT_NAMES[14:21])
RIGHT_HAND_JOINT_NAMES = tuple(EXPECTED_JOINT_NAMES[21:28])

ARM_DOF = 14
HAND_DOF = 7
CONTROL_HZ = 30.0
INITIALIZATION_MODES = ("measured", "xr-home", "pose-file")
INITIAL_POSE_SCHEMA_VERSION = 1

# These are deliberately fixed deployment ceilings, not tuning flags.  They need
# hardware qualification before being relaxed.
# CHANGEDSAFETY: original local adapter default was 0.05 rad; current is 0.10 rad.
# This is the raw 30 Hz ceiling when conditioning is disabled and remains a
# hard final-command backstop when conditioning is enabled. The 100 Hz
# conditioner derives a stricter 0.03 rad ceiling to preserve the same 3 rad/s
# ordinary effective slew; a minimum inward recovery from a measured-only
# tolerance may exceed it but remains below this hard backstop. It is not an
# official Unitree velocity limit.
MAX_ARM_STEP_RAD = 0.10
# CHANGEDSAFETY: original local adapter default was 0.10 rad; current is 0.60 rad.
# This is the raw 30 Hz ceiling when conditioning is disabled and remains a
# hard final-command backstop when conditioning is enabled. The default 100 Hz
# conditioner uses the stricter per-joint velocity maxima from the checked-in
# Unitree Dex3 URDF (6.857 rad/s for thumb0, 12 rad/s otherwise); a minimum
# inward recovery from a measured-only tolerance may exceed that ordinary
# conditioner ceiling but remains below this hard backstop. This 0.60-rad
# backstop itself is not an official Unitree velocity limit.
MAX_HAND_STEP_RAD = 0.60
JOINT_LIMIT_MARGIN_RAD = 0.03
HAND_LIMIT_TOLERANCE_RAD = 0.01
MEASURED_LIMIT_TOLERANCE_RAD = 0.2

# OFFICIAL: Unitree-derived g1_body29_hand14.urdf position limits, reordered into
# the G1_29_JointArmIndex / recorded-dataset action order.
ARM_LOWER = np.array(
    [
        -3.0892,
        -1.5882,
        -2.6180,
        -1.0472,
        -1.972222054,
        -1.614429558,
        -1.614429558,
        -3.0892,
        -2.2515,
        -2.6180,
        -1.0472,
        -1.972222054,
        -1.614429558,
        -1.614429558,
    ],
    dtype=np.float64,
)
# OFFICIAL: Unitree-derived g1_body29_hand14.urdf upper arm position bounds.
ARM_UPPER = np.array(
    [
        2.6704,
        2.2515,
        2.6180,
        2.0944,
        1.972222054,
        1.614429558,
        1.614429558,
        2.6704,
        1.5882,
        2.6180,
        2.0944,
        1.972222054,
        1.614429558,
        1.614429558,
    ],
    dtype=np.float64,
)
# OFFICIAL: Unitree-derived g1_body29_hand14.urdf left Dex3 lower position bounds.
LEFT_HAND_LOWER = np.array(
    [-1.04719755, -0.72431163, 0.0, -1.57079632, -1.74532925, -1.57079632, -1.74532925],
    dtype=np.float64,
)
# OFFICIAL: Unitree-derived g1_body29_hand14.urdf left Dex3 upper position bounds.
LEFT_HAND_UPPER = np.array([1.04719755, 1.04719755, 1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# OFFICIAL: Unitree-derived g1_body29_hand14.urdf right Dex3 lower position bounds.
RIGHT_HAND_LOWER = np.array([-1.04719755, -1.04719755, -1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# OFFICIAL: Unitree-derived g1_body29_hand14.urdf right Dex3 upper position bounds.
RIGHT_HAND_UPPER = np.array(
    [1.04719755, 0.72431163, 0.0, 1.57079632, 1.74532925, 1.57079632, 1.74532925],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ModelContract:
    action_horizon: int
    video_keys: tuple[str, ...] = COLOUR_VIDEO_KEYS

    @property
    def requires_depth(self) -> bool:
        """Whether the live camera must provide atomic aligned depth."""

        return self.video_keys in (RGBD_VIDEO_KEYS, SURFACE_NORMAL_VIDEO_KEYS)

    @property
    def requires_depth_gray(self) -> bool:
        return self.video_keys == RGBD_VIDEO_KEYS

    @property
    def requires_surface_normals(self) -> bool:
        return self.video_keys == SURFACE_NORMAL_VIDEO_KEYS


@dataclass(frozen=True)
class DepthEncodingContract:
    near_m: float
    far_m: float


@dataclass(frozen=True)
class SurfaceNormalEncodingContract:
    intrinsics: PinholeIntrinsics
    max_neighbor_depth_delta_m: float


@dataclass(frozen=True)
class ActionChunk:
    arm: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray

    @property
    def length(self) -> int:
        return int(self.arm.shape[0])


@dataclass(frozen=True)
class InitializationSpec:
    """Validated initialization targets; ``None`` means preserve measured q."""

    mode: str
    label: str
    arm: np.ndarray | None
    left_hand: np.ndarray | None
    right_hand: np.ndarray | None

    @property
    def moves(self) -> bool:
        return any(target is not None for target in (self.arm, self.left_hand, self.right_hand))

    @property
    def moves_hands(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


def _validate_depth_metadata(contract: dict[str, Any]) -> DepthEncodingContract:
    video_shapes = contract.get("video_shapes")
    if not isinstance(video_shapes, dict):
        raise DeploymentError("Deployment dataset contract has no video_shapes for the RGBD model")
    expected_shapes = {
        "ego_view": EXPECTED_EGO_VIEW_SHAPE,
        DEPTH_OUTPUT_KEY: EXPECTED_DEPTH_VIEW_SHAPE,
    }
    for key, expected_shape in expected_shapes.items():
        if video_shapes.get(key) != expected_shape:
            raise DeploymentError(
                f"Deployment dataset contract mismatch for video_shapes.{key}: "
                f"got {video_shapes.get(key)!r}, expected {expected_shape!r}"
            )

    encoding = contract.get("depth_encoding")
    if not isinstance(encoding, dict):
        raise DeploymentError("Deployment dataset contract has no depth_encoding for the RGBD model")
    expected = {
        "source_key": DEPTH_SOURCE_KEY,
        "feature_key": f"observation.images.{DEPTH_OUTPUT_KEY}",
        "encoding": DEPTH_ENCODING,
        "invalid_value": 0,
        "valid_value_range": [1, 255],
    }
    for field, value in expected.items():
        if encoding.get(field) != value:
            raise DeploymentError(
                f"Unsupported depth encoding for {field}: got {encoding.get(field)!r}, expected {value!r}"
            )
    try:
        near_m = float(encoding["near_m"])
        far_m = float(encoding["far_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("Depth encoding near_m/far_m are missing or non-numeric") from exc
    if not np.isfinite(near_m) or not np.isfinite(far_m) or near_m < 0.0 or far_m <= near_m:
        raise DeploymentError(f"Invalid depth encoding bounds: near_m={near_m!r}, far_m={far_m!r}")
    return DepthEncodingContract(near_m=near_m, far_m=far_m)


def _validate_surface_normal_metadata(contract: dict[str, Any]) -> SurfaceNormalEncodingContract:
    video_shapes = contract.get("video_shapes")
    if not isinstance(video_shapes, dict):
        raise DeploymentError("Deployment dataset contract has no video_shapes for the surface-normal model")
    expected_shapes = {
        "ego_view": EXPECTED_EGO_VIEW_SHAPE,
        SURFACE_NORMAL_OUTPUT_KEY: EXPECTED_SURFACE_NORMAL_VIEW_SHAPE,
    }
    for key, expected_shape in expected_shapes.items():
        if video_shapes.get(key) != expected_shape:
            raise DeploymentError(
                f"Deployment dataset contract mismatch for video_shapes.{key}: "
                f"got {video_shapes.get(key)!r}, expected {expected_shape!r}"
            )

    encoding = contract.get("surface_normals_encoding")
    if not isinstance(encoding, dict):
        raise DeploymentError(
            "Deployment dataset contract has no surface_normals_encoding for the surface-normal model"
        )
    expected = surface_normals_encoding_metadata()
    for field, value in expected.items():
        if encoding.get(field) != value:
            raise DeploymentError(
                f"Unsupported surface-normal encoding for {field}: "
                f"got {encoding.get(field)!r}, expected {value!r}"
            )
    return SurfaceNormalEncodingContract(
        intrinsics=DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        max_neighbor_depth_delta_m=DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
    )


def validate_policy_metadata(
    metadata: dict[str, Any],
    *,
    requires_depth: bool = False,
    requires_surface_normals: bool = False,
) -> DepthEncodingContract | SurfaceNormalEncodingContract | None:
    if metadata.get("protocol_version") != 1:
        raise DeploymentError(f"Unsupported GR00T deployment protocol {metadata.get('protocol_version')!r}")
    if metadata.get("embodiment_tag") != EXPECTED_TRAINING_TAG:
        raise DeploymentError(
            f"This G1/Dex3 adapter requires GR00T training tag "
            f"'{EXPECTED_TRAINING_TAG}', got {metadata.get('embodiment_tag')!r}"
        )
    contract = metadata.get("dataset_contract")
    if not isinstance(contract, dict):
        raise DeploymentError(
            "GR00T server has no deployment dataset contract; start it with --deployment-dataset-path"
        )
    expected = {
        "robot_type": EXPECTED_ROBOT_TYPE,
        "fps": CONTROL_HZ,
        "observation_state_names": EXPECTED_JOINT_NAMES,
        "action_names": EXPECTED_JOINT_NAMES,
        "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise DeploymentError(
                f"Deployment dataset contract mismatch for {field}: got {contract.get(field)!r}, expected {value!r}"
            )
    action_output = metadata.get("action_output_contract")
    if not isinstance(action_output, dict):
        raise DeploymentError(
            "GR00T server has no checkpoint action output contract; update the server before deployment"
        )
    for field, value in EXPECTED_ACTION_OUTPUT_CONTRACT.items():
        if action_output.get(field) != value:
            raise DeploymentError(
                "Unsupported checkpoint action output contract for "
                f"{field}: got {action_output.get(field)!r}, expected {value!r}"
            )
    if requires_depth and requires_surface_normals:
        raise DeploymentError("A checkpoint cannot request depth_gray_view and surface_normals_view together")
    if requires_surface_normals:
        return _validate_surface_normal_metadata(contract)
    if requires_depth:
        return _validate_depth_metadata(contract)
    video_shapes = contract.get("video_shapes")
    if isinstance(video_shapes, dict) and video_shapes.get("ego_view") != EXPECTED_EGO_VIEW_SHAPE:
        raise DeploymentError(
            "Deployment dataset contract mismatch for video_shapes.ego_view: "
            f"got {video_shapes.get('ego_view')!r}, expected {EXPECTED_EGO_VIEW_SHAPE!r}"
        )
    return None


def _config_field(config: dict[str, Any], modality: str, field: str) -> Any:
    try:
        value = config[modality][field]
    except (KeyError, TypeError) as exc:
        raise DeploymentError(f"Server modality config is missing {modality}.{field}") from exc
    return value


def validate_model_contract(config: dict[str, Any]) -> ModelContract:
    """Accept only the supported G1/Dex3 colour and geometry contracts."""

    expected = {
        "state": STATE_KEYS,
        "action": ACTION_KEYS,
        "language": LANGUAGE_KEYS,
    }
    video_keys = tuple(_config_field(config, "video", "modality_keys"))
    if video_keys not in SUPPORTED_VIDEO_KEYS:
        raise DeploymentError(
            f"Unsupported video keys from GR00T server: {video_keys}; expected exactly "
            f"{COLOUR_VIDEO_KEYS}, {RGBD_VIDEO_KEYS}, or {SURFACE_NORMAL_VIDEO_KEYS}."
        )
    for modality, keys in expected.items():
        actual = tuple(_config_field(config, modality, "modality_keys"))
        if actual != keys:
            raise DeploymentError(f"Unsupported {modality} keys from GR00T server: {actual}; expected {keys}.")

    for modality in ("video", "state", "language"):
        delta_indices = list(_config_field(config, modality, "delta_indices"))
        if delta_indices != [0]:
            raise DeploymentError(f"Unsupported {modality} history {delta_indices}; this runner supplies only index 0")

    action_indices = list(_config_field(config, "action", "delta_indices"))
    if not action_indices or action_indices != list(range(len(action_indices))):
        raise DeploymentError(f"Unsupported action delta indices {action_indices}; expected consecutive indices from 0")
    action_configs = _config_field(config, "action", "action_configs")
    if not isinstance(action_configs, list) or len(action_configs) != len(ACTION_KEYS):
        raise DeploymentError("Server action config must describe all four joint action keys")
    for key, action_config in zip(ACTION_KEYS, action_configs, strict=True):
        if not isinstance(action_config, dict):
            raise DeploymentError(f"Server action config for '{key}' is malformed")
        representation = action_config.get("rep")
        action_type = action_config.get("type")
        action_format = action_config.get("format")
        state_key = action_config.get("state_key")
        # GR00T's policy postprocessor converts RELATIVE actions back to absolute
        # joint positions.  DELTA has different semantics and is not converted by
        # that path, so treating it as an absolute position here would be unsafe.
        if representation not in {"RELATIVE", "ABSOLUTE"}:
            raise DeploymentError(f"Unsupported action representation for '{key}': {representation!r}")
        if action_type != "NON_EEF" or action_format != "DEFAULT":
            raise DeploymentError(
                f"Action '{key}' is not a default joint-space action: type={action_type!r}, format={action_format!r}"
            )
        if state_key not in (None, key):
            raise DeploymentError(f"Action '{key}' refers to unexpected state key {state_key!r}")
    return ModelContract(action_horizon=len(action_indices), video_keys=video_keys)


def make_observation(
    rgb: np.ndarray,
    arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    instruction: str,
    *,
    video_keys: tuple[str, ...] = COLOUR_VIDEO_KEYS,
    depth_gray: np.ndarray | None = None,
    surface_normals: np.ndarray | None = None,
    allow_custom_instruction: bool = False,
) -> dict[str, Any]:
    """Build exactly the video/state/language inputs selected by the checkpoint."""

    rgb = np.asarray(rgb)
    arm = np.asarray(arm)
    left_hand = np.asarray(left_hand)
    right_hand = np.asarray(right_hand)
    if list(rgb.shape) != EXPECTED_EGO_VIEW_SHAPE or rgb.dtype != np.uint8:
        raise DeploymentError(
            f"Expected uint8 RGB image with shape {tuple(EXPECTED_EGO_VIEW_SHAPE)}, got {rgb.shape} {rgb.dtype}"
        )
    if arm.shape != (ARM_DOF,):
        raise DeploymentError(f"Expected 14 arm joints, got {arm.shape}")
    if left_hand.shape != (HAND_DOF,) or right_hand.shape != (HAND_DOF,):
        raise DeploymentError(f"Expected two 7-DoF hands, got {left_hand.shape} and {right_hand.shape}")
    if not all(np.all(np.isfinite(values)) for values in (arm, left_hand, right_hand)):
        raise DeploymentError("Robot observation contains NaN or infinity")
    if not isinstance(instruction, str) or not instruction or len(instruction) > 256:
        raise DeploymentError("Instruction must be a non-empty string of at most 256 characters")
    if any(ord(character) < 32 for character in instruction):
        raise DeploymentError("Instruction contains control characters")
    if not allow_custom_instruction and instruction not in TASKS.values():
        raise DeploymentError("Instruction is not in the trained task allowlist")

    if video_keys not in SUPPORTED_VIDEO_KEYS:
        raise DeploymentError(f"Cannot build an observation for unsupported video keys {video_keys}")
    video = {"ego_view": np.ascontiguousarray(rgb)[None, None]}
    if video_keys == RGBD_VIDEO_KEYS:
        if depth_gray is None:
            raise DeploymentError("RGBD checkpoint requires depth_gray_view")
        if surface_normals is not None:
            raise DeploymentError("Depth checkpoint must not receive surface_normals_view")
        depth_gray = np.asarray(depth_gray)
        if list(depth_gray.shape) != EXPECTED_DEPTH_VIEW_SHAPE or depth_gray.dtype != np.uint8:
            raise DeploymentError(
                "Expected uint8 depth_gray_view with shape "
                f"{tuple(EXPECTED_DEPTH_VIEW_SHAPE)}, got {depth_gray.shape} {depth_gray.dtype}"
            )
        video[DEPTH_OUTPUT_KEY] = np.ascontiguousarray(depth_gray)[None, None]
    elif video_keys == SURFACE_NORMAL_VIDEO_KEYS:
        if surface_normals is None:
            raise DeploymentError("Surface-normal checkpoint requires surface_normals_view")
        if depth_gray is not None:
            raise DeploymentError("Surface-normal checkpoint must not receive depth_gray_view")
        surface_normals = np.asarray(surface_normals)
        if (
            list(surface_normals.shape) != EXPECTED_SURFACE_NORMAL_VIEW_SHAPE
            or surface_normals.dtype != np.uint8
        ):
            raise DeploymentError(
                "Expected uint8 surface_normals_view with shape "
                f"{tuple(EXPECTED_SURFACE_NORMAL_VIEW_SHAPE)}, got "
                f"{surface_normals.shape} {surface_normals.dtype}"
            )
        video[SURFACE_NORMAL_OUTPUT_KEY] = np.ascontiguousarray(surface_normals)[None, None]
    elif depth_gray is not None or surface_normals is not None:
        raise DeploymentError("Colour-only checkpoint must not receive a geometry view")

    return {
        "video": video,
        "state": {
            "left_arm": arm[:7].astype(np.float32)[None, None],
            "right_arm": arm[7:].astype(np.float32)[None, None],
            "left_hand": left_hand.astype(np.float32)[None, None],
            "right_hand": right_hand.astype(np.float32)[None, None],
        },
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def _numeric_action(action: dict[str, Any], key: str, model_horizon: int) -> np.ndarray:
    if key not in action:
        raise DeploymentError(f"Model action is missing '{key}'")
    value = np.asarray(action[key])
    if value.dtype.kind not in "iuf":
        raise DeploymentError(f"Action '{key}' has unsafe dtype {value.dtype}")
    if value.shape != (1, model_horizon, 7):
        raise DeploymentError(f"Action '{key}' must have shape (1, {model_horizon}, 7), got {value.shape}")
    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise DeploymentError(f"Action '{key}' contains NaN or infinity")
    return value[0]


def _check_limits(
    name: str,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    joint_names: tuple[str, ...],
    lower_constant: str,
    upper_constant: str,
    margin: float = 0.0,
    tolerance: float = 0.0,
    margin_constant: str | None = None,
    tolerance_constant: str | None = None,
) -> None:
    safe_lower = lower + margin - tolerance
    safe_upper = upper - margin + tolerance
    bad = np.argwhere((values < safe_lower) | (values > safe_upper))
    if bad.size:
        step, joint = (int(index) for index in bad[0])
        value = float(values[step, joint])
        lower_violation = value < safe_lower[joint]
        boundary_constant = lower_constant if lower_violation else upper_constant
        boundary = float(lower[joint] if lower_violation else upper[joint])
        effective_boundary = float(safe_lower[joint] if lower_violation else safe_upper[joint])
        direction = "minimum" if lower_violation else "maximum"
        terms = [f"{boundary_constant}[{joint}]={boundary:.4f} rad"]
        if margin_constant is not None:
            operator = "+" if lower_violation else "-"
            terms.append(f"{operator} {margin_constant}={margin:.4f} rad")
        if tolerance_constant is not None:
            operator = "-" if lower_violation else "+"
            terms.append(f"{operator} {tolerance_constant}={tolerance:.4f} rad")
        raise DeploymentError(
            f"{name} target is outside its safety-margined joint range at step {step}, "
            f"joint {joint} ({joint_names[joint]}): {value:.4f} rad; effective {direction} "
            f"is {effective_boundary:.4f} rad from {' '.join(terms)}"
        )


def _initial_pose_vector(
    value: Any,
    *,
    field: str,
    size: int,
    allow_measured: bool,
) -> np.ndarray | None:
    if allow_measured and value == "measured":
        return None
    if not isinstance(value, list) or len(value) != size:
        measured_hint = " or the literal 'measured'" if allow_measured else ""
        raise DeploymentError(f"Initial pose field '{field}' must contain exactly {size} numbers{measured_hint}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise DeploymentError(f"Initial pose field '{field}' must contain only numbers (not booleans)")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise DeploymentError(f"Initial pose field '{field}' must contain only numbers")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise DeploymentError(f"Initial pose field '{field}' contains NaN or infinity")
    return array


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"Initial pose JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def validate_initialization_spec(spec: InitializationSpec) -> None:
    if spec.mode not in INITIALIZATION_MODES:
        raise DeploymentError(f"Unsupported initialization mode {spec.mode!r}")
    if not isinstance(spec.label, str) or not spec.label.strip() or len(spec.label) > 120:
        raise DeploymentError("Initialization label must be a non-empty string of at most 120 characters")
    if spec.mode == "measured" and any(target is not None for target in (spec.arm, spec.left_hand, spec.right_hand)):
        raise DeploymentError("Measured initialization must preserve every measured target")
    if spec.mode == "xr-home":
        expected_sizes = (ARM_DOF, HAND_DOF, HAND_DOF)
        for name, target, size in zip(
            ("arm", "left hand", "right hand"),
            (spec.arm, spec.left_hand, spec.right_hand),
            expected_sizes,
            strict=True,
        ):
            if target is None or not np.array_equal(np.asarray(target), np.zeros(size)):
                raise DeploymentError(f"XR-home {name} target must be exactly joint zero")
    if spec.mode == "pose-file":
        if spec.arm is None:
            raise DeploymentError("Pose-file initialization requires an explicit arm target")
        if (spec.left_hand is None) != (spec.right_hand is None):
            raise DeploymentError("Pose-file initialization must preserve both hands or explicitly target both")
    targets = (
        (
            "arm",
            spec.arm,
            ARM_DOF,
            ARM_LOWER,
            ARM_UPPER,
            ARM_JOINT_NAMES,
            "ARM_LOWER",
            "ARM_UPPER",
            JOINT_LIMIT_MARGIN_RAD,
            0.0,
            "JOINT_LIMIT_MARGIN_RAD",
            None,
        ),
        (
            "left hand",
            spec.left_hand,
            HAND_DOF,
            LEFT_HAND_LOWER,
            LEFT_HAND_UPPER,
            LEFT_HAND_JOINT_NAMES,
            "LEFT_HAND_LOWER",
            "LEFT_HAND_UPPER",
            0.0,
            HAND_LIMIT_TOLERANCE_RAD,
            None,
            "HAND_LIMIT_TOLERANCE_RAD",
        ),
        (
            "right hand",
            spec.right_hand,
            HAND_DOF,
            RIGHT_HAND_LOWER,
            RIGHT_HAND_UPPER,
            RIGHT_HAND_JOINT_NAMES,
            "RIGHT_HAND_LOWER",
            "RIGHT_HAND_UPPER",
            0.0,
            HAND_LIMIT_TOLERANCE_RAD,
            None,
            "HAND_LIMIT_TOLERANCE_RAD",
        ),
    )
    for (
        name,
        target,
        size,
        lower,
        upper,
        joint_names,
        lower_constant,
        upper_constant,
        margin,
        tolerance,
        margin_constant,
        tolerance_constant,
    ) in targets:
        if target is None:
            continue
        target = np.asarray(target)
        if target.shape != (size,) or target.dtype.kind not in "iuf" or not np.all(np.isfinite(target)):
            raise DeploymentError(f"Initialization {name} target is not a finite {size}-element numeric vector")
        _check_limits(
            name,
            target.astype(np.float64, copy=False)[None],
            lower,
            upper,
            joint_names=joint_names,
            lower_constant=lower_constant,
            upper_constant=upper_constant,
            margin=margin,
            tolerance=tolerance,
            margin_constant=margin_constant,
            tolerance_constant=tolerance_constant,
        )


def load_initialization_spec(
    mode: str,
    *,
    task_name: str,
    pose_file: str | Path | None = None,
) -> InitializationSpec:
    """Load one explicit initialization choice without consulting robot state.

    ``xr-home`` faithfully reproduces XR startup targets: fourteen arm zeros and
    seven zeros for each Dex3 hand.  ``pose-file`` is deliberately task-bound;
    its hand policy is either explicit for both hands or preserves both measured
    hand poses.
    """

    if mode not in INITIALIZATION_MODES:
        raise DeploymentError(f"Unsupported initialization mode {mode!r}; expected one of {INITIALIZATION_MODES}")
    if mode == "pose-file" and task_name not in TASKS:
        raise DeploymentError("Pose-file initialization requires one of the exact trained tasks")
    if mode != "pose-file" and pose_file is not None:
        raise DeploymentError("--initial-pose-file is valid only with --initialization pose-file")

    if mode == "measured":
        return InitializationSpec(
            mode=mode,
            label="preserve freshly measured pose",
            arm=None,
            left_hand=None,
            right_hand=None,
        )
    if mode == "xr-home":
        spec = InitializationSpec(
            mode=mode,
            label="XR joint-zero home (arms and both hands)",
            arm=np.zeros(ARM_DOF, dtype=np.float64),
            left_hand=np.zeros(HAND_DOF, dtype=np.float64),
            right_hand=np.zeros(HAND_DOF, dtype=np.float64),
        )
        validate_initialization_spec(spec)
        return spec

    if pose_file is None:
        raise DeploymentError("--initialization pose-file requires --initial-pose-file")
    path = Path(pose_file).expanduser()
    try:
        if not path.is_file():
            raise DeploymentError(f"Initial pose file does not exist or is not a regular file: {path}")
        if path.stat().st_size > 64 * 1024:
            raise DeploymentError(f"Initial pose file is unexpectedly large: {path}")
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except DeploymentError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"Could not read initial pose JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise DeploymentError("Initial pose JSON must be an object")
    required = {
        "schema_version",
        "name",
        "robot_type",
        "task",
        "instruction",
        "joint_names",
        "arm",
        "hands",
        "source",
    }
    missing = sorted(required - document.keys())
    unexpected = sorted(document.keys() - required)
    if missing or unexpected:
        raise DeploymentError(f"Initial pose JSON fields are not exact (missing={missing}, unexpected={unexpected})")
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != INITIAL_POSE_SCHEMA_VERSION
    ):
        raise DeploymentError(
            f"Initial pose schema_version must be {INITIAL_POSE_SCHEMA_VERSION}, got {document['schema_version']!r}"
        )
    name = document["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise DeploymentError("Initial pose name must be a non-empty string of at most 120 characters")
    if document["robot_type"] != EXPECTED_ROBOT_TYPE:
        raise DeploymentError(
            f"Initial pose robot_type must be {EXPECTED_ROBOT_TYPE!r}, got {document['robot_type']!r}"
        )
    if document["task"] != task_name or document["instruction"] != TASKS[task_name]:
        raise DeploymentError("Initial pose task/instruction does not exactly match the selected trained task")
    if document["joint_names"] != EXPECTED_JOINT_NAMES:
        raise DeploymentError("Initial pose joint_names do not exactly match the deployment joint order")

    source = document["source"]
    if not isinstance(source, dict) or set(source) != {"dataset_path", "episode_index", "frame_index"}:
        raise DeploymentError("Initial pose source must contain exactly dataset_path, episode_index, and frame_index")
    dataset_path = source["dataset_path"]
    if not isinstance(dataset_path, str) or not dataset_path or not Path(dataset_path).is_absolute():
        raise DeploymentError("Initial pose source.dataset_path must be a non-empty absolute path")
    episode_index = source["episode_index"]
    frame_index = source["frame_index"]
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise DeploymentError("Initial pose source.episode_index must be a non-negative integer")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index != 0:
        raise DeploymentError("Initial pose source.frame_index must be integer 0")

    hands = document["hands"]
    if not isinstance(hands, dict) or hands.get("policy") not in {"measured", "explicit"}:
        raise DeploymentError("Initial pose hands.policy must be exactly 'measured' or 'explicit'")
    if hands["policy"] == "measured":
        if set(hands) != {"policy"}:
            raise DeploymentError("Measured-hands pose must contain only hands.policy")
        left_hand = None
        right_hand = None
    else:
        if set(hands) != {"policy", "left", "right"}:
            raise DeploymentError("Explicit-hands pose must contain exactly hands.policy, hands.left, and hands.right")
        left_hand = _initial_pose_vector(hands["left"], field="hands.left", size=HAND_DOF, allow_measured=False)
        right_hand = _initial_pose_vector(hands["right"], field="hands.right", size=HAND_DOF, allow_measured=False)

    spec = InitializationSpec(
        mode=mode,
        label=name.strip(),
        arm=_initial_pose_vector(document["arm"], field="arm", size=ARM_DOF, allow_measured=False),
        left_hand=left_hand,
        right_hand=right_hand,
    )
    validate_initialization_spec(spec)
    return spec


def validate_measured_state(
    arm: np.ndarray,
    arm_dq: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> None:
    """Reject malformed or physically implausible DDS state before it is reused."""

    arrays = {
        "arm": (np.asarray(arm, dtype=np.float64), (ARM_DOF,)),
        "arm velocity": (np.asarray(arm_dq, dtype=np.float64), (ARM_DOF,)),
        "left hand": (np.asarray(left_hand, dtype=np.float64), (HAND_DOF,)),
        "right hand": (np.asarray(right_hand, dtype=np.float64), (HAND_DOF,)),
    }
    for name, (values, expected_shape) in arrays.items():
        if values.shape != expected_shape:
            raise DeploymentError(f"Measured {name} has shape {values.shape}; expected {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise DeploymentError(f"Measured {name} contains NaN or infinity")

    limits = (
        ("arm", arrays["arm"][0], ARM_LOWER, ARM_UPPER, ARM_JOINT_NAMES, "ARM_LOWER", "ARM_UPPER"),
        (
            "left hand",
            arrays["left hand"][0],
            LEFT_HAND_LOWER,
            LEFT_HAND_UPPER,
            LEFT_HAND_JOINT_NAMES,
            "LEFT_HAND_LOWER",
            "LEFT_HAND_UPPER",
        ),
        (
            "right hand",
            arrays["right hand"][0],
            RIGHT_HAND_LOWER,
            RIGHT_HAND_UPPER,
            RIGHT_HAND_JOINT_NAMES,
            "RIGHT_HAND_LOWER",
            "RIGHT_HAND_UPPER",
        ),
    )
    for name, values, lower, upper, joint_names, lower_constant, upper_constant in limits:
        bad = np.flatnonzero(
            (values < lower - MEASURED_LIMIT_TOLERANCE_RAD) | (values > upper + MEASURED_LIMIT_TOLERANCE_RAD)
        )
        if bad.size:
            joint = int(bad[0])
            lower_violation = values[joint] < lower[joint] - MEASURED_LIMIT_TOLERANCE_RAD
            boundary_constant = lower_constant if lower_violation else upper_constant
            boundary = float(lower[joint] if lower_violation else upper[joint])
            effective_boundary = boundary + (
                -MEASURED_LIMIT_TOLERANCE_RAD if lower_violation else MEASURED_LIMIT_TOLERANCE_RAD
            )
            direction = "minimum" if lower_violation else "maximum"
            raise DeploymentError(
                f"Measured {name} joint {joint} ({joint_names[joint]}) is outside its physical range: "
                f"{values[joint]:.4f} rad; effective {direction} is {effective_boundary:.4f} rad from "
                f"{boundary_constant}[{joint}]={boundary:.4f} rad "
                f"{'-' if lower_violation else '+'} MEASURED_LIMIT_TOLERANCE_RAD="
                f"{MEASURED_LIMIT_TOLERANCE_RAD:.4f} rad"
            )


def _check_step_size(
    name: str,
    values: np.ndarray,
    current: np.ndarray,
    max_step: float,
    *,
    joint_names: tuple[str, ...],
    constant_name: str,
) -> None:
    deltas = np.diff(np.vstack((current, values)), axis=0)
    bad = np.argwhere(np.abs(deltas) > max_step)
    if bad.size:
        step, joint = (int(index) for index in bad[0])
        raise DeploymentError(
            f"{name} target jump is too large at step {step}, joint {joint} "
            f"({joint_names[joint]}): {deltas[step, joint]:+.4f} rad; "
            f"{constant_name}={max_step:.4f} rad"
        )


def validate_action_chunk_limits(chunk: ActionChunk) -> None:
    """Validate policy-action structure, finiteness, and absolute joint limits.

    This deliberately excludes target-to-target slew.  A deployment-side
    conditioner may turn a discontinuous but finite, in-range policy plan into
    the commands that are actually published.  Those final commands must still
    pass :func:`validate_action_chunk` before they reach DDS.
    """

    arm = np.asarray(chunk.arm, dtype=np.float64)
    left = np.asarray(chunk.left_hand, dtype=np.float64)
    right = np.asarray(chunk.right_hand, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != ARM_DOF or arm.shape[0] < 1:
        raise DeploymentError(f"Invalid arm chunk shape {arm.shape}")
    if left.shape != (arm.shape[0], HAND_DOF) or right.shape != left.shape:
        raise DeploymentError(f"Invalid hand chunk shapes {left.shape} and {right.shape}")
    if not all(np.all(np.isfinite(values)) for values in (arm, left, right)):
        raise DeploymentError("Action chunk contains NaN or infinity")

    _check_limits(
        "arm",
        arm,
        ARM_LOWER,
        ARM_UPPER,
        joint_names=ARM_JOINT_NAMES,
        lower_constant="ARM_LOWER",
        upper_constant="ARM_UPPER",
        margin=JOINT_LIMIT_MARGIN_RAD,
        margin_constant="JOINT_LIMIT_MARGIN_RAD",
    )
    _check_limits(
        "left hand",
        left,
        LEFT_HAND_LOWER,
        LEFT_HAND_UPPER,
        joint_names=LEFT_HAND_JOINT_NAMES,
        lower_constant="LEFT_HAND_LOWER",
        upper_constant="LEFT_HAND_UPPER",
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
        tolerance_constant="HAND_LIMIT_TOLERANCE_RAD",
    )
    _check_limits(
        "right hand",
        right,
        RIGHT_HAND_LOWER,
        RIGHT_HAND_UPPER,
        joint_names=RIGHT_HAND_JOINT_NAMES,
        lower_constant="RIGHT_HAND_LOWER",
        upper_constant="RIGHT_HAND_UPPER",
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
        tolerance_constant="HAND_LIMIT_TOLERANCE_RAD",
    )


def validate_action_chunk(
    chunk: ActionChunk,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
) -> None:
    """Validate an already parsed chunk and every commanded target transition."""

    validate_action_chunk_limits(chunk)
    arm = np.asarray(chunk.arm, dtype=np.float64)
    left = np.asarray(chunk.left_hand, dtype=np.float64)
    right = np.asarray(chunk.right_hand, dtype=np.float64)
    current_arm = np.asarray(current_arm, dtype=np.float64)
    current_left = np.asarray(current_left, dtype=np.float64)
    current_right = np.asarray(current_right, dtype=np.float64)
    if current_arm.shape != (ARM_DOF,) or current_left.shape != (HAND_DOF,) or current_right.shape != (HAND_DOF,):
        raise DeploymentError("Fresh robot state has the wrong shape")
    if not all(np.all(np.isfinite(values)) for values in (current_arm, current_left, current_right)):
        raise DeploymentError("Robot state contains NaN or infinity")
    _check_step_size(
        "arm",
        arm,
        current_arm,
        MAX_ARM_STEP_RAD,
        joint_names=ARM_JOINT_NAMES,
        constant_name="MAX_ARM_STEP_RAD",
    )
    _check_step_size(
        "left hand",
        left,
        current_left,
        MAX_HAND_STEP_RAD,
        joint_names=LEFT_HAND_JOINT_NAMES,
        constant_name="MAX_HAND_STEP_RAD",
    )
    _check_step_size(
        "right hand",
        right,
        current_right,
        MAX_HAND_STEP_RAD,
        joint_names=RIGHT_HAND_JOINT_NAMES,
        constant_name="MAX_HAND_STEP_RAD",
    )


def _parse_full_action(action: dict[str, Any], model_horizon: int) -> ActionChunk:
    """Parse a complete model prediction and enforce its absolute joint ranges."""

    chunks = {key: _numeric_action(action, key, model_horizon) for key in ACTION_KEYS}
    full_arm = np.concatenate((chunks["left_arm"], chunks["right_arm"]), axis=1)
    _check_limits(
        "arm",
        full_arm,
        ARM_LOWER,
        ARM_UPPER,
        joint_names=ARM_JOINT_NAMES,
        lower_constant="ARM_LOWER",
        upper_constant="ARM_UPPER",
        margin=JOINT_LIMIT_MARGIN_RAD,
        margin_constant="JOINT_LIMIT_MARGIN_RAD",
    )
    _check_limits(
        "left hand",
        chunks["left_hand"],
        LEFT_HAND_LOWER,
        LEFT_HAND_UPPER,
        joint_names=LEFT_HAND_JOINT_NAMES,
        lower_constant="LEFT_HAND_LOWER",
        upper_constant="LEFT_HAND_UPPER",
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
        tolerance_constant="HAND_LIMIT_TOLERANCE_RAD",
    )
    _check_limits(
        "right hand",
        chunks["right_hand"],
        RIGHT_HAND_LOWER,
        RIGHT_HAND_UPPER,
        joint_names=RIGHT_HAND_JOINT_NAMES,
        lower_constant="RIGHT_HAND_LOWER",
        upper_constant="RIGHT_HAND_UPPER",
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
        tolerance_constant="HAND_LIMIT_TOLERANCE_RAD",
    )

    return ActionChunk(
        arm=np.ascontiguousarray(full_arm),
        left_hand=np.ascontiguousarray(chunks["left_hand"]),
        right_hand=np.ascontiguousarray(chunks["right_hand"]),
    )


def parse_action_plan(
    action: dict[str, Any],
    model_horizon: int,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    validate_initial_step: bool = True,
    validate_target_steps: bool = True,
) -> ActionChunk:
    """Parse and rate-check the full prediction horizon used by RTC."""

    result = _parse_full_action(action, model_horizon)
    if validate_target_steps:
        if validate_initial_step:
            validation_state = (current_arm, current_left, current_right)
        else:
            validation_state = (result.arm[0], result.left_hand[0], result.right_hand[0])
        validate_action_chunk(result, *validation_state)
    return result


def parse_action_chunk(
    action: dict[str, Any],
    model_horizon: int,
    execution_horizon: int,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    validate_initial_step: bool = True,
    validate_target_steps: bool = True,
) -> ActionChunk:
    if not 1 <= execution_horizon <= model_horizon:
        raise DeploymentError(
            f"Execution horizon {execution_horizon} must be 1..{model_horizon}; "
            f"checkpoint action horizon={model_horizon}"
        )

    full = _parse_full_action(action, model_horizon)

    result = ActionChunk(
        arm=np.ascontiguousarray(full.arm[:execution_horizon]),
        left_hand=np.ascontiguousarray(full.left_hand[:execution_horizon]),
        right_hand=np.ascontiguousarray(full.right_hand[:execution_horizon]),
    )
    if validate_target_steps:
        if validate_initial_step:
            validation_state = (current_arm, current_left, current_right)
        else:
            # This result is either an unexecuted shadow prediction or a discarded
            # preflight/warm-start prediction.  Still enforce every shape,
            # finite/range check and every within-prefix rate step, but do not
            # interpret measured-q -> action[0] as a commanded transition.
            validation_state = (result.arm[0], result.left_hand[0], result.right_hand[0])
        validate_action_chunk(result, *validation_state)
    return result
