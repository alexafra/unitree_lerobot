"""Pure GR00T/Unitree observation and action contract checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.utils.depth_encoding import (
    DEPTH_ENCODING,
    DEPTH_OUTPUT_KEY,
    DEPTH_SOURCE_KEY,
)


TASKS = {
    "pick-toothpaste": "pick up the cylinder toothepaste.",
    "put-toothpaste": "put down the cylinder toothepaste.",
    "pick-red-cup": "pick up the red cup.",
    "put-red-cup": "put down the red cup.",
}

COLOUR_VIDEO_KEYS = ("ego_view",)
RGBD_VIDEO_KEYS = ("ego_view", DEPTH_OUTPUT_KEY)
SUPPORTED_VIDEO_KEYS = (COLOUR_VIDEO_KEYS, RGBD_VIDEO_KEYS)
# Backwards-compatible public name used by existing callers/tests.
VIDEO_KEYS = COLOUR_VIDEO_KEYS
STATE_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand")
ACTION_KEYS = STATE_KEYS
LANGUAGE_KEYS = ("annotation.human.task_description",)
EXPECTED_TRAINING_TAG = "new_embodiment"
EXPECTED_ROBOT_TYPE = "Unitree_G1_Dex3_HeadOnly"
EXPECTED_EGO_VIEW_SHAPE = [480, 640, 3]
EXPECTED_DEPTH_VIEW_SHAPE = [480, 640, 3]
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

ARM_DOF = 14
HAND_DOF = 7
CONTROL_HZ = 30.0
MAX_EXECUTION_HORIZON = 8

# These are deliberately fixed deployment ceilings, not tuning flags.  They need
# hardware qualification before being relaxed.
MAX_ARM_STEP_RAD = 0.05
MAX_HAND_STEP_RAD = 0.10
JOINT_LIMIT_MARGIN_RAD = 0.03
HAND_LIMIT_TOLERANCE_RAD = 0.002
MEASURED_LIMIT_TOLERANCE_RAD = 0.01

# G1 29-DoF limits from g1_body29_hand14.urdf in G1_29_JointArmIndex order.
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
LEFT_HAND_LOWER = np.array(
    [-1.04719755, -0.72431163, 0.0, -1.57079632, -1.74532925, -1.57079632, -1.74532925],
    dtype=np.float64,
)
LEFT_HAND_UPPER = np.array([1.04719755, 1.04719755, 1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
RIGHT_HAND_LOWER = np.array([-1.04719755, -1.04719755, -1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
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
        return self.video_keys == RGBD_VIDEO_KEYS


@dataclass(frozen=True)
class DepthEncodingContract:
    near_m: float
    far_m: float


@dataclass(frozen=True)
class ActionChunk:
    arm: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray

    @property
    def length(self) -> int:
        return int(self.arm.shape[0])


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


def validate_policy_metadata(
    metadata: dict[str, Any],
    *,
    requires_depth: bool = False,
) -> DepthEncodingContract | None:
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
    """Accept only the explicitly supported G1/Dex3 colour or RGBD contracts."""

    expected = {
        "state": STATE_KEYS,
        "action": ACTION_KEYS,
        "language": LANGUAGE_KEYS,
    }
    video_keys = tuple(_config_field(config, "video", "modality_keys"))
    if video_keys not in SUPPORTED_VIDEO_KEYS:
        raise DeploymentError(
            f"Unsupported video keys from GR00T server: {video_keys}; expected exactly "
            f"{COLOUR_VIDEO_KEYS} or {RGBD_VIDEO_KEYS}."
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
    if instruction not in TASKS.values():
        raise DeploymentError("Instruction is not in the trained task allowlist")

    if video_keys not in SUPPORTED_VIDEO_KEYS:
        raise DeploymentError(f"Cannot build an observation for unsupported video keys {video_keys}")
    video = {"ego_view": np.ascontiguousarray(rgb)[None, None]}
    if video_keys == RGBD_VIDEO_KEYS:
        if depth_gray is None:
            raise DeploymentError("RGBD checkpoint requires depth_gray_view")
        depth_gray = np.asarray(depth_gray)
        if list(depth_gray.shape) != EXPECTED_DEPTH_VIEW_SHAPE or depth_gray.dtype != np.uint8:
            raise DeploymentError(
                "Expected uint8 depth_gray_view with shape "
                f"{tuple(EXPECTED_DEPTH_VIEW_SHAPE)}, got {depth_gray.shape} {depth_gray.dtype}"
            )
        video[DEPTH_OUTPUT_KEY] = np.ascontiguousarray(depth_gray)[None, None]
    elif depth_gray is not None:
        raise DeploymentError("Colour-only checkpoint must not receive depth_gray_view")

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
    margin: float = 0.0,
    tolerance: float = 0.0,
) -> None:
    safe_lower = lower + margin - tolerance
    safe_upper = upper - margin + tolerance
    bad = np.argwhere((values < safe_lower) | (values > safe_upper))
    if bad.size:
        step, joint = (int(index) for index in bad[0])
        raise DeploymentError(
            f"{name} target is outside its safety-margined joint range at step {step}, "
            f"joint {joint}: {values[step, joint]:.4f} rad"
        )


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
        ("arm", arrays["arm"][0], ARM_LOWER, ARM_UPPER),
        ("left hand", arrays["left hand"][0], LEFT_HAND_LOWER, LEFT_HAND_UPPER),
        ("right hand", arrays["right hand"][0], RIGHT_HAND_LOWER, RIGHT_HAND_UPPER),
    )
    for name, values, lower, upper in limits:
        bad = np.flatnonzero(
            (values < lower - MEASURED_LIMIT_TOLERANCE_RAD) | (values > upper + MEASURED_LIMIT_TOLERANCE_RAD)
        )
        if bad.size:
            joint = int(bad[0])
            raise DeploymentError(
                f"Measured {name} joint {joint} is outside its physical range: {values[joint]:.4f} rad"
            )


def _check_step_size(name: str, values: np.ndarray, current: np.ndarray, max_step: float) -> None:
    deltas = np.diff(np.vstack((current, values)), axis=0)
    bad = np.argwhere(np.abs(deltas) > max_step)
    if bad.size:
        step, joint = (int(index) for index in bad[0])
        raise DeploymentError(
            f"{name} target jump is too large at step {step}, joint {joint}: "
            f"{deltas[step, joint]:+.4f} rad (limit {max_step:.4f})"
        )


def validate_action_chunk(
    chunk: ActionChunk,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
) -> None:
    """Independently validate an already parsed chunk against fresh state."""

    arm = np.asarray(chunk.arm, dtype=np.float64)
    left = np.asarray(chunk.left_hand, dtype=np.float64)
    right = np.asarray(chunk.right_hand, dtype=np.float64)
    current_arm = np.asarray(current_arm, dtype=np.float64)
    current_left = np.asarray(current_left, dtype=np.float64)
    current_right = np.asarray(current_right, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != ARM_DOF or arm.shape[0] < 1:
        raise DeploymentError(f"Invalid arm chunk shape {arm.shape}")
    if left.shape != (arm.shape[0], HAND_DOF) or right.shape != left.shape:
        raise DeploymentError(f"Invalid hand chunk shapes {left.shape} and {right.shape}")
    if current_arm.shape != (ARM_DOF,) or current_left.shape != (HAND_DOF,) or current_right.shape != (HAND_DOF,):
        raise DeploymentError("Fresh robot state has the wrong shape")
    if not all(np.all(np.isfinite(values)) for values in (arm, left, right, current_arm, current_left, current_right)):
        raise DeploymentError("Action chunk or robot state contains NaN or infinity")

    _check_limits("arm", arm, ARM_LOWER, ARM_UPPER, JOINT_LIMIT_MARGIN_RAD)
    _check_limits(
        "left hand",
        left,
        LEFT_HAND_LOWER,
        LEFT_HAND_UPPER,
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
    )
    _check_limits(
        "right hand",
        right,
        RIGHT_HAND_LOWER,
        RIGHT_HAND_UPPER,
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
    )
    _check_step_size("arm", arm, current_arm, MAX_ARM_STEP_RAD)
    _check_step_size("left hand", left, current_left, MAX_HAND_STEP_RAD)
    _check_step_size("right hand", right, current_right, MAX_HAND_STEP_RAD)


def parse_action_chunk(
    action: dict[str, Any],
    model_horizon: int,
    execution_horizon: int,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
) -> ActionChunk:
    if not 1 <= execution_horizon <= min(model_horizon, MAX_EXECUTION_HORIZON):
        raise DeploymentError(f"Execution horizon must be 1..{min(model_horizon, MAX_EXECUTION_HORIZON)}")

    chunks = {key: _numeric_action(action, key, model_horizon) for key in ACTION_KEYS}
    # Reject gross invalid values anywhere in the model response, including the
    # unexecuted tail, then apply rate checks to the executed prefix.
    full_arm = np.concatenate((chunks["left_arm"], chunks["right_arm"]), axis=1)
    _check_limits("arm", full_arm, ARM_LOWER, ARM_UPPER, JOINT_LIMIT_MARGIN_RAD)
    _check_limits(
        "left hand",
        chunks["left_hand"],
        LEFT_HAND_LOWER,
        LEFT_HAND_UPPER,
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
    )
    _check_limits(
        "right hand",
        chunks["right_hand"],
        RIGHT_HAND_LOWER,
        RIGHT_HAND_UPPER,
        tolerance=HAND_LIMIT_TOLERANCE_RAD,
    )

    result = ActionChunk(
        arm=np.ascontiguousarray(full_arm[:execution_horizon]),
        left_hand=np.ascontiguousarray(chunks["left_hand"][:execution_horizon]),
        right_hand=np.ascontiguousarray(chunks["right_hand"][:execution_horizon]),
    )
    validate_action_chunk(result, current_arm, current_left, current_right)
    return result
