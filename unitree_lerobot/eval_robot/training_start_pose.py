"""Frozen demonstrated start pose shared by deployment warmup and diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from unitree_lerobot.eval_robot.g1_end_effectors import get_end_effector_profile
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ARM_JOINT_NAMES,
    InitializationSpec,
    TRAINING_START_MODE,
    validate_initialization_spec,
)


DEX3_TRAINING_START_SOURCE = {
    "dataset_path": (
        "/home/alex/Development/Datasets/lerobot2/"
        "atomic_combined_09_08_And_10_08/train"
    ),
    "episode_index": 0,
    "frame_index": 0,
    "timestamp_s": 0.0,
    "task": "pick up the cereal box.",
}

# Frozen directly from observation.state, not action, in the source frame above.
# Order is the deployment contract: left arm 7, right arm 7, left hand 7,
# right hand 7. Values retain the exact float32 values stored in Parquet.
DEX3_TRAINING_START_JOINTS_RAD = np.array(
    [
        -0.35650673508644104,
        0.17410682141780853,
        0.19246666133403778,
        1.0689928531646729,
        -0.0953824520111084,
        -0.9570353031158447,
        -0.2767454981803894,
        -0.3830517828464508,
        -0.17801368236541748,
        -0.027348002418875694,
        0.8752079606056213,
        -0.17540112137794495,
        -0.7583771347999573,
        -0.17424488067626953,
        -0.537041425704956,
        0.796114981174469,
        0.021243207156658173,
        -0.347329705953598,
        -0.033961232751607895,
        -0.2210468202829361,
        -0.022243322804570198,
        -0.3697550594806671,
        -0.8164053559303284,
        -0.02797570452094078,
        0.2404455542564392,
        0.03841176629066467,
        0.12233000993728638,
        0.018584586679935455,
    ],
    dtype=np.float64,
)
DEX3_TRAINING_START_JOINTS_RAD.setflags(write=False)

# Keep the original public names as Dex3 aliases. Downstream diagnostics and
# scripts imported these names before training-start poses became
# end-effector-specific.
TRAINING_START_SOURCE = DEX3_TRAINING_START_SOURCE
TRAINING_START_JOINTS_RAD = DEX3_TRAINING_START_JOINTS_RAD


INSPIRE_TRAINING_START_SOURCE = {
    "dataset_path": (
        "/home/alex/Development/Datasets/lerobot2/inspire/"
        "all_tasks_713eps_20260917_normals_range_mask_v2/train"
    ),
    # LeRobot episode indices are zero-based. The split manifest maps this
    # converted episode to stack_red_cups_09_15 source episode_0084.
    "episode_index": 428,
    "split_episode": "episode_000428",
    "source_episode": "episode_0084",
    "data_json_sha256": "4675ad296f03ad83feff7811ae01dc1aad0e740fed5f734408aad2f0f256774b",
    "frame_index": 0,
    "timestamp_s": 0.0,
    "task": "stack the three red cups.",
    "selection": "medoid of all 115 stack-task training episode-start states",
}

# Reviewed demonstrated medoid from observation.state, not action, in the
# source frame above. Order is left arm 7, right arm 7, left Inspire hand 6,
# right Inspire hand 6. Arm values are radians and hand values are normalized
# open fractions. Values retain the exact float32 values stored in Parquet.
INSPIRE_TRAINING_START_JOINTS = np.array(
    [
        -0.4093931317329407,
        0.20677582919597626,
        0.26564234495162964,
        0.9761511087417603,
        -0.22572287917137146,
        -0.8953654170036316,
        -0.4879257380962372,
        -0.35373836755752563,
        -0.11657056212425232,
        -0.11414974182844162,
        0.6982728838920593,
        0.049195244908332825,
        -0.6964153051376343,
        0.1946837455034256,
        0.7450000047683716,
        0.8410000205039978,
        0.8820000290870667,
        0.8960000276565552,
        0.9990000128746033,
        0.7990000247955322,
        0.6690000295639038,
        0.7620000243186951,
        0.8169999718666077,
        0.8429999947547913,
        0.906000018119812,
        0.4830000102519989,
    ],
    dtype=np.float64,
)
INSPIRE_TRAINING_START_JOINTS.setflags(write=False)


def training_start_source(end_effector: str = "dex3") -> dict[str, object]:
    """Return a copy of the reviewed source metadata for one hand profile."""

    if end_effector == "dex3":
        return dict(DEX3_TRAINING_START_SOURCE)
    if end_effector in {"inspire-dfx", "inspire-ftp"}:
        return dict(INSPIRE_TRAINING_START_SOURCE)
    raise ValueError(f"No reviewed training-start pose for end effector {end_effector!r}")


def training_start_spec(end_effector: str = "dex3") -> InitializationSpec:
    """Return a fresh validated copy of the profile's demonstrated target."""

    if end_effector == "dex3":
        values = DEX3_TRAINING_START_JOINTS_RAD
        hand_dof = 7
        mode = "pose-file"
        label = "Warmup1: training episode 0 frame 0 measured pose"
    elif end_effector in {"inspire-dfx", "inspire-ftp"}:
        values = INSPIRE_TRAINING_START_JOINTS
        hand_dof = 6
        mode = TRAINING_START_MODE
        label = "Warmup1: Inspire stack training episode 428 frame 0 measured pose"
    else:
        raise ValueError(f"No reviewed training-start pose for end effector {end_effector!r}")

    spec = InitializationSpec(
        mode=mode,
        label=label,
        arm=values[:14].copy(),
        left_hand=values[14 : 14 + hand_dof].copy(),
        right_hand=values[14 + hand_dof :].copy(),
        end_effector=end_effector,
    )
    validate_initialization_spec(
        spec,
        allow_training_start=mode == TRAINING_START_MODE,
    )
    return spec


# User-selected pyramid frame-0 measured poses. Kept in code so named presets
# do not depend on a reports folder, dataset mount, or user-supplied JSON path.
# Full vectors preserve observation.state exactly: 14 arm radians then 6+6
# Inspire normalized open fractions. Both Inspire transports share this order.
WARMUP1_POSE_NAMES = ("default", "pyramid1", "pyramid2")
_PYRAMID_WARMUP1_POSES = {
    "pyramid1": (
        (
            -0.3238137662410736,
            0.11266370117664337,
            0.260237455368042,
            1.071677327156067,
            -0.34684744477272034,
            -1.010689616203308,
            -0.844001054763794,
            -0.3882769048213959,
            -0.19183149933815002,
            -0.057380471378564835,
            1.0802819728851318,
            -0.0228659026324749,
            -0.9447644352912903,
            0.28523653745651245,
            0.6859999895095825,
            0.777999997138977,
            0.8489999771118164,
            0.8619999885559082,
            0.9990000128746033,
            0.8289999961853027,
            0.6710000038146973,
            0.7749999761581421,
            0.8259999752044678,
            0.871999979019165,
            0.8809999823570251,
            0.6759999990463257,
        ),
        {
            "dataset_path": "/home/alex/Development/Datasets/lerobot2/inspire/pyramid_sideways_09_19_normals_range_mask_v2/train",
            "episode_index": 11,
            "frame_index": 0,
            "quantity": "observation.state",
            "source_episode": "episode_0018",
            "task": "build a cup pyramid left-to-right.",
            "timestamp_s": 0
        },
    ),
    "pyramid2": (
        (
            -0.33673277497291565,
            0.17343570291996002,
            0.3664536476135254,
            1.0695921182632446,
            -0.3935379981994629,
            -0.870522141456604,
            -0.8195532560348511,
            -0.4010041654109955,
            -0.16189490258693695,
            -0.16660469770431519,
            0.9246788620948792,
            0.20451080799102783,
            -0.859544575214386,
            0.5700176954269409,
            0.6880000233650208,
            0.7870000004768372,
            0.8550000190734863,
            0.8730000257492065,
            0.9959999918937683,
            0.9039999842643738,
            0.5289999842643738,
            0.625,
            0.6909999847412109,
            0.7670000195503235,
            0.8610000014305115,
            0.7839999794960022,
        ),
        {
            "dataset_path": "/home/alex/Development/Datasets/lerobot2/inspire/pyramid_sideways_09_19_normals_range_mask_v2/train",
            "episode_index": 46,
            "frame_index": 0,
            "quantity": "observation.state",
            "source_episode": "episode_0068",
            "task": "build a cup pyramid left-to-right.",
            "timestamp_s": 0
        },
    ),
}


WARMUP1_POSE_MAX_BYTES = 64 * 1024
_WARMUP1_POSE_FIELDS = {
    "schema_version", "name", "robot_type", "end_effector", "arm_unit",
    "hand_unit", "joint_names", "arm", "left_hand", "right_hand", "source",
}
_WARMUP1_SOURCE_REQUIRED = {
    "dataset_path", "episode_index", "frame_index", "quantity",
}
_WARMUP1_SOURCE_OPTIONAL = {"source_episode", "task", "timestamp_s"}


def _warmup1_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"Warmup1 pose JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _warmup1_reject_json_constant(value: str) -> None:
    raise DeploymentError(f"Warmup1 pose JSON contains non-finite constant {value!r}")


def _warmup1_fields(
    value: object,
    required: set[str],
    *,
    field: str,
    optional: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DeploymentError(f"Warmup1 pose {field} must be a JSON object")
    allowed = required | (optional or set())
    if set(value) - allowed or required - set(value):
        raise DeploymentError(
            f"Warmup1 pose {field} fields do not match the schema: "
            f"missing={sorted(required - set(value))}, extra={sorted(set(value) - allowed)}"
        )
    return value


def _warmup1_text(value: object, *, field: str, max_length: int | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 32 for character in value)
        or (max_length is not None and len(value) > max_length)
    ):
        suffix = f" of at most {max_length} characters" if max_length is not None else ""
        raise DeploymentError(f"Warmup1 pose {field} must be a non-empty single-line string{suffix}")
    return value


def _warmup1_vector(value: object, *, field: str, size: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise DeploymentError(f"Warmup1 pose {field} must be a {size}-element numeric list")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (OverflowError, ValueError, TypeError) as exc:
        raise DeploymentError(f"Warmup1 pose {field} is not a finite numeric vector") from exc
    if not np.all(np.isfinite(array)):
        raise DeploymentError(f"Warmup1 pose {field} contains NaN or infinity")
    return array


def load_warmup1_pose(
    end_effector: str = "dex3",
    pose_file: str | Path | None = None,
    *,
    pose_name: str | None = None,
) -> tuple[InitializationSpec, dict[str, object]]:
    """Load a complete, profile-bound measured pose without changing any limits.

    No selection preserves the exact built-in pose and its validation path. A file
    is only a different target for Warmup1, not a robot command or a relaxation
    of the guarded trajectory checks. Provenance paths are historical metadata
    and need not still exist. File-backed arrays are frozen after validation.
    """
    if pose_name is not None and pose_file is not None:
        raise DeploymentError("Select either --warmup1-pose or --warmup1-pose-file, not both")
    if pose_name is not None and pose_name not in WARMUP1_POSE_NAMES:
        raise DeploymentError(f"Unknown Warmup1 pose {pose_name!r}; choose from {WARMUP1_POSE_NAMES}")
    if pose_name in _PYRAMID_WARMUP1_POSES:
        if end_effector not in {"inspire-ftp", "inspire-dfx"}:
            raise DeploymentError(f"Warmup1 pose {pose_name!r} requires Inspire hands, not {end_effector!r}")
        values, recorded_source = _PYRAMID_WARMUP1_POSES[pose_name]
        joints = np.asarray(values, dtype=np.float64)
        spec = InitializationSpec(
            mode=TRAINING_START_MODE,
            label=f"Warmup1: {pose_name} (pyramid episode {recorded_source['episode_index']} frame 0 measured pose)",
            arm=joints[:14].copy(),
            left_hand=joints[14:20].copy(),
            right_hand=joints[20:].copy(),
            end_effector=end_effector,
        )
        validate_initialization_spec(spec, allow_training_start=True)
        for array in (spec.arm, spec.left_hand, spec.right_hand):
            array.setflags(write=False)
        return spec, {**recorded_source, "pose_name": pose_name}

    if pose_file is None:
        source = training_start_source(end_effector)
        source["pose_name"] = "default"
        return training_start_spec(end_effector), source

    try:
        profile = get_end_effector_profile(end_effector)
    except ValueError as exc:
        raise DeploymentError(f"Unsupported Warmup1 end effector {end_effector!r}") from exc
    try:
        path = Path(pose_file).expanduser().resolve(strict=True)
        if not path.is_file():
            raise DeploymentError("Warmup1 pose path must identify a regular file")
        with path.open("rb") as stream:
            content = stream.read(WARMUP1_POSE_MAX_BYTES + 1)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise DeploymentError(f"Cannot read Warmup1 pose file {pose_file!r}: {exc}") from exc
    if len(content) > WARMUP1_POSE_MAX_BYTES:
        raise DeploymentError("Warmup1 pose JSON exceeds the 64 KiB size limit")
    try:
        document = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_warmup1_json_object,
            parse_constant=_warmup1_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise DeploymentError(f"Invalid Warmup1 pose JSON: {exc}") from exc
    document = _warmup1_fields(document, _WARMUP1_POSE_FIELDS, field="document")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise DeploymentError("Warmup1 pose schema_version must be integer 1")
    name = _warmup1_text(document["name"], field="name", max_length=100)
    for field, expected in (
        ("robot_type", "g1"),
        ("end_effector", profile.name),
        ("arm_unit", "rad"),
        ("hand_unit", profile.value_unit),
    ):
        if document[field] != expected:
            raise DeploymentError(f"Warmup1 pose {field} must be exactly {expected!r}")
    joint_names = _warmup1_fields(
        document["joint_names"], {"arm", "left_hand", "right_hand"}, field="joint_names"
    )
    for field, expected in (
        ("arm", ARM_JOINT_NAMES),
        ("left_hand", profile.left_joint_names),
        ("right_hand", profile.right_joint_names),
    ):
        if joint_names[field] != list(expected):
            raise DeploymentError(f"Warmup1 pose joint_names.{field} must match the exact profile order")

    source = _warmup1_fields(
        document["source"], _WARMUP1_SOURCE_REQUIRED,
        field="source", optional=_WARMUP1_SOURCE_OPTIONAL,
    )
    _warmup1_text(source["dataset_path"], field="source.dataset_path")
    for field in ("episode_index", "frame_index"):
        if type(source[field]) is not int or source[field] < 0:
            raise DeploymentError(f"Warmup1 pose source.{field} must be a nonnegative integer")
    if source["quantity"] != "observation.state":
        raise DeploymentError("Warmup1 pose source.quantity must be 'observation.state', not action")
    for field in ("source_episode", "task"):
        if field in source:
            _warmup1_text(source[field], field=f"source.{field}")
    if "timestamp_s" in source:
        timestamp = source["timestamp_s"]
        try:
            valid_timestamp = (
                not isinstance(timestamp, bool)
                and isinstance(timestamp, (int, float))
                and math.isfinite(timestamp)
                and timestamp >= 0
            )
        except OverflowError:
            valid_timestamp = False
        if not valid_timestamp:
            raise DeploymentError("Warmup1 pose source.timestamp_s must be finite and nonnegative")

    spec = InitializationSpec(
        mode="pose-file" if profile.name == "dex3" else TRAINING_START_MODE,
        label=f"Warmup1: {name}",
        arm=_warmup1_vector(document["arm"], field="arm", size=len(ARM_JOINT_NAMES)),
        left_hand=_warmup1_vector(document["left_hand"], field="left_hand", size=profile.hand_dof),
        right_hand=_warmup1_vector(document["right_hand"], field="right_hand", size=profile.hand_dof),
        end_effector=profile.name,
    )
    validate_initialization_spec(spec, allow_training_start=spec.mode == TRAINING_START_MODE)
    for array in (spec.arm, spec.left_hand, spec.right_hand):
        array.setflags(write=False)
    provenance = dict(source)
    provenance.update(pose_name=name, pose_file=str(path), sha256=hashlib.sha256(content).hexdigest())
    return spec, provenance
