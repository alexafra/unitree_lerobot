"""Frozen demonstrated start pose shared by deployment warmup and diagnostics."""

from __future__ import annotations

import numpy as np

from unitree_lerobot.eval_robot.groot_contract import (
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
