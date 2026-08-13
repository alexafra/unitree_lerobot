"""Frozen demonstrated start pose shared by deployment warmup and diagnostics."""

from __future__ import annotations

import numpy as np

from unitree_lerobot.eval_robot.groot_contract import (
    InitializationSpec,
    validate_initialization_spec,
)


TRAINING_START_SOURCE = {
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
TRAINING_START_JOINTS_RAD = np.array(
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
TRAINING_START_JOINTS_RAD.setflags(write=False)


def training_start_spec() -> InitializationSpec:
    """Return a fresh validated copy of the demonstrated frame-zero target."""

    spec = InitializationSpec(
        mode="pose-file",
        label="Warmup1: training episode 0 frame 0 measured pose",
        arm=TRAINING_START_JOINTS_RAD[:14].copy(),
        left_hand=TRAINING_START_JOINTS_RAD[14:21].copy(),
        right_hand=TRAINING_START_JOINTS_RAD[21:].copy(),
    )
    validate_initialization_spec(spec)
    return spec
