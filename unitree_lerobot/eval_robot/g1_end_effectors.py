"""Static end-effector contracts for the guarded G1 deployment client.

Selecting a profile changes only the hand transport and data contract.  Dex3
remains the default; Inspire DFX must be requested explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UNQUALIFIED_INSPIRE_DFX_COMMAND_MAX_STEP = 0.2


@dataclass(frozen=True)
class EndEffectorProfile:
    name: str
    robot_type: str
    hand_dof: int
    left_joint_names: tuple[str, ...]
    right_joint_names: tuple[str, ...]
    left_lower: np.ndarray
    left_upper: np.ndarray
    right_lower: np.ndarray
    right_upper: np.ndarray
    home: np.ndarray | None
    value_unit: str
    transport: str
    limit_tolerance: float
    measured_limit_tolerance: float
    max_step: float | None
    conditioned_step: np.ndarray | None
    initialization_speed: float | None
    initialization_tolerance: float | None
    tracking_warning: float | None
    tracking_clear: float | None
    tracking_hard: float | None
    supports_simulation: bool
    supports_gravity_feedforward: bool
    has_motor_stop: bool
    state_topic: str | None = None
    command_topic: str | None = None
    left_state_topic: str | None = None
    right_state_topic: str | None = None
    left_command_topic: str | None = None
    right_command_topic: str | None = None
    uses_lost_counters: bool = False

    def __post_init__(self) -> None:
        if self.hand_dof < 1:
            raise ValueError("hand_dof must be positive")
        if len(self.left_joint_names) != self.hand_dof or len(self.right_joint_names) != self.hand_dof:
            raise ValueError(f"{self.name}: joint-name count does not match hand_dof")
        for field_name in (
            "left_lower",
            "left_upper",
            "right_lower",
            "right_upper",
        ):
            value = np.array(getattr(self, field_name), dtype=np.float64, copy=True)
            if value.shape != (self.hand_dof,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{self.name}: {field_name} must be a finite ({self.hand_dof},) vector")
            value.flags.writeable = False
            object.__setattr__(self, field_name, value)
        for field_name in ("home", "conditioned_step"):
            original = getattr(self, field_name)
            if original is None:
                continue
            value = np.array(original, dtype=np.float64, copy=True)
            if value.shape != (self.hand_dof,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{self.name}: {field_name} must be a finite ({self.hand_dof},) vector")
            value.flags.writeable = False
            object.__setattr__(self, field_name, value)
        if self.conditioned_step is not None and np.any(self.conditioned_step <= 0.0):
            raise ValueError(f"{self.name}: conditioned_step must be strictly positive")
        if np.any(self.left_lower > self.left_upper) or np.any(self.right_lower > self.right_upper):
            raise ValueError(f"{self.name}: lower hand bounds exceed upper bounds")
        if self.home is not None:
            if np.any(self.home < self.left_lower) or np.any(self.home > self.left_upper):
                raise ValueError(f"{self.name}: home is outside left-hand bounds")
            if np.any(self.home < self.right_lower) or np.any(self.home > self.right_upper):
                raise ValueError(f"{self.name}: home is outside right-hand bounds")
        if self.transport == "dex3":
            if not all(
                (
                    self.left_state_topic,
                    self.right_state_topic,
                    self.left_command_topic,
                    self.right_command_topic,
                )
            ):
                raise ValueError("Dex3 profile requires separate per-hand DDS topics")
        elif self.transport == "inspire-dfx":
            if not self.state_topic or not self.command_topic or not self.uses_lost_counters:
                raise ValueError("Inspire DFX profile requires combined DDS topics and lost counters")
        else:
            raise ValueError(f"Unsupported hand transport {self.transport!r}")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.left_joint_names + self.right_joint_names


DEX3_PROFILE = EndEffectorProfile(
    name="dex3",
    robot_type="Unitree_G1_Dex3_HeadOnly",
    hand_dof=7,
    left_joint_names=(
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
    ),
    right_joint_names=(
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ),
    left_lower=np.array(
        [-1.04719755, -0.72431163, 0.0, -1.57079632, -1.74532925, -1.57079632, -1.74532925],
        dtype=np.float64,
    ),
    left_upper=np.array([1.04719755, 1.04719755, 1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    right_lower=np.array([-1.04719755, -1.04719755, -1.74532925, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    right_upper=np.array(
        [1.04719755, 0.72431163, 0.0, 1.57079632, 1.74532925, 1.57079632, 1.74532925],
        dtype=np.float64,
    ),
    home=np.zeros(7, dtype=np.float64),
    value_unit="rad",
    transport="dex3",
    limit_tolerance=0.01,
    measured_limit_tolerance=0.2,
    max_step=0.60,
    conditioned_step=np.array([0.06857, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12], dtype=np.float64),
    initialization_speed=0.5,
    initialization_tolerance=0.8,
    tracking_warning=0.5,
    tracking_clear=0.4,
    tracking_hard=2.0,
    supports_simulation=True,
    supports_gravity_feedforward=True,
    has_motor_stop=True,
    left_state_topic="rt/dex3/left/state",
    right_state_topic="rt/dex3/right/state",
    left_command_topic="rt/dex3/left/cmd",
    right_command_topic="rt/dex3/right/cmd",
)


INSPIRE_DFX_PROFILE = EndEffectorProfile(
    name="inspire-dfx",
    robot_type="Unitree_G1_Inspire_HeadOnly",
    hand_dof=6,
    left_joint_names=(
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
    ),
    right_joint_names=(
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ),
    left_lower=np.zeros(6, dtype=np.float64),
    left_upper=np.ones(6, dtype=np.float64),
    right_lower=np.zeros(6, dtype=np.float64),
    right_upper=np.ones(6, dtype=np.float64),
    # This is the exact hand target used by the original XR teleop startup:
    # Inspire DFX uses 0=closed and 1=open, so its XR home is fully open. The
    # guarded client still defaults to measured initialization; this target is
    # used only when the operator explicitly selects ``--initialization xr-home``.
    home=np.ones(6, dtype=np.float64),
    value_unit="normalized_open_fraction",
    transport="inspire-dfx",
    limit_tolerance=0.0,
    measured_limit_tolerance=0.0,
    # Raw 30 Hz model targets may move anywhere inside [0, 1]. The 0.2 ceiling
    # applies only after conditioning and again at the final DDS writer.
    max_step=None,
    conditioned_step=np.full(
        6,
        UNQUALIFIED_INSPIRE_DFX_COMMAND_MAX_STEP,
        dtype=np.float64,
    ),
    initialization_speed=None,
    initialization_tolerance=None,
    tracking_warning=None,
    tracking_clear=None,
    tracking_hard=None,
    supports_simulation=False,
    # This intentionally follows the existing Inspire teleop path, which uses
    # the same g1_body29_hand14 model as Dex3. It is parity, not a newly
    # identified Inspire payload model.
    supports_gravity_feedforward=True,
    has_motor_stop=False,
    state_topic="rt/inspire/state",
    command_topic="rt/inspire/cmd",
    uses_lost_counters=True,
)


END_EFFECTOR_PROFILES = {
    DEX3_PROFILE.name: DEX3_PROFILE,
    INSPIRE_DFX_PROFILE.name: INSPIRE_DFX_PROFILE,
}


def inspire_dfx_dataset_contract() -> dict[str, object]:
    """Return the exact converted-dataset provenance required for DFX."""

    return {
        "schema_version": 1,
        "type": "inspire",
        "protocol": "dfx",
        "hand_dof": INSPIRE_DFX_PROFILE.hand_dof,
        "value_unit": "normalized_open_fraction",
        "value_range": [0.0, 1.0],
        "zero_semantics": "fully_closed",
        "one_semantics": "fully_open",
        "left_joint_names": list(INSPIRE_DFX_PROFILE.left_joint_names),
        "right_joint_names": list(INSPIRE_DFX_PROFILE.right_joint_names),
        "canonical_order": "left_then_right",
    }


def get_end_effector_profile(name: str) -> EndEffectorProfile:
    try:
        return END_EFFECTOR_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(END_EFFECTOR_PROFILES)
        raise ValueError(f"Unsupported end effector {name!r}; expected one of: {choices}") from exc
