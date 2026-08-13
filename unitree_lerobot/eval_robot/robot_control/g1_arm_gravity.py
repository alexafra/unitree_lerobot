"""Fail-closed XR-compatible gravity feed-forward for the G1 dual arm.

XR teleoperation computes the 14 arm feed-forward torques with Pinocchio RNEA
after reducing the checked-in G1 model by locking the legs, waist, and Dex3
joints at zero.  Deployment intentionally reproduces that model and joint order;
it does not infer dynamics from policy data or from the live robot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError


G1_ARM_GRAVITY_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# This is deliberately narrower than the URDF effort limits
# [25, 25, 25, 25, 25, 5, 5] Nm per arm.  It covers the XR RNEA torques across
# the recorded training actions (observed maxima below 7.71 Nm) while making a
# corrupt model, joint permutation, or implausible result fail before DDS Write.
G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM = np.array(
    [10.0, 5.0, 3.0, 5.0, 2.0, 2.0, 1.0] * 2,
    dtype=np.float64,
)
G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM.setflags(write=False)

# Pin the dynamics source, not just its filename.  A mass/inertia edit must be
# reviewed together with this safety envelope before it can command hardware.
G1_ARM_GRAVITY_URDF_SHA256 = "63b8d1178113a1311de3c89a1d0db98d4366c076d0ec57727ae85f307d553b1e"

# Exact locked-joint semantics used by Unitree XR's G1_29_ArmIK.  Mode 6 locks
# waist roll/pitch; locking yaw at zero is gravity-equivalent because yaw is
# about the gravity axis.  Dex3 joints are also fixed at zero exactly as in XR.
_XR_LOCKED_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)


def default_g1_gravity_urdf_path() -> Path:
    """Return the reviewed, checked-in G1/Dex3 dynamics asset."""

    return Path(__file__).resolve().parent.parent / "assets" / "g1" / "g1_body29_hand14.urdf"


class G1ArmGravityCompensator:
    """Compute bounded static arm gravity torques with XR's reduced model."""

    def __init__(self, urdf_path: Path | str | None = None) -> None:
        path = default_g1_gravity_urdf_path() if urdf_path is None else Path(urdf_path)
        try:
            model_bytes = path.read_bytes()
        except OSError as exc:
            raise DeploymentError(f"Could not read the reviewed G1 gravity URDF {path}: {exc}") from exc
        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != G1_ARM_GRAVITY_URDF_SHA256:
            raise DeploymentError(
                "G1 gravity URDF digest is not the reviewed value: "
                f"path={path}, expected={G1_ARM_GRAVITY_URDF_SHA256}, got={digest}"
            )

        try:
            import pinocchio as pin
        except (ImportError, OSError) as exc:
            raise DeploymentError(
                "Pinocchio is required for XR-compatible G1 arm gravity feed-forward; "
                "refusing to create command publishers"
            ) from exc

        try:
            robot = pin.RobotWrapper.BuildFromURDF(str(path), str(path.parent))
            reference = np.zeros(robot.model.nq, dtype=np.float64)
            reduced = robot.buildReducedRobot(
                list_of_joints_to_lock=list(_XR_LOCKED_JOINT_NAMES),
                reference_configuration=reference,
            )
        except Exception as exc:
            raise DeploymentError(f"Could not build the reviewed G1 arm gravity model: {exc}") from exc

        model = reduced.model
        joint_names = tuple(str(name) for name in model.names[1:])
        if model.nq != 14 or model.nv != 14 or joint_names != G1_ARM_GRAVITY_JOINT_NAMES:
            raise DeploymentError(
                "G1 arm gravity model has an unsafe reduced-model contract: "
                f"nq={model.nq}, nv={model.nv}, joints={joint_names!r}"
            )
        effort_limits = np.asarray(model.effortLimit, dtype=np.float64)
        if effort_limits.shape != (14,) or not np.all(np.isfinite(effort_limits)):
            raise DeploymentError(
                f"G1 arm gravity model has invalid URDF effort limits: {effort_limits!r}"
            )
        torque_envelope = np.asarray(G1_ARM_GRAVITY_TORQUE_ENVELOPE_NM, dtype=np.float64)
        if (
            torque_envelope.shape != (14,)
            or not np.all(np.isfinite(torque_envelope))
            or np.any(torque_envelope <= 0.0)
        ):
            raise DeploymentError(
                f"G1 arm gravity safety envelope must contain 14 finite positive values: {torque_envelope!r}"
            )
        if np.any(torque_envelope > effort_limits):
            raise DeploymentError(
                "G1 arm gravity safety envelope exceeds the reviewed URDF effort limits: "
                f"envelope={torque_envelope.tolist()}, "
                f"limits={effort_limits.tolist()}"
            )

        self.urdf_path = path
        self._pin: Any = pin
        self._model: Any = model
        self._data: Any = model.createData()
        self._zero_velocity = np.zeros(14, dtype=np.float64)
        self._zero_acceleration = np.zeros(14, dtype=np.float64)
        self._torque_envelope_nm = torque_envelope.copy()
        self._torque_envelope_nm.setflags(write=False)

        # Exercise RNEA and all result checks during construction.  The command
        # backend constructs this object before it creates any DDS publisher.
        self.compute(np.zeros(14, dtype=np.float64))

    def compute(self, arm_q: np.ndarray) -> np.ndarray:
        """Return gravity feed-forward in deployment's left-7/right-7 order."""

        q = np.asarray(arm_q, dtype=np.float64)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise DeploymentError(
                f"G1 arm gravity input must be 14 finite joint positions; got shape={q.shape}"
            )
        try:
            tau = np.asarray(
                self._pin.rnea(
                    self._model,
                    self._data,
                    q,
                    self._zero_velocity,
                    self._zero_acceleration,
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            raise DeploymentError(f"G1 arm gravity RNEA failed: {exc}") from exc
        if tau.shape != (14,) or not np.all(np.isfinite(tau)):
            raise DeploymentError(f"G1 arm gravity RNEA returned an invalid vector: {tau!r}")
        violation = np.abs(tau) - self._torque_envelope_nm
        if np.any(violation > 1e-12):
            joint = int(np.argmax(violation))
            raise DeploymentError(
                "G1 arm gravity torque exceeded its reviewed envelope at "
                f"joint {joint} ({G1_ARM_GRAVITY_JOINT_NAMES[joint]}): "
                f"tau={tau[joint]:+.4f} Nm, "
                f"limit={self._torque_envelope_nm[joint]:.4f} Nm"
            )
        return np.ascontiguousarray(tau).copy()
