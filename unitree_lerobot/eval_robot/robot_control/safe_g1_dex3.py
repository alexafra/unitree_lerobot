"""Guarded G1-29 + Dex3 state, camera, and command runtime.

Shadow use creates only subscribers.  Live command publishers exist in a spawned
child process with an independent heartbeat deadline, so blocked inference cannot
leave an action chunk advancing indefinitely.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.queues import Queue as MpQueue
import queue
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.image_server.rgbd_protocol import RGBD_PROTOCOL
from unitree_lerobot.eval_robot.groot_contract import (
    ActionChunk,
    ARM_DOF,
    ARM_JOINT_NAMES,
    ARM_LOWER,
    ARM_UPPER,
    CONTROL_HZ,
    DepthEncodingContract,
    EXPECTED_DEPTH_VIEW_SHAPE,
    EXPECTED_EGO_VIEW_SHAPE,
    HAND_LIMIT_TOLERANCE_RAD,
    InitializationSpec,
    HAND_DOF,
    JOINT_LIMIT_MARGIN_RAD,
    LEFT_HAND_LOWER,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_UPPER,
    MAX_ARM_STEP_RAD,
    MEASURED_LIMIT_TOLERANCE_RAD,
    RIGHT_HAND_LOWER,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_HAND_UPPER,
    validate_action_chunk,
    validate_action_chunk_limits,
    validate_initialization_spec,
    validate_measured_state,
)
from unitree_lerobot.utils.depth_encoding import encode_depth_gray_rgb


LOGGER = logging.getLogger(__name__)

STATE_MAX_AGE_S = 0.25
ACTUATOR_ARM_STATE_MAX_AGE_S = 0.075
ACTUATOR_HAND_STATE_WARNING_AGE_S = 0.075
ACTUATOR_HAND_STATE_MAX_AGE_S = 0.250
ACTUATOR_HAND_RECOVERY_SAMPLES = 5
# Backward-compatible name for tests/internal imports.  It remains the hard
# arm-state deadline; hand state has its own limits above.
ACTUATOR_STATE_MAX_AGE_S = ACTUATOR_ARM_STATE_MAX_AGE_S
HEARTBEAT_TIMEOUT_S = 1.0
CHUNK_MAX_AGE_S = 0.25
ARM_AUTHORITY_RAMP_S = 1.5
# CHANGEDSAFETY: original local adapter default was 1.0 s; current is 1.5 s.
# This is the orderly arm_sdk authority ramp-down duration.
ARM_RELEASE_RAMP_S = 1.5
ACTUATOR_RELEASE_SOFT_TIMEOUT_S = ARM_RELEASE_RAMP_S + 2.0
ACTUATOR_RELEASE_HARD_TIMEOUT_S = ARM_RELEASE_RAMP_S + 5.0
PUBLISH_HZ = 100.0
DDS_WRITE_TIMEOUT_S = 0.5
# CHANGEDSAFETY: original local adapter default was 6.0 rad/s, it was experimentally
# relaxed to 12.0 rad/s, and the current reviewed value restores the original 6.0 rad/s.
# This is the local measured arm-dq watchdog ceiling, not an official Unitree limit.
MAX_ARM_DQ_RAD_S = 6.0
MAX_ARM_TRACKING_ERROR_RAD = 0.35
# CHANGEDSAFETY: original local adapter default was 0.50 rad; current is 1.50 rad.
# This is max abs(measured hand q - commanded hand q), not a speed limit.
MAX_HAND_TRACKING_ERROR_RAD = 1.5
# The original 0.50-rad threshold remains a warning-only diagnostic.  It must
# persist across distinct hand-state samples for 0.20 s; 0.40 rad hysteresis
# prevents repeated warnings at the boundary.  Only the aligned 1.50-rad gate
# above is an actuator fault.
HAND_TRACKING_WARNING_RAD = 0.50
HAND_TRACKING_WARNING_CLEAR_RAD = 0.40
HAND_TRACKING_WARNING_DWELL_S = 0.20
HAND_COMMAND_HISTORY_SIZE = 128
TRACKING_GRACE_S = 0.50
MAX_ACTION_LATENESS_S = 0.02

# OFFICIAL: Unitree G1 asset mapping identifies mode_machine 6 as
# g1_29dof_lock_waist_with_hand_rev_1_0: waist yaw remains active while
# roll/pitch are locked (eval_robot/assets/g1/README.md). This matches the
# embodiment used to collect and train the deployed policy.
QUALIFIED_REAL_MODE_MACHINE = 6
PREARM_STATIONARY_DWELL_S = 0.5
PREARM_STATE_MAX_AGE_S = 0.05
PREARM_MAX_ARM_DQ_RAD_S = 0.10
PREARM_MAX_POSITION_DRIFT_RAD = 0.02
PREARM_MIN_DISTINCT_SAMPLES = 5
INITIALIZATION_ARM_SPEED_RAD_S = 0.25
INITIALIZATION_HAND_SPEED_RAD_S = 0.50
INITIALIZATION_MAX_ARM_STEP_RAD = INITIALIZATION_ARM_SPEED_RAD_S / PUBLISH_HZ
INITIALIZATION_MAX_HAND_STEP_RAD = INITIALIZATION_HAND_SPEED_RAD_S / PUBLISH_HZ
INITIALIZATION_MIN_MOVE_S = 0.50
INITIALIZATION_MAX_DURATION_S = 30.0
INITIALIZATION_START_TIMEOUT_S = 3.0
INITIALIZATION_START_DWELL_S = 0.50
INITIALIZATION_CONVERGENCE_TIMEOUT_S = 10.0
INITIALIZATION_CONVERGENCE_DWELL_S = 0.50
INITIALIZATION_MIN_DISTINCT_SAMPLES = 5
INITIALIZATION_ARM_TOLERANCE_RAD = 0.05
INITIALIZATION_HAND_TOLERANCE_RAD = 0.10
INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S = 0.10
INITIALIZATION_MAX_POSITION_DRIFT_RAD = 0.02
INITIALIZATION_COMMAND_MAX_AGE_S = 0.25

# CHANGEDSAFETY: the original deployment adapter had no policy-output
# conditioner. XR conditioning is now the CLI default; ``none`` preserves the
# original fail-on-raw-step behavior for controlled comparisons.
# XR's recorded arm action is already the post-IK, four-sample moving-average
# target.  The final real-arm publisher target was not recorded: at 250 Hz it
# globally rescaled desired-minus-measured q to a 0.08 rad lead, ramping to
# 0.12 rad over five seconds.  Preserve those command-lead values at this
# adapter's 100 Hz publisher instead of incorrectly multiplying 20/30 rad/s by
# the slower period.
XR_ARM_INITIAL_COMMAND_LEAD_RAD = 20.0 / 250.0
XR_ARM_FINAL_COMMAND_LEAD_RAD = 30.0 / 250.0
XR_ARM_COMMAND_LEAD_RAMP_S = 5.0
# Dex3 targets in the dataset already passed XR's alpha=0.2 retargeting filter.
# Do not apply that filter again here. The policy target feeds the final range
# and command-slew conditioner directly.
# CHANGEDSAFETY: the immediately pre-conditioner configuration checked 0.10 rad
# arm and 0.60 rad hand target steps at 30 Hz and had no 100 Hz conditioner.
# Preserve the arm's 3 rad/s effective target slew at the publisher rate.
MAX_CONDITIONED_ARM_STEP_RAD = MAX_ARM_STEP_RAD * CONTROL_HZ / PUBLISH_HZ
# OFFICIAL: velocity maxima from both checked-in Unitree Dex3 URDFs. This is a
# better-grounded outgoing ceiling than the prior invented 18 rad/s scalar, but
# a URDF maximum is not by itself a completed real-hardware qualification.
DEX3_URDF_MAX_VELOCITY_RAD_S = np.array([6.857, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
MAX_CONDITIONED_HAND_STEP_RAD = DEX3_URDF_MAX_VELOCITY_RAD_S / PUBLISH_HZ
COMMAND_CONDITIONING_MODES = ("none", "xr")

# IsaacLab publishes/consumes its right hand as thumb, middle, index.  The real
# Dex3 and recorded dataset use thumb, index, middle.  This permutation is its own
# inverse and must be applied to simulation states and commands only.
SIM_RIGHT_HAND_PERMUTATION = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)
TELEIMAGER_CONFIG_PORT = 60000
TELEIMAGER_CONFIG_TIMEOUT_S = 1.0
RGBD_MAX_RECEIVE_AGE_S = 0.15


class RtcTerminalEvent(DeploymentError):
    """The child has already entered powered HOLD or completed its RTC budget."""

    def __init__(self, outcome: str, detail: Any):
        super().__init__(f"RTC {outcome}: {detail}")
        self.outcome = outcome
        self.detail = detail


class ImmediateControlEvent(DeploymentError):
    """An operator STOP/release interrupted a blocking actuator operation."""

    def __init__(self, action: str):
        if action not in {"hold", "release"}:
            raise ValueError(f"Unknown immediate operator action {action!r}")
        super().__init__(f"Operator requested immediate {action}")
        self.action = action


@dataclass(frozen=True)
class RobotState:
    captured_at: float
    mode_machine: int
    arm: np.ndarray
    arm_dq: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    # DDS hand messages have no qualified capture timestamp. These are local
    # monotonic callback-receipt times, kept separately so a 50 Hz hand sample
    # is not compared with a newer 100 Hz command or counted twice.
    left_hand_received_at: float | None = None
    right_hand_received_at: float | None = None
    arm_received_at: float | None = None


@dataclass(frozen=True)
class HandFreshnessResult:
    ready: bool
    entered: bool = False
    recovered: bool = False
    pause_s: float = 0.0
    stale_hands: tuple[str, ...] = ()
    max_age_s: float = 0.0


class HandStateFreshnessGate:
    """Turn short Dex3 delivery gaps into a bounded motion pause."""

    def __init__(self) -> None:
        self._active = False
        self._started_at = 0.0
        self._fresh_samples = 0
        self._last_left_at = 0.0
        self._last_right_at = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._started_at = 0.0
        self._fresh_samples = 0
        self._last_left_at = 0.0
        self._last_right_at = 0.0

    def check(self, state: RobotState, *, now: float | None = None) -> HandFreshnessResult:
        checked_at = time.monotonic() if now is None else float(now)
        left_at = state.captured_at if state.left_hand_received_at is None else state.left_hand_received_at
        right_at = state.captured_at if state.right_hand_received_at is None else state.right_hand_received_at
        ages = {
            "left": max(0.0, checked_at - float(left_at)),
            "right": max(0.0, checked_at - float(right_at)),
        }
        stale_hands = tuple(
            name for name, age_s in ages.items() if age_s > ACTUATOR_HAND_STATE_WARNING_AGE_S
        )
        max_age_s = max(ages.values())
        if stale_hands:
            entered = not self._active
            if entered:
                self._active = True
                self._started_at = checked_at
            self._fresh_samples = 0
            self._last_left_at = float(left_at)
            self._last_right_at = float(right_at)
            return HandFreshnessResult(
                ready=False,
                entered=entered,
                stale_hands=stale_hands,
                max_age_s=max_age_s,
            )

        if not self._active:
            return HandFreshnessResult(ready=True, max_age_s=max_age_s)

        # Count actual paired DDS updates, not repeated 100 Hz reads of one
        # cached sample.
        if float(left_at) <= self._last_left_at or float(right_at) <= self._last_right_at:
            return HandFreshnessResult(ready=False, max_age_s=max_age_s)
        self._last_left_at = float(left_at)
        self._last_right_at = float(right_at)
        self._fresh_samples += 1
        if self._fresh_samples < ACTUATOR_HAND_RECOVERY_SAMPLES:
            return HandFreshnessResult(ready=False, max_age_s=max_age_s)

        pause_s = checked_at - self._started_at
        self.reset()
        return HandFreshnessResult(
            ready=True,
            recovered=True,
            pause_s=pause_s,
            max_age_s=max_age_s,
        )


@dataclass(frozen=True)
class CameraImages:
    rgb: np.ndarray
    depth_gray: np.ndarray | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class RtcExecutionSnapshot:
    sequence: int
    action_index: int
    plan_length: int
    total_actions: int
    action_budget: int


@dataclass(frozen=True)
class PublishedHandTarget:
    """Canonical hand target recorded after its DDS Write succeeded."""

    completed_at: float
    target: np.ndarray


class HandTrackingWatchdog:
    """Time-align hand feedback to successful DDS targets and track warnings."""

    def __init__(self) -> None:
        self._last_sample_at = np.full(2, -np.inf, dtype=np.float64)
        self._violation_since = np.full((2, HAND_DOF), np.nan, dtype=np.float64)
        self._warning_active = np.zeros((2, HAND_DOF), dtype=bool)

    def reset(self, backend: Any | None = None) -> None:
        self._last_sample_at.fill(-np.inf)
        self._violation_since.fill(np.nan)
        self._warning_active.fill(False)
        if backend is not None:
            reset_history = getattr(backend, "reset_hand_publish_history", None)
            if reset_history is not None:
                reset_history()

    @staticmethod
    def _aligned_target(history: Any, sample_at: float) -> PublishedHandTarget | None:
        if history is None:
            return None
        for entry in reversed(history):
            if entry.completed_at <= sample_at:
                return entry
        return None

    @staticmethod
    def _target_slew(history: Any, joint: int) -> float | None:
        if history is None or len(history) < 2:
            return None
        latest = history[-1]
        previous = history[-2]
        elapsed = latest.completed_at - previous.completed_at
        if elapsed <= 0.0:
            return None
        return float((latest.target[joint] - previous.target[joint]) / elapsed)

    def enforce(
        self,
        backend: Any,
        state: RobotState,
        *,
        now: float | None = None,
        context: str = "unknown",
        desired_left: np.ndarray | None = None,
        desired_right: np.ndarray | None = None,
    ) -> bool:
        """Check each newly received hand sample; return whether history was used."""

        histories = (
            getattr(backend, "_left_hand_publish_history", None),
            getattr(backend, "_right_hand_publish_history", None),
        )
        if histories[0] is None or histories[1] is None:
            return False

        checked_at = time.monotonic() if now is None else float(now)
        hands = (
            (
                "left hand",
                np.asarray(state.left_hand, dtype=np.float64),
                state.left_hand_received_at,
                LEFT_HAND_JOINT_NAMES,
                desired_left,
            ),
            (
                "right hand",
                np.asarray(state.right_hand, dtype=np.float64),
                state.right_hand_received_at,
                RIGHT_HAND_JOINT_NAMES,
                desired_right,
            ),
        )
        for hand_index, ((hand_name, measured, received_at, joint_names, desired), history) in enumerate(
            zip(hands, histories, strict=True)
        ):
            sample_at = float(state.captured_at if received_at is None else received_at)
            if sample_at <= self._last_sample_at[hand_index]:
                continue
            self._last_sample_at[hand_index] = sample_at

            aligned = self._aligned_target(history, sample_at)
            if aligned is None:
                # This is expected only for feedback received before the first
                # successful post-reset hand command. A subsequent fresh sample
                # will have a matching history entry.
                continue

            errors = measured - aligned.target
            joint = int(np.argmax(np.abs(errors)))
            error = float(errors[joint])
            if abs(error) > MAX_HAND_TRACKING_ERROR_RAD:
                latest = history[-1]
                latest_target = float(latest.target[joint])
                latest_error = float(measured[joint] - latest_target)
                raw_desired = None if desired is None else float(np.asarray(desired)[joint])
                slew = self._target_slew(history, joint)
                violation_since = self._violation_since[hand_index, joint]
                violation_duration = 0.0 if np.isnan(violation_since) else sample_at - violation_since
                raise DeploymentError(
                    f"{hand_name.title()} time-aligned tracking error at joint {joint} "
                    f"({joint_names[joint]}) is {error:+.3f} rad (measured minus aligned target); "
                    f"MAX_HAND_TRACKING_ERROR_RAD={MAX_HAND_TRACKING_ERROR_RAD:.3f} rad; "
                    f"measured_q={float(measured[joint]):+.4f}, "
                    f"aligned_target_q={float(aligned.target[joint]):+.4f}, "
                    f"latest_published_q={latest_target:+.4f}, latest_error={latest_error:+.4f}, "
                    f"raw_desired_q={raw_desired!r}, target_slew_rad_s={slew!r}, "
                    f"state_age_s={max(0.0, checked_at - sample_at):.4f}, "
                    f"aligned_command_age_s={max(0.0, sample_at - aligned.completed_at):.4f}, "
                    f"violation_duration_s={max(0.0, violation_duration):.4f}, context={context}"
                )

            magnitudes = np.abs(errors)
            for joint_index, magnitude in enumerate(magnitudes):
                if magnitude >= HAND_TRACKING_WARNING_RAD:
                    since = self._violation_since[hand_index, joint_index]
                    if np.isnan(since):
                        self._violation_since[hand_index, joint_index] = sample_at
                    elif (
                        sample_at - since >= HAND_TRACKING_WARNING_DWELL_S
                        and not self._warning_active[hand_index, joint_index]
                    ):
                        self._warning_active[hand_index, joint_index] = True
                        LOGGER.warning(
                            "%s tracking error persisted for %.3fs at joint %d (%s): "
                            "%+.3f rad measured-minus-aligned-target; warning threshold=%.3f rad; %s",
                            hand_name.title(),
                            sample_at - since,
                            joint_index,
                            joint_names[joint_index],
                            float(errors[joint_index]),
                            HAND_TRACKING_WARNING_RAD,
                            context,
                        )
                elif magnitude <= HAND_TRACKING_WARNING_CLEAR_RAD:
                    if self._warning_active[hand_index, joint_index]:
                        LOGGER.info(
                            "%s tracking warning recovered at joint %d (%s): %+.3f rad",
                            hand_name.title(),
                            joint_index,
                            joint_names[joint_index],
                            float(errors[joint_index]),
                        )
                    self._violation_since[hand_index, joint_index] = np.nan
                    self._warning_active[hand_index, joint_index] = False
        return True


class XrPolicyOutputConditioner:
    """Fast stateful XR-style final-command conditioner for policy targets.

    Raw model plans are validated separately for shape, finiteness, and joint
    position limits.  This object owns only deployment-time continuity.  It is
    reset by initialization, warm-start, STOP/HOLD, and terminal RTC paths, but
    not by normal synchronous chunks or RTC replacements.
    """

    def __init__(self) -> None:
        self._desired_arm: np.ndarray | None = None
        self._desired_left: np.ndarray | None = None
        self._desired_right: np.ndarray | None = None
        self._started_at: float | None = None

    def reset(self, arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in (arm, left, right))
        if tuple(value.shape for value in arrays) != ((ARM_DOF,), (HAND_DOF,), (HAND_DOF,)):
            raise DeploymentError("XR policy-output conditioner reset target has the wrong shape")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise DeploymentError("XR policy-output conditioner reset target is not finite")
        self._desired_arm = arrays[0].copy()
        self._desired_left = arrays[1].copy()
        self._desired_right = arrays[2].copy()
        self._started_at = None

    def set_desired(self, arm: np.ndarray, left: np.ndarray, right: np.ndarray, *, now: float) -> None:
        target = ActionChunk(
            arm=np.asarray(arm, dtype=np.float64)[None],
            left_hand=np.asarray(left, dtype=np.float64)[None],
            right_hand=np.asarray(right, dtype=np.float64)[None],
        )
        validate_action_chunk_limits(target)
        self._desired_arm = target.arm[0].copy()
        self._desired_left = target.left_hand[0].copy()
        self._desired_right = target.right_hand[0].copy()
        if self._started_at is None:
            self._started_at = float(now)

    def next_command(
        self,
        measured_arm: np.ndarray,
        current_arm: np.ndarray,
        current_left: np.ndarray,
        current_right: np.ndarray,
        *,
        now: float,
    ) -> ActionChunk:
        if self._desired_arm is None or self._desired_left is None or self._desired_right is None:
            raise DeploymentError("XR policy-output conditioner has not been seeded")
        measured_arm = np.asarray(measured_arm, dtype=np.float64)
        current_arm = np.asarray(current_arm, dtype=np.float64)
        current_left = np.asarray(current_left, dtype=np.float64)
        current_right = np.asarray(current_right, dtype=np.float64)
        if self._started_at is None:
            ramp = 0.0
        else:
            ramp = np.clip((float(now) - self._started_at) / XR_ARM_COMMAND_LEAD_RAMP_S, 0.0, 1.0)
        arm_lead = XR_ARM_INITIAL_COMMAND_LEAD_RAD + ramp * (
            XR_ARM_FINAL_COMMAND_LEAD_RAD - XR_ARM_INITIAL_COMMAND_LEAD_RAD
        )
        arm_delta = self._desired_arm - measured_arm
        arm_scale = max(1.0, float(np.max(np.abs(arm_delta))) / arm_lead)
        arm = measured_arm + arm_delta / arm_scale
        left = self._desired_left.copy()
        right = self._desired_right.copy()
        # The desired policy target has already passed hard range validation.
        # Project only the measured-dependent conditioned candidate.  This lets
        # a hand state admitted by the wider measured-state tolerance move
        # inward without making an out-of-range DDS target.  The subsequent
        # command-step limiter remains authoritative.
        arm_lower = ARM_LOWER + JOINT_LIMIT_MARGIN_RAD
        arm_upper = ARM_UPPER - JOINT_LIMIT_MARGIN_RAD
        left_lower = LEFT_HAND_LOWER - HAND_LIMIT_TOLERANCE_RAD
        left_upper = LEFT_HAND_UPPER + HAND_LIMIT_TOLERANCE_RAD
        right_lower = RIGHT_HAND_LOWER - HAND_LIMIT_TOLERANCE_RAD
        right_upper = RIGHT_HAND_UPPER + HAND_LIMIT_TOLERANCE_RAD
        arm = np.clip(arm, arm_lower, arm_upper)
        left = np.clip(
            left,
            left_lower,
            left_upper,
        )
        right = np.clip(
            right,
            right_lower,
            right_upper,
        )
        # Feedback-relative lead limiting alone does not bound a command
        # reversal: measured+lead -> measured-lead could jump by twice the lead.
        # Rescale each command group once more against the last published target
        # so conditioning happens before, rather than merely tripping, the hard
        # final step invariant.
        arm_command_delta = arm - current_arm
        arm_command_scale = max(
            1.0,
            float(np.max(np.abs(arm_command_delta))) / MAX_CONDITIONED_ARM_STEP_RAD,
        )
        arm = current_arm + arm_command_delta / arm_command_scale
        left_delta = left - current_left
        left_scale = max(
            1.0,
            float(np.max(np.abs(left_delta) / MAX_CONDITIONED_HAND_STEP_RAD)),
        )
        left = current_left + left_delta / left_scale
        right_delta = right - current_right
        right_scale = max(
            1.0,
            float(np.max(np.abs(right_delta) / MAX_CONDITIONED_HAND_STEP_RAD)),
        )
        right = current_right + right_delta / right_scale
        # A group-wide slew scale can otherwise leave a different joint just
        # outside its target range after reset from a measured-only tolerance.
        # Project once more after slew limiting. This changes an ordinary
        # in-range step by zero; for an out-of-range held seed it performs only
        # the minimum inward recovery needed to make the outgoing target legal.
        arm = np.clip(arm, arm_lower, arm_upper)
        left = np.clip(left, left_lower, left_upper)
        right = np.clip(right, right_lower, right_upper)
        result = ActionChunk(
            arm=np.ascontiguousarray(arm[None]),
            left_hand=np.ascontiguousarray(left[None]),
            right_hand=np.ascontiguousarray(right[None]),
        )
        # This is the final outgoing command.  It must satisfy both absolute
        # limits and target-to-target step ceilings before DDS sees it.
        validate_action_chunk(result, current_arm, current_left, current_right)
        arm_recovery = np.maximum.reduce((arm_lower - current_arm, current_arm - arm_upper, np.zeros(ARM_DOF)))
        left_recovery = np.maximum.reduce((left_lower - current_left, current_left - left_upper, np.zeros(HAND_DOF)))
        right_recovery = np.maximum.reduce(
            (right_lower - current_right, current_right - right_upper, np.zeros(HAND_DOF))
        )
        if np.any(np.abs(result.arm[0] - current_arm) > np.maximum(MAX_CONDITIONED_ARM_STEP_RAD, arm_recovery) + 1e-12):
            raise DeploymentError("XR conditioned arm command exceeded its 100 Hz slew ceiling")
        if np.any(
            np.abs(result.left_hand[0] - current_left)
            > np.maximum(MAX_CONDITIONED_HAND_STEP_RAD, left_recovery) + 1e-12
        ) or np.any(
            np.abs(result.right_hand[0] - current_right)
            > np.maximum(MAX_CONDITIONED_HAND_STEP_RAD, right_recovery) + 1e-12
        ):
            raise DeploymentError("XR conditioned hand command exceeded its 100 Hz slew ceiling")
        return result


def _largest_named_value(
    groups: tuple[tuple[str, np.ndarray, tuple[str, ...]], ...],
) -> tuple[str, int, str, float]:
    """Return the signed value with the largest magnitude and its joint identity."""

    best: tuple[str, int, str, float] | None = None
    for group_name, raw_values, joint_names in groups:
        values = np.asarray(raw_values, dtype=np.float64)
        joint = int(np.argmax(np.abs(values)))
        candidate = (group_name, joint, joint_names[joint], float(values[joint]))
        if best is None or abs(candidate[3]) > abs(best[3]):
            best = candidate
    assert best is not None
    return best


def _smooth_initialization_path(
    current: np.ndarray,
    target: np.ndarray,
    steps: int,
) -> np.ndarray:
    phase = np.arange(1, steps + 1, dtype=np.float64) / steps
    blend = phase * phase * (3.0 - 2.0 * phase)
    return current[None] + blend[:, None] * (target - current)[None]


def _strict_hand_target_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return lower - HAND_LIMIT_TOLERANCE_RAD, upper + HAND_LIMIT_TOLERANCE_RAD


def _validate_initialization_hand_recovery(
    name: str,
    values: np.ndarray,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: tuple[str, ...],
) -> None:
    """Allow only a bounded inward transition from measured-only tolerance.

    Policy targets use the stricter hand target range.  Initialization is the
    one place where its starting state can legitimately be outside that range:
    measured state admits a wider tolerance so encoder noise at a URDF limit
    does not fault the reader.  Until a joint first enters the strict target
    range, every initialization command must stay inside the measured range and
    move monotonically inward.  Once inside, it may never leave again.
    """

    strict_lower, strict_upper = _strict_hand_target_bounds(lower, upper)
    measured_lower = lower - MEASURED_LIMIT_TOLERANCE_RAD
    measured_upper = upper + MEASURED_LIMIT_TOLERANCE_RAD
    previous = np.asarray(current, dtype=np.float64).copy()
    entered_strict_range = (previous >= strict_lower) & (previous <= strict_upper)

    for step, target in enumerate(np.asarray(values, dtype=np.float64)):
        bad_measured = np.flatnonzero((target < measured_lower) | (target > measured_upper))
        if bad_measured.size:
            joint = int(bad_measured[0])
            raise DeploymentError(
                f"Initialization {name} recovery left the measured-state range at step {step}, "
                f"joint {joint} ({joint_names[joint]}): {target[joint]:.4f} rad"
            )

        delta = target - previous
        bad_step = np.flatnonzero(np.abs(delta) > INITIALIZATION_MAX_HAND_STEP_RAD + 1e-12)
        if bad_step.size:
            joint = int(bad_step[0])
            raise DeploymentError(
                f"Initialization {name} recovery step is too large at step {step}, joint {joint} "
                f"({joint_names[joint]}): {delta[joint]:+.4f} rad; "
                f"INITIALIZATION_MAX_HAND_STEP_RAD={INITIALIZATION_MAX_HAND_STEP_RAD:.4f} rad"
            )

        below = target < strict_lower
        above = target > strict_upper
        for joint in np.flatnonzero(below):
            if entered_strict_range[joint] or previous[joint] >= strict_lower[joint] or delta[joint] < -1e-12:
                raise DeploymentError(
                    f"Initialization {name} recovery is not monotonic inward at step {step}, "
                    f"joint {joint} ({joint_names[joint]})"
                )
        for joint in np.flatnonzero(above):
            if entered_strict_range[joint] or previous[joint] <= strict_upper[joint] or delta[joint] > 1e-12:
                raise DeploymentError(
                    f"Initialization {name} recovery is not monotonic inward at step {step}, "
                    f"joint {joint} ({joint_names[joint]})"
                )

        entered_strict_range |= ~(below | above)
        previous = target

    if not np.all(entered_strict_range):
        joint = int(np.flatnonzero(~entered_strict_range)[0])
        raise DeploymentError(
            f"Initialization {name} recovery did not enter the strict target range at joint {joint} "
            f"({joint_names[joint]})"
        )


def _validate_moving_initialization_chunk(
    chunk: ActionChunk,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
) -> None:
    """Validate a moving init without weakening the policy-action contract."""

    left_lower, left_upper = _strict_hand_target_bounds(LEFT_HAND_LOWER, LEFT_HAND_UPPER)
    right_lower, right_upper = _strict_hand_target_bounds(RIGHT_HAND_LOWER, RIGHT_HAND_UPPER)

    # Reuse the ordinary strict validator for structure, arm limits, final hand
    # limits, and the hard command-step backstop.  Only initialization's
    # measured-origin hand samples are projected in this validation surrogate;
    # the actual samples are checked against the narrower recovery rules below.
    strict_chunk = ActionChunk(
        arm=chunk.arm,
        left_hand=np.clip(chunk.left_hand, left_lower, left_upper),
        right_hand=np.clip(chunk.right_hand, right_lower, right_upper),
    )
    validate_action_chunk(strict_chunk, current_arm, current_left, current_right)
    _validate_initialization_hand_recovery(
        "left hand",
        chunk.left_hand,
        current_left,
        LEFT_HAND_LOWER,
        LEFT_HAND_UPPER,
        LEFT_HAND_JOINT_NAMES,
    )
    _validate_initialization_hand_recovery(
        "right hand",
        chunk.right_hand,
        current_right,
        RIGHT_HAND_LOWER,
        RIGHT_HAND_UPPER,
        RIGHT_HAND_JOINT_NAMES,
    )


def build_initialization_chunk(state: RobotState, spec: InitializationSpec) -> ActionChunk:
    """Resolve measured targets and create a bounded smooth joint-space path."""

    validate_initialization_spec(spec)
    validate_measured_state(state.arm, state.arm_dq, state.left_hand, state.right_hand)
    current = (
        np.asarray(state.arm, dtype=np.float64),
        np.asarray(state.left_hand, dtype=np.float64),
        np.asarray(state.right_hand, dtype=np.float64),
    )
    if spec.mode == "measured":
        # This is an exact no-motion hold, not a policy action.  The measured
        # state was accepted by validate_measured_state above and must not be
        # rejected merely because the policy target tolerance is narrower.
        return ActionChunk(
            arm=np.ascontiguousarray(current[0][None]).copy(),
            left_hand=np.ascontiguousarray(current[1][None]).copy(),
            right_hand=np.ascontiguousarray(current[2][None]).copy(),
        )

    left_lower, left_upper = _strict_hand_target_bounds(LEFT_HAND_LOWER, LEFT_HAND_UPPER)
    right_lower, right_upper = _strict_hand_target_bounds(RIGHT_HAND_LOWER, RIGHT_HAND_UPPER)
    targets = (
        current[0].copy() if spec.arm is None else np.asarray(spec.arm, dtype=np.float64).copy(),
        (
            np.clip(current[1], left_lower, left_upper)
            if spec.left_hand is None
            else np.asarray(spec.left_hand, dtype=np.float64).copy()
        ),
        (
            np.clip(current[2], right_lower, right_upper)
            if spec.right_hand is None
            else np.asarray(spec.right_hand, dtype=np.float64).copy()
        ),
    )
    max_steps = (INITIALIZATION_MAX_ARM_STEP_RAD, INITIALIZATION_MAX_HAND_STEP_RAD, INITIALIZATION_MAX_HAND_STEP_RAD)
    # A cubic smoothstep has a maximum slope of 1.5.  This initial estimate is
    # checked below against the actual discrete path before it can be used.
    movement = any(np.any(target != measured) for measured, target in zip(current, targets, strict=True))
    steps = max(
        1,
        *(
            int(np.ceil(1.5 * float(np.max(np.abs(target - measured))) / max_step))
            for measured, target, max_step in zip(current, targets, max_steps, strict=True)
        ),
    )
    if movement:
        steps = max(steps, round(INITIALIZATION_MIN_MOVE_S * PUBLISH_HZ))
    while True:
        paths = tuple(
            _smooth_initialization_path(measured, target, steps)
            for measured, target in zip(current, targets, strict=True)
        )
        actual_steps = tuple(
            float(np.max(np.abs(np.diff(np.vstack((measured, path)), axis=0))))
            for measured, path in zip(current, paths, strict=True)
        )
        if all(actual <= limit + 1e-12 for actual, limit in zip(actual_steps, max_steps, strict=True)):
            break
        steps += 1

    duration_s = steps / PUBLISH_HZ
    if duration_s > INITIALIZATION_MAX_DURATION_S:
        raise DeploymentError(
            f"Initialization path needs {duration_s:.1f}s at the fixed conservative rate; "
            f"the limit is {INITIALIZATION_MAX_DURATION_S:.1f}s"
        )
    chunk = ActionChunk(
        arm=np.ascontiguousarray(paths[0]),
        left_hand=np.ascontiguousarray(paths[1]),
        right_hand=np.ascontiguousarray(paths[2]),
    )
    _validate_moving_initialization_chunk(chunk, *current)
    return chunk


def initialize_dds(simulation: bool, network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    domain = 1 if simulation else 0
    if network_interface:
        ChannelFactoryInitialize(domain, networkInterface=network_interface)
    else:
        ChannelFactoryInitialize(domain)


class G1Dex3StateReader:
    """Read arm and hand state without constructing command publishers."""

    def __init__(
        self,
        simulation: bool = False,
        max_age_s: float = STATE_MAX_AGE_S,
        hand_max_age_s: float | None = None,
    ):
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex
        from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
            Dex3_1_Left_JointIndex,
            Dex3_1_Right_JointIndex,
        )
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        self._simulation = simulation
        self._arm_max_age_s = float(max_age_s)
        self._hand_max_age_s = float(max_age_s if hand_max_age_s is None else hand_max_age_s)
        if self._arm_max_age_s <= 0.0 or self._hand_max_age_s <= 0.0:
            raise ValueError("State freshness limits must be positive")
        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        self._left_indices = tuple(int(index) for index in Dex3_1_Left_JointIndex)
        self._right_indices = tuple(int(index) for index in Dex3_1_Right_JointIndex)
        self._lock = threading.Lock()
        self._messages: dict[str, Any] = {"arm": None, "left": None, "right": None}
        self._updated_at = {key: 0.0 for key in self._messages}
        self._rejected_zero_hand_frames = {"left": 0, "right": 0}
        self._subscribers = {
            "arm": ChannelSubscriber("rt/lowstate", LowState_),
            "left": ChannelSubscriber("rt/dex3/left/state", HandState_),
            "right": ChannelSubscriber("rt/dex3/right/state", HandState_),
        }
        for key, subscriber in self._subscribers.items():
            subscriber.Init(handler=self._make_handler(key))

    def _make_handler(self, key: str):
        def update(message: Any) -> None:
            if message is None:
                return
            # A failed Dex3 boot can keep publishing a syntactically valid,
            # all-zero placeholder at a high callback rate.  Treating those
            # frames as fresh poisoned measured initialization on the real
            # robot.  Preserve the last valid sample instead; sustained
            # placeholders then naturally cross the soft/hard age gates.
            if not self._simulation and key in {"left", "right"}:
                indices = self._left_indices if key == "left" else self._right_indices
                try:
                    all_zero = all(float(message.motor_state[index].q) == 0.0 for index in indices)
                except (AttributeError, IndexError, TypeError, ValueError):
                    all_zero = False
                if all_zero:
                    with self._lock:
                        self._rejected_zero_hand_frames[key] += 1
                    return
            with self._lock:
                self._messages[key] = message
                self._updated_at[key] = time.monotonic()

        return update

    def latest(self) -> RobotState:
        now = time.monotonic()
        with self._lock:
            messages = dict(self._messages)
            updated_at = dict(self._updated_at)
            rejected_zero = dict(self._rejected_zero_hand_frames)
        limits = {
            "arm": self._arm_max_age_s,
            "left": self._hand_max_age_s,
            "right": self._hand_max_age_s,
        }
        stale = []
        for key, message in messages.items():
            if message is None:
                detail = f", rejected {rejected_zero[key]} all-zero frames" if key in rejected_zero else ""
                stale.append(f"{key} (missing{detail})")
                continue
            age_s = now - updated_at[key]
            if age_s > limits[key]:
                detail = f", rejected {rejected_zero[key]} all-zero frames" if key in rejected_zero else ""
                stale.append(f"{key} (age {age_s:.3f}s > {limits[key]:.3f}s{detail})")
        if stale:
            raise TimeoutError(f"Stale Unitree state: {', '.join(stale)}")

        arm_message = messages["arm"]
        left_message = messages["left"]
        right_message = messages["right"]
        arm = np.array([arm_message.motor_state[index].q for index in self._arm_indices], dtype=np.float64)
        arm_dq = np.array([arm_message.motor_state[index].dq for index in self._arm_indices], dtype=np.float64)
        left = np.array([left_message.motor_state[index].q for index in self._left_indices], dtype=np.float64)
        right = np.array([right_message.motor_state[index].q for index in self._right_indices], dtype=np.float64)
        if self._simulation:
            right = right[SIM_RIGHT_HAND_PERMUTATION]
        validate_measured_state(arm, arm_dq, left, right)
        return RobotState(
            captured_at=min(updated_at.values()),
            mode_machine=int(getattr(arm_message, "mode_machine", 0)),
            arm=arm,
            arm_dq=arm_dq,
            left_hand=left,
            right_hand=right,
            left_hand_received_at=updated_at["left"],
            right_hand_received_at=updated_at["right"],
            arm_received_at=updated_at["arm"],
        )

    def read(self, timeout_s: float = 3.0) -> RobotState:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.latest()
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.002)
        raise TimeoutError(f"Timed out waiting for fresh Unitree state ({last_error})")

    def close(self) -> None:
        for subscriber in self._subscribers.values():
            with contextlib.suppress(Exception):
                subscriber.Close()


def _decode_color_jpeg_rgb(jpg: bytes | None, camera_config: dict[str, Any]) -> np.ndarray:
    if not jpg:
        raise TimeoutError("Head-camera transport has no fresh JPEG")
    encoded = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise DeploymentError("TeleImager returned an undecodable head-camera JPEG")

    try:
        head = camera_config["head_camera"]
        configured_height, configured_width = (int(value) for value in head["image_shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("TeleImager head-camera configuration is incomplete") from exc
    if bgr.shape != (configured_height, configured_width, 3):
        raise DeploymentError(
            f"Head frame {bgr.shape} does not match TeleImager config {(configured_height, configured_width, 3)}"
        )
    if head.get("binocular", False):
        if configured_width % 2:
            raise DeploymentError(f"Binocular frame width {configured_width} is not even")
        bgr = bgr[:, : configured_width // 2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    if list(rgb.shape) != EXPECTED_EGO_VIEW_SHAPE:
        raise DeploymentError(
            f"Decoded color_0 shape {rgb.shape} does not match the training contract {tuple(EXPECTED_EGO_VIEW_SHAPE)}"
        )
    return rgb


def decode_color_0_rgb(frame: Any, camera_config: dict[str, Any]) -> np.ndarray:
    """Decode a fresh TeleImage JPEG and reproduce recorded camera ``color_0``."""

    return _decode_color_jpeg_rgb(getattr(frame, "jpg", None), camera_config)


def _decode_depth_png_u16(encoded_png: bytes) -> np.ndarray:
    depth = cv2.imdecode(np.frombuffer(encoded_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise DeploymentError("RGBD packet contains an undecodable aligned-depth PNG")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise DeploymentError(
            f"RGBD aligned depth must decode to an HxW uint16 image; got shape={depth.shape}, dtype={depth.dtype}"
        )
    return depth


def request_live_camera_config(
    host: str,
    port: int = TELEIMAGER_CONFIG_PORT,
    timeout_s: float = TELEIMAGER_CONFIG_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch TeleImager config without its requester's local-YAML fallback."""

    timeout_ms = max(1, round(timeout_s * 1000.0))
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(f"tcp://{host}:{port}")
        socket.send(b"GET_DATA")
        if not socket.poll(timeout_ms, zmq.POLLIN):
            raise DeploymentError(
                f"Live TeleImager config server {host}:{port} did not respond; "
                "refusing the client's local YAML fallback"
            )
        config = socket.recv_json()
        if not isinstance(config, dict):
            raise DeploymentError("Live TeleImager config response is not a JSON object")
        return config
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(f"Could not fetch live TeleImager config from {host}:{port}") from exc
    finally:
        socket.close(linger=0)
        context.term()


def _validate_live_head_config(
    config: dict[str, Any],
    *,
    requires_depth: bool,
) -> tuple[int, float | None]:
    head = config.get("head_camera")
    if not isinstance(head, dict):
        raise DeploymentError("TeleImager configuration has no head_camera object")
    if not head.get("enable_zmq", False):
        raise DeploymentError("TeleImager head-camera ZMQ stream is disabled")
    try:
        fps = float(head["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("TeleImager head camera has no numeric FPS") from exc
    if not np.isfinite(fps) or fps != CONTROL_HZ:
        raise DeploymentError(
            f"TeleImager head camera is configured for {head.get('fps')!r} FPS; "
            f"the training contract requires {CONTROL_HZ:g} FPS"
        )

    port_key = "zmq_port"
    depth_scale: float | None = None
    if requires_depth:
        if str(head.get("type", "")).lower() != "realsense":
            raise DeploymentError("RGBD checkpoint requires a RealSense TeleImager head camera")
        if not head.get("enable_depth", False):
            raise DeploymentError("RGBD checkpoint requires TeleImager aligned depth")
        if head.get("binocular", False):
            raise DeploymentError("RGBD checkpoint does not support a binocular head-camera layout")
        if head.get("rgbd_protocol") != RGBD_PROTOCOL:
            raise DeploymentError(
                f"RGBD checkpoint requires rgbd_protocol={RGBD_PROTOCOL!r}; got {head.get('rgbd_protocol')!r}"
            )
        port_key = "rgbd_zmq_port"
        try:
            depth_scale = float(head["depth_scale_m_per_unit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("TeleImager has no valid live RealSense depth scale") from exc
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise DeploymentError(f"TeleImager reported invalid depth scale {depth_scale!r}")
    try:
        port = int(head[port_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError(f"TeleImager has no valid {port_key}") from exc
    if not 1 <= port <= 65535:
        raise DeploymentError(f"TeleImager reported invalid {port_key} {port!r}")
    return port, depth_scale


class TeleimagerCamera:
    """TeleImager client selected automatically for colour or atomic RGBD."""

    def __init__(self, host: str, depth_encoding: DepthEncodingContract | None = None):
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

        self._client = None
        self._depth_encoding = depth_encoding
        self._requires_depth = depth_encoding is not None
        self._last_rgbd_sequence: int | None = None
        live_config = request_live_camera_config(host)
        stream_port, depth_scale = _validate_live_head_config(
            live_config,
            requires_depth=self._requires_depth,
        )
        if depth_scale is not None:
            self._depth_scale_m_per_unit = depth_scale
        try:
            self._client = ImageClient(
                host=host,
                request_bgr=False,
                request_rgbd=self._requires_depth,
            )
        except Exception:
            self.close()
            raise
        self.config = self._client.get_cam_config()
        if self.config != live_config:
            self.close()
            raise DeploymentError("TeleImager ImageClient config differs from the config returned by the live server")
        try:
            stream_key = (host, stream_port)
            self._head_subscriber = self._client._subscriber_manager._subscriber_threads[stream_key]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.close()
            raise DeploymentError("Could not verify the live TeleImager head subscriber") from exc
        if not self._head_subscriber.is_alive():
            self.close()
            raise DeploymentError("TeleImager head subscriber stopped during startup")
        self._reported_stream_fps = False

    def _assert_subscriber_alive(self) -> None:
        if not self._head_subscriber.is_alive():
            raise DeploymentError("TeleImager head subscriber stopped")

    def read_rgb(self, timeout_s: float = 3.0) -> np.ndarray:
        if getattr(self, "_requires_depth", False):
            raise DeploymentError("RGBD camera must be read atomically with read()")
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._assert_subscriber_alive()
                frame = self._client.get_head_frame()
                try:
                    measured_fps = float(getattr(frame, "fps", 0.0))
                except (TypeError, ValueError) as exc:
                    raise TimeoutError("TeleImager stream FPS is not numeric") from exc
                if not np.isfinite(measured_fps) or measured_fps <= 0.0:
                    raise TimeoutError("TeleImager stream has not established a live rolling FPS")
                rgb = decode_color_0_rgb(frame, self.config)
                if not self._reported_stream_fps:
                    LOGGER.info("TeleImager head stream is live at %.1f measured FPS", measured_fps)
                    self._reported_stream_fps = True
                return rgb
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for a fresh TeleImager frame ({last_error})")

    def _read_rgbd(self, timeout_s: float) -> CameraImages:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._assert_subscriber_alive()
                frame = self._client.get_head_rgbd_frame()
                if frame is None:
                    raise TimeoutError("TeleImager RGBD transport has no fresh packet")
                try:
                    measured_fps = float(self._client.get_head_rgbd_fps())
                except (TypeError, ValueError) as exc:
                    raise TimeoutError("TeleImager RGBD stream FPS is not numeric") from exc
                if not np.isfinite(measured_fps) or measured_fps <= 0.0:
                    raise TimeoutError("TeleImager RGBD stream has not established a live rolling FPS")
                if frame.received_monotonic_ns is None:
                    raise DeploymentError("TeleImager RGBD packet has no local receive timestamp")
                age_s = (time.monotonic_ns() - frame.received_monotonic_ns) / 1_000_000_000.0
                if age_s < 0.0 or age_s > RGBD_MAX_RECEIVE_AGE_S:
                    raise TimeoutError(f"TeleImager RGBD packet is stale ({age_s:.3f}s old)")
                if self._last_rgbd_sequence is not None:
                    if frame.sequence < self._last_rgbd_sequence:
                        raise DeploymentError(
                            "TeleImager RGBD sequence regressed from "
                            f"{self._last_rgbd_sequence} to {frame.sequence}; restart the policy runner"
                        )
                    if frame.sequence == self._last_rgbd_sequence:
                        raise TimeoutError(f"TeleImager RGBD sequence {frame.sequence} is not new")

                rgb = _decode_color_jpeg_rgb(frame.color_jpeg, self.config)
                depth_u16 = _decode_depth_png_u16(frame.aligned_depth_png)
                if list(depth_u16.shape) != EXPECTED_DEPTH_VIEW_SHAPE[:2]:
                    raise DeploymentError(
                        f"Aligned depth shape {depth_u16.shape} does not match the training contract "
                        f"{tuple(EXPECTED_DEPTH_VIEW_SHAPE[:2])}"
                    )
                assert self._depth_encoding is not None
                try:
                    depth_gray = encode_depth_gray_rgb(
                        depth_u16,
                        scale_m_per_unit=self._depth_scale_m_per_unit,
                        near_m=self._depth_encoding.near_m,
                        far_m=self._depth_encoding.far_m,
                    )
                except ValueError as exc:
                    raise DeploymentError(f"Could not encode aligned depth: {exc}") from exc
                self._last_rgbd_sequence = frame.sequence
                if not self._reported_stream_fps:
                    LOGGER.info("TeleImager atomic RGBD stream is live at %.1f measured FPS", measured_fps)
                    self._reported_stream_fps = True
                return CameraImages(rgb=rgb, depth_gray=depth_gray, sequence=frame.sequence)
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.005)
            except (TypeError, ValueError) as exc:
                raise DeploymentError(f"Invalid TeleImager RGBD packet: {exc}") from exc
        raise TimeoutError(f"Timed out waiting for a fresh TeleImager RGBD frame ({last_error})")

    def read(self, timeout_s: float = 3.0) -> CameraImages:
        if self._requires_depth:
            return self._read_rgbd(timeout_s)
        return CameraImages(rgb=self.read_rgb(timeout_s=timeout_s))

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            self._client = None
            client.close()


# Existing colour-only imports remain valid.  The default constructor still
# selects the legacy head-colour stream.
TeleimagerColourCamera = TeleimagerCamera


class _G1Dex3CommandBackend:
    """DDS publishers used only inside the actuator child process."""

    _supports_cleanup_phases = True

    def __init__(self, simulation: bool, network_interface: str | None):
        initialize_dds(simulation, network_interface)
        self.simulation = simulation
        self.reader = G1Dex3StateReader(
            simulation=simulation,
            max_age_s=ACTUATOR_ARM_STATE_MAX_AGE_S,
            hand_max_age_s=ACTUATOR_HAND_STATE_MAX_AGE_S,
        )
        initial = self.reader.read(timeout_s=5.0)
        if not simulation and initial.mode_machine != QUALIFIED_REAL_MODE_MACHINE:
            raise DeploymentError(
                f"Real G1 mode_machine is {initial.mode_machine}; this adapter is qualified only "
                f"for mode {QUALIFIED_REAL_MODE_MACHINE} "
                "(g1_29dof_lock_waist_with_hand_rev_1_0)"
            )

        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex
        from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
            Dex3_1_Left_JointIndex,
            Dex3_1_Right_JointIndex,
        )
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        self._left_indices = tuple(int(index) for index in Dex3_1_Left_JointIndex)
        self._right_indices = tuple(int(index) for index in Dex3_1_Right_JointIndex)
        arm_topic = "rt/lowcmd" if simulation else "rt/arm_sdk"

        # Construct every resource before the first Write.  A partial constructor
        # failure therefore cannot acquire arm authority or move a hand.
        self._arm_publisher = ChannelPublisher(arm_topic, LowCmd_)
        self._left_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self._right_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._arm_publisher.Init()
        self._left_publisher.Init()
        self._right_publisher.Init()
        self._arm_message = unitree_hg_msg_dds__LowCmd_()
        self._left_message = unitree_hg_msg_dds__HandCmd_()
        self._right_message = unitree_hg_msg_dds__HandCmd_()
        self._crc = CRC()
        self._weight = 0.0
        self._released = False
        self._has_published = False
        # Enabled only by the dedicated authority-ramp diagnostic.  Keeping the
        # timers dormant avoids changing the normal deployment hot path.
        self._authority_ramp_timing_enabled = False
        self._last_publish_timing_ms: dict[str, float] = {}
        self._arm_target = initial.arm.copy()
        self._left_target = initial.left_hand.copy()
        self._right_target = initial.right_hand.copy()
        self._left_hand_publish_history: deque[PublishedHandTarget] = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        self._right_hand_publish_history: deque[PublishedHandTarget] = deque(maxlen=HAND_COMMAND_HISTORY_SIZE)
        self._configure_messages(initial)

    def _configure_messages(self, initial: RobotState) -> None:
        self._arm_message.mode_pr = 0
        self._arm_message.mode_machine = initial.mode_machine
        wrist_offsets = {4, 5, 6, 11, 12, 13}
        for offset, index in enumerate(self._arm_indices):
            command = self._arm_message.motor_cmd[index]
            command.mode = 1
            command.q = float(initial.arm[offset])
            command.dq = 0.0
            command.tau = 0.0
            command.kp = 40.0 if offset in wrist_offsets else 80.0
            command.kd = 1.5 if offset in wrist_offsets else 3.0

        for message, indices in (
            (self._left_message, self._left_indices),
            (self._right_message, self._right_indices),
        ):
            for index in indices:
                command = message.motor_cmd[index]
                command.mode = (index & 0x0F) | (0x01 << 4)
                command.q = 0.0
                command.dq = 0.0
                command.tau = 0.0
                command.kp = 1.5
                command.kd = 0.2

    def _validate_runtime_state(self, state: RobotState) -> RobotState:
        if not self.simulation and state.mode_machine != QUALIFIED_REAL_MODE_MACHINE:
            raise DeploymentError(
                f"Robot mode_machine changed from QUALIFIED_REAL_MODE_MACHINE="
                f"{QUALIFIED_REAL_MODE_MACHINE} to {state.mode_machine}"
            )
        joint = int(np.argmax(np.abs(state.arm_dq)))
        velocity = float(state.arm_dq[joint])
        if abs(velocity) > MAX_ARM_DQ_RAD_S:
            raise DeploymentError(
                f"Arm velocity at joint {joint} ({ARM_JOINT_NAMES[joint]}) is {velocity:+.3f} rad/s; "
                f"MAX_ARM_DQ_RAD_S={MAX_ARM_DQ_RAD_S:.3f} rad/s"
            )
        return state

    def state(self) -> RobotState:
        return self._validate_runtime_state(self.reader.latest())

    def prepare_measured_hold(self) -> RobotState:
        state = self._validate_runtime_state(self.reader.read(timeout_s=0.5))
        if not self.simulation:
            reference = state
            last_captured_at = float("-inf")
            distinct_samples = 0
            deadline = time.monotonic() + PREARM_STATIONARY_DWELL_S
            while time.monotonic() < deadline:
                state = self._validate_runtime_state(self.reader.read(timeout_s=0.1))
                state_age = time.monotonic() - state.captured_at
                if state_age > PREARM_STATE_MAX_AGE_S:
                    raise DeploymentError(
                        f"Robot state age is {state_age:.3f}s; PREARM_STATE_MAX_AGE_S={PREARM_STATE_MAX_AGE_S:.3f}s"
                    )
                joint = int(np.argmax(np.abs(state.arm_dq)))
                velocity = float(state.arm_dq[joint])
                if abs(velocity) > PREARM_MAX_ARM_DQ_RAD_S:
                    raise DeploymentError(
                        f"Arm is not stationary at joint {joint} ({ARM_JOINT_NAMES[joint]}): "
                        f"{velocity:+.3f} rad/s; "
                        f"PREARM_MAX_ARM_DQ_RAD_S={PREARM_MAX_ARM_DQ_RAD_S:.3f} rad/s"
                    )
                group, joint, joint_name, drift = _largest_named_value(
                    (
                        ("arm", state.arm - reference.arm, ARM_JOINT_NAMES),
                        ("left hand", state.left_hand - reference.left_hand, LEFT_HAND_JOINT_NAMES),
                        ("right hand", state.right_hand - reference.right_hand, RIGHT_HAND_JOINT_NAMES),
                    )
                )
                if abs(drift) > PREARM_MAX_POSITION_DRIFT_RAD:
                    raise DeploymentError(
                        f"Pre-arm {group} position drift at joint {joint} ({joint_name}) is "
                        f"{drift:+.4f} rad; PREARM_MAX_POSITION_DRIFT_RAD="
                        f"{PREARM_MAX_POSITION_DRIFT_RAD:.4f} rad"
                    )
                if state.captured_at > last_captured_at:
                    distinct_samples += 1
                    last_captured_at = state.captured_at
                time.sleep(0.005)
            if distinct_samples < PREARM_MIN_DISTINCT_SAMPLES:
                raise DeploymentError(
                    f"Only {distinct_samples} distinct robot-state samples arrived during the pre-arm dwell; "
                    f"PREARM_MIN_DISTINCT_SAMPLES={PREARM_MIN_DISTINCT_SAMPLES}"
                )
        self._arm_message.mode_machine = state.mode_machine if self.simulation else QUALIFIED_REAL_MODE_MACHINE
        self.set_target(state.arm, state.left_hand, state.right_hand)
        return state

    def set_target(self, arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self._arm_target = np.asarray(arm, dtype=np.float64).copy()
        self._left_target = np.asarray(left, dtype=np.float64).copy()
        self._right_target = np.asarray(right, dtype=np.float64).copy()

    def reset_hand_publish_history(self) -> None:
        self._left_hand_publish_history.clear()
        self._right_hand_publish_history.clear()

    def _publish_arm(self, require_qualified_state: bool = True) -> None:
        timing_enabled = self._authority_ramp_timing_enabled
        timing = self._last_publish_timing_ms
        for offset, index in enumerate(self._arm_indices):
            self._arm_message.motor_cmd[index].q = float(self._arm_target[offset])
        if not self.simulation:
            # Never copy an unqualified/transient mode into a real arm command.
            # Operational writes also re-check the latest state immediately
            # before publishing.  Cleanup can skip the freshness dependency so
            # an attempted authority release is not defeated by the same state
            # fault that triggered it.
            self._arm_message.mode_machine = QUALIFIED_REAL_MODE_MACHINE
            if require_qualified_state:
                started_ns = time.monotonic_ns()
                self._validate_runtime_state(self.reader.latest())
                if timing_enabled:
                    timing["arm_state_check"] = (time.monotonic_ns() - started_ns) / 1e6
            self._arm_message.motor_cmd[29].q = float(self._weight)
        started_ns = time.monotonic_ns()
        self._arm_message.crc = self._crc.Crc(self._arm_message)
        if timing_enabled:
            timing["arm_crc"] = (time.monotonic_ns() - started_ns) / 1e6
        started_ns = time.monotonic_ns()
        write_ok = self._arm_publisher.Write(self._arm_message, timeout=DDS_WRITE_TIMEOUT_S)
        if timing_enabled:
            timing["arm_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if write_ok is not True:
            raise DeploymentError("Arm DDS Write failed")
        self._has_published = True

    def _publish_hands(self) -> None:
        # Snapshot canonical-order targets before either Write. History is
        # appended independently only after the corresponding Write succeeds.
        left_target = self._left_target.copy()
        right_target = self._right_target.copy()
        left_command = left_target
        right_command = right_target
        if self.simulation:
            right_command = right_command[SIM_RIGHT_HAND_PERMUTATION]
        for offset, index in enumerate(self._left_indices):
            self._left_message.motor_cmd[index].q = float(left_command[offset])
        for offset, index in enumerate(self._right_indices):
            self._right_message.motor_cmd[index].q = float(right_command[offset])

        timing_enabled = self._authority_ramp_timing_enabled
        timing = self._last_publish_timing_ms
        started_ns = time.monotonic_ns()
        left_ok = self._left_publisher.Write(self._left_message, timeout=DDS_WRITE_TIMEOUT_S)
        if timing_enabled:
            timing["left_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if left_ok is not True:
            raise DeploymentError("Left Dex3 DDS Write failed")
        self._left_hand_publish_history.append(PublishedHandTarget(completed_at=time.monotonic(), target=left_target))
        started_ns = time.monotonic_ns()
        right_ok = self._right_publisher.Write(self._right_message, timeout=DDS_WRITE_TIMEOUT_S)
        if timing_enabled:
            timing["right_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if right_ok is not True:
            raise DeploymentError("Right Dex3 DDS Write failed")
        self._right_hand_publish_history.append(PublishedHandTarget(completed_at=time.monotonic(), target=right_target))

    def _stop_hands(self, phase_callback: Any | None = None) -> None:
        """Send Unitree's documented Dex3 ``stopMotors`` command once per hand."""

        for message, indices in (
            (self._left_message, self._left_indices),
            (self._right_message, self._right_indices),
        ):
            for index in indices:
                command = message.motor_cmd[index]
                command.mode = (index & 0x0F) | (0x01 << 4) | (0x01 << 7)
                command.q = 0.0
                command.dq = 0.0
                command.tau = 0.0
                command.kp = 0.0
                command.kd = 0.0
        failures = []
        for name, publisher, message in (
            ("Left", self._left_publisher, self._left_message),
            ("Right", self._right_publisher, self._right_message),
        ):
            if phase_callback is not None:
                phase_callback(f"{name.lower()}_hand_stop_begin", {})
            started = time.monotonic()
            try:
                if publisher.Write(message, timeout=DDS_WRITE_TIMEOUT_S) is not True:
                    failures.append(f"{name} Dex3 stop Write failed")
            except Exception as exc:
                failures.append(f"{name} Dex3 stop Write raised {exc!r}")
            finally:
                if phase_callback is not None:
                    phase_callback(
                        f"{name.lower()}_hand_stop_end",
                        {"elapsed_s": time.monotonic() - started},
                    )
        if failures:
            raise DeploymentError("; ".join(failures))

    def publish(self) -> None:
        timing_enabled = self._authority_ramp_timing_enabled
        if timing_enabled:
            self._last_publish_timing_ms = {}
        started_ns = time.monotonic_ns()
        try:
            self._publish_arm()
            self._publish_hands()
        finally:
            if timing_enabled:
                self._last_publish_timing_ms["publish_total"] = (time.monotonic_ns() - started_ns) / 1e6

    def set_weight(self, weight: float) -> None:
        self._weight = float(np.clip(weight, 0.0, 1.0))

    def release(self, phase_callback: Any | None = None) -> None:
        if self._released:
            return
        self._released = True
        if not self._has_published:
            if phase_callback is not None:
                phase_callback("release_skipped_no_writes", {})
            return
        if self.simulation:
            try:
                measured = self.reader.latest()
                self.set_target(measured.arm, measured.left_hand, measured.right_hand)
                self.publish()
            except Exception as exc:
                raise DeploymentError(f"Failed to leave simulator holding measured position: {exc}") from exc
            return
        failures = []
        start_weight = self._weight
        period = 1.0 / PUBLISH_HZ
        duration = max(float(ARM_RELEASE_RAMP_S), period)
        ramp_started = time.monotonic()
        next_write_at = ramp_started
        arm_writes = 0
        skipped_ticks = 0
        max_arm_cycle_s = 0.0
        last_successful_weight = start_weight
        # Release does not depend on policy, camera, fresh state, or IK.
        LOGGER.info("Arm authority release started at weight %.4f", start_weight)
        if phase_callback is not None:
            phase_callback("arm_release_begin", {"start_weight": start_weight})
        while True:
            now = time.monotonic()
            if now < next_write_at:
                time.sleep(next_write_at - now)
            cycle_started = time.monotonic()
            progress = 1.0 if start_weight == 0.0 else min(
                1.0,
                (cycle_started - ramp_started + period) / duration,
            )
            self.set_weight(start_weight * (1.0 - progress))
            try:
                self._publish_arm(require_qualified_state=False)
            except Exception as exc:
                failures.append(f"arm_sdk authority release failed: {exc}")
                break
            cycle_completed = time.monotonic()
            arm_writes += 1
            last_successful_weight = self._weight
            max_arm_cycle_s = max(max_arm_cycle_s, cycle_completed - cycle_started)
            if progress >= 1.0:
                break
            next_write_at += period
            if next_write_at <= cycle_completed:
                missed = int((cycle_completed - next_write_at) // period) + 1
                skipped_ticks += missed
                next_write_at += missed * period
        # If an intermediate ramp write failed, make one explicit best-effort
        # zero-weight attempt before stopping the hands.
        if last_successful_weight != 0.0:
            self.set_weight(0.0)
            try:
                self._publish_arm(require_qualified_state=False)
                arm_writes += 1
                last_successful_weight = 0.0
            except Exception as exc:
                failures.append(f"final zero-weight arm_sdk Write failed: {exc}")
        arm_release_s = time.monotonic() - ramp_started
        LOGGER.info(
            "Arm authority release ended in %.3fs: writes=%d, skipped 100 Hz ticks=%d, "
            "max arm cycle=%.3fs, last successful weight=%.4f",
            arm_release_s,
            arm_writes,
            skipped_ticks,
            max_arm_cycle_s,
            last_successful_weight,
        )
        if phase_callback is not None:
            phase_callback(
                "arm_release_end",
                {
                    "elapsed_s": arm_release_s,
                    "writes": arm_writes,
                    "skipped_ticks": skipped_ticks,
                    "last_successful_weight": last_successful_weight,
                },
            )
        # Do this after the arm release so a blocked hand DDS Write cannot
        # prevent the higher-priority arm_sdk weight ramp from being attempted.
        hand_stop_started = time.monotonic()
        try:
            self._stop_hands(phase_callback)
        except Exception as exc:
            failures.append(f"Dex3 stopMotors failed: {exc}")
        LOGGER.info("Dex3 stopMotors phase finished in %.3fs", time.monotonic() - hand_stop_started)
        if failures:
            raise DeploymentError("; ".join(failures))

    def close(self) -> None:
        self.reader.close()
        for publisher in (
            self._arm_publisher,
            self._left_publisher,
            self._right_publisher,
        ):
            with contextlib.suppress(Exception):
                publisher.Close()


def _heartbeat_age(heartbeat: Any) -> float:
    with heartbeat.get_lock():
        last = float(heartbeat.value)
    return time.monotonic() - last


def _status(status_queue: MpQueue, kind: str, payload: Any = None) -> None:
    try:
        status_queue.put((kind, payload), timeout=0.2)
    except queue.Full:
        pass


def _status_nonblocking(status_queue: MpQueue, kind: str, payload: Any = None) -> None:
    """Best-effort diagnostic status that must never delay the control loop."""

    try:
        status_queue.put_nowait((kind, payload))
    except queue.Full:
        pass


def _observe_hand_freshness(
    gate: HandStateFreshnessGate,
    state: RobotState,
    status_queue: MpQueue,
    *,
    context: str,
) -> HandFreshnessResult:
    result = gate.check(state)
    if result.entered:
        payload = {
            "context": context,
            "hands": result.stale_hands,
            "age_s": result.max_age_s,
            "warning_age_s": ACTUATOR_HAND_STATE_WARNING_AGE_S,
            "hard_age_s": ACTUATOR_HAND_STATE_MAX_AGE_S,
        }
        LOGGER.warning(
            "Dex3 state pause in %s: %s age %.3fs exceeded soft %.3fs; "
            "freezing commands (hard fault at %.3fs)",
            context,
            "/".join(result.stale_hands),
            result.max_age_s,
            ACTUATOR_HAND_STATE_WARNING_AGE_S,
            ACTUATOR_HAND_STATE_MAX_AGE_S,
        )
        _status_nonblocking(status_queue, "hand_state_pause", payload)
    elif result.recovered:
        payload = {
            "context": context,
            "pause_s": result.pause_s,
            "fresh_samples": ACTUATOR_HAND_RECOVERY_SAMPLES,
        }
        LOGGER.info(
            "Dex3 state recovered in %s after %.3fs and %d consecutive fresh samples",
            context,
            result.pause_s,
            ACTUATOR_HAND_RECOVERY_SAMPLES,
        )
        # Recovery gates future motion, so unlike the entry diagnostic this
        # acknowledgment is delivered through the bounded reliable path.
        _status(status_queue, "hand_state_recovered", payload)
    return result


def _authority_ramp_timing_summary(
    records: list[dict[str, float]],
    *,
    event: str,
    elapsed_s: float,
    total_steps: int,
) -> dict[str, Any]:
    """Return a compact timing summary without doing I/O in the 100 Hz loop."""

    summary: dict[str, Any] = {
        "event": event,
        "completed_steps": len(records),
        "total_steps": total_steps,
        "elapsed_ms": elapsed_s * 1e3,
    }
    if not records:
        return summary
    summary["last_weight"] = records[-1]["weight"]
    summary["heartbeat_age_max_ms"] = max(record["heartbeat_age_ms"] for record in records)
    summary["state_age_max_ms"] = max(record["state_age_ms"] for record in records)
    summary["arm_drift_max_rad"] = max(record["arm_drift_rad"] for record in records)
    summary["arm_dq_max_rad_s"] = max(record["arm_dq_rad_s"] for record in records)
    summary["work_over_10ms"] = sum(record["work_ms"] > 10.0 for record in records)
    for key in (
        "state_lookup_ms",
        "arm_state_check_ms",
        "arm_crc_ms",
        "arm_write_ms",
        "left_write_ms",
        "right_write_ms",
        "publish_total_ms",
        "work_ms",
        "cycle_ms",
    ):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_p95"] = float(np.percentile(values, 95))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def _format_authority_ramp_timing(payload: dict[str, Any]) -> str:
    preferred = (
        "event",
        "completed_steps",
        "total_steps",
        "elapsed_ms",
        "last_weight",
        "heartbeat_age_max_ms",
        "state_age_max_ms",
        "arm_drift_max_rad",
        "arm_dq_max_rad_s",
        "publish_total_ms_p95",
        "publish_total_ms_max",
        "arm_write_ms_max",
        "left_write_ms_max",
        "right_write_ms_max",
        "cycle_ms_p95",
        "cycle_ms_max",
        "work_over_10ms",
    )
    fields = []
    for key in preferred:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, float):
            fields.append(f"{key}={value:.3f}")
        else:
            fields.append(f"{key}={value}")
    return " ".join(fields)


def _tracking_errors(backend: _G1Dex3CommandBackend, state: RobotState) -> tuple[float, float]:
    arm_error = float(np.max(np.abs(state.arm - backend._arm_target)))
    hand_error = float(
        max(
            np.max(np.abs(state.left_hand - backend._left_target)),
            np.max(np.abs(state.right_hand - backend._right_target)),
        )
    )
    return arm_error, hand_error


def _enforce_tracking(
    backend: _G1Dex3CommandBackend,
    state: RobotState,
    hand_watchdog: HandTrackingWatchdog | None = None,
    *,
    now: float | None = None,
    context: str = "unknown",
    desired_left: np.ndarray | None = None,
    desired_right: np.ndarray | None = None,
) -> None:
    _, arm_joint, arm_name, arm_delta = _largest_named_value(
        (("arm", state.arm - backend._arm_target, ARM_JOINT_NAMES),)
    )
    if abs(arm_delta) > MAX_ARM_TRACKING_ERROR_RAD:
        raise DeploymentError(
            f"Arm tracking error at joint {arm_joint} ({arm_name}) is {arm_delta:+.3f} rad "
            f"(measured minus target); MAX_ARM_TRACKING_ERROR_RAD="
            f"{MAX_ARM_TRACKING_ERROR_RAD:.3f} rad"
        )
    if hand_watchdog is not None and hand_watchdog.enforce(
        backend,
        state,
        now=now,
        context=context,
        desired_left=desired_left,
        desired_right=desired_right,
    ):
        return

    # Compatibility fallback for focused tests and older internal fake
    # backends that do not expose successful-publish history.
    hand_group, hand_joint, hand_name, hand_delta = _largest_named_value(
        (
            ("left hand", state.left_hand - backend._left_target, LEFT_HAND_JOINT_NAMES),
            ("right hand", state.right_hand - backend._right_target, RIGHT_HAND_JOINT_NAMES),
        )
    )
    if abs(hand_delta) > MAX_HAND_TRACKING_ERROR_RAD:
        raise DeploymentError(
            f"{hand_group.title()} tracking error at joint {hand_joint} ({hand_name}) is "
            f"{hand_delta:+.3f} rad (measured minus target); MAX_HAND_TRACKING_ERROR_RAD="
            f"{MAX_HAND_TRACKING_ERROR_RAD:.3f} rad"
        )


def _position_drift(state: RobotState, reference: RobotState) -> float:
    return float(
        max(
            np.max(np.abs(state.arm - reference.arm)),
            np.max(np.abs(state.left_hand - reference.left_hand)),
            np.max(np.abs(state.right_hand - reference.right_hand)),
        )
    )


def _wait_for_initialization_start(
    backend: _G1Dex3CommandBackend,
    stop_event: Any,
    heartbeat: Any,
    hand_watchdog: HandTrackingWatchdog | None = None,
    hand_freshness_gate: HandStateFreshnessGate | None = None,
    status_queue: MpQueue | None = None,
) -> RobotState | None:
    """Hold the current command until fresh measured state is stationary again."""

    period = 1.0 / PUBLISH_HZ
    deadline = time.monotonic() + INITIALIZATION_START_TIMEOUT_S
    stationary_since: float | None = None
    stationary_reference: RobotState | None = None
    distinct_samples = 0
    last_capture = float("-inf")
    latest: RobotState | None = None
    while not stop_event.is_set() and time.monotonic() < deadline:
        loop_started = time.monotonic()
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired before initialization movement")
        latest = backend.state()
        now = time.monotonic()
        if hand_freshness_gate is not None and status_queue is not None:
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                latest,
                status_queue,
                context="initialization-start dwell",
            )
            if not freshness.ready:
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue
            if freshness.recovered:
                _enforce_tracking(
                    backend,
                    latest,
                    hand_watchdog,
                    now=now,
                    context="initialization-start recovery",
                )
                deadline += freshness.pause_s
                if hand_watchdog is not None:
                    hand_watchdog.reset(backend)
                stationary_since = None
                stationary_reference = None
                distinct_samples = 0
                last_capture = float("-inf")
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue
        arm_received_at = latest.captured_at if latest.arm_received_at is None else latest.arm_received_at
        if now - arm_received_at > PREARM_STATE_MAX_AGE_S:
            raise DeploymentError("Arm state is not fresh enough to start initialization")
        _enforce_tracking(
            backend,
            latest,
            hand_watchdog,
            now=now,
            context="initialization-start dwell",
        )
        arm_error, hand_error = _tracking_errors(backend, latest)

        stationary = (
            (backend.simulation or float(np.max(np.abs(latest.arm_dq))) <= INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S)
            and arm_error <= INITIALIZATION_ARM_TOLERANCE_RAD
            and hand_error <= INITIALIZATION_HAND_TOLERANCE_RAD
        )
        if stationary:
            if stationary_since is None:
                stationary_since = now
                stationary_reference = latest
                distinct_samples = 0
                last_capture = float("-inf")
            assert stationary_reference is not None
            if _position_drift(latest, stationary_reference) > INITIALIZATION_MAX_POSITION_DRIFT_RAD:
                stationary_since = now
                stationary_reference = latest
                distinct_samples = 0
                last_capture = float("-inf")
            if latest.captured_at > last_capture:
                distinct_samples += 1
                last_capture = latest.captured_at
            if (
                now - stationary_since >= INITIALIZATION_START_DWELL_S
                and distinct_samples >= INITIALIZATION_MIN_DISTINCT_SAMPLES
            ):
                return latest
        else:
            stationary_since = None
            stationary_reference = None
            distinct_samples = 0
            last_capture = float("-inf")

        backend.publish()
        elapsed = time.monotonic() - loop_started
        stop_event.wait(max(0.0, period - elapsed))
    if stop_event.is_set():
        return None
    if latest is None:
        raise DeploymentError("No fresh robot state arrived before initialization")
    arm_error, hand_error = _tracking_errors(backend, latest)
    raise DeploymentError(
        "Robot did not become stationary at the held target before initialization: "
        f"arm error={arm_error:.3f} rad, hand error={hand_error:.3f} rad, "
        f"max arm dq={float(np.max(np.abs(latest.arm_dq))):.3f} rad/s"
    )


def _execute_initialization(
    backend: _G1Dex3CommandBackend,
    chunk: ActionChunk,
    stop_event: Any,
    heartbeat: Any,
    tracking_checks_after: float,
    hand_watchdog: HandTrackingWatchdog | None = None,
    hand_freshness_gate: HandStateFreshnessGate | None = None,
    status_queue: MpQueue | None = None,
    *,
    context: str = "initialization",
) -> bool:
    """Execute and verify one bounded initialization path in the DDS owner."""

    period = 1.0 / PUBLISH_HZ
    path_index = 0
    endpoint_deadline: float | None = None
    converged_since: float | None = None
    converged_reference: RobotState | None = None
    distinct_converged_samples = 0
    last_converged_capture = float("-inf")

    while not stop_event.is_set():
        loop_started = time.monotonic()
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired during initialization")

        state = backend.state()
        now = time.monotonic()
        if hand_freshness_gate is not None and status_queue is not None:
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                state,
                status_queue,
                context=context,
            )
            if not freshness.ready:
                # Preserve the last command exactly.  In particular, never
                # replace a hand target with an old measurement while its DDS
                # stream is paused.
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue
            if freshness.recovered:
                if endpoint_deadline is not None:
                    endpoint_deadline += freshness.pause_s
                if np.isfinite(tracking_checks_after):
                    tracking_checks_after += freshness.pause_s
                _enforce_tracking(
                    backend,
                    state,
                    hand_watchdog,
                    now=now,
                    context=f"{context} feedback recovery",
                )
                if hand_watchdog is not None:
                    hand_watchdog.reset(backend)
                # Resume on the next publisher tick; no path point is consumed
                # in the recovery iteration.
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue
        if path_index < chunk.length:
            # Advance exactly once per publisher iteration.  A late loop slows
            # initialization; it never skips or bursts targets to catch up.
            backend.set_target(
                chunk.arm[path_index],
                chunk.left_hand[path_index],
                chunk.right_hand[path_index],
            )
            path_index += 1
            if path_index == chunk.length:
                endpoint_deadline = now + INITIALIZATION_CONVERGENCE_TIMEOUT_S

        if now >= tracking_checks_after:
            _enforce_tracking(backend, state, hand_watchdog, now=now, context=context)

        if endpoint_deadline is not None:
            arm_error, hand_error = _tracking_errors(backend, state)
            in_tolerance = (
                arm_error <= INITIALIZATION_ARM_TOLERANCE_RAD
                and hand_error <= INITIALIZATION_HAND_TOLERANCE_RAD
                and (backend.simulation or float(np.max(np.abs(state.arm_dq))) <= INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S)
            )
            if in_tolerance:
                if converged_since is None:
                    converged_since = now
                    converged_reference = state
                    distinct_converged_samples = 0
                    last_converged_capture = float("-inf")
                assert converged_reference is not None
                if _position_drift(state, converged_reference) > INITIALIZATION_MAX_POSITION_DRIFT_RAD:
                    converged_since = now
                    converged_reference = state
                    distinct_converged_samples = 0
                    last_converged_capture = float("-inf")
                if state.captured_at > last_converged_capture:
                    distinct_converged_samples += 1
                    last_converged_capture = state.captured_at
                if (
                    now - converged_since >= INITIALIZATION_CONVERGENCE_DWELL_S
                    and distinct_converged_samples >= INITIALIZATION_MIN_DISTINCT_SAMPLES
                ):
                    return True
            else:
                converged_since = None
                converged_reference = None
                distinct_converged_samples = 0
                last_converged_capture = float("-inf")
            if now > endpoint_deadline:
                _, arm_joint, arm_name, arm_delta = _largest_named_value(
                    (("arm", state.arm - backend._arm_target, ARM_JOINT_NAMES),)
                )
                raise DeploymentError(
                    "Initialization target did not converge: "
                    f"arm error={arm_error:.3f} rad at joint {arm_joint} ({arm_name}), "
                    f"measured minus target={arm_delta:+.3f} rad, "
                    f"max arm dq={float(np.max(np.abs(state.arm_dq))):.3f} rad/s, "
                    f"hand error={hand_error:.3f} rad"
                )

        backend.publish()
        elapsed = time.monotonic() - loop_started
        stop_event.wait(max(0.0, period - elapsed))
    return False


def _set_direct_target(
    backend: _G1Dex3CommandBackend,
    conditioner: XrPolicyOutputConditioner | None,
    hand_watchdog: HandTrackingWatchdog | None,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    """Set a non-policy target and make it the conditioner's new origin."""

    backend.set_target(arm, left, right)
    if conditioner is not None:
        conditioner.reset(backend._arm_target, backend._left_target, backend._right_target)
    if hand_watchdog is not None:
        hand_watchdog.reset(backend)


def _validate_policy_target_input(
    chunk: ActionChunk,
    current_arm: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    conditioner: XrPolicyOutputConditioner | None,
    *,
    context: str,
) -> None:
    """Keep raw value checks hard while making raw slew diagnostic in XR mode."""

    if conditioner is None:
        validate_action_chunk(chunk, current_arm, current_left, current_right)
        return
    validate_action_chunk_limits(chunk)
    try:
        validate_action_chunk(chunk, current_arm, current_left, current_right)
    except DeploymentError as exc:
        LOGGER.warning("%s raw target discontinuity will be conditioned before DDS: %s", context, exc)


def _ramp_real_arm_authority(
    backend: _G1Dex3CommandBackend,
    stop_event: Any,
    heartbeat: Any,
) -> bool:
    """Publish a measured hold while gradually acquiring ``arm_sdk`` authority.

    Dex3 absolute targets do not need to be rewritten for every arm authority
    step.  Writing each hand at every step made the nominal 1.5-second ramp
    perform 450 serial DDS writes and allowed DDS latency to push arming past
    the parent's timeout.  First publish the measured arm hold at zero weight,
    then send the measured hand hold once and ramp only the arm command.  The
    initial arm write also establishes the cleanup invariant before any hand
    write: a subsequent cancellation/fault must run arm release and Dex3
    ``stopMotors``.  Arm weight follows elapsed wall time and missed 100 Hz
    ticks are skipped instead of accumulating an extra sleep after slow writes.

    ``False`` is an orderly cancellation requested through ``stop_event``;
    heartbeat expiry remains a fault with a distinct diagnostic.
    """

    if stop_event.is_set():
        LOGGER.info("Arm authority ramp cancelled before the first DDS write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired before arm authority ramp")

    backend.set_weight(0.0)
    authority_started = time.monotonic()
    initial_arm_write_started = authority_started
    backend._publish_arm()
    initial_arm_write_s = time.monotonic() - initial_arm_write_started
    if stop_event.is_set():
        LOGGER.info("Arm authority ramp cancelled after the zero-weight arm hold write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired after zero-weight arm hold write")

    hand_write_started = time.monotonic()
    backend._publish_hands()
    hand_write_s = time.monotonic() - hand_write_started
    if stop_event.is_set():
        LOGGER.info("Arm authority ramp cancelled after the measured hand hold write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired after measured hand hold write")

    period = 1.0 / PUBLISH_HZ
    duration = max(float(ARM_AUTHORITY_RAMP_S), period)
    ramp_started = time.monotonic()
    next_write_at = ramp_started
    arm_writes = 0
    skipped_ticks = 0
    max_arm_cycle_s = 0.0

    while True:
        if stop_event.is_set():
            LOGGER.info("Arm authority ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm authority")

        now = time.monotonic()
        if now < next_write_at and stop_event.wait(next_write_at - now):
            LOGGER.info("Arm authority ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm authority")

        cycle_started = time.monotonic()
        measured = backend.state()
        # Keep the exact hand targets that were written above.  Only the arm
        # target tracks measured q while authority weight rises.
        backend.set_target(measured.arm, backend._left_target, backend._right_target)
        weight = min(1.0, (cycle_started - ramp_started + period) / duration)
        backend.set_weight(weight)
        backend._publish_arm()
        cycle_completed = time.monotonic()
        arm_writes += 1
        max_arm_cycle_s = max(max_arm_cycle_s, cycle_completed - cycle_started)

        # Cleanup requests are not heartbeat faults.  Check the stop event
        # first, including after a potentially blocking DDS Write.
        if stop_event.is_set():
            LOGGER.info("Arm authority ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm authority")
        if weight >= 1.0:
            break

        next_write_at += period
        if next_write_at <= cycle_completed:
            missed = int((cycle_completed - next_write_at) // period) + 1
            skipped_ticks += missed
            next_write_at += missed * period

    completed_at = time.monotonic()
    LOGGER.info(
        "Arm authority acquisition completed in %.3fs: weight ramp=%.3fs, "
        "zero-weight arm write=%.3fs, hand hold write=%.3fs, arm writes=%d, "
        "skipped 100 Hz ticks=%d, max arm cycle=%.3fs",
        completed_at - authority_started,
        completed_at - ramp_started,
        initial_arm_write_s,
        hand_write_s,
        arm_writes,
        skipped_ticks,
        max_arm_cycle_s,
    )
    return True


def _actuator_main(
    simulation: bool,
    network_interface: str | None,
    command_queue: MpQueue,
    status_queue: MpQueue,
    stop_event: Any,
    heartbeat: Any,
    urgent_hold_event: Any | None = None,
    command_conditioning: str = "none",
    authority_ramp_diagnostics: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if threading.current_thread() is threading.main_thread():

        def stop_on_signal(_signum: int, _frame: object) -> None:
            stop_event.set()

        for signal_name in ("SIGTERM", "SIGINT", "SIGHUP"):
            signum = getattr(signal, signal_name, None)
            if signum is not None:
                signal.signal(signum, stop_on_signal)
    backend: _G1Dex3CommandBackend | None = None
    if urgent_hold_event is None:
        # Keeps direct/threaded tests and older internal callers compatible.
        urgent_hold_event = threading.Event()
    if command_conditioning not in COMMAND_CONDITIONING_MODES:
        _status(status_queue, "fault", f"Unknown command conditioning mode {command_conditioning!r}")
        _status(status_queue, "stopped")
        return

    last_sequence = 0
    tracking_checks_after = float("inf")
    conditioner: XrPolicyOutputConditioner | None = None
    hand_watchdog = HandTrackingWatchdog()
    hand_freshness_gate = HandStateFreshnessGate()
    try:
        backend = _G1Dex3CommandBackend(simulation, network_interface)
        backend._authority_ramp_timing_enabled = authority_ramp_diagnostics
        if command_conditioning == "xr":
            conditioner = XrPolicyOutputConditioner()
        _status(status_queue, "ready")

        # Publishers now exist but no message has been written.  Wait for the
        # parent's explicit arm request.
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.05)
            except queue.Empty:
                if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                    raise DeploymentError("Parent heartbeat expired before arm request")
                continue
            if not isinstance(command, tuple) or command != ("arm",):
                raise DeploymentError("Unexpected actuator command before arm request")
            break
        else:
            return

        prearm_started = time.monotonic()
        if authority_ramp_diagnostics:
            _status(
                status_queue,
                "authority_ramp_timing",
                {"event": "prearm_begin", "elapsed_ms": 0.0},
            )
        ramp_reference = backend.prepare_measured_hold()
        if authority_ramp_diagnostics:
            _status(
                status_queue,
                "authority_ramp_timing",
                {
                    "event": "prearm_complete",
                    "elapsed_ms": (time.monotonic() - prearm_started) * 1e3,
                },
            )

        if not simulation:
            if not _ramp_real_arm_authority(backend, stop_event, heartbeat):
                return
        tracking_checks_after = time.monotonic() + TRACKING_GRACE_S
        _status(status_queue, "armed")

        period = 1.0 / PUBLISH_HZ
        action_period = 1.0 / CONTROL_HZ

        # Initialization is a mandatory state transition.  Until the parent
        # requests it, hold the measured target while keeping every watchdog
        # active.  Policy chunks are structurally rejected in this phase.
        while not stop_event.is_set():
            loop_started = time.monotonic()
            if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                raise DeploymentError("Parent heartbeat expired before initialization")
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command is not None:
                if not isinstance(command, tuple) or len(command) != 3 or command[0] != "initialize":
                    kind = command[0] if isinstance(command, tuple) and command else None
                    raise DeploymentError(f"Unexpected actuator command before initialization: {kind!r}")
                _, created_at, initialization = command
                if not isinstance(initialization, InitializationSpec):
                    raise DeploymentError("Malformed initialization command")
                try:
                    command_age = time.monotonic() - float(created_at)
                except (TypeError, ValueError) as exc:
                    raise DeploymentError("Initialization command timestamp is invalid") from exc
                if not np.isfinite(command_age) or not 0.0 <= command_age <= INITIALIZATION_COMMAND_MAX_AGE_S:
                    raise DeploymentError("Initialization command expired before execution")
                hand_watchdog.reset(backend)
                state = _wait_for_initialization_start(
                    backend,
                    stop_event,
                    heartbeat,
                    hand_watchdog,
                    hand_freshness_gate,
                    status_queue,
                )
                if state is None:
                    _status(status_queue, "initialization_cancelled", initialization.mode)
                    return
                # Build from the already-published hold target, not directly
                # from measured q.  This keeps the first initialization command
                # increment bounded even when normal tracking error is nonzero.
                command_start = RobotState(
                    captured_at=state.captured_at,
                    mode_machine=state.mode_machine,
                    arm=backend._arm_target.copy(),
                    arm_dq=state.arm_dq.copy(),
                    left_hand=backend._left_target.copy(),
                    right_hand=backend._right_target.copy(),
                )
                initialization_chunk = build_initialization_chunk(command_start, initialization)
                _status(
                    status_queue,
                    "initializing",
                    {
                        "mode": initialization.mode,
                        "steps": initialization_chunk.length,
                        "duration_s": initialization_chunk.length / PUBLISH_HZ,
                    },
                )
                initialized = _execute_initialization(
                    backend,
                    initialization_chunk,
                    stop_event,
                    heartbeat,
                    tracking_checks_after,
                    hand_watchdog,
                    hand_freshness_gate,
                    status_queue,
                    context=f"initialization mode={initialization.mode}",
                )
                if not initialized:
                    _status(status_queue, "initialization_cancelled", initialization.mode)
                    return
                if conditioner is not None:
                    conditioner.reset(backend._arm_target, backend._left_target, backend._right_target)
                hand_watchdog.reset(backend)
                tracking_checks_after = time.monotonic()
                _status(status_queue, "initialized", initialization.mode)
                break

            state = backend.state()
            now = time.monotonic()
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                state,
                status_queue,
                context="pre-initialization hold",
            )
            if not freshness.ready:
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue
            if freshness.recovered:
                _set_direct_target(
                    backend,
                    conditioner,
                    hand_watchdog,
                    state.arm,
                    backend._left_target,
                    backend._right_target,
                )
                tracking_checks_after = now
            if now >= tracking_checks_after:
                _enforce_tracking(
                    backend,
                    state,
                    hand_watchdog,
                    context="pre-initialization hold",
                )
            backend.publish()
            elapsed = time.monotonic() - loop_started
            stop_event.wait(max(0.0, period - elapsed))
        else:
            return

        chunk: ActionChunk | None = None
        chunk_sequence = 0
        chunk_index = 0
        next_action_at = 0.0
        holding = True
        rtc_mode = False
        rtc_total_actions = 0
        rtc_action_budget = 0
        urgent_hold_active = False
        paused_sync_sequence: int | None = None

        while not stop_event.is_set():
            loop_started = time.monotonic()
            if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                raise DeploymentError("Parent heartbeat expired")

            # Sample freshness before reading a queued plan or advancing the
            # 30 Hz scheduler.  A short Dex3 gap therefore cannot consume an
            # action that was never written.
            state = backend.state()
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                state,
                status_queue,
                context=(
                    f"active sequence={chunk_sequence} next_action_index={chunk_index} "
                    f"rtc={rtc_mode} holding={holding}"
                ),
            )
            if freshness.entered:
                had_sync_motion = chunk is not None and not rtc_mode
                had_rtc_motion = chunk is not None and rtc_mode
                if had_sync_motion:
                    paused_sync_sequence = chunk_sequence
                if had_rtc_motion:
                    _status(
                        status_queue,
                        "rtc_rejected",
                        {
                            "reason": "hand_state_soft_stale",
                            "sequence": chunk_sequence,
                            "action_index": chunk_index,
                            "total_actions": rtc_total_actions,
                        },
                    )
                # Retain the exact targets most recently published.  Resetting
                # the conditioner to those targets fences every unconsumed
                # policy command without copying stale hand measurements.
                frozen_arm = backend._arm_target.copy()
                frozen_left = backend._left_target.copy()
                frozen_right = backend._right_target.copy()
                _set_direct_target(
                    backend,
                    conditioner,
                    hand_watchdog,
                    frozen_arm,
                    frozen_left,
                    frozen_right,
                )
                chunk = None
                chunk_index = 0
                rtc_mode = False
                rtc_total_actions = 0
                rtc_action_budget = 0
                holding = True

            if not freshness.ready:
                # STOP/release remains higher priority than recovery.  A STOP
                # barrier may be queued behind an RTC request, so discard all
                # queued motion while the independent urgent latch is set and
                # acknowledge the barrier when it is reached.
                if urgent_hold_event.is_set():
                    while True:
                        try:
                            paused_command = command_queue.get_nowait()
                        except queue.Empty:
                            break
                        if paused_command == ("urgent_hold_barrier",):
                            urgent_hold_event.clear()
                            urgent_hold_active = False
                            _status(status_queue, "urgent_holding", last_sequence)
                            break
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue

            if freshness.recovered:
                # Arm feedback remained under its hard 75 ms gate.  Capture
                # its current pose while preserving the exact frozen Dex3
                # targets; the discarded plan is never resumed.
                _enforce_tracking(
                    backend,
                    state,
                    hand_watchdog,
                    now=time.monotonic(),
                    context="Dex3 feedback recovery HOLD",
                )
                _set_direct_target(
                    backend,
                    conditioner,
                    hand_watchdog,
                    state.arm,
                    backend._left_target,
                    backend._right_target,
                )
                tracking_checks_after = time.monotonic() + TRACKING_GRACE_S
                if paused_sync_sequence is not None:
                    _status(status_queue, "holding", paused_sync_sequence)
                    paused_sync_sequence = None
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue

            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command is not None:
                kind = command[0] if isinstance(command, tuple) and command else None
                if kind == "urgent_hold_barrier":
                    if command != ("urgent_hold_barrier",):
                        raise DeploymentError("Malformed urgent STOP barrier")
                    # The barrier is queued only after every concurrently
                    # submitted motion command. Reaching it proves that no
                    # pre-STOP plan remains hidden in multiprocessing.Queue's
                    # feeder thread. Capture the final measured pose, then
                    # acknowledge the fully serialized powered STOP.
                    state = backend.state()
                    _set_direct_target(
                        backend,
                        conditioner,
                        hand_watchdog,
                        state.arm,
                        state.left_hand,
                        state.right_hand,
                    )
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    urgent_hold_active = False
                    urgent_hold_event.clear()
                    _status(status_queue, "urgent_holding", last_sequence)
                    continue

                if kind == "hold":
                    if not isinstance(command, tuple) or len(command) != 2:
                        raise DeploymentError("Malformed hold command")
                    if chunk is not None and not rtc_mode:
                        raise DeploymentError("Cannot enter hold before the current chunk completes")
                    _, created_at = command
                    try:
                        command_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("Hold command timestamp is invalid") from exc
                    if not np.isfinite(command_age) or not 0.0 <= command_age <= CHUNK_MAX_AGE_S:
                        raise DeploymentError("Hold command expired before execution")
                    state = backend.state()
                    _set_direct_target(
                        backend,
                        conditioner,
                        hand_watchdog,
                        state.arm,
                        state.left_hand,
                        state.right_hand,
                    )
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    _status(status_queue, "holding", last_sequence)
                    continue

                if kind == "warm_start":
                    if not isinstance(command, tuple) or len(command) != 3:
                        raise DeploymentError("Malformed policy warm-start command")
                    if chunk is not None or not holding:
                        raise DeploymentError("Policy warm-start requires an acknowledged hold with no active chunk")
                    _, created_at, warm_start = command
                    if not isinstance(warm_start, InitializationSpec):
                        raise DeploymentError("Malformed policy warm-start target")
                    try:
                        command_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("Policy warm-start timestamp is invalid") from exc
                    if not np.isfinite(command_age) or not 0.0 <= command_age <= INITIALIZATION_COMMAND_MAX_AGE_S:
                        raise DeploymentError("Policy warm-start command expired before execution")

                    hand_watchdog.reset(backend)
                    state = _wait_for_initialization_start(
                        backend,
                        stop_event,
                        heartbeat,
                        hand_watchdog,
                        hand_freshness_gate,
                        status_queue,
                    )
                    if state is None:
                        _status(status_queue, "warm_start_cancelled")
                        return
                    command_start = RobotState(
                        captured_at=state.captured_at,
                        mode_machine=state.mode_machine,
                        arm=backend._arm_target.copy(),
                        arm_dq=state.arm_dq.copy(),
                        left_hand=backend._left_target.copy(),
                        right_hand=backend._right_target.copy(),
                    )
                    warm_start_chunk = build_initialization_chunk(command_start, warm_start)
                    _status(
                        status_queue,
                        "warm_starting",
                        {
                            "steps": warm_start_chunk.length,
                            "duration_s": warm_start_chunk.length / PUBLISH_HZ,
                        },
                    )
                    completed = _execute_initialization(
                        backend,
                        warm_start_chunk,
                        stop_event,
                        heartbeat,
                        tracking_checks_after,
                        hand_watchdog,
                        hand_freshness_gate,
                        status_queue,
                        context="policy warm-start",
                    )
                    if not completed:
                        _status(status_queue, "warm_start_cancelled")
                        return
                    if conditioner is not None:
                        conditioner.reset(backend._arm_target, backend._left_target, backend._right_target)
                    hand_watchdog.reset(backend)
                    tracking_checks_after = time.monotonic()
                    holding = False
                    _status(status_queue, "warm_started", warm_start.label)
                    continue

                if kind == "rtc_snapshot":
                    if not isinstance(command, tuple) or len(command) != 2:
                        raise DeploymentError("Malformed RTC snapshot command")
                    _, expected_sequence = command
                    if (
                        not rtc_mode
                        or chunk is None
                        or isinstance(expected_sequence, bool)
                        or not isinstance(expected_sequence, int)
                        or expected_sequence != chunk_sequence
                    ):
                        _status(status_queue, "rtc_rejected", "snapshot has no matching active plan")
                        continue
                    if chunk_index >= chunk.length:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        tracking_checks_after = time.monotonic()
                        terminal_kind = "rtc_completed" if rtc_total_actions >= rtc_action_budget else "rtc_underrun"
                        _status(status_queue, terminal_kind, rtc_total_actions)
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        continue
                    _status(
                        status_queue,
                        "rtc_snapshot",
                        {
                            "sequence": chunk_sequence,
                            "action_index": chunk_index,
                            "plan_length": chunk.length,
                            "total_actions": rtc_total_actions,
                            "action_budget": rtc_action_budget,
                        },
                    )
                    # Do not continue: an action that is due in this publisher
                    # iteration must still advance after the snapshot.  The
                    # replacement path accounts for that action in its child-
                    # measured delay.

                elif kind == "rtc_start":
                    if not isinstance(command, tuple) or len(command) != 8:
                        raise DeploymentError("Malformed RTC start command")
                    _, sequence, created_at, action_budget, arm, left, right, expected_horizon = command
                    if chunk is not None or rtc_mode:
                        raise DeploymentError("RTC can start only with no active plan")
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != last_sequence + 1:
                        raise DeploymentError(f"Stale/out-of-order RTC plan {sequence!r}; expected {last_sequence + 1}")
                    if isinstance(action_budget, bool) or not isinstance(action_budget, int) or action_budget < 1:
                        raise DeploymentError("RTC action budget must be a positive integer")
                    try:
                        plan_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("RTC plan timestamp is invalid") from exc
                    if not np.isfinite(plan_age) or not 0.0 <= plan_age <= CHUNK_MAX_AGE_S:
                        raise DeploymentError(f"RTC plan {sequence} expired before execution")
                    proposed = ActionChunk(arm=arm, left_hand=left, right_hand=right)
                    if proposed.length != expected_horizon:
                        raise DeploymentError("RTC plan length changed in transit")
                    state = backend.state()
                    _validate_policy_target_input(
                        proposed,
                        state.arm,
                        state.left_hand,
                        state.right_hand,
                        conditioner,
                        context="Initial RTC plan",
                    )
                    chunk = proposed
                    chunk_sequence = int(sequence)
                    chunk_index = 0
                    next_action_at = time.monotonic()
                    last_sequence = sequence
                    holding = False
                    rtc_mode = True
                    rtc_total_actions = 0
                    rtc_action_budget = action_budget
                    _status(status_queue, "rtc_started", sequence)

                elif kind == "rtc_replace":
                    if not isinstance(command, tuple) or len(command) != 10:
                        raise DeploymentError("Malformed RTC replacement command")
                    (
                        _,
                        sequence,
                        created_at,
                        expected_sequence,
                        request_index,
                        expected_overlap,
                        arm,
                        left,
                        right,
                        expected_horizon,
                    ) = command
                    if rtc_mode and rtc_total_actions >= rtc_action_budget:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        tracking_checks_after = time.monotonic()
                        _status(status_queue, "rtc_completed", rtc_total_actions)
                        continue
                    stale = (
                        not rtc_mode
                        or chunk is None
                        or expected_sequence != chunk_sequence
                        or isinstance(request_index, bool)
                        or not isinstance(request_index, int)
                        or request_index < 0
                        or request_index > chunk_index
                        or expected_overlap != (chunk.length - request_index)
                    )
                    if stale:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        tracking_checks_after = time.monotonic()
                        _status(status_queue, "rtc_rejected", "stale plan generation or request index")
                        continue
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != last_sequence + 1:
                        raise DeploymentError(f"Stale/out-of-order RTC plan {sequence!r}; expected {last_sequence + 1}")
                    try:
                        plan_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("RTC replacement timestamp is invalid") from exc
                    if not np.isfinite(plan_age) or not 0.0 <= plan_age <= CHUNK_MAX_AGE_S:
                        raise DeploymentError(f"RTC replacement {sequence} expired before execution")

                    replacement = ActionChunk(arm=arm, left_hand=left, right_hand=right)
                    if replacement.length != expected_horizon:
                        raise DeploymentError("RTC replacement length changed in transit")
                    # Validate every raw value, including the portion that elapsed
                    # during inference. Raw step jumps are diagnostic only when the
                    # final-command conditioner is enabled.
                    try:
                        _validate_policy_target_input(
                            replacement,
                            replacement.arm[0],
                            replacement.left_hand[0],
                            replacement.right_hand[0],
                            conditioner,
                            context="RTC replacement plan",
                        )
                    except DeploymentError as exc:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        tracking_checks_after = time.monotonic()
                        _status(status_queue, "rtc_rejected", f"invalid replacement plan: {exc}")
                        continue
                    elapsed_actions = chunk_index - request_index
                    if elapsed_actions >= expected_overlap or elapsed_actions >= replacement.length:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        tracking_checks_after = time.monotonic()
                        _status(
                            status_queue,
                            "rtc_underrun",
                            {
                                "sequence": expected_sequence,
                                "elapsed_actions": elapsed_actions,
                                "overlap": expected_overlap,
                            },
                        )
                        continue

                    suffix = ActionChunk(
                        arm=np.ascontiguousarray(replacement.arm[elapsed_actions:]),
                        left_hand=np.ascontiguousarray(replacement.left_hand[elapsed_actions:]),
                        right_hand=np.ascontiguousarray(replacement.right_hand[elapsed_actions:]),
                    )
                    try:
                        _validate_policy_target_input(
                            suffix,
                            backend._arm_target,
                            backend._left_target,
                            backend._right_target,
                            conditioner,
                            context="RTC handoff",
                        )
                    except DeploymentError as exc:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                        tracking_checks_after = time.monotonic()
                        _status(status_queue, "rtc_rejected", f"unsafe handoff boundary: {exc}")
                        continue
                    chunk = replacement
                    chunk_sequence = int(sequence)
                    chunk_index = elapsed_actions
                    last_sequence = sequence
                    holding = False
                    _status(
                        status_queue,
                        "rtc_replaced",
                        {
                            "sequence": sequence,
                            "action_index": elapsed_actions,
                            "previous_sequence": expected_sequence,
                        },
                    )

                elif not isinstance(command, tuple) or len(command) != 6 or kind != "chunk":
                    raise DeploymentError(f"Unexpected or malformed actuator command {kind!r}")
                elif chunk is not None:
                    raise DeploymentError("Received a new chunk before the prior chunk completed")
                else:
                    _, sequence, created_at, arm, left, right = command
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != last_sequence + 1:
                        raise DeploymentError(
                            f"Stale/out-of-order action chunk {sequence!r}; expected {last_sequence + 1}"
                        )
                    try:
                        chunk_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("Action chunk timestamp is invalid") from exc
                    if not np.isfinite(chunk_age) or not 0.0 <= chunk_age <= CHUNK_MAX_AGE_S:
                        raise DeploymentError(f"Action chunk {sequence} expired before execution")
                    state = backend.state()
                    proposed = ActionChunk(arm=arm, left_hand=left, right_hand=right)
                    _validate_policy_target_input(
                        proposed,
                        state.arm,
                        state.left_hand,
                        state.right_hand,
                        conditioner,
                        context="Synchronous plan",
                    )
                    chunk = proposed
                    chunk_sequence = int(sequence)
                    chunk_index = 0
                    next_action_at = time.monotonic()
                    last_sequence = sequence
                    holding = False

            # This is deliberately checked after accepting any queued plan, but
            # before advancing its next 30 Hz target.  An operator stop can
            # therefore cancel synchronous and RTC motion without racing an
            # unconsumed plan command in the queue.  The event is independent
            # of the normal command queue so it cannot be delayed by a full
            # queue.
            if urgent_hold_event.is_set():
                if not urgent_hold_active or chunk is not None or rtc_mode or not holding:
                    state = backend.state()
                    _set_direct_target(
                        backend,
                        conditioner,
                        hand_watchdog,
                        state.arm,
                        state.left_hand,
                        state.right_hand,
                    )
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    _status(status_queue, "holding", last_sequence)
                urgent_hold_active = True
                now = time.monotonic()
                state = backend.state()
                if now >= tracking_checks_after:
                    _enforce_tracking(
                        backend,
                        state,
                        hand_watchdog,
                        now=now,
                        context=f"powered STOP sequence={last_sequence}",
                    )
                backend.publish()
                elapsed = time.monotonic() - loop_started
                stop_event.wait(max(0.0, period - elapsed))
                continue

            now = time.monotonic()
            if chunk is not None and now >= next_action_at:
                if chunk_index < chunk.length:
                    if now - next_action_at > MAX_ACTION_LATENESS_S:
                        raise DeploymentError(f"Action scheduler is {now - next_action_at:.3f}s late")
                    if conditioner is None:
                        backend.set_target(
                            chunk.arm[chunk_index],
                            chunk.left_hand[chunk_index],
                            chunk.right_hand[chunk_index],
                        )
                    else:
                        conditioner.set_desired(
                            chunk.arm[chunk_index],
                            chunk.left_hand[chunk_index],
                            chunk.right_hand[chunk_index],
                            now=now,
                        )
                    chunk_index += 1
                    if rtc_mode:
                        rtc_total_actions += 1
                        if rtc_total_actions >= rtc_action_budget:
                            # The completion status is emitted on the next 30 Hz
                            # boundary, after the final target has occupied one
                            # complete controller period.
                            chunk_index = chunk.length
                    # Keep the 30 Hz phase on the 100 Hz publication loop.  A
                    # lateness ceiling above prevents burst catch-up after a stall.
                    next_action_at += action_period
                else:
                    if rtc_mode:
                        state = backend.state()
                        _set_direct_target(
                            backend,
                            conditioner,
                            hand_watchdog,
                            state.arm,
                            state.left_hand,
                            state.right_hand,
                        )
                        tracking_checks_after = time.monotonic()
                        if rtc_total_actions >= rtc_action_budget:
                            _status(status_queue, "rtc_completed", rtc_total_actions)
                        else:
                            _status(
                                status_queue,
                                "rtc_underrun",
                                {
                                    "sequence": chunk_sequence,
                                    "elapsed_actions": chunk_index,
                                    "overlap": 0,
                                },
                            )
                        chunk = None
                        chunk_index = 0
                        rtc_mode = False
                        holding = True
                    else:
                        _status(status_queue, "completed", chunk_sequence)
                        chunk = None

            if now >= tracking_checks_after:
                # In conditioned mode this compares against the last target that
                # was actually published. The next conditioned command is formed
                # only below, so a fresh policy step is not misreported as servo lag.
                desired_left = None if conditioner is None else conditioner._desired_left
                desired_right = None if conditioner is None else conditioner._desired_right
                _enforce_tracking(
                    backend,
                    state,
                    hand_watchdog,
                    now=now,
                    context=(
                        f"active sequence={chunk_sequence} next_action_index={chunk_index} "
                        f"rtc={rtc_mode} holding={holding}"
                    ),
                    desired_left=desired_left,
                    desired_right=desired_right,
                )

            if conditioner is not None and not holding:
                conditioned = conditioner.next_command(
                    state.arm,
                    backend._arm_target,
                    backend._left_target,
                    backend._right_target,
                    now=now,
                )
                backend.set_target(
                    conditioned.arm[0],
                    conditioned.left_hand[0],
                    conditioned.right_hand[0],
                )

            backend.publish()
            elapsed = time.monotonic() - loop_started
            stop_event.wait(max(0.0, period - elapsed))
    except BaseException as exc:
        _status(status_queue, "fault", f"{type(exc).__name__}: {exc}")
    finally:
        release_ok = backend is None
        if backend is not None:
            def cleanup_phase(event: str, payload: dict[str, Any]) -> None:
                _status(
                    status_queue,
                    "cleanup_phase",
                    {"event": event, **payload},
                )

            cleanup_phase(
                "cleanup_begin",
                {"has_published": bool(getattr(backend, "_has_published", True))},
            )
            try:
                if getattr(backend, "_supports_cleanup_phases", False):
                    backend.release(cleanup_phase)
                else:
                    # Focused fake backends predate structured cleanup phases.
                    backend.release()
                release_ok = True
                _status(status_queue, "release_complete")
            except BaseException as exc:
                _status(status_queue, "release_failed", f"{type(exc).__name__}: {exc}")
            cleanup_phase("backend_close_begin", {})
            try:
                backend.close()
            except BaseException as exc:
                _status(status_queue, "close_failed", f"{type(exc).__name__}: {exc}")
            else:
                cleanup_phase("backend_close_complete", {})
        elif release_ok:
            _status(status_queue, "release_complete", "no backend was constructed")
        _status(status_queue, "stopped")


class SafeG1Dex3Actuator:
    """Parent-side handle for the watchdog-owning actuator process."""

    def __init__(
        self,
        simulation: bool,
        network_interface: str | None,
        command_conditioning: str = "none",
        authority_ramp_diagnostics: bool = False,
    ):
        if command_conditioning not in COMMAND_CONDITIONING_MODES:
            raise DeploymentError(f"Unknown command conditioning mode {command_conditioning!r}")
        context = mp.get_context("spawn")
        self._command_queue = context.Queue(maxsize=1)
        self._status_queue = context.Queue(maxsize=32)
        self._stop_event = context.Event()
        self._urgent_hold_event = context.Event()
        self._heartbeat = context.Value("d", time.monotonic())
        process_args = (
            simulation,
            network_interface,
            self._command_queue,
            self._status_queue,
            self._stop_event,
            self._heartbeat,
            self._urgent_hold_event,
            command_conditioning,
        )
        if authority_ramp_diagnostics:
            process_args += (True,)
        self._process = context.Process(
            target=_actuator_main,
            args=process_args,
            name="groot-g1-dex3-actuator",
        )
        self._sequence = 0
        self._started = False
        self._armed = False
        self._initialized = False
        self._warm_started = False
        self._holding = False
        self._chunk_in_flight = False
        self._pending_sequence: int | None = None
        self._rtc_active = False
        self._rtc_terminal: tuple[str, Any] | None = None
        self._command_conditioning = command_conditioning
        self._authority_ramp_diagnostics = authority_ramp_diagnostics
        self._last_authority_ramp_timing: dict[str, Any] | None = None
        self._control_lock = threading.Lock()
        self._immediate_hold_requested = threading.Event()
        self._immediate_release_requested = threading.Event()
        self._stopped_acknowledged = False
        self._hand_state_paused = False
        self._last_hand_state_event: Any = None
        self._release_completed = False
        self._last_cleanup_phase: Any = None
        self._closed = False

    def heartbeat(self) -> None:
        with self._heartbeat.get_lock():
            self._heartbeat.value = time.monotonic()

    def _record_auxiliary_status(self, kind: str, value: Any) -> bool:
        if kind == "hand_state_pause":
            self._hand_state_paused = True
            self._last_hand_state_event = value
            LOGGER.warning("Actuator paused for short Dex3 feedback loss: %s", value)
            return True
        if kind == "hand_state_recovered":
            self._hand_state_paused = False
            self._last_hand_state_event = value
            LOGGER.info("Actuator Dex3 feedback recovered: %s", value)
            return True
        if kind == "cleanup_phase":
            self._last_cleanup_phase = value
            return True
        if kind == "release_complete":
            self._release_completed = True
            return True
        return False

    def _wait_for_hand_feedback(self) -> None:
        """Do not enqueue new motion while the child is in a soft hand pause."""

        self.assert_healthy()
        if not getattr(self, "_hand_state_paused", False):
            return
        self._wait_status(
            "hand_state_recovered",
            timeout_s=ACTUATOR_HAND_STATE_MAX_AGE_S + 0.25,
        )

    def _wait_status(self, expected: str, timeout_s: float, payload: Any = None) -> Any:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            immediate = self.immediate_control_requested()
            if immediate is not None and expected != "urgent_holding":
                raise ImmediateControlEvent(immediate)
            if immediate == "release":
                raise ImmediateControlEvent("release")
            self.heartbeat()
            try:
                kind, value = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                if not self._process.is_alive():
                    raise DeploymentError("Actuator process exited unexpectedly")
                continue
            if kind == "fault":
                raise DeploymentError(f"Actuator fault: {value}")
            if kind in {"hand_state_pause", "hand_state_recovered"}:
                self._record_auxiliary_status(kind, value)
                if kind == expected and (payload is None or value == payload):
                    return value
                continue
            if self._record_auxiliary_status(kind, value):
                continue
            if kind == "authority_ramp_timing":
                self._last_authority_ramp_timing = value
                LOGGER.warning("ACTUATOR_TIMING %s", _format_authority_ramp_timing(value))
                continue
            if kind == "release_failed":
                raise DeploymentError(f"Actuator release failed: {value}")
            if kind == "close_failed":
                raise DeploymentError(f"Actuator resource cleanup failed: {value}")
            if kind == "stopped":
                self._stopped_acknowledged = True
                if self._immediate_release_requested.is_set():
                    raise ImmediateControlEvent("release")
                raise DeploymentError("Actuator stopped before the requested operation completed")
            if kind in {"rtc_underrun", "rtc_rejected"} and kind != expected and expected != "urgent_holding":
                self._rtc_terminal = ("hold", value)
                self._chunk_in_flight = False
                self._pending_sequence = None
                self._rtc_active = False
                self._holding = True
                raise RtcTerminalEvent("hold", value)
            if kind == "rtc_completed" and kind != expected and expected != "urgent_holding":
                self._rtc_terminal = ("complete", value)
                self._chunk_in_flight = False
                self._pending_sequence = None
                self._rtc_active = False
                self._holding = True
                raise RtcTerminalEvent("complete", value)
            if kind == expected and (payload is None or value == payload):
                return value
        if expected == "armed" and self._authority_ramp_diagnostics:
            heartbeat_age_ms = _heartbeat_age(self._heartbeat) * 1e3
            LOGGER.error(
                "ACTUATOR_TIMING event=parent_timeout expected=armed timeout_ms=%.3f "
                "heartbeat_age_ms=%.3f child_alive=%s last=%s",
                timeout_s * 1e3,
                heartbeat_age_ms,
                self._process.is_alive(),
                _format_authority_ramp_timing(self._last_authority_ramp_timing or {}),
            )
        raise TimeoutError(f"Timed out waiting for actuator status '{expected}'")

    def start(self) -> None:
        if self._started:
            raise DeploymentError("Actuator has already been started")
        self._process.start()
        self._wait_status("ready", timeout_s=8.0)
        self._started = True

    def arm(self, timeout_s: float | None = None) -> None:
        if not self._started or self._armed:
            raise DeploymentError("Actuator must be started exactly once before arming")
        self.heartbeat()
        self._command_queue.put(("arm",), timeout=0.2)
        if timeout_s is None:
            timeout_s = ARM_AUTHORITY_RAMP_S + 3.0
        try:
            self._wait_status("armed", timeout_s=timeout_s)
        except TimeoutError:
            # Do not wait for an outer finally block to tell the child to stop.
            self._stop_event.set()
            raise
        self._armed = True

    def initialize(self, spec: InitializationSpec) -> None:
        if not self._armed or self._initialized:
            raise DeploymentError("Actuator must be armed and not yet initialized")
        validate_initialization_spec(spec)
        self._wait_for_hand_feedback()
        self.heartbeat()
        try:
            self._command_queue.put(("initialize", time.monotonic(), spec), timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing initialization") from exc
        self._wait_status(
            "initialized",
            timeout_s=(
                INITIALIZATION_START_TIMEOUT_S
                + INITIALIZATION_MAX_DURATION_S
                + INITIALIZATION_CONVERGENCE_TIMEOUT_S
                + 3.0
            ),
            payload=spec.mode,
        )
        self._initialized = True
        self._holding = True

    def assert_healthy(self) -> None:
        immediate = self.immediate_control_requested()
        if immediate is not None:
            raise ImmediateControlEvent(immediate)
        issue = None
        try:
            while True:
                kind, value = self._status_queue.get_nowait()
                if self._record_auxiliary_status(kind, value):
                    continue
                if kind == "fault":
                    issue = f"fault: {value}"
                elif kind == "release_failed":
                    issue = f"release failed: {value}"
                elif kind == "close_failed":
                    issue = f"resource cleanup failed: {value}"
                elif kind == "stopped":
                    self._stopped_acknowledged = True
                    if issue is None:
                        issue = "stopped"
                elif kind == "rtc_completed":
                    self._rtc_terminal = ("complete", value)
                elif kind in {"rtc_underrun", "rtc_rejected"}:
                    self._rtc_terminal = ("hold", value)
        except queue.Empty:
            pass
        if issue is not None:
            if self._immediate_release_requested.is_set():
                raise ImmediateControlEvent("release")
            raise DeploymentError(f"Actuator process is unhealthy: {issue}")
        if not self._process.is_alive():
            if self._immediate_release_requested.is_set():
                raise ImmediateControlEvent("release")
            raise DeploymentError("Actuator process stopped")

    def warm_start(self, chunk: ActionChunk) -> None:
        """Smoothly reach a first policy target from an acknowledged hold."""

        if not self._initialized or not self._holding or self._chunk_in_flight:
            raise DeploymentError("Policy warm-start requires initialized HOLD with no chunk in flight")
        if getattr(self, "_command_conditioning", "none") == "xr":
            validate_action_chunk_limits(chunk)
        else:
            validate_action_chunk(chunk, chunk.arm[0], chunk.left_hand[0], chunk.right_hand[0])
        target = InitializationSpec(
            mode="pose-file",
            label="first policy target",
            arm=np.ascontiguousarray(chunk.arm[0], dtype=np.float64),
            left_hand=np.ascontiguousarray(chunk.left_hand[0], dtype=np.float64),
            right_hand=np.ascontiguousarray(chunk.right_hand[0], dtype=np.float64),
        )
        validate_initialization_spec(target)
        self._wait_for_hand_feedback()
        self.heartbeat()
        try:
            self._command_queue.put(("warm_start", time.monotonic(), target), timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing policy warm-start") from exc
        self._wait_status(
            "warm_started",
            timeout_s=(
                INITIALIZATION_START_TIMEOUT_S
                + INITIALIZATION_MAX_DURATION_S
                + INITIALIZATION_CONVERGENCE_TIMEOUT_S
                + 3.0
            ),
            payload=target.label,
        )
        self._warm_started = True
        self._holding = False

    def submit(self, chunk: ActionChunk) -> int:
        if not self._initialized:
            raise DeploymentError("Actuator must complete initialization before policy actions")
        if self._chunk_in_flight:
            raise DeploymentError("A policy chunk is already in flight")
        self._wait_for_hand_feedback()
        with self._control_lock:
            immediate = self.immediate_control_requested()
            if immediate is not None:
                raise ImmediateControlEvent(immediate)
            self._sequence += 1
            self.heartbeat()
            command = (
                "chunk",
                self._sequence,
                time.monotonic(),
                chunk.arm,
                chunk.left_hand,
                chunk.right_hand,
            )
            try:
                self._command_queue.put(command, timeout=0.2)
            except queue.Full as exc:
                raise DeploymentError("Actuator command queue is full; refusing to buffer") from exc
            self._chunk_in_flight = True
            self._pending_sequence = self._sequence
            self._holding = False
            return self._sequence

    def start_rtc(self, plan: ActionChunk, *, action_budget: int) -> int:
        """Start one full-horizon plan under the child-owned RTC scheduler."""

        if not self._initialized:
            raise DeploymentError("Actuator must complete initialization before RTC")
        if self._chunk_in_flight or self._rtc_active:
            raise DeploymentError("An action plan is already active")
        if isinstance(action_budget, bool) or not isinstance(action_budget, int) or action_budget < 1:
            raise DeploymentError("RTC action budget must be a positive integer")
        if getattr(self, "_command_conditioning", "none") == "xr":
            validate_action_chunk_limits(plan)
        else:
            validate_action_chunk(plan, plan.arm[0], plan.left_hand[0], plan.right_hand[0])
        self._wait_for_hand_feedback()
        with self._control_lock:
            immediate = self.immediate_control_requested()
            if immediate is not None:
                raise ImmediateControlEvent(immediate)
            next_sequence = self._sequence + 1
            self.heartbeat()
            command = (
                "rtc_start",
                next_sequence,
                time.monotonic(),
                action_budget,
                plan.arm,
                plan.left_hand,
                plan.right_hand,
                plan.length,
            )
            try:
                self._command_queue.put(command, timeout=0.2)
            except queue.Full as exc:
                raise DeploymentError("Actuator command queue is full; refusing RTC plan") from exc
        self._wait_status("rtc_started", timeout_s=1.0, payload=next_sequence)
        self._sequence = next_sequence
        self._chunk_in_flight = True
        self._pending_sequence = next_sequence
        self._rtc_active = True
        self._rtc_terminal = None
        self._holding = False
        return next_sequence

    def rtc_snapshot(self) -> RtcExecutionSnapshot:
        """Obtain the child scheduler's exact active plan index."""

        if not self._rtc_active or not self._chunk_in_flight or self._pending_sequence is None:
            raise DeploymentError("There is no active RTC plan to snapshot")
        event = self.poll_rtc_event()
        if event is not None:
            raise RtcTerminalEvent(*event)
        self.heartbeat()
        try:
            self._command_queue.put(("rtc_snapshot", self._pending_sequence), timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing RTC snapshot") from exc
        value = self._wait_status("rtc_snapshot", timeout_s=0.5)
        if not isinstance(value, dict):
            raise DeploymentError("Actuator returned a malformed RTC snapshot")
        try:
            snapshot = RtcExecutionSnapshot(
                sequence=int(value["sequence"]),
                action_index=int(value["action_index"]),
                plan_length=int(value["plan_length"]),
                total_actions=int(value["total_actions"]),
                action_budget=int(value["action_budget"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("Actuator returned an incomplete RTC snapshot") from exc
        if snapshot.sequence != self._pending_sequence:
            raise DeploymentError("RTC snapshot generation changed unexpectedly")
        return snapshot

    def replace_rtc(
        self,
        plan: ActionChunk,
        *,
        expected_sequence: int,
        request_index: int,
        expected_overlap: int,
    ) -> tuple[int, int]:
        """Atomically replace the unconsumed plan at the next 30 Hz action slot."""

        if not self._rtc_active or expected_sequence != self._pending_sequence:
            raise DeploymentError("RTC response is stale for the active plan generation")
        if getattr(self, "_command_conditioning", "none") == "xr":
            validate_action_chunk_limits(plan)
        else:
            validate_action_chunk(plan, plan.arm[0], plan.left_hand[0], plan.right_hand[0])
        event = self.poll_rtc_event()
        if event is not None:
            raise RtcTerminalEvent(*event)
        with self._control_lock:
            immediate = self.immediate_control_requested()
            if immediate is not None:
                raise ImmediateControlEvent(immediate)
            next_sequence = self._sequence + 1
            self.heartbeat()
            command = (
                "rtc_replace",
                next_sequence,
                time.monotonic(),
                expected_sequence,
                request_index,
                expected_overlap,
                plan.arm,
                plan.left_hand,
                plan.right_hand,
                plan.length,
            )
            try:
                self._command_queue.put(command, timeout=0.2)
            except queue.Full as exc:
                raise DeploymentError("Actuator command queue is full; refusing RTC replacement") from exc
        value = self._wait_status("rtc_replaced", timeout_s=0.5)
        if not isinstance(value, dict) or value.get("sequence") != next_sequence:
            raise DeploymentError("Actuator returned a malformed RTC replacement acknowledgment")
        try:
            action_index = int(value["action_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("RTC replacement acknowledgment has no action index") from exc
        self._sequence = next_sequence
        self._pending_sequence = next_sequence
        return next_sequence, action_index

    def poll_rtc_event(self) -> tuple[str, Any] | None:
        """Poll completion/HOLD/fault without allowing a worker to touch actuator state."""

        immediate = self.immediate_control_requested()
        if immediate is not None:
            raise ImmediateControlEvent(immediate)
        event = self._rtc_terminal
        self._rtc_terminal = None
        try:
            while True:
                kind, value = self._status_queue.get_nowait()
                if self._record_auxiliary_status(kind, value):
                    continue
                if kind == "fault":
                    raise DeploymentError(f"Actuator fault: {value}")
                if kind == "release_failed":
                    raise DeploymentError(f"Actuator release failed: {value}")
                if kind == "close_failed":
                    raise DeploymentError(f"Actuator resource cleanup failed: {value}")
                if kind == "stopped":
                    self._stopped_acknowledged = True
                    if self._immediate_release_requested.is_set():
                        raise ImmediateControlEvent("release")
                    raise DeploymentError("Actuator stopped during RTC")
                if kind == "rtc_completed":
                    event = ("complete", value)
                elif kind in {"rtc_underrun", "rtc_rejected"}:
                    event = ("hold", value)
        except queue.Empty:
            pass
        if event is not None:
            self._chunk_in_flight = False
            self._pending_sequence = None
            self._rtc_active = False
            self._holding = True
        if not self._process.is_alive():
            if self._immediate_release_requested.is_set():
                raise ImmediateControlEvent("release")
            raise DeploymentError("Actuator process stopped")
        return event

    def immediate_control_requested(self) -> str | None:
        """Return the pending thread-safe operator control, with release priority."""

        if self._immediate_release_requested.is_set():
            return "release"
        if self._immediate_hold_requested.is_set():
            return "hold"
        return None

    def request_immediate_hold(self) -> None:
        """Request a child-side measured-pose stop without using the command queue."""

        if not self._initialized:
            raise DeploymentError("Actuator must complete initialization before operator STOP")
        if self._closed or self._immediate_release_requested.is_set():
            return
        self._immediate_hold_requested.set()
        self._urgent_hold_event.set()

    def request_immediate_release(self) -> None:
        """Start the existing orderly release path without waiting for the main thread."""

        self._immediate_release_requested.set()
        self._stop_event.set()

    def finish_immediate_hold(self) -> str:
        """Fence queued plans, consume the child STOP ack, and update parent state."""

        if self._immediate_release_requested.is_set():
            return "release"
        if not self._immediate_hold_requested.is_set() and self._holding:
            return "hold"
        self._rtc_terminal = None
        # This command is queued after every submission that could have raced
        # the raw-key thread. The child keeps its STOP latch set until it
        # consumes this barrier, so none of those plans can advance a target.
        with self._control_lock:
            try:
                self._command_queue.put(("urgent_hold_barrier",), timeout=0.2)
            except queue.Full as exc:
                if self._immediate_release_requested.is_set():
                    return "release"
                raise DeploymentError("Actuator command queue is full; could not fence operator STOP") from exc
        try:
            acknowledged_sequence = self._wait_status("urgent_holding", timeout_s=1.0)
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                return "release"
            raise
        if isinstance(acknowledged_sequence, bool) or not isinstance(acknowledged_sequence, int):
            raise DeploymentError("Actuator returned a malformed urgent STOP acknowledgment")
        if acknowledged_sequence < self._sequence:
            raise DeploymentError("Urgent STOP acknowledgment regressed the action sequence")
        self._sequence = acknowledged_sequence
        self._immediate_hold_requested.clear()
        self._holding = True
        self._warm_started = False
        self._chunk_in_flight = False
        self._pending_sequence = None
        self._rtc_active = False
        return "release" if self._immediate_release_requested.is_set() else "hold"

    def wait_completed(self, sequence: int, timeout_s: float) -> str:
        if not self._chunk_in_flight or sequence != self._pending_sequence:
            raise DeploymentError(f"Action chunk {sequence} is not the pending sequence")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            immediate = self.immediate_control_requested()
            if immediate == "release":
                return "release"
            if immediate == "hold":
                return self.finish_immediate_hold()

            self.heartbeat()
            try:
                kind, value = self._status_queue.get(timeout=0.02)
            except queue.Empty:
                if not self._process.is_alive():
                    raise DeploymentError("Actuator process exited unexpectedly")
                continue
            if self._record_auxiliary_status(kind, value):
                continue
            if kind == "fault":
                raise DeploymentError(f"Actuator fault: {value}")
            if kind == "release_failed":
                raise DeploymentError(f"Actuator release failed: {value}")
            if kind == "close_failed":
                raise DeploymentError(f"Actuator resource cleanup failed: {value}")
            if kind == "stopped":
                self._stopped_acknowledged = True
                if self._immediate_release_requested.is_set():
                    return "release"
                raise DeploymentError("Actuator stopped before the action chunk completed")
            if kind == "holding" and value == sequence:
                # Soft-stale synchronous HOLD is emitted only after the five
                # distinct-sample recovery gate has completed.
                self._hand_state_paused = False
                self._immediate_hold_requested.clear()
                self._holding = True
                self._warm_started = False
                self._chunk_in_flight = False
                self._pending_sequence = None
                self._rtc_active = False
                return "hold"
            if kind == "completed" and value == sequence:
                self._chunk_in_flight = False
                self._pending_sequence = None
                return "complete"
            if kind in {"rtc_underrun", "rtc_rejected"}:
                self._rtc_terminal = ("hold", value)
                raise RtcTerminalEvent("hold", value)
            if kind == "rtc_completed":
                self._rtc_terminal = ("complete", value)
                raise RtcTerminalEvent("complete", value)
        raise TimeoutError(f"Timed out waiting for action chunk {sequence}")

    def hold(self) -> None:
        """Capture the measured pose and keep publishing it under watchdog control."""

        if not self._initialized:
            raise DeploymentError("Actuator must complete initialization before HOLD")
        if self._chunk_in_flight and not self._rtc_active:
            raise DeploymentError("Cannot enter HOLD before the current action chunk completes")
        if self._holding:
            return
        self._wait_for_hand_feedback()
        self.heartbeat()
        try:
            self._command_queue.put(("hold", time.monotonic()), timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing HOLD") from exc
        self._wait_status("holding", timeout_s=1.0, payload=self._sequence)
        self._holding = True
        self._warm_started = False
        self._chunk_in_flight = False
        self._pending_sequence = None
        self._rtc_active = False

    def close(self) -> None:
        if self._closed:
            return
        self.heartbeat()
        self._stop_event.set()
        if self._process.pid is None:
            self._closed = True
            return
        forced_kill = False
        runtime_faults: list[str] = []
        release_issues: list[str] = []
        close_issues: list[str] = []
        stopped_acknowledged = getattr(self, "_stopped_acknowledged", False)

        def consume(kind: str, value: Any) -> None:
            nonlocal stopped_acknowledged
            if self._record_auxiliary_status(kind, value):
                return
            if kind == "stopped":
                stopped_acknowledged = True
            elif kind == "release_failed":
                release_issues.append(str(value))
            elif kind == "close_failed":
                close_issues.append(str(value))
            elif kind == "fault":
                runtime_faults.append(str(value))
            elif kind == "authority_ramp_timing":
                self._last_authority_ramp_timing = value
                LOGGER.warning("ACTUATOR_TIMING %s", _format_authority_ramp_timing(value))

        started = time.monotonic()
        soft_deadline = started + ACTUATOR_RELEASE_SOFT_TIMEOUT_S
        hard_deadline = started + ACTUATOR_RELEASE_HARD_TIMEOUT_S
        soft_warning_emitted = False
        while self._process.is_alive() and time.monotonic() < hard_deadline:
            try:
                while True:
                    consume(*self._status_queue.get_nowait())
            except queue.Empty:
                pass
            now = time.monotonic()
            if now >= soft_deadline and not soft_warning_emitted:
                soft_warning_emitted = True
                LOGGER.critical(
                    "Actuator release exceeded %.1fs soft deadline; waiting to %.1fs hard deadline; last phase=%s",
                    ACTUATOR_RELEASE_SOFT_TIMEOUT_S,
                    ACTUATOR_RELEASE_HARD_TIMEOUT_S,
                    self._last_cleanup_phase,
                )
            self._process.join(timeout=min(0.05, max(0.0, hard_deadline - now)))
        if self._process.is_alive():
            LOGGER.critical(
                "Actuator is unresponsive at hard release deadline; forcing child exit; last phase=%s",
                self._last_cleanup_phase,
            )
            forced_kill = True
            self._process.kill()
            self._process.join(timeout=1.0)
        self._closed = True

        # Drain terminal queue records after the multiprocessing feeder has
        # observed child exit.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                kind, value = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            consume(kind, value)

        if self._process.is_alive():
            release_issues.append("actuator process remains alive after forced shutdown")
        if not getattr(self, "_release_completed", False):
            release_issues.append("no local release-complete acknowledgment was received")
        if not stopped_acknowledged:
            release_issues.append("no orderly stopped acknowledgment was received")
        if forced_kill:
            release_issues.append("actuator required SIGKILL")
        if self._process.exitcode not in (None, 0):
            release_issues.append(f"actuator exited with code {self._process.exitcode}")
        if release_issues:
            detail = "; ".join(release_issues)
            last_cleanup_phase = getattr(self, "_last_cleanup_phase", None)
            if last_cleanup_phase is not None:
                detail += f"; last cleanup phase={last_cleanup_phase}"
            raise DeploymentError("DDS release is unconfirmed: " + detail)
        if close_issues:
            raise DeploymentError(
                "DDS release completed, but actuator resource cleanup failed: " + "; ".join(close_issues)
            )
        if runtime_faults:
            raise DeploymentError(
                "Actuator fault; local DDS release completed: " + "; ".join(runtime_faults)
            )
