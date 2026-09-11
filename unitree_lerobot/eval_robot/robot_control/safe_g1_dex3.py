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
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.run_logging import (
    configure_process_logging,
    diagnostic_json_path,
    write_json,
)
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
    SurfaceNormalEncodingContract,
    validate_action_chunk,
    validate_action_chunk_limits,
    validate_initialization_spec,
    validate_measured_state,
)
from unitree_lerobot.eval_robot.robot_control.g1_arm_gravity import G1ArmGravityCompensator
from unitree_lerobot.utils.depth_encoding import encode_depth_gray_rgb
from unitree_lerobot.utils.surface_normal_encoding import encode_surface_normals_rgb


LOGGER = logging.getLogger(__name__)

STATE_MAX_AGE_S = 0.25
ACTUATOR_ARM_STATE_MAX_AGE_S = 0.100
ACTUATOR_HAND_STATE_PAUSE_AGE_S = 0.100
ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S = 1.250
ACTUATOR_HAND_STATE_MAX_AGE_S = 3.0
ACTUATOR_HAND_RECOVERY_SAMPLES = 3
# Parent waits that overlap an active plan need enough room for the permitted
# feedback pause plus the three-sample recovery/stable-publisher gates.
ACTUATOR_HAND_RECOVERY_WAIT_GRACE_S = ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S + 0.25
# Backward-compatible name for tests/internal imports that predate the explicit
# pause/operator-HOLD split.
ACTUATOR_HAND_STATE_WARNING_AGE_S = ACTUATOR_HAND_STATE_PAUSE_AGE_S
# Backward-compatible name for tests/internal imports.  It remains the hard
# arm-state deadline; hand state has its own limits above.
ACTUATOR_STATE_MAX_AGE_S = ACTUATOR_ARM_STATE_MAX_AGE_S
HEARTBEAT_TIMEOUT_S = 1.0 #SAFETYCHANGE was 1
CHUNK_MAX_AGE_S = 0.25
# Take arm_sdk authority in one matched-pose, zero-feed-forward write.  The
# robot may retain an earlier arm_sdk weight after an unconfirmed shutdown, so
# beginning a new process at weight zero is not a safe or idempotent handoff.
# Once the matched-pose hold owns the arm, introduce gravity feed-forward
# independently so authority and model torque never step at the same instant.
ARM_GRAVITY_RAMP_S = 1.5
ARM_TAKEOVER_SETTLE_TIMEOUT_S = 3.0
ARM_TAKEOVER_SETTLE_DWELL_S = 0.50
# CHANGEDSAFETY: original local adapter default was 1.0 s; current is 3.0 s.
# This is the orderly arm_sdk authority ramp-down duration.
ARM_RELEASE_RAMP_S = 3.0
ACTUATOR_RELEASE_SOFT_TIMEOUT_S = ARM_RELEASE_RAMP_S + 2.0
ACTUATOR_RELEASE_HARD_TIMEOUT_S = ARM_RELEASE_RAMP_S + 5.0
PUBLISH_HZ = 100.0
DDS_WRITE_TIMEOUT_S = 0.5
# CHANGEDSAFETY: original local adapter default was 6.0 rad/s, it was experimentally
# relaxed to 12.0 rad/s, and the current reviewed value restores the original 6.0 rad/s.
# This is the local measured arm-dq watchdog ceiling, not an official Unitree limit.
MAX_ARM_DQ_RAD_S = 6.0
MAX_ARM_TRACKING_ERROR_RAD = 0.35
# CHANGEDSAFETY: original local adapter default was 0.50 rad; current is 2.0 rad.
# This is max abs(measured hand q - commanded hand q), not a speed limit.
MAX_HAND_TRACKING_ERROR_RAD = 2
# The original 0.50-rad threshold remains a warning-only diagnostic.  It must
# persist across distinct hand-state samples for 0.20 s; 0.40 rad hysteresis
# prevents repeated warnings at the boundary.  Only the aligned 2.0-rad gate
# above is an actuator fault.
HAND_TRACKING_WARNING_RAD = 0.50
HAND_TRACKING_WARNING_CLEAR_RAD = 0.40
HAND_TRACKING_WARNING_DWELL_S = 0.20
HAND_COMMAND_HISTORY_SIZE = 128
TRACKING_GRACE_S = 0.50
MAX_ACTION_LATENESS_S = 0.02 #SAFETYCHANGE was 0.02
# Missing one complete 30 Hz target-residency window invalidates the remaining
# time-indexed plan even when the broader local lateness ceiling was relaxed.
# The child freezes the last command and asks the parent for a fresh observation
# rather than replaying overdue targets at the 100 Hz publisher rate.
ACTION_REPLAN_LATENESS_S = 1.0 / CONTROL_HZ
ACTION_REPLAN_RECOVERY_SAMPLES = 5
ACTION_REPLAN_RECOVERY_MAX_LOOP_GAP_S = 2.0 / PUBLISH_HZ

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
INITIALIZATION_ARM_TOLERANCE_RAD = 0.20 #increased from 0.05 SAFETYCHANGE
INITIALIZATION_HAND_TOLERANCE_RAD = 0.80 #SAFETYCHANGE increased from 0.4
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
ACTIVE_TIMING_RING_CAPACITY = 3_000
ACTIVE_TIMING_SLOW_LATENESS_MS = (20.0, 50.0, 100.0)


class RtcTerminalEvent(DeploymentError):
    """The child has already entered powered HOLD or completed its RTC budget."""

    def __init__(self, outcome: str, detail: Any):
        super().__init__(f"RTC {outcome}: {detail}")
        self.outcome = outcome
        self.detail = detail


class HandFeedbackReplan(DeploymentError):
    """A policy result was invalidated by a newer Dex3 feedback-pause epoch."""

    def __init__(self, detail: Any):
        super().__init__(f"Dex3 feedback pause invalidated the policy result: {detail}")
        self.detail = detail


class HandFeedbackOperatorHold(DeploymentError):
    """The 1.25 s Dex3 boundary revoked automatic task resume."""

    def __init__(self, detail: Any):
        super().__init__(f"Dex3 feedback crossed the operator-HOLD boundary: {detail}")
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
    # monotonic callback-receipt times, kept separately so one cached hand sample
    # is not compared with a newer 100 Hz command or counted twice.
    left_hand_received_at: float | None = None
    right_hand_received_at: float | None = None
    arm_received_at: float | None = None
    # Inspire DFX exposes one uint32 ``lost`` counter per motor.  Dex3 leaves
    # these unset; read-only DFX diagnostics preserve the latest exact values.
    left_hand_lost: tuple[int, ...] | None = None
    right_hand_lost: tuple[int, ...] | None = None


@dataclass(frozen=True)
class HandFreshnessResult:
    ready: bool
    entered: bool = False
    recovered: bool = False
    recovery_progressed: bool = False
    operator_hold_entered: bool = False
    pause_s: float = 0.0
    stale_hands: tuple[str, ...] = ()
    max_age_s: float = 0.0
    fresh_samples: int = 0


class HandStateFreshnessGate:
    """Turn short Dex3 delivery gaps into a bounded motion pause."""

    def __init__(self) -> None:
        self._active = False
        self._started_at = 0.0
        self._fresh_samples = 0
        self._operator_hold_active = False
        self._last_left_at = 0.0
        self._last_right_at = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._started_at = 0.0
        self._fresh_samples = 0
        self._operator_hold_active = False
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
            name for name, age_s in ages.items() if age_s > ACTUATOR_HAND_STATE_PAUSE_AGE_S
        )
        max_age_s = max(ages.values())
        if stale_hands:
            entered = not self._active
            if entered:
                self._active = True
                self._started_at = checked_at
            operator_hold_entered = (
                not self._operator_hold_active
                and max_age_s > ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S
            )
            if operator_hold_entered:
                self._operator_hold_active = True
            self._fresh_samples = 0
            self._last_left_at = float(left_at)
            self._last_right_at = float(right_at)
            return HandFreshnessResult(
                ready=False,
                entered=entered,
                operator_hold_entered=operator_hold_entered,
                stale_hands=stale_hands,
                max_age_s=max_age_s,
            )

        if not self._active:
            return HandFreshnessResult(ready=True, max_age_s=max_age_s)

        # Count actual paired DDS updates, not repeated 100 Hz reads of one
        # cached sample.
        if float(left_at) <= self._last_left_at or float(right_at) <= self._last_right_at:
            return HandFreshnessResult(
                ready=False,
                max_age_s=max_age_s,
                fresh_samples=self._fresh_samples,
            )
        self._last_left_at = float(left_at)
        self._last_right_at = float(right_at)
        self._fresh_samples += 1
        if self._fresh_samples < ACTUATOR_HAND_RECOVERY_SAMPLES:
            return HandFreshnessResult(
                ready=False,
                recovery_progressed=True,
                max_age_s=max_age_s,
                fresh_samples=self._fresh_samples,
            )

        pause_s = checked_at - self._started_at
        fresh_samples = self._fresh_samples
        self.reset()
        return HandFreshnessResult(
            ready=True,
            recovered=True,
            recovery_progressed=True,
            operator_hold_entered=False,
            pause_s=pause_s,
            max_age_s=max_age_s,
            fresh_samples=fresh_samples,
        )


class DdsHoldTimingAccumulator:
    """Collect low-overhead timing samples for the measured-HOLD diagnostic."""

    _FIELDS = (
        "cycle_ms",
        "lateness_ms",
        "work_ms",
        "state_lookup_ms",
        "arm_age_ms",
        "left_age_ms",
        "right_age_ms",
        "arm_state_check_ms",
        "arm_crc_ms",
        "arm_write_ms",
        "left_write_ms",
        "right_write_ms",
        "publish_total_ms",
    )
    _THRESHOLDS_MS = (20.0, 50.0, 75.0, 100.0)
    _SLOWEST_LIMIT = 20

    def __init__(self, *, period_s: float) -> None:
        self._period_ms = float(period_s) * 1e3
        self._started_at = time.monotonic()
        self._previous_loop_started: float | None = None
        self._values: dict[str, list[float]] = {name: [] for name in self._FIELDS}
        self._slowest: list[dict[str, Any]] = []
        self._samples = 0
        self._pause_count = 0
        self._recovery_count = 0
        self._max_pause_s = 0.0

    def note_freshness(self, result: HandFreshnessResult) -> None:
        if result.entered:
            self._pause_count += 1
        if result.recovered:
            self._recovery_count += 1
            self._max_pause_s = max(self._max_pause_s, float(result.pause_s))

    def record(
        self,
        *,
        loop_started: float,
        state_lookup_ms: float,
        state: RobotState,
        publish_timing_ms: dict[str, float],
        completed_at: float,
    ) -> None:
        now = float(completed_at)
        cycle_ms = (
            float("nan")
            if self._previous_loop_started is None
            else (float(loop_started) - self._previous_loop_started) * 1e3
        )
        self._previous_loop_started = float(loop_started)
        arm_at = state.captured_at if state.arm_received_at is None else state.arm_received_at
        left_at = state.captured_at if state.left_hand_received_at is None else state.left_hand_received_at
        right_at = state.captured_at if state.right_hand_received_at is None else state.right_hand_received_at
        record = {
            "sample": self._samples + 1,
            "at_s": now - self._started_at,
            "cycle_ms": cycle_ms,
            "lateness_ms": max(0.0, cycle_ms - self._period_ms) if np.isfinite(cycle_ms) else float("nan"),
            "work_ms": (now - float(loop_started)) * 1e3,
            "state_lookup_ms": float(state_lookup_ms),
            "arm_age_ms": max(0.0, now - float(arm_at)) * 1e3,
            "left_age_ms": max(0.0, now - float(left_at)) * 1e3,
            "right_age_ms": max(0.0, now - float(right_at)) * 1e3,
            "arm_state_check_ms": float(publish_timing_ms.get("arm_state_check", float("nan"))),
            "arm_crc_ms": float(publish_timing_ms.get("arm_crc", float("nan"))),
            "arm_write_ms": float(publish_timing_ms.get("arm_write", float("nan"))),
            "left_write_ms": float(publish_timing_ms.get("left_write", float("nan"))),
            "right_write_ms": float(publish_timing_ms.get("right_write", float("nan"))),
            "publish_total_ms": float(publish_timing_ms.get("publish_total", float("nan"))),
        }
        self._samples += 1
        for name in self._FIELDS:
            value = float(record[name])
            if np.isfinite(value):
                self._values[name].append(value)

        stage_names = (
            "state_lookup_ms",
            "arm_state_check_ms",
            "arm_crc_ms",
            "arm_write_ms",
            "left_write_ms",
            "right_write_ms",
        )
        finite_stages = [(name, float(record[name])) for name in stage_names if np.isfinite(record[name])]
        worst_stage, worst_ms = max(finite_stages, key=lambda item: item[1], default=("none", 0.0))

        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        slow_record = {
            "sample": record["sample"],
            "at_s": record["at_s"],
            "cycle_ms": finite_or_none(record["cycle_ms"]),
            "work_ms": finite_or_none(record["work_ms"]),
            "arm_age_ms": finite_or_none(record["arm_age_ms"]),
            "left_age_ms": finite_or_none(record["left_age_ms"]),
            "right_age_ms": finite_or_none(record["right_age_ms"]),
            "worst_stage": worst_stage,
            "worst_stage_ms": worst_ms,
            "arm_write_ms": finite_or_none(record["arm_write_ms"]),
            "left_write_ms": finite_or_none(record["left_write_ms"]),
            "right_write_ms": finite_or_none(record["right_write_ms"]),
        }
        score = max(
            value
            for value in (record["cycle_ms"], record["work_ms"], worst_ms)
            if np.isfinite(value)
        )
        slow_record["score_ms"] = float(score)
        self._slowest.append(slow_record)
        self._slowest.sort(key=lambda item: float(item["score_ms"]), reverse=True)
        del self._slowest[self._SLOWEST_LIMIT :]

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(np.max(array)),
        }

    def summary(self, *, completed_at: float | None = None) -> dict[str, Any]:
        ended_at = time.monotonic() if completed_at is None else float(completed_at)
        elapsed_s = max(0.0, ended_at - self._started_at)
        metrics = {name: self._stats(values) for name, values in self._values.items()}
        cycle_values = self._values["cycle_ms"]
        work_values = self._values["work_ms"]
        overruns = {}
        for threshold in self._THRESHOLDS_MS:
            label = str(int(threshold))
            overruns[f"cycle_over_{label}ms"] = sum(value > threshold for value in cycle_values)
            overruns[f"work_over_{label}ms"] = sum(value > threshold for value in work_values)
        return {
            "samples": self._samples,
            "elapsed_s": elapsed_s,
            "achieved_hz": self._samples / elapsed_s if elapsed_s > 0.0 else 0.0,
            "period_ms": self._period_ms,
            "hand_pause_count": self._pause_count,
            "hand_recovery_count": self._recovery_count,
            "max_hand_pause_s": self._max_pause_s,
            "metrics": metrics,
            "overruns": overruns,
            "slowest": list(self._slowest),
        }


class ActiveTimingRing:
    """Bounded, allocation-light timing history for the live actuator loop."""

    def __init__(self, *, capacity: int = ACTIVE_TIMING_RING_CAPACITY) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Active timing capacity must be a positive integer")
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._capacity = capacity
        self._total = 0
        self._started_at = time.monotonic()
        self._previous_loop_started: float | None = None
        self._current: dict[str, Any] | None = None
        self._last_state: RobotState | None = None

    def begin(
        self,
        *,
        loop_started: float,
        heartbeat_age_s: float | None,
        sequence: int,
        action_index: int,
        chunk_length: int | None,
        rtc: bool,
        rtc_total_actions: int,
        rtc_action_budget: int,
        holding: bool,
    ) -> None:
        self.finish(completed_at=loop_started)
        cycle_ms = (
            None
            if self._previous_loop_started is None
            else (float(loop_started) - self._previous_loop_started) * 1e3
        )
        self._previous_loop_started = float(loop_started)
        self._total += 1
        record: dict[str, Any] = {
            "sample": self._total,
            "at_s": float(loop_started) - self._started_at,
            "cycle_ms": cycle_ms,
            "heartbeat_age_ms": (
                None if heartbeat_age_s is None else max(0.0, float(heartbeat_age_s)) * 1e3
            ),
            "heartbeat_lookup_ms": None,
            "sequence": int(sequence),
            "next_action_index": int(action_index),
            "chunk_length": None if chunk_length is None else int(chunk_length),
            "rtc": bool(rtc),
            "rtc_total_actions": int(rtc_total_actions),
            "rtc_action_budget": int(rtc_action_budget),
            "holding": bool(holding),
            "event": "loop",
            "command_kind": None,
            "command_queue_get_ms": None,
            "command_processing_ms": None,
            "state_lookup_ms": None,
            "arm_age_ms": None,
            "left_age_ms": None,
            "right_age_ms": None,
            "max_arm_dq_rad_s": None,
            "freshness_ready": None,
            "freshness_entered": False,
            "freshness_recovered": False,
            "freshness_pause_ms": None,
            "freshness_max_age_ms": None,
            "freshness_stale_hands": [],
            "scheduler_due": False,
            "scheduler_lateness_ms": None,
            "raw_arm_step_rad": None,
            "raw_left_step_rad": None,
            "raw_right_step_rad": None,
            "conditioned_arm_step_rad": None,
            "conditioned_left_step_rad": None,
            "conditioned_right_step_rad": None,
            "tracking_ms": None,
            "tracking_arm_error_rad": None,
            "tracking_left_error_rad": None,
            "tracking_right_error_rad": None,
            "conditioner_ms": None,
            "publish_called": False,
            "arm_state_check_ms": None,
            "arm_crc_ms": None,
            "arm_write_ms": None,
            "left_write_ms": None,
            "right_write_ms": None,
            "publish_total_ms": None,
            "work_ms": None,
            "requested_wait_ms": None,
            "actual_wait_ms": None,
            "wait_overshoot_ms": None,
            "thread_cpu_ms": None,
            "exception": None,
            "_loop_started": float(loop_started),
            "_thread_cpu_started_ns": time.thread_time_ns(),
            "_command_processing_started_ns": None,
            "_finished": False,
        }
        self._records.append(record)
        self._current = record

    def update(self, **values: Any) -> None:
        if self._current is not None:
            self._current.update(values)

    def note_state(self, state: RobotState, *, completed_at: float, lookup_ms: float) -> None:
        self._last_state = state
        arm_at = state.captured_at if state.arm_received_at is None else state.arm_received_at
        left_at = state.captured_at if state.left_hand_received_at is None else state.left_hand_received_at
        right_at = state.captured_at if state.right_hand_received_at is None else state.right_hand_received_at
        self.update(
            state_lookup_ms=float(lookup_ms),
            arm_age_ms=max(0.0, completed_at - float(arm_at)) * 1e3,
            left_age_ms=max(0.0, completed_at - float(left_at)) * 1e3,
            right_age_ms=max(0.0, completed_at - float(right_at)) * 1e3,
            max_arm_dq_rad_s=float(np.max(np.abs(state.arm_dq))),
        )

    def note_freshness(self, result: HandFreshnessResult) -> None:
        self.update(
            freshness_ready=bool(result.ready),
            freshness_entered=bool(result.entered),
            freshness_recovered=bool(result.recovered),
            freshness_pause_ms=float(result.pause_s) * 1e3 if result.recovered else None,
            freshness_max_age_ms=float(result.max_age_s) * 1e3,
            freshness_stale_hands=list(result.stale_hands),
        )

    def start_command(self, kind: Any) -> None:
        if self._current is None:
            return
        self._current["command_kind"] = repr(kind)
        self._current["_command_processing_started_ns"] = time.monotonic_ns()

    def finish_command(self) -> None:
        if self._current is None:
            return
        started_ns = self._current.get("_command_processing_started_ns")
        if started_ns is not None and self._current.get("command_processing_ms") is None:
            self._current["command_processing_ms"] = (time.monotonic_ns() - int(started_ns)) / 1e6
        self._current["_command_processing_started_ns"] = None

    def note_publish(self, timing_ms: dict[str, float]) -> None:
        self.update(
            publish_called=True,
            arm_state_check_ms=_finite_float_or_none(timing_ms.get("arm_state_check")),
            arm_crc_ms=_finite_float_or_none(timing_ms.get("arm_crc")),
            arm_write_ms=_finite_float_or_none(timing_ms.get("arm_write")),
            left_write_ms=_finite_float_or_none(timing_ms.get("left_write")),
            right_write_ms=_finite_float_or_none(timing_ms.get("right_write")),
            publish_total_ms=_finite_float_or_none(timing_ms.get("publish_total")),
        )

    def wait(self, stop_event: Any, *, requested_s: float) -> None:
        requested = max(0.0, float(requested_s))
        wait_started = time.monotonic()
        self.update(
            work_ms=(wait_started - self._current["_loop_started"]) * 1e3 if self._current else None,
            requested_wait_ms=requested * 1e3,
        )
        stop_event.wait(requested)
        completed_at = time.monotonic()
        actual = completed_at - wait_started
        self.update(
            actual_wait_ms=actual * 1e3,
            wait_overshoot_ms=(actual - requested) * 1e3,
        )
        self.finish(completed_at=completed_at)

    def fault(self, exc: BaseException, *, event: str | None = None, **values: Any) -> None:
        if event is not None:
            values["event"] = event
        values["exception"] = f"{type(exc).__name__}: {exc}"
        self.update(**values)
        self.finish(completed_at=time.monotonic())

    def finish(self, *, completed_at: float) -> None:
        record = self._current
        if record is None or record.get("_finished"):
            return
        if record.get("work_ms") is None:
            record["work_ms"] = (float(completed_at) - float(record["_loop_started"])) * 1e3
        command_started_ns = record.get("_command_processing_started_ns")
        if command_started_ns is not None and record.get("command_processing_ms") is None:
            record["command_processing_ms"] = (time.monotonic_ns() - int(command_started_ns)) / 1e6
        record["thread_cpu_ms"] = (time.thread_time_ns() - int(record["_thread_cpu_started_ns"])) / 1e6
        record["_finished"] = True
        self._current = None

    def snapshot(
        self,
        *,
        trigger: str,
        error: str | None,
        backend: Any,
        command_conditioning: str,
        target_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        self.finish(completed_at=time.monotonic())
        records = []
        lateness_counts = {f"over_{int(threshold)}ms": 0 for threshold in ACTIVE_TIMING_SLOW_LATENESS_MS}
        for source in self._records:
            record = {key: value for key, value in source.items() if not key.startswith("_")}
            lateness = record.get("scheduler_lateness_ms")
            if isinstance(lateness, (int, float)) and np.isfinite(lateness):
                for threshold in ACTIVE_TIMING_SLOW_LATENESS_MS:
                    if lateness > threshold:
                        lateness_counts[f"over_{int(threshold)}ms"] += 1
            records.append(record)
        state = self._last_state
        state_payload = None
        if state is not None:
            state_payload = {
                "captured_at": float(state.captured_at),
                "mode_machine": int(state.mode_machine),
                "arm": np.asarray(state.arm, dtype=np.float64).tolist(),
                "arm_dq": np.asarray(state.arm_dq, dtype=np.float64).tolist(),
                "left_hand": np.asarray(state.left_hand, dtype=np.float64).tolist(),
                "right_hand": np.asarray(state.right_hand, dtype=np.float64).tolist(),
                "arm_received_at": state.arm_received_at,
                "left_hand_received_at": state.left_hand_received_at,
                "right_hand_received_at": state.right_hand_received_at,
            }
        targets = None
        if target_snapshot is not None:
            arm_target, left_target, right_target = target_snapshot
            targets = {
                "arm": np.asarray(arm_target, dtype=np.float64).tolist(),
                "left_hand": np.asarray(left_target, dtype=np.float64).tolist(),
                "right_hand": np.asarray(right_target, dtype=np.float64).tolist(),
            }
        elif backend is not None:
            targets = {
                "arm": np.asarray(getattr(backend, "_arm_target", []), dtype=np.float64).tolist(),
                "left_hand": np.asarray(getattr(backend, "_left_target", []), dtype=np.float64).tolist(),
                "right_hand": np.asarray(getattr(backend, "_right_target", []), dtype=np.float64).tolist(),
            }
        payload = {
            "schema_version": 1,
            "trigger": trigger,
            "error": error,
            "command_conditioning": command_conditioning,
            "publish_hz": PUBLISH_HZ,
            "control_hz": CONTROL_HZ,
            "max_action_lateness_s": MAX_ACTION_LATENESS_S,
            "action_replan_lateness_s": min(
                MAX_ACTION_LATENESS_S,
                ACTION_REPLAN_LATENESS_S,
            ),
            "capacity": self._capacity,
            "total_records": self._total,
            "retained_records": len(records),
            "dropped_records": max(0, self._total - len(records)),
            "scheduler_lateness_counts": lateness_counts,
            "last_state": state_payload,
            "last_targets": targets,
            "records": records,
        }
        normalized = _json_safe(payload)
        assert isinstance(normalized, dict)
        return normalized


def _finite_float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _json_safe(value: Any) -> Any:
    """Convert a diagnostic payload to strict, portable JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class CameraImages:
    rgb: np.ndarray
    depth_gray: np.ndarray | None = None
    surface_normals: np.ndarray | None = None
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

    def __init__(self, *, emit_logs: bool = True) -> None:
        self._emit_logs = bool(emit_logs)
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
                        if self._emit_logs:
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
                    if self._warning_active[hand_index, joint_index] and self._emit_logs:
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
            raise DeploymentError(
                "XR conditioned arm command exceeded its 100 Hz slew ceiling; "
                f"MAX_CONDITIONED_ARM_STEP_RAD={MAX_CONDITIONED_ARM_STEP_RAD:.4f} rad"
            )
        if np.any(
            np.abs(result.left_hand[0] - current_left)
            > np.maximum(MAX_CONDITIONED_HAND_STEP_RAD, left_recovery) + 1e-12
        ) or np.any(
            np.abs(result.right_hand[0] - current_right)
            > np.maximum(MAX_CONDITIONED_HAND_STEP_RAD, right_recovery) + 1e-12
        ):
            raise DeploymentError(
                "XR conditioned hand command exceeded its 100 Hz slew ceiling; "
                "MAX_CONDITIONED_HAND_STEP_RAD="
                f"{np.array2string(MAX_CONDITIONED_HAND_STEP_RAD, precision=4)} rad"
            )
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
                f"joint {joint} ({joint_names[joint]}): {target[joint]:.4f} rad; "
                f"MEASURED_LIMIT_TOLERANCE_RAD={MEASURED_LIMIT_TOLERANCE_RAD:.4f} rad"
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
                    f"joint {joint} ({joint_names[joint]}); "
                    f"HAND_LIMIT_TOLERANCE_RAD={HAND_LIMIT_TOLERANCE_RAD:.4f} rad"
                )
        for joint in np.flatnonzero(above):
            if entered_strict_range[joint] or previous[joint] <= strict_upper[joint] or delta[joint] > 1e-12:
                raise DeploymentError(
                    f"Initialization {name} recovery is not monotonic inward at step {step}, "
                    f"joint {joint} ({joint_names[joint]}); "
                    f"HAND_LIMIT_TOLERANCE_RAD={HAND_LIMIT_TOLERANCE_RAD:.4f} rad"
                )

        entered_strict_range |= ~(below | above)
        previous = target

    if not np.all(entered_strict_range):
        joint = int(np.flatnonzero(~entered_strict_range)[0])
        raise DeploymentError(
            f"Initialization {name} recovery did not enter the strict target range at joint {joint} "
            f"({joint_names[joint]}); "
            f"HAND_LIMIT_TOLERANCE_RAD={HAND_LIMIT_TOLERANCE_RAD:.4f} rad"
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
            f"INITIALIZATION_MAX_DURATION_S={INITIALIZATION_MAX_DURATION_S:.1f}s"
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
        *,
        max_age_constant: str = "STATE_MAX_AGE_S",
        hand_max_age_constant: str | None = None,
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
        self._arm_max_age_constant = max_age_constant
        self._hand_max_age_constant = (
            max_age_constant if hand_max_age_constant is None else hand_max_age_constant
        )
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
        limit_names = {
            "arm": self._arm_max_age_constant,
            "left": self._hand_max_age_constant,
            "right": self._hand_max_age_constant,
        }
        stale = []
        for key, message in messages.items():
            if message is None:
                detail = f", rejected {rejected_zero[key]} all-zero frames" if key in rejected_zero else ""
                stale.append(
                    f"{key} (missing; {limit_names[key]}={limits[key]:.3f}s{detail})"
                )
                continue
            age_s = now - updated_at[key]
            if age_s > limits[key]:
                detail = f", rejected {rejected_zero[key]} all-zero frames" if key in rejected_zero else ""
                stale.append(
                    f"{key} (age {age_s:.3f}s > {limits[key]:.3f}s; "
                    f"{limit_names[key]}={limits[key]:.3f}s{detail})"
                )
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
        raise DeploymentError("TeleImager returned an undecodable aligned-depth PNG")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise DeploymentError(
            f"Aligned depth must decode to an HxW uint16 image; got shape={depth.shape}, dtype={depth.dtype}"
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
) -> tuple[int, int | None, float | None]:
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

    try:
        port = int(head["zmq_port"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("TeleImager has no valid zmq_port") from exc
    if not 1 <= port <= 65535:
        raise DeploymentError(f"TeleImager reported invalid zmq_port {port!r}")

    depth_port: int | None = None
    depth_scale: float | None = None
    if requires_depth:
        if head.get("image_shape") != EXPECTED_DEPTH_VIEW_SHAPE[:2]:
            raise DeploymentError(
                "Aligned-depth geometry requires the calibrated 640x480 color frame; "
                f"TeleImager reported image_shape={head.get('image_shape')!r}"
            )
        if str(head.get("type", "")).lower() != "realsense":
            raise DeploymentError("Aligned-depth geometry requires a RealSense TeleImager head camera")
        if not head.get("enable_depth", False):
            raise DeploymentError("Geometry checkpoint requires TeleImager aligned depth")
        if head.get("binocular", False):
            raise DeploymentError("Geometry checkpoint does not support a binocular head-camera layout")
        try:
            depth_port = int(head["depth_zmq_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("TeleImager has no valid aligned-depth depth_zmq_port") from exc
        if not 1 <= depth_port <= 65535:
            raise DeploymentError(f"TeleImager reported invalid depth_zmq_port {depth_port!r}")
        try:
            depth_scale = float(head["depth_scale_m_per_unit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("TeleImager has no valid live RealSense depth scale") from exc
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise DeploymentError(f"TeleImager reported invalid depth scale {depth_scale!r}")
    return port, depth_port, depth_scale


class TeleimagerCamera:
    """TeleImager client for colour or client-encoded legacy aligned-depth geometry."""

    def __init__(
        self,
        host: str,
        depth_encoding: DepthEncodingContract | None = None,
        surface_normal_encoding: SurfaceNormalEncodingContract | None = None,
    ):
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

        if depth_encoding is not None and surface_normal_encoding is not None:
            raise DeploymentError("Camera cannot emit depth_gray_view and surface_normals_view together")
        self._client = None
        self._depth_encoding = depth_encoding
        self._surface_normal_encoding = surface_normal_encoding
        self._requires_depth = depth_encoding is not None or surface_normal_encoding is not None
        self._last_color_received_ns: int | None = None
        self._last_depth_received_ns: int | None = None
        live_config = request_live_camera_config(host)
        stream_port, depth_port, depth_scale = _validate_live_head_config(
            live_config,
            requires_depth=self._requires_depth,
        )
        if depth_scale is not None:
            self._depth_scale_m_per_unit = depth_scale
        try:
            self._client = ImageClient(
                host=host,
                request_bgr=False,
                request_depth=self._requires_depth,
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
            self._depth_subscriber = (
                self._client._subscriber_manager._subscriber_threads[(host, depth_port)]
                if depth_port is not None
                else None
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.close()
            raise DeploymentError("Could not verify the live TeleImager head subscriber") from exc
        if not self._head_subscriber.is_alive():
            self.close()
            raise DeploymentError("TeleImager head subscriber stopped during startup")
        if self._depth_subscriber is not None and not self._depth_subscriber.is_alive():
            self.close()
            raise DeploymentError("TeleImager aligned-depth subscriber stopped during startup")
        self._reported_stream_fps = False
        if self._requires_depth:
            LOGGER.warning(
                "Using legacy independent TeleImager RGB and aligned-depth streams on ports %d and %d, "
                "matching the training data collection transport",
                stream_port,
                depth_port,
            )

    def _assert_subscriber_alive(self) -> None:
        if not self._head_subscriber.is_alive():
            raise DeploymentError("TeleImager head subscriber stopped")
        depth_subscriber = getattr(self, "_depth_subscriber", None)
        if depth_subscriber is not None and not depth_subscriber.is_alive():
            raise DeploymentError("TeleImager aligned-depth subscriber stopped")

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

    def _read_geometry(self, timeout_s: float) -> CameraImages:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._assert_subscriber_alive()
                color_frame = self._client.get_head_frame()
                depth_frame = self._client.get_head_depth_frame()
                frames = (("RGB", color_frame), ("aligned-depth", depth_frame))
                measured_fps: dict[str, float] = {}
                received_ns: dict[str, int] = {}
                now_ns = time.monotonic_ns()
                for name, frame in frames:
                    try:
                        measured_fps[name] = float(getattr(frame, "fps", 0.0))
                    except (TypeError, ValueError) as exc:
                        raise TimeoutError(f"TeleImager {name} stream FPS is not numeric") from exc
                    if not np.isfinite(measured_fps[name]) or measured_fps[name] <= 0.0:
                        raise TimeoutError(
                            f"TeleImager {name} stream has not established a live rolling FPS"
                        )
                    timestamp = getattr(frame, "received_monotonic_ns", None)
                    if timestamp is None:
                        raise DeploymentError(
                            f"TeleImager {name} frame has no local receive timestamp"
                        )
                    received_ns[name] = int(timestamp)
                    age_s = (now_ns - received_ns[name]) / 1_000_000_000.0
                    if age_s < 0.0 or age_s > RGBD_MAX_RECEIVE_AGE_S:
                        raise TimeoutError(
                            f"TeleImager {name} frame is stale ({age_s:.3f}s old); "
                            f"RGBD_MAX_RECEIVE_AGE_S={RGBD_MAX_RECEIVE_AGE_S:.3f}s"
                        )

                previous = {
                    "RGB": self._last_color_received_ns,
                    "aligned-depth": self._last_depth_received_ns,
                }
                for name in ("RGB", "aligned-depth"):
                    if previous[name] is not None and received_ns[name] < previous[name]:
                        raise DeploymentError(
                            f"TeleImager {name} receive timestamp regressed; restart the policy runner"
                        )
                    if previous[name] == received_ns[name]:
                        raise TimeoutError(f"TeleImager {name} frame is not new")

                rgb = decode_color_0_rgb(color_frame, self.config)
                depth_bytes = getattr(depth_frame, "jpg", None)
                if not depth_bytes:
                    raise TimeoutError("TeleImager aligned-depth stream has no fresh PNG")
                depth_u16 = _decode_depth_png_u16(depth_bytes)
                if list(depth_u16.shape) != EXPECTED_DEPTH_VIEW_SHAPE[:2]:
                    raise DeploymentError(
                        f"Aligned depth shape {depth_u16.shape} does not match the training contract "
                        f"{tuple(EXPECTED_DEPTH_VIEW_SHAPE[:2])}"
                    )
                try:
                    depth_encoding = getattr(self, "_depth_encoding", None)
                    surface_normal_encoding = getattr(self, "_surface_normal_encoding", None)
                    if depth_encoding is not None:
                        depth_gray = encode_depth_gray_rgb(
                            depth_u16,
                            scale_m_per_unit=self._depth_scale_m_per_unit,
                            near_m=depth_encoding.near_m,
                            far_m=depth_encoding.far_m,
                        )
                        surface_normals = None
                    elif surface_normal_encoding is not None:
                        depth_gray = None
                        surface_normals = encode_surface_normals_rgb(
                            depth_u16,
                            scale_m_per_unit=self._depth_scale_m_per_unit,
                            intrinsics=surface_normal_encoding.intrinsics,
                            max_neighbor_depth_delta_m=(
                                surface_normal_encoding.max_neighbor_depth_delta_m
                            ),
                        )
                    else:
                        raise DeploymentError("Geometry camera has no selected encoding")
                except ValueError as exc:
                    raise DeploymentError(f"Could not encode aligned depth geometry: {exc}") from exc
                self._last_color_received_ns = received_ns["RGB"]
                self._last_depth_received_ns = received_ns["aligned-depth"]
                if not self._reported_stream_fps:
                    LOGGER.info(
                        "TeleImager legacy streams are live: RGB %.1f FPS, aligned depth %.1f FPS",
                        measured_fps["RGB"],
                        measured_fps["aligned-depth"],
                    )
                    self._reported_stream_fps = True
                return CameraImages(
                    rgb=rgb,
                    depth_gray=depth_gray,
                    surface_normals=surface_normals,
                )
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.005)
            except (TypeError, ValueError) as exc:
                raise DeploymentError(f"Invalid TeleImager geometry frame: {exc}") from exc
        raise TimeoutError(f"Timed out waiting for fresh TeleImager geometry ({last_error})")

    def read(self, timeout_s: float = 3.0) -> CameraImages:
        if self._requires_depth:
            return self._read_geometry(timeout_s)
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

    def __init__(
        self,
        simulation: bool,
        network_interface: str | None,
        gravity_feedforward: bool = True,
    ):
        if not isinstance(gravity_feedforward, bool):
            raise DeploymentError("gravity_feedforward must be a bool")
        # CHANGEDSAFETY: the original deployment adapter published zero arm
        # feed-forward torque.  Unitree XR instead publishes static RNEA torque
        # on every command; restore that demonstrated controller contract.
        # Build and exercise the reviewed XR dynamics model before DDS is
        # initialized and, critically, before any command publisher exists.
        # Missing Pinocchio or a changed/invalid URDF therefore fails closed.
        self._gravity_feedforward = gravity_feedforward
        self._arm_gravity = G1ArmGravityCompensator() if gravity_feedforward else None
        if self._arm_gravity is None:
            # The explicit opt-out restores the pre-feed-forward command
            # contract.  Do not import/build Pinocchio when it is disabled.
            LOGGER.warning("G1 arm gravity feed-forward is disabled; outgoing arm tau is zero")
        else:
            LOGGER.info(
                "XR-compatible G1 arm gravity feed-forward is ready: %s",
                self._arm_gravity.urdf_path,
            )
        initialize_dds(simulation, network_interface)
        self.simulation = simulation
        self.reader = G1Dex3StateReader(
            simulation=simulation,
            max_age_s=ACTUATOR_ARM_STATE_MAX_AGE_S,
            hand_max_age_s=ACTUATOR_HAND_STATE_MAX_AGE_S,
            max_age_constant="ACTUATOR_ARM_STATE_MAX_AGE_S",
            hand_max_age_constant="ACTUATOR_HAND_STATE_MAX_AGE_S",
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
        self._last_published_arm_q: np.ndarray | None = None
        self._last_published_arm_tau: np.ndarray | None = None
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

    def _validate_prearm_takeover_state(self, state: RobotState) -> None:
        """Apply the existing pre-arm limits to a prospective takeover sample."""

        state_age = time.monotonic() - state.captured_at
        if not np.isfinite(state_age) or not 0.0 <= state_age <= PREARM_STATE_MAX_AGE_S:
            raise DeploymentError(
                f"Robot state age before arm takeover is {state_age:.3f}s; "
                f"PREARM_STATE_MAX_AGE_S={PREARM_STATE_MAX_AGE_S:.3f}s"
            )
        arm_dq_joint = int(np.argmax(np.abs(state.arm_dq)))
        arm_dq = float(state.arm_dq[arm_dq_joint])
        if abs(arm_dq) > PREARM_MAX_ARM_DQ_RAD_S:
            raise DeploymentError(
                f"Arm is not stationary before takeover at joint {arm_dq_joint} "
                f"({ARM_JOINT_NAMES[arm_dq_joint]}): {arm_dq:+.3f} rad/s; "
                f"PREARM_MAX_ARM_DQ_RAD_S={PREARM_MAX_ARM_DQ_RAD_S:.3f} rad/s"
            )
        drift_group, drift_joint, drift_name, drift = _largest_named_value(
            (
                ("arm", state.arm - self._arm_target, ARM_JOINT_NAMES),
                ("left hand", state.left_hand - self._left_target, LEFT_HAND_JOINT_NAMES),
                ("right hand", state.right_hand - self._right_target, RIGHT_HAND_JOINT_NAMES),
            )
        )
        if abs(drift) > PREARM_MAX_POSITION_DRIFT_RAD:
            raise DeploymentError(
                f"Pre-takeover {drift_group} position drift at joint {drift_joint} "
                f"({drift_name}) is {drift:+.4f} rad; "
                f"PREARM_MAX_POSITION_DRIFT_RAD={PREARM_MAX_POSITION_DRIFT_RAD:.4f} rad"
            )

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

    def _write_arm_message(
        self,
        *,
        require_qualified_state: bool,
        require_prearm_takeover_state: bool = False,
    ) -> None:
        """CRC and write the already-populated arm message once."""

        timing_enabled = getattr(self, "_authority_ramp_timing_enabled", False)
        timing = getattr(self, "_last_publish_timing_ms", {})
        if not self.simulation:
            # Never copy an unqualified/transient mode into a real arm command.
            # Operational writes also re-check the latest state immediately
            # before publishing.  Cleanup can skip the freshness dependency so
            # an attempted authority release is not defeated by the same state
            # fault that triggered it.
            self._arm_message.mode_machine = QUALIFIED_REAL_MODE_MACHINE
            if require_qualified_state:
                started_ns = time.monotonic_ns()
                try:
                    state = self._validate_runtime_state(self.reader.latest())
                    if require_prearm_takeover_state:
                        self._validate_prearm_takeover_state(state)
                finally:
                    if timing_enabled:
                        timing["arm_state_check"] = (time.monotonic_ns() - started_ns) / 1e6
            self._arm_message.motor_cmd[29].q = float(self._weight)
        started_ns = time.monotonic_ns()
        try:
            self._arm_message.crc = self._crc.Crc(self._arm_message)
        finally:
            if timing_enabled:
                timing["arm_crc"] = (time.monotonic_ns() - started_ns) / 1e6
        started_ns = time.monotonic_ns()
        try:
            write_ok = self._arm_publisher.Write(self._arm_message, timeout=DDS_WRITE_TIMEOUT_S)
        finally:
            if timing_enabled:
                timing["arm_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if write_ok is not True:
            raise DeploymentError(
                f"Arm DDS Write failed; DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
            )
        self._has_published = True

    def _publish_arm(
        self,
        require_qualified_state: bool = True,
        *,
        gravity_scale: float = 1.0,
        require_prearm_takeover_state: bool = False,
    ) -> None:
        # Reproduce XR teleoperation's Pinocchio RNEA feed-forward for the final
        # outgoing q, including initialization, warmup, HOLD, authority ramps,
        # and simulation.  compute() checks finiteness, joint order, and a
        # conservative per-joint torque envelope before the DDS message changes.
        # Allocate both cache candidates before touching the message or calling
        # DDS.  After a successful Write, publishing the cache is only two
        # reference assignments and cannot fail due to an array allocation.
        if (
            isinstance(gravity_scale, bool)
            or not isinstance(gravity_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(float(gravity_scale))
            or not 0.0 <= float(gravity_scale) <= 1.0
        ):
            raise DeploymentError(
                f"gravity_scale must be finite and inside [0, 1], got {gravity_scale!r}"
            )
        gravity_scale = float(gravity_scale)
        candidate_q = self._arm_target.copy()
        if self._arm_gravity is None:
            full_gravity_tau = np.zeros(ARM_DOF, dtype=np.float64)
        else:
            # Validate the dynamics result and torque envelope before the first
            # full-authority Write even when that packet intentionally sends
            # zero feed-forward torque.
            full_gravity_tau = self._arm_gravity.compute(candidate_q)
        gravity_tau = gravity_scale * full_gravity_tau
        for offset, index in enumerate(self._arm_indices):
            command = self._arm_message.motor_cmd[index]
            command.q = float(candidate_q[offset])
            command.tau = float(gravity_tau[offset])
        self._write_arm_message(
            require_qualified_state=require_qualified_state,
            require_prearm_takeover_state=require_prearm_takeover_state,
        )
        # Cache only after Write succeeds.  Cleanup can then shed arm_sdk
        # authority even when the dynamics computation that triggered a fault
        # is no longer usable.  The cached pair is exactly what DDS accepted.
        self._last_published_arm_q = candidate_q
        self._last_published_arm_tau = gravity_tau

    def _publish_last_arm_for_release(self) -> None:
        """Write the last successful q/tau while changing only authority weight."""

        if self._last_published_arm_q is None or self._last_published_arm_tau is None:
            raise DeploymentError("No successful arm q/tau is available for authority release")
        for offset, index in enumerate(self._arm_indices):
            command = self._arm_message.motor_cmd[index]
            command.q = float(self._last_published_arm_q[offset])
            command.tau = float(self._last_published_arm_tau[offset])
        self._write_arm_message(require_qualified_state=False)

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

        timing_enabled = getattr(self, "_authority_ramp_timing_enabled", False)
        timing = getattr(self, "_last_publish_timing_ms", {})
        started_ns = time.monotonic_ns()
        try:
            left_ok = self._left_publisher.Write(self._left_message, timeout=DDS_WRITE_TIMEOUT_S)
        finally:
            if timing_enabled:
                timing["left_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if left_ok is not True:
            raise DeploymentError(
                f"Left Dex3 DDS Write failed; DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
            )
        self._left_hand_publish_history.append(PublishedHandTarget(completed_at=time.monotonic(), target=left_target))
        started_ns = time.monotonic_ns()
        try:
            right_ok = self._right_publisher.Write(self._right_message, timeout=DDS_WRITE_TIMEOUT_S)
        finally:
            if timing_enabled:
                timing["right_write"] = (time.monotonic_ns() - started_ns) / 1e6
        if right_ok is not True:
            raise DeploymentError(
                f"Right Dex3 DDS Write failed; DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
            )
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
                    failures.append(
                        f"{name} Dex3 stop Write failed; "
                        f"DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
                    )
            except Exception as exc:
                failures.append(
                    f"{name} Dex3 stop Write raised {exc!r}; "
                    f"DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
                )
            finally:
                if phase_callback is not None:
                    phase_callback(
                        f"{name.lower()}_hand_stop_end",
                        {"elapsed_s": time.monotonic() - started},
                    )
        if failures:
            raise DeploymentError("; ".join(failures))

    def publish(self) -> None:
        timing_enabled = getattr(self, "_authority_ramp_timing_enabled", False)
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
        # Preserve one constant weight-slope across both normal full-authority
        # shutdowns and faults during acquisition.  A fault after only a tiny
        # amount of authority was acquired must not keep a suspect command
        # alive for the entire full-weight release duration.
        full_weight_duration = max(float(ARM_RELEASE_RAMP_S), period)
        duration = max(full_weight_duration * start_weight, period)
        ramp_started = time.monotonic()
        next_write_at = ramp_started
        arm_writes = 0
        skipped_ticks = 0
        max_arm_cycle_s = 0.0
        last_successful_weight = start_weight
        # Release does not depend on policy, camera, fresh state, or IK.
        LOGGER.info(
            "Arm authority release started at weight %.4f: scheduled ramp %.3fs "
            "(full-weight ramp %.3fs)",
            start_weight,
            duration,
            full_weight_duration,
        )
        if phase_callback is not None:
            phase_callback(
                "arm_release_begin",
                {
                    "start_weight": start_weight,
                    "scheduled_ramp_s": duration,
                    "full_weight_ramp_s": full_weight_duration,
                },
            )
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
                self._publish_last_arm_for_release()
            except Exception as exc:
                failures.append(
                    f"arm_sdk authority release failed: {exc}; "
                    f"DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
                )
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
                self._publish_last_arm_for_release()
                arm_writes += 1
                last_successful_weight = 0.0
            except Exception as exc:
                failures.append(
                    f"final zero-weight arm_sdk Write failed: {exc}; "
                    f"DDS_WRITE_TIMEOUT_S={DDS_WRITE_TIMEOUT_S:.3f}s"
                )
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


def _hand_pause_generation_value(generation: Any | None) -> int:
    if generation is None:
        return 0
    with generation.get_lock():
        return int(generation.value)


def _advance_hand_pause_generation(generation: Any | None) -> int:
    if generation is None:
        return 0
    with generation.get_lock():
        generation.value += 1
        return int(generation.value)


def _policy_generation_matches(expected: Any, generation: Any | None) -> bool:
    if expected is None:
        return True
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise DeploymentError("Policy command has an invalid Dex3 pause generation")
    return expected == _hand_pause_generation_value(generation)


def _status(status_queue: MpQueue, kind: str, payload: Any = None) -> None:
    try:
        status_queue.put((kind, payload), timeout=0.2)
    except queue.Full:
        pass


def _status_nonblocking(status_queue: MpQueue, kind: str, payload: Any = None) -> bool:
    """Best-effort diagnostic status that must never delay the control loop."""

    try:
        status_queue.put_nowait((kind, payload))
    except queue.Full:
        return False
    return True


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
            "pause_age_s": ACTUATOR_HAND_STATE_PAUSE_AGE_S,
            "operator_hold_age_s": ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
            "hard_age_s": ACTUATOR_HAND_STATE_MAX_AGE_S,
        }
        _status_nonblocking(status_queue, "hand_state_pause", payload)
    if result.operator_hold_entered:
        payload = {
            "context": context,
            "hands": result.stale_hands,
            "age_s": result.max_age_s,
            "operator_hold_age_s": ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
            "hard_age_s": ACTUATOR_HAND_STATE_MAX_AGE_S,
        }
        _status(status_queue, "hand_state_operator_hold", payload)
    if result.recovery_progressed:
        payload = {
            "context": context,
            "fresh_samples": result.fresh_samples,
            "required_samples": ACTUATOR_HAND_RECOVERY_SAMPLES,
        }
        _status_nonblocking(status_queue, "hand_state_recovery_progress", payload)
    if result.recovered:
        payload = {
            "context": context,
            "pause_s": result.pause_s,
            "fresh_samples": result.fresh_samples,
            "required_samples": ACTUATOR_HAND_RECOVERY_SAMPLES,
        }
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


def _enforce_arm_tracking(
    backend: _G1Dex3CommandBackend,
    state: RobotState,
    *,
    context: str,
) -> None:
    """Reject motion away from the fixed outgoing arm target."""

    _, arm_joint, arm_name, arm_delta = _largest_named_value(
        (("arm", state.arm - backend._arm_target, ARM_JOINT_NAMES),)
    )
    if abs(arm_delta) > MAX_ARM_TRACKING_ERROR_RAD:
        raise DeploymentError(
            f"Arm tracking error during {context} at joint {arm_joint} ({arm_name}) is "
            f"{arm_delta:+.3f} rad (measured minus target); "
            f"MAX_ARM_TRACKING_ERROR_RAD={MAX_ARM_TRACKING_ERROR_RAD:.3f} rad"
        )


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
    _enforce_arm_tracking(backend, state, context=context)
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
                hand_group, hand_joint, hand_name, hand_delta = _largest_named_value(
                    (
                        (
                            "left hand",
                            state.left_hand - backend._left_target,
                            LEFT_HAND_JOINT_NAMES,
                        ),
                        (
                            "right hand",
                            state.right_hand - backend._right_target,
                            RIGHT_HAND_JOINT_NAMES,
                        ),
                    )
                )
                arm_dq_joint = int(np.argmax(np.abs(state.arm_dq)))
                arm_dq = float(state.arm_dq[arm_dq_joint])
                failed_gates: list[str] = []
                if arm_error > INITIALIZATION_ARM_TOLERANCE_RAD:
                    failed_gates.append(
                        f"arm error={arm_error:.3f} rad at joint {arm_joint} ({arm_name}), "
                        f"measured minus target={arm_delta:+.3f} rad > "
                        f"INITIALIZATION_ARM_TOLERANCE_RAD="
                        f"{INITIALIZATION_ARM_TOLERANCE_RAD:.3f} rad"
                    )
                if hand_error > INITIALIZATION_HAND_TOLERANCE_RAD:
                    failed_gates.append(
                        f"hand error={hand_error:.3f} rad at {hand_group} joint {hand_joint} "
                        f"({hand_name}), measured minus target={hand_delta:+.3f} rad > "
                        f"INITIALIZATION_HAND_TOLERANCE_RAD="
                        f"{INITIALIZATION_HAND_TOLERANCE_RAD:.3f} rad"
                    )
                if not backend.simulation and abs(arm_dq) > INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S:
                    failed_gates.append(
                        f"max arm dq={abs(arm_dq):.3f} rad/s at joint {arm_dq_joint} "
                        f"({ARM_JOINT_NAMES[arm_dq_joint]}), signed dq={arm_dq:+.3f} rad/s > "
                        f"INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S="
                        f"{INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S:.3f} rad/s"
                    )
                if not failed_gates:
                    dwell_elapsed = 0.0 if converged_since is None else max(0.0, now - converged_since)
                    failed_gates.append(
                        "instantaneous position/velocity gates passed but the stability dwell was incomplete: "
                        f"dwell={dwell_elapsed:.3f}s / INITIALIZATION_CONVERGENCE_DWELL_S="
                        f"{INITIALIZATION_CONVERGENCE_DWELL_S:.3f}s, "
                        f"distinct samples={distinct_converged_samples} / "
                        f"INITIALIZATION_MIN_DISTINCT_SAMPLES={INITIALIZATION_MIN_DISTINCT_SAMPLES}"
                    )
                raise DeploymentError(
                    "Initialization target did not converge before "
                    f"INITIALIZATION_CONVERGENCE_TIMEOUT_S="
                    f"{INITIALIZATION_CONVERGENCE_TIMEOUT_S:.3f}s; failed gate(s): "
                    + "; ".join(failed_gates)
                    + "; observed: "
                    f"arm error={arm_error:.3f} rad at joint {arm_joint} ({arm_name}), "
                    f"measured minus target={arm_delta:+.3f} rad; "
                    f"max arm dq={abs(arm_dq):.3f} rad/s at joint {arm_dq_joint} "
                    f"({ARM_JOINT_NAMES[arm_dq_joint]}); "
                    f"hand error={hand_error:.3f} rad at {hand_group} joint {hand_joint} "
                    f"({hand_name}), measured minus target={hand_delta:+.3f} rad"
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


def _set_powered_hold_target(
    backend: _G1Dex3CommandBackend,
    conditioner: XrPolicyOutputConditioner | None,
    hand_watchdog: HandTrackingWatchdog | None,
    state: RobotState,
) -> None:
    """Stop the arms at measured q without relaxing the current hand grip.

    A loaded Dex3 finger normally trails its commanded position.  Capturing
    measured hand q as the HOLD target removes that position error and can
    loosen a grasp.  Preserve the exact last outgoing hand targets instead;
    only the arm targets are replaced with measured q to stop arm motion.
    """

    _set_direct_target(
        backend,
        conditioner,
        hand_watchdog,
        state.arm,
        backend._left_target,
        backend._right_target,
    )


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
    except DeploymentError:
        # The final-command conditioner owns the hard DDS slew bound. Avoid
        # synchronous terminal/file logging inside the 100 Hz actuator loop;
        # the parent preflight and active timing artifact retain diagnostics.
        pass


def _ramp_real_arm_authority(
    backend: _G1Dex3CommandBackend,
    stop_event: Any,
    heartbeat: Any,
) -> bool:
    """Acquire ``arm_sdk`` at measured q, then ramp only gravity feed-forward.

    Dex3 absolute targets do not need to be rewritten for every arm authority
    step.  Writing each hand at every step made the nominal 1.5-second ramp
    perform 450 serial DDS writes and allowed DDS latency to push arming past
    the parent's timeout.  First publish the measured arm hold at full weight
    and zero feed-forward torque, then send the measured hand hold once and
    ramp only the arm gravity term.  The
    initial arm write also establishes the cleanup invariant before any hand
    write: a subsequent cancellation/fault must run arm release and Dex3
    ``stopMotors``.  The first arm packet uses weight one, the captured measured
    pose and zero feed-forward torque.  This is a bumpless takeover whether the
    robot applied a previous process's release or retained its old arm_sdk
    weight.  The captured pose remains fixed while gravity feed-forward rises
    from zero to its normal value; XR-zero remains a separate, subsequently
    confirmed bounded initialization path.  Missed 100 Hz ticks are skipped
    instead of accumulating an extra sleep after slow writes.

    ``False`` is an orderly cancellation requested through ``stop_event``;
    heartbeat expiry remains a fault with a distinct diagnostic.
    """

    if stop_event.is_set():
        LOGGER.info("Arm takeover cancelled before the first DDS write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired before arm takeover")

    # Reapply the existing pre-arm contract immediately before the first
    # full-authority write. The robot can move, or feedback can age, after the
    # dwell's final sample; a failed final guard must produce no command.
    backend._validate_prearm_takeover_state(backend.state())
    if stop_event.is_set():
        LOGGER.info("Arm takeover cancelled after the final state guard")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired after the final arm takeover state guard")

    backend.set_weight(1.0)
    authority_started = time.monotonic()
    initial_arm_write_started = authority_started
    backend._publish_arm(
        gravity_scale=0.0,
        require_prearm_takeover_state=True,
    )
    initial_arm_write_s = time.monotonic() - initial_arm_write_started
    if stop_event.is_set():
        LOGGER.info("Arm takeover cancelled after the matched-pose zero-torque write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired after matched-pose zero-torque write")

    hand_write_started = time.monotonic()
    backend._publish_hands()
    hand_write_s = time.monotonic() - hand_write_started
    if stop_event.is_set():
        LOGGER.info("Arm takeover cancelled after the measured hand hold write")
        return False
    if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
        raise DeploymentError("Parent heartbeat expired after measured hand hold write")

    period = 1.0 / PUBLISH_HZ
    duration = max(float(ARM_GRAVITY_RAMP_S), period)
    ramp_started = time.monotonic()
    next_write_at = ramp_started
    arm_writes = 0
    skipped_ticks = 0
    max_arm_cycle_s = 0.0

    while True:
        if stop_event.is_set():
            LOGGER.info("Arm gravity ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm gravity feed-forward")

        now = time.monotonic()
        if now < next_write_at and stop_event.wait(next_write_at - now):
            LOGGER.info("Arm gravity ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm gravity feed-forward")

        cycle_started = time.monotonic()
        # Validate fresh/mode/velocity state on every takeover cycle, but keep
        # q fixed at the measured pose captured before the first Write.  Chasing
        # measured q would hide motion instead of holding the handoff point.
        state = backend.state()
        _enforce_arm_tracking(backend, state, context="matched-pose gravity ramp")
        gravity_scale = min(1.0, (cycle_started - ramp_started + period) / duration)
        backend._publish_arm(gravity_scale=gravity_scale)
        cycle_completed = time.monotonic()
        arm_writes += 1
        max_arm_cycle_s = max(max_arm_cycle_s, cycle_completed - cycle_started)

        # Cleanup requests are not heartbeat faults.  Check the stop event
        # first, including after a potentially blocking DDS Write.
        if stop_event.is_set():
            LOGGER.info("Arm gravity ramp cancelled after %d arm writes", arm_writes)
            return False
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while ramping arm gravity feed-forward")
        if gravity_scale >= 1.0:
            break

        next_write_at += period
        if next_write_at <= cycle_completed:
            missed = int((cycle_completed - next_write_at) // period) + 1
            skipped_ticks += missed
            next_write_at += missed * period

    # Do not report the actuator as armed merely because the torque blend
    # finished.  Continue holding the captured pose at full gravity torque and
    # require fresh, low-velocity state to settle before the parent can offer
    # the XR-zero transition.  This check occurs after the zero-torque phase,
    # so its tight stationary limits do not turn expected initial PD sag into
    # an immediate fault.
    ramp_completed_at = time.monotonic()
    settle_started = ramp_completed_at
    settle_deadline = settle_started + ARM_TAKEOVER_SETTLE_TIMEOUT_S
    stationary_since: float | None = None
    stationary_reference: np.ndarray | None = None
    distinct_samples = 0
    last_arm_received_at = float("-inf")
    last_arm_error = float("inf")
    last_arm_dq = float("inf")
    while not stop_event.is_set() and time.monotonic() < settle_deadline:
        loop_started = time.monotonic()
        if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
            raise DeploymentError("Parent heartbeat expired while verifying arm takeover")
        state = backend.state()
        _enforce_arm_tracking(backend, state, context="matched-pose takeover settle")
        now = time.monotonic()
        arm_received_at = (
            state.captured_at if state.arm_received_at is None else state.arm_received_at
        )
        if now - arm_received_at > ACTUATOR_ARM_STATE_MAX_AGE_S:
            raise DeploymentError("Arm state is not fresh enough to verify arm takeover")
        last_arm_error = float(np.max(np.abs(state.arm - backend._arm_target)))
        last_arm_dq = float(np.max(np.abs(state.arm_dq)))
        stationary = (
            last_arm_error <= INITIALIZATION_ARM_TOLERANCE_RAD
            and last_arm_dq <= INITIALIZATION_MAX_FINAL_ARM_DQ_RAD_S
        )
        if stationary:
            if stationary_since is None:
                stationary_since = now
                stationary_reference = state.arm.copy()
                distinct_samples = 0
                last_arm_received_at = float("-inf")
            assert stationary_reference is not None
            if (
                float(np.max(np.abs(state.arm - stationary_reference)))
                > INITIALIZATION_MAX_POSITION_DRIFT_RAD
            ):
                stationary_since = now
                stationary_reference = state.arm.copy()
                distinct_samples = 0
                last_arm_received_at = float("-inf")
            if arm_received_at > last_arm_received_at:
                distinct_samples += 1
                last_arm_received_at = arm_received_at
            if (
                now - stationary_since >= ARM_TAKEOVER_SETTLE_DWELL_S
                and distinct_samples >= INITIALIZATION_MIN_DISTINCT_SAMPLES
            ):
                break
        else:
            stationary_since = None
            stationary_reference = None
            distinct_samples = 0
            last_arm_received_at = float("-inf")
        backend._publish_arm(gravity_scale=1.0)
        elapsed = time.monotonic() - loop_started
        stop_event.wait(max(0.0, period - elapsed))
    else:
        if stop_event.is_set():
            LOGGER.info("Arm takeover cancelled during the stationary verification dwell")
            return False
        raise DeploymentError(
            "Arm did not settle after matched-pose takeover: "
            f"arm error={last_arm_error:.3f} rad, max arm dq={last_arm_dq:.3f} rad/s, "
            f"required dwell={ARM_TAKEOVER_SETTLE_DWELL_S:.3f}s"
        )

    completed_at = time.monotonic()
    LOGGER.info(
        "Arm authority acquisition completed in %.3fs: gravity ramp=%.3fs, "
        "stationary verification=%.3fs, "
        "matched-pose zero-torque arm write=%.3fs, hand hold write=%.3fs, arm writes=%d, "
        "skipped 100 Hz ticks=%d, max arm cycle=%.3fs",
        completed_at - authority_started,
        ramp_completed_at - ramp_started,
        completed_at - settle_started,
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
    dds_hold_diagnostics: bool = False,
    run_log_dir: str | None = None,
    gravity_feedforward: bool = True,
    hand_pause_generation: Any | None = None,
) -> None:
    if run_log_dir is not None:
        configure_process_logging(Path(run_log_dir) / "actuator.log")
    else:
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
    if not isinstance(gravity_feedforward, bool):
        _status(status_queue, "fault", "gravity_feedforward must be a bool")
        _status(status_queue, "stopped")
        return

    last_sequence = 0
    tracking_checks_after = float("inf")
    conditioner: XrPolicyOutputConditioner | None = None
    # The spawned actuator process must never perform terminal/file I/O from
    # its 100 Hz loop. Parent-owned statuses and the active timing ring retain
    # the same diagnostics without blocking command publication.
    hand_watchdog = HandTrackingWatchdog(emit_logs=False)
    hand_freshness_gate = HandStateFreshnessGate()
    dds_hold_timing: DdsHoldTimingAccumulator | None = None
    active_timing: ActiveTimingRing | None = None
    active_timing_snapshot: dict[str, Any] | None = None
    active_timing_capture_error: str | None = None
    active_timing_trigger: str | None = None
    active_timing_error: str | None = None
    active_fault_trigger: str | None = None
    timing_dump_errors: list[str] = []
    timing_dump_dropped = 0
    # A Python writer thread shares this process's GIL and can perturb the
    # 100 Hz actuator loop while normalizing/encoding a full timing ring.
    # Rotate rings in O(1), retain a bounded number, and persist them only
    # after command authority has been released.
    deferred_timing_dumps: list[
        tuple[
            ActiveTimingRing,
            str,
            str | None,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
    ] = []
    try:
        if gravity_feedforward:
            # Preserve the original two-argument construction shape for older
            # internal backend adapters; its API default is feed-forward on.
            backend = _G1Dex3CommandBackend(simulation, network_interface)
        else:
            backend = _G1Dex3CommandBackend(
                simulation,
                network_interface,
                gravity_feedforward=False,
            )
        backend._authority_ramp_timing_enabled = authority_ramp_diagnostics or dds_hold_diagnostics
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
        backend.prepare_measured_hold()
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
        if dds_hold_diagnostics:
            dds_hold_timing = DdsHoldTimingAccumulator(period_s=period)
            _status(status_queue, "dds_hold_timing_started")

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

            state_lookup_started = time.monotonic_ns()
            state = backend.state()
            state_lookup_ms = (time.monotonic_ns() - state_lookup_started) / 1e6
            now = time.monotonic()
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                state,
                status_queue,
                context="pre-initialization hold",
            )
            if dds_hold_timing is not None:
                dds_hold_timing.note_freshness(freshness)
            if freshness.ready:
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
            try:
                backend.publish()
            finally:
                if dds_hold_timing is not None:
                    dds_hold_timing.record(
                        loop_started=loop_started,
                        state_lookup_ms=state_lookup_ms,
                        state=state,
                        publish_timing_ms=backend._last_publish_timing_ms,
                        completed_at=time.monotonic(),
                    )
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
        last_discontinued_sequence: int | None = None
        pending_replan_detail: dict[str, Any] | None = None
        replan_freeze_active = False
        replan_last_loop_started: float | None = None
        replan_stable_samples = 0
        urgent_hold_active = False
        pending_hand_replan_detail: dict[str, Any] | None = None
        active_timing = ActiveTimingRing()
        # Per-DDS-stage monotonic timers are enabled only after initialization,
        # where the active policy loop needs them. They add no file I/O to the
        # 100 Hz path; records remain in the bounded in-memory ring.
        backend._authority_ramp_timing_enabled = True

        def publish_active() -> None:
            publish_started_ns = time.monotonic_ns()
            try:
                backend.publish()
            finally:
                assert active_timing is not None
                timing_ms = dict(getattr(backend, "_last_publish_timing_ms", {}))
                timing_ms["publish_total"] = (time.monotonic_ns() - publish_started_ns) / 1e6
                active_timing.note_publish(timing_ms)

        def wait_active(loop_started: float) -> None:
            assert active_timing is not None
            elapsed = time.monotonic() - loop_started
            active_timing.wait(stop_event, requested_s=max(0.0, period - elapsed))

        while not stop_event.is_set():
            loop_started = time.monotonic()
            replan_loop_gap_s = (
                None
                if replan_last_loop_started is None
                else loop_started - replan_last_loop_started
            )
            if pending_replan_detail is not None:
                replan_last_loop_started = loop_started
            active_timing.begin(
                loop_started=loop_started,
                heartbeat_age_s=None,
                sequence=chunk_sequence,
                action_index=chunk_index,
                chunk_length=None if chunk is None else chunk.length,
                rtc=rtc_mode,
                rtc_total_actions=rtc_total_actions,
                rtc_action_budget=rtc_action_budget,
                holding=holding,
            )
            heartbeat_started_ns = time.monotonic_ns()
            heartbeat_age_s = _heartbeat_age(heartbeat)
            active_timing.update(
                heartbeat_age_ms=max(0.0, heartbeat_age_s) * 1e3,
                heartbeat_lookup_ms=(time.monotonic_ns() - heartbeat_started_ns) / 1e6,
            )
            if heartbeat_age_s > HEARTBEAT_TIMEOUT_S:
                raise DeploymentError("Parent heartbeat expired")

            # Sample freshness before reading a queued plan or advancing the
            # 30 Hz scheduler.  A short Dex3 gap therefore cannot consume an
            # action that was never written.
            state_lookup_started = time.monotonic_ns()
            try:
                state = backend.state()
            finally:
                state_lookup_ms = (time.monotonic_ns() - state_lookup_started) / 1e6
                active_timing.update(state_lookup_ms=state_lookup_ms)
            active_timing.note_state(
                state,
                completed_at=time.monotonic(),
                lookup_ms=state_lookup_ms,
            )
            freshness = _observe_hand_freshness(
                hand_freshness_gate,
                state,
                status_queue,
                context=(
                    f"active sequence={chunk_sequence} next_action_index={chunk_index} "
                    f"rtc={rtc_mode} holding={holding}"
                ),
            )
            active_timing.note_freshness(freshness)
            if freshness.entered:
                pause_generation = _advance_hand_pause_generation(hand_pause_generation)
                had_motion = chunk is not None
                if had_motion:
                    pending_hand_replan_detail = {
                        "reason": "hand_state_soft_stale",
                        "sequence": int(chunk_sequence),
                        "action_index": int(chunk_index),
                        "chunk_length": int(chunk.length),
                        "discarded_actions": int(chunk.length - chunk_index),
                        "rtc": bool(rtc_mode),
                        "rtc_total_actions": int(rtc_total_actions),
                        "rtc_action_budget": int(rtc_action_budget),
                        "feedback_age_s": float(freshness.max_age_s),
                        "pause_age_s": ACTUATOR_HAND_STATE_PAUSE_AGE_S,
                        "operator_hold_age_s": ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
                        "hand_pause_generation": pause_generation,
                    }
                    last_discontinued_sequence = int(chunk_sequence)
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
                holding = not had_motion
                replan_freeze_active = had_motion

            if not freshness.ready:
                if freshness.operator_hold_entered and (
                    pending_hand_replan_detail is not None or pending_replan_detail is not None
                ):
                    terminal_detail = dict(
                        pending_hand_replan_detail
                        if pending_hand_replan_detail is not None
                        else pending_replan_detail
                    )
                    terminal_detail.update(
                        reason="hand_state_operator_hold",
                        feedback_age_s=float(freshness.max_age_s),
                        operator_hold_age_s=ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
                    )
                    pending_hand_replan_detail = None
                    pending_replan_detail = None
                    replan_freeze_active = False
                    replan_last_loop_started = None
                    replan_stable_samples = 0
                    holding = True
                    if bool(terminal_detail.get("rtc", False)):
                        _status(status_queue, "rtc_rejected", terminal_detail)
                    else:
                        _status(status_queue, "holding", int(terminal_detail["sequence"]))
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
                            # Operator STOP supersedes an automatic scheduler
                            # replan even while Dex3 feedback is paused.  Do
                            # not let the old replan surface after recovery or
                            # leave the child logically outside HOLD.
                            pending_replan_detail = None
                            pending_hand_replan_detail = None
                            replan_freeze_active = False
                            replan_last_loop_started = None
                            replan_stable_samples = 0
                            holding = True
                            urgent_hold_event.clear()
                            urgent_hold_active = False
                            _status(status_queue, "urgent_holding", last_sequence)
                            break
                publish_active()
                wait_active(loop_started)
                continue

            if freshness.recovered:
                # Arm feedback remained under its hard 100 ms gate.  Capture
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
                if pending_hand_replan_detail is not None:
                    pending_hand_replan_detail.update(
                        reason="hand_state_recovered",
                        pause_s=float(freshness.pause_s),
                        recovery_samples=int(freshness.fresh_samples),
                    )
                    pending_replan_detail = pending_hand_replan_detail
                    pending_hand_replan_detail = None
                    replan_freeze_active = True
                    holding = False
                    replan_last_loop_started = loop_started
                    replan_stable_samples = 0
                publish_active()
                wait_active(loop_started)
                continue

            if pending_replan_detail is not None:
                if (
                    replan_loop_gap_s is not None
                    and replan_loop_gap_s <= ACTION_REPLAN_RECOVERY_MAX_LOOP_GAP_S
                ):
                    replan_stable_samples += 1
                else:
                    replan_stable_samples = 0
                if replan_stable_samples >= ACTION_REPLAN_RECOVERY_SAMPLES:
                    pending_replan_detail.setdefault("recovery_samples", replan_stable_samples)
                    pending_replan_detail["publisher_recovery_samples"] = replan_stable_samples
                    if _status_nonblocking(
                        status_queue,
                        "replan_required",
                        pending_replan_detail,
                    ):
                        pending_replan_detail = None
                        replan_last_loop_started = None
                        replan_stable_samples = 0
                if pending_replan_detail is not None:
                    # Do not accept or advance any policy work until the
                    # publisher loop has demonstrated stable timing and the
                    # parent has been notified that a fresh request is valid.
                    publish_active()
                    wait_active(loop_started)
                    continue

            queue_get_started_ns = time.monotonic_ns()
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            finally:
                active_timing.update(
                    command_queue_get_ms=(time.monotonic_ns() - queue_get_started_ns) / 1e6,
                )
            if command is not None:
                kind = command[0] if isinstance(command, tuple) and command else None
                active_timing.start_command(kind)
                if kind == "urgent_hold_barrier":
                    if command != ("urgent_hold_barrier",):
                        raise DeploymentError("Malformed urgent STOP barrier")
                    # The barrier is queued only after every concurrently
                    # submitted motion command. Reaching it proves that no
                    # pre-STOP plan remains hidden in multiprocessing.Queue's
                    # feeder thread. Capture measured arm q while preserving
                    # the last hand grip targets, then acknowledge STOP.
                    state = backend.state()
                    _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    replan_freeze_active = False
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
                    _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    replan_freeze_active = False
                    _status(status_queue, "holding", last_sequence)
                    continue

                if kind == "warmup_pose":
                    if not isinstance(command, tuple) or len(command) != 3:
                        raise DeploymentError("Malformed guarded warmup-pose command")
                    if chunk is not None or not holding:
                        raise DeploymentError(
                            "Guarded warmup pose requires an acknowledged hold with no active chunk"
                        )
                    _, created_at, warmup_pose = command
                    if not isinstance(warmup_pose, InitializationSpec):
                        raise DeploymentError("Malformed guarded warmup-pose target")
                    try:
                        command_age = time.monotonic() - float(created_at)
                    except (TypeError, ValueError) as exc:
                        raise DeploymentError("Guarded warmup-pose timestamp is invalid") from exc
                    if not np.isfinite(command_age) or not 0.0 <= command_age <= INITIALIZATION_COMMAND_MAX_AGE_S:
                        raise DeploymentError("Guarded warmup-pose command expired before execution")

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
                        _status(status_queue, "warmup_pose_cancelled", warmup_pose.label)
                        return
                    command_start = RobotState(
                        captured_at=state.captured_at,
                        mode_machine=state.mode_machine,
                        arm=backend._arm_target.copy(),
                        arm_dq=state.arm_dq.copy(),
                        left_hand=backend._left_target.copy(),
                        right_hand=backend._right_target.copy(),
                    )
                    warmup_pose_chunk = build_initialization_chunk(command_start, warmup_pose)
                    _status(
                        status_queue,
                        "warmup_pose_started",
                        {
                            "label": warmup_pose.label,
                            "steps": warmup_pose_chunk.length,
                            "duration_s": warmup_pose_chunk.length / PUBLISH_HZ,
                        },
                    )
                    completed = _execute_initialization(
                        backend,
                        warmup_pose_chunk,
                        stop_event,
                        heartbeat,
                        tracking_checks_after,
                        hand_watchdog,
                        hand_freshness_gate,
                        status_queue,
                        context=f"guarded warmup pose={warmup_pose.label}",
                    )
                    if not completed:
                        _status(status_queue, "warmup_pose_cancelled", warmup_pose.label)
                        return
                    if conditioner is not None:
                        conditioner.reset(backend._arm_target, backend._left_target, backend._right_target)
                    hand_watchdog.reset(backend)
                    tracking_checks_after = time.monotonic()
                    holding = True
                    _status(status_queue, "warmup_pose_completed", warmup_pose.label)
                    continue

                if kind == "warm_start":
                    if not isinstance(command, tuple) or len(command) not in {3, 4}:
                        raise DeploymentError("Malformed policy warm-start command")
                    if chunk is not None or not holding:
                        raise DeploymentError("Policy warm-start requires an acknowledged hold with no active chunk")
                    if len(command) == 4:
                        _, created_at, expected_pause_generation, warm_start = command
                    else:
                        _, created_at, warm_start = command
                        expected_pause_generation = None
                    if not isinstance(warm_start, InitializationSpec):
                        raise DeploymentError("Malformed policy warm-start target")
                    if not _policy_generation_matches(
                        expected_pause_generation,
                        hand_pause_generation,
                    ):
                        _status(
                            status_queue,
                            "warm_start_invalidated",
                            {
                                "expected_generation": expected_pause_generation,
                                "current_generation": _hand_pause_generation_value(
                                    hand_pause_generation
                                ),
                            },
                        )
                        continue
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
                        if expected_sequence == last_discontinued_sequence:
                            _status_nonblocking(
                                status_queue,
                                "rtc_obsolete",
                                {
                                    "operation": "snapshot",
                                    "sequence": expected_sequence,
                                },
                            )
                            continue
                        _status(status_queue, "rtc_rejected", "snapshot has no matching active plan")
                        continue
                    if chunk_index >= chunk.length:
                        state = backend.state()
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                    if not isinstance(command, tuple) or len(command) not in {8, 9}:
                        raise DeploymentError("Malformed RTC start command")
                    if len(command) == 9:
                        (
                            _,
                            sequence,
                            created_at,
                            expected_pause_generation,
                            action_budget,
                            arm,
                            left,
                            right,
                            expected_horizon,
                        ) = command
                    else:
                        _, sequence, created_at, action_budget, arm, left, right, expected_horizon = command
                        expected_pause_generation = None
                    if chunk is not None or rtc_mode:
                        raise DeploymentError("RTC can start only with no active plan")
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != last_sequence + 1:
                        raise DeploymentError(f"Stale/out-of-order RTC plan {sequence!r}; expected {last_sequence + 1}")
                    if isinstance(action_budget, bool) or not isinstance(action_budget, int) or action_budget < 1:
                        raise DeploymentError("RTC action budget must be a positive integer")
                    if not _policy_generation_matches(
                        expected_pause_generation,
                        hand_pause_generation,
                    ):
                        last_sequence = int(sequence)
                        last_discontinued_sequence = int(sequence)
                        _status(
                            status_queue,
                            "replan_required",
                            {
                                "reason": "hand_pause_generation_changed",
                                "sequence": int(sequence),
                                "action_index": 0,
                                "chunk_length": int(expected_horizon),
                                "discarded_actions": int(expected_horizon),
                                "rtc": True,
                                "rtc_total_actions": 0,
                                "rtc_action_budget": int(action_budget),
                                "plan_installed": False,
                                "expected_generation": expected_pause_generation,
                                "current_generation": _hand_pause_generation_value(
                                    hand_pause_generation
                                ),
                            },
                        )
                        continue
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
                        backend._arm_target if replan_freeze_active else state.arm,
                        backend._left_target if replan_freeze_active else state.left_hand,
                        backend._right_target if replan_freeze_active else state.right_hand,
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
                    replan_freeze_active = False
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
                    if not rtc_mode and expected_sequence == last_discontinued_sequence:
                        _status_nonblocking(
                            status_queue,
                            "rtc_obsolete",
                            {
                                "operation": "replace",
                                "sequence": expected_sequence,
                            },
                        )
                        continue
                    if rtc_mode and rtc_total_actions >= rtc_action_budget:
                        state = backend.state()
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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

                elif not isinstance(command, tuple) or len(command) not in {6, 7} or kind != "chunk":
                    raise DeploymentError(f"Unexpected or malformed actuator command {kind!r}")
                elif chunk is not None:
                    raise DeploymentError("Received a new chunk before the prior chunk completed")
                else:
                    if len(command) == 7:
                        _, sequence, created_at, expected_pause_generation, arm, left, right = command
                    else:
                        _, sequence, created_at, arm, left, right = command
                        expected_pause_generation = None
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != last_sequence + 1:
                        raise DeploymentError(
                            f"Stale/out-of-order action chunk {sequence!r}; expected {last_sequence + 1}"
                        )
                    if not _policy_generation_matches(
                        expected_pause_generation,
                        hand_pause_generation,
                    ):
                        discarded_actions = int(np.asarray(arm).shape[0])
                        last_sequence = int(sequence)
                        last_discontinued_sequence = int(sequence)
                        _status(
                            status_queue,
                            "replan_required",
                            {
                                "reason": "hand_pause_generation_changed",
                                "sequence": int(sequence),
                                "action_index": 0,
                                "chunk_length": discarded_actions,
                                "discarded_actions": discarded_actions,
                                "rtc": False,
                                "plan_installed": False,
                                "expected_generation": expected_pause_generation,
                                "current_generation": _hand_pause_generation_value(
                                    hand_pause_generation
                                ),
                            },
                        )
                        continue
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
                        backend._arm_target if replan_freeze_active else state.arm,
                        backend._left_target if replan_freeze_active else state.left_hand,
                        backend._right_target if replan_freeze_active else state.right_hand,
                        conditioner,
                        context="Synchronous plan",
                    )
                    chunk = proposed
                    chunk_sequence = int(sequence)
                    chunk_index = 0
                    next_action_at = time.monotonic()
                    last_sequence = sequence
                    holding = False
                    replan_freeze_active = False

                active_timing.finish_command()

            # This is deliberately checked after accepting any queued plan, but
            # before advancing its next 30 Hz target.  An operator stop can
            # therefore cancel synchronous and RTC motion without racing an
            # unconsumed plan command in the queue.  The event is independent
            # of the normal command queue so it cannot be delayed by a full
            # queue.
            if urgent_hold_event.is_set():
                if not urgent_hold_active or chunk is not None or rtc_mode or not holding:
                    state = backend.state()
                    _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
                    chunk = None
                    chunk_index = 0
                    rtc_mode = False
                    tracking_checks_after = time.monotonic()
                    holding = True
                    replan_freeze_active = False
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
                publish_active()
                wait_active(loop_started)
                continue

            now = time.monotonic()
            if chunk is not None and now >= next_action_at:
                scheduler_lateness_s = now - next_action_at
                scheduler_replan_threshold_s = min(
                    MAX_ACTION_LATENESS_S,
                    ACTION_REPLAN_LATENESS_S,
                )
                active_timing.update(
                    scheduler_due=True,
                    scheduler_lateness_ms=scheduler_lateness_s * 1e3,
                    sequence=chunk_sequence,
                    next_action_index=chunk_index,
                    chunk_length=chunk.length,
                    rtc=rtc_mode,
                    rtc_total_actions=rtc_total_actions,
                    rtc_action_budget=rtc_action_budget,
                    holding=holding,
                )
                if chunk_index < chunk.length:
                    if scheduler_lateness_s >= scheduler_replan_threshold_s:
                        replan_detail = {
                            "sequence": int(chunk_sequence),
                            "action_index": int(chunk_index),
                            "chunk_length": int(chunk.length),
                            "discarded_actions": int(chunk.length - chunk_index),
                            "lateness_s": float(scheduler_lateness_s),
                            "threshold_s": float(scheduler_replan_threshold_s),
                            "rtc": bool(rtc_mode),
                            "rtc_total_actions": int(rtc_total_actions),
                            "rtc_action_budget": int(rtc_action_budget),
                        }
                        active_timing.update(event="action_scheduler_replan")
                        active_timing.finish(completed_at=time.monotonic())
                        replan_timing = active_timing
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
                        last_discontinued_sequence = int(chunk_sequence)
                        chunk = None
                        chunk_index = 0
                        next_action_at = 0.0
                        rtc_mode = False
                        rtc_total_actions = 0
                        rtc_action_budget = 0
                        # This is a transparent replan-idle state, not the
                        # operator's measured-pose HOLD.  Continue publishing
                        # the exact last command until a fresh plan arrives.
                        holding = False
                        replan_freeze_active = True
                        pending_replan_detail = replan_detail
                        replan_last_loop_started = loop_started
                        replan_stable_samples = 0
                        if run_log_dir is not None:
                            if len(deferred_timing_dumps) < 4:
                                deferred_timing_dumps.append(
                                    (
                                        replan_timing,
                                        "action_scheduler_replan",
                                        None,
                                        (
                                            frozen_arm.copy(),
                                            frozen_left.copy(),
                                            frozen_right.copy(),
                                        ),
                                    )
                                )
                            else:
                                timing_dump_dropped += 1
                        active_timing = ActiveTimingRing()
                        publish_active()
                        wait_active(loop_started)
                        continue
                    if scheduler_lateness_s * 1e3 > ACTIVE_TIMING_SLOW_LATENESS_MS[0]:
                        active_timing.update(event="action_scheduler_slow")
                    raw_arm = chunk.arm[chunk_index]
                    raw_left = chunk.left_hand[chunk_index]
                    raw_right = chunk.right_hand[chunk_index]
                    raw_arm_step_rad = float(np.max(np.abs(raw_arm - backend._arm_target)))
                    raw_left_step_rad = float(np.max(np.abs(raw_left - backend._left_target)))
                    raw_right_step_rad = float(np.max(np.abs(raw_right - backend._right_target)))
                    active_timing.update(
                        raw_arm_step_rad=raw_arm_step_rad,
                        raw_left_step_rad=raw_left_step_rad,
                        raw_right_step_rad=raw_right_step_rad,
                    )
                    if conditioner is None:
                        backend.set_target(
                            raw_arm,
                            raw_left,
                            raw_right,
                        )
                        active_timing.update(
                            conditioned_arm_step_rad=raw_arm_step_rad,
                            conditioned_left_step_rad=raw_left_step_rad,
                            conditioned_right_step_rad=raw_right_step_rad,
                        )
                    else:
                        conditioner.set_desired(
                            raw_arm,
                            raw_left,
                            raw_right,
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
                        _set_powered_hold_target(backend, conditioner, hand_watchdog, state)
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
                tracking_started = time.monotonic_ns()
                try:
                    active_timing.update(
                        tracking_arm_error_rad=float(np.max(np.abs(state.arm - backend._arm_target))),
                        tracking_left_error_rad=float(np.max(np.abs(state.left_hand - backend._left_target))),
                        tracking_right_error_rad=float(np.max(np.abs(state.right_hand - backend._right_target))),
                    )
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
                finally:
                    active_timing.update(tracking_ms=(time.monotonic_ns() - tracking_started) / 1e6)

            if conditioner is not None and not holding and not replan_freeze_active:
                conditioner_started = time.monotonic_ns()
                previous_arm = backend._arm_target.copy()
                previous_left = backend._left_target.copy()
                previous_right = backend._right_target.copy()
                try:
                    conditioned = conditioner.next_command(
                        state.arm,
                        backend._arm_target,
                        backend._left_target,
                        backend._right_target,
                        now=now,
                    )
                finally:
                    active_timing.update(conditioner_ms=(time.monotonic_ns() - conditioner_started) / 1e6)
                backend.set_target(
                    conditioned.arm[0],
                    conditioned.left_hand[0],
                    conditioned.right_hand[0],
                )
                active_timing.update(
                    conditioned_arm_step_rad=float(np.max(np.abs(conditioned.arm[0] - previous_arm))),
                    conditioned_left_step_rad=float(np.max(np.abs(conditioned.left_hand[0] - previous_left))),
                    conditioned_right_step_rad=float(np.max(np.abs(conditioned.right_hand[0] - previous_right))),
                )

            publish_active()
            wait_active(loop_started)
    except BaseException as exc:
        if active_timing is not None:
            active_timing_trigger = active_fault_trigger or "active_fault"
            active_timing_error = f"{type(exc).__name__}: {exc}"
            try:
                active_timing.fault(exc, event=active_timing_trigger)
            except BaseException as diagnostic_exc:
                active_timing_capture_error = (
                    "Could not freeze active timing after actuator fault: "
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
        _status(status_queue, "fault", f"{type(exc).__name__}: {exc}")
    finally:
        if active_timing is not None and active_timing_trigger is None:
            active_timing_trigger = "orderly_shutdown"
            try:
                active_timing.finish(completed_at=time.monotonic())
            except BaseException as diagnostic_exc:
                active_timing_capture_error = (
                    "Could not freeze active timing during orderly shutdown: "
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
        dds_hold_completed_at = time.monotonic() if dds_hold_timing is not None else None
        cleanup_reporting_errors: list[str] = []
        local_release_complete = backend is None
        local_close_complete = backend is None

        def cleanup_status(kind: str, payload: Any = None) -> None:
            try:
                _status(status_queue, kind, payload)
            except BaseException as status_exc:
                cleanup_reporting_errors.append(
                    f"status {kind!r} failed: {type(status_exc).__name__}: {status_exc}"
                )

        if backend is not None:
            def cleanup_phase(event: str, payload: dict[str, Any]) -> None:
                cleanup_status("cleanup_phase", {"event": event, **payload})

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
            except BaseException as exc:
                cleanup_status("release_failed", f"{type(exc).__name__}: {exc}")
            else:
                local_release_complete = True
                cleanup_status("release_complete")
            cleanup_phase("backend_close_begin", {})
            try:
                backend.close()
            except BaseException as exc:
                cleanup_status("close_failed", f"{type(exc).__name__}: {exc}")
            else:
                local_close_complete = True
                cleanup_phase("backend_close_complete", {})
        else:
            cleanup_status("release_complete", "no backend was constructed")

        # Everything below is post-release diagnostics. No logging or status
        # serialization above is allowed to bypass backend release/close.
        if run_log_dir is not None and local_release_complete and local_close_complete:
            for ring, trigger, error, targets in deferred_timing_dumps:
                try:
                    payload = ring.snapshot(
                        trigger=trigger,
                        error=error,
                        backend=None,
                        command_conditioning=command_conditioning,
                        target_snapshot=targets,
                    )
                    path = diagnostic_json_path(run_log_dir, trigger)
                    write_json(path, payload)
                    LOGGER.warning("ACTIVE_TIMING_DUMP=%s", path)
                except BaseException as diagnostic_exc:
                    timing_dump_errors.append(
                        "Deferred active timing dump failed after release: "
                        f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                    )
        elif deferred_timing_dumps:
            timing_dump_errors.append(
                "Skipped deferred active timing dumps because local release/close was not confirmed"
            )
        if timing_dump_dropped:
            timing_dump_errors.append(
                f"Dropped {timing_dump_dropped} active timing dump request(s) because the writer queue was full"
            )
        dds_hold_timing_summary: dict[str, Any] | None = None
        if dds_hold_timing is not None:
            try:
                dds_hold_timing_summary = dds_hold_timing.summary(completed_at=dds_hold_completed_at)
            except BaseException as diagnostic_exc:
                message = (
                    "Could not summarize DDS HOLD timing: "
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
                active_timing_capture_error = (
                    message
                    if active_timing_capture_error is None
                    else f"{active_timing_capture_error}; {message}"
                )
        if dds_hold_timing_summary is not None:
            cleanup_status("dds_hold_timing", dds_hold_timing_summary)
        try:
            if active_timing is not None and active_timing_trigger is not None:
                active_timing_snapshot = active_timing.snapshot(
                    trigger=active_timing_trigger,
                    error=active_timing_error,
                    backend=backend,
                    command_conditioning=command_conditioning,
                )
            if active_timing_snapshot is not None:
                if run_log_dir is None:
                    LOGGER.warning(
                        "ACTIVE_TIMING_DUMP unavailable because no run log directory was configured; "
                        "trigger=%s retained=%s total=%s",
                        active_timing_snapshot["trigger"],
                        active_timing_snapshot["retained_records"],
                        active_timing_snapshot["total_records"],
                    )
                else:
                    timing_path = diagnostic_json_path(run_log_dir, str(active_timing_snapshot["trigger"]))
                    write_json(timing_path, active_timing_snapshot)
                    LOGGER.warning("ACTIVE_TIMING_DUMP=%s", timing_path)
            if active_timing_capture_error is not None:
                LOGGER.error("%s", active_timing_capture_error)
            for timing_error in timing_dump_errors:
                LOGGER.error("%s", timing_error)
            for reporting_error in cleanup_reporting_errors:
                LOGGER.error("Cleanup reporting error after release: %s", reporting_error)
        except BaseException as diagnostic_exc:
            # Release/close has already completed. Diagnostic persistence must
            # never suppress the terminal stopped acknowledgment.
            try:
                LOGGER.error(
                    "Post-release diagnostic persistence failed: %s: %s",
                    type(diagnostic_exc).__name__,
                    diagnostic_exc,
                )
            except BaseException:
                pass
        finally:
            _status(status_queue, "stopped")


class SafeG1Dex3Actuator:
    """Parent-side handle for the watchdog-owning actuator process."""

    def __init__(
        self,
        simulation: bool,
        network_interface: str | None,
        command_conditioning: str = "none",
        authority_ramp_diagnostics: bool = False,
        dds_hold_diagnostics: bool = False,
        run_log_dir: str | None = None,
        gravity_feedforward: bool = True,
    ):
        if command_conditioning not in COMMAND_CONDITIONING_MODES:
            raise DeploymentError(f"Unknown command conditioning mode {command_conditioning!r}")
        if not isinstance(gravity_feedforward, bool):
            raise DeploymentError("gravity_feedforward must be a bool")
        context = mp.get_context("spawn")
        self._command_queue = context.Queue(maxsize=1)
        self._status_queue = context.Queue(maxsize=32)
        self._stop_event = context.Event()
        self._urgent_hold_event = context.Event()
        self._hand_pause_generation = context.Value("Q", 0)
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
        if (
            authority_ramp_diagnostics
            or dds_hold_diagnostics
            or run_log_dir is not None
            or not gravity_feedforward
        ):
            process_args += (
                authority_ramp_diagnostics,
                dds_hold_diagnostics,
                run_log_dir,
                gravity_feedforward,
            )
        self._process = context.Process(
            target=_actuator_main,
            args=process_args,
            kwargs={"hand_pause_generation": self._hand_pause_generation},
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
        self._rtc_fenced_through_sequence = 0
        self._last_replan_detail: dict[str, Any] | None = None
        self._command_conditioning = command_conditioning
        self._gravity_feedforward = gravity_feedforward
        self._authority_ramp_diagnostics = authority_ramp_diagnostics
        self._last_authority_ramp_timing: dict[str, Any] | None = None
        self._dds_hold_diagnostics = dds_hold_diagnostics
        self._last_dds_hold_timing: dict[str, Any] | None = None
        self._control_lock = threading.Lock()
        self._immediate_hold_requested = threading.Event()
        self._immediate_release_requested = threading.Event()
        self._stopped_acknowledged = False
        self._hand_state_paused = False
        self._hand_operator_hold_pending = False
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
            LOGGER.warning(
                "DDS drop — waiting for reconnection",
                extra={"terminal_yellow": True},
            )
            return True
        if kind == "hand_state_recovery_progress":
            self._last_hand_state_event = value
            LOGGER.warning(
                "Dex3 feedback recovery safety reading %s/%s: %s",
                value.get("fresh_samples", "?") if isinstance(value, dict) else "?",
                value.get("required_samples", "?") if isinstance(value, dict) else "?",
                value,
            )
            return True
        if kind == "hand_state_operator_hold":
            pending_sequence = getattr(self, "_pending_sequence", None)
            if isinstance(pending_sequence, int) and not isinstance(pending_sequence, bool):
                self._rtc_fenced_through_sequence = max(
                    int(getattr(self, "_rtc_fenced_through_sequence", 0)),
                    pending_sequence,
                )
            # The child has already discarded the time-indexed plan and is
            # publishing a powered HOLD by the time this reliable status is
            # observed. Mirror that terminal transition atomically in the
            # parent. Otherwise a later goal can be rejected as though the
            # abandoned plan were still active.
            self._hand_state_paused = True
            self._hand_operator_hold_pending = True
            self._last_hand_state_event = value
            self._holding = True
            self._warm_started = False
            self._chunk_in_flight = False
            self._pending_sequence = None
            self._rtc_active = False
            self._rtc_terminal = None
            LOGGER.warning(
                "Dex3 feedback remained stale through the operator-HOLD deadline: %s",
                value,
            )
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
        if kind == "dds_hold_timing":
            self._last_dds_hold_timing = value
            return True
        if kind == "dds_hold_timing_started":
            return True
        if kind == "rtc_obsolete":
            LOGGER.info("Discarded obsolete RTC control message after scheduler replan: %s", value)
            return True
        return False

    def _rtc_terminal_status_is_fenced(self, kind: str, value: Any) -> bool:
        """Return whether a delayed RTC terminal belongs to an abandoned plan."""

        if kind not in {"rtc_completed", "rtc_underrun", "rtc_rejected"}:
            return False
        if not isinstance(value, dict):
            return False
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            return False
        fenced_through = int(getattr(self, "_rtc_fenced_through_sequence", 0))
        if sequence > fenced_through:
            return False
        LOGGER.info(
            "Discarded delayed %s for fenced RTC sequence %d (fenced through %d)",
            kind,
            sequence,
            fenced_through,
        )
        return True

    def _accept_rtc_terminal_status(self, kind: str, value: Any) -> tuple[str, Any] | None:
        """Reconcile one RTC terminal status without touching a newer plan."""

        if self._rtc_terminal_status_is_fenced(kind, value):
            return None
        sequence = None
        if isinstance(value, dict):
            candidate = value.get("sequence")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                sequence = candidate
        target_sequence = (
            self._pending_sequence
            if getattr(self, "_rtc_active", False) and getattr(self, "_chunk_in_flight", False)
            else None
        )
        if sequence is not None:
            if target_sequence is None:
                if sequence <= self._sequence:
                    LOGGER.info(
                        "Discarded delayed %s for inactive RTC sequence %d",
                        kind,
                        sequence,
                    )
                    return None
                raise DeploymentError(
                    f"Unexpected {kind} sequence {sequence}; no RTC plan is active"
                )
            if sequence < target_sequence:
                LOGGER.info(
                    "Discarded delayed %s for RTC sequence %d; active sequence is %d",
                    kind,
                    sequence,
                    target_sequence,
                )
                return None
            if sequence > target_sequence:
                raise DeploymentError(
                    f"{kind} sequence {sequence} does not match active RTC sequence "
                    f"{target_sequence}"
                )
        outcome = "complete" if kind == "rtc_completed" else "hold"
        event = (outcome, value)
        self._rtc_terminal = event
        self._chunk_in_flight = False
        self._pending_sequence = None
        self._rtc_active = False
        self._holding = True
        if outcome == "hold":
            self._warm_started = False
        return event

    def _accept_replan_status(
        self,
        value: Any,
        *,
        expected_sequence: int | None = None,
    ) -> dict[str, Any] | None:
        """Accept one sequence-qualified scheduler replan terminal event."""

        if not isinstance(value, dict):
            raise DeploymentError("Actuator returned malformed scheduler replan detail")
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise DeploymentError("Actuator scheduler replan has no valid sequence")
        target_sequence = self._pending_sequence if expected_sequence is None else expected_sequence
        if target_sequence is None:
            if sequence <= self._sequence:
                # A duplicated terminal status from an already fenced plan may
                # arrive after its replacement started.  Never clear the newer
                # generation.
                return None
            raise DeploymentError(
                f"Unexpected scheduler replan sequence {sequence}; no plan is pending"
            )
        if sequence != target_sequence:
            if sequence < target_sequence:
                return None
            raise DeploymentError(
                f"Scheduler replan sequence {sequence} does not match pending {target_sequence}"
            )
        self._last_replan_detail = dict(value)
        self._chunk_in_flight = False
        self._pending_sequence = None
        self._rtc_active = False
        self._rtc_terminal = None
        # The child is publishing the last successfully commanded target, but
        # this is not the operator's measured-pose HOLD state.
        self._holding = not bool(value.get("plan_installed", True))
        return self._last_replan_detail

    @property
    def last_replan_detail(self) -> dict[str, Any] | None:
        return None if self._last_replan_detail is None else dict(self._last_replan_detail)

    @property
    def last_dds_hold_timing(self) -> dict[str, Any] | None:
        return self._last_dds_hold_timing

    @property
    def hand_state_paused(self) -> bool:
        return bool(self._hand_state_paused)

    @property
    def hand_pause_generation(self) -> int:
        return _hand_pause_generation_value(getattr(self, "_hand_pause_generation", None))

    def acknowledge_hand_operator_hold(self) -> bool:
        """Acknowledge the terminal transition before an explicit new goal."""

        pending = bool(getattr(self, "_hand_operator_hold_pending", False))
        self._hand_operator_hold_pending = False
        if pending:
            # Keep acknowledgment idempotently aligned with the child HOLD.
            # It grants permission to prepare a fresh goal; it never revives
            # the abandoned sequence.
            self._holding = True
            self._warm_started = False
            self._chunk_in_flight = False
            self._pending_sequence = None
            self._rtc_active = False
            self._rtc_terminal = None
        return pending

    def wait_for_hand_feedback(self) -> None:
        """Block policy observation/motion until the recovery sample gate passes."""

        self._wait_for_hand_feedback()

    def _wait_for_hand_feedback(self) -> None:
        """Do not enqueue new motion while the child is in a soft hand pause."""

        self.assert_healthy()
        if getattr(self, "_hand_operator_hold_pending", False):
            raise HandFeedbackOperatorHold(self._last_hand_state_event)
        if not getattr(self, "_hand_state_paused", False):
            return
        self._wait_status(
            "hand_state_recovered",
            timeout_s=ACTUATOR_HAND_STATE_MAX_AGE_S + 0.25,
        )

    def _wait_status(self, expected: str, timeout_s: float, payload: Any = None) -> Any:
        deadline = time.monotonic() + timeout_s
        if getattr(self, "_hand_state_paused", False):
            deadline += ACTUATOR_HAND_RECOVERY_WAIT_GRACE_S
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
            if kind == "replan_required":
                detail = self._accept_replan_status(value)
                if detail is not None:
                    raise RtcTerminalEvent("replan", detail)
                continue
            if kind in {"hand_state_pause", "hand_state_recovered"}:
                self._record_auxiliary_status(kind, value)
                if kind == "hand_state_pause":
                    deadline += ACTUATOR_HAND_RECOVERY_WAIT_GRACE_S
                if kind == expected and (payload is None or value == payload):
                    return value
                continue
            if self._record_auxiliary_status(kind, value):
                if kind == "hand_state_operator_hold":
                    raise HandFeedbackOperatorHold(value)
                continue
            if kind == "authority_ramp_timing":
                self._last_authority_ramp_timing = value
                LOGGER.warning("ACTUATOR_TIMING %s", _format_authority_ramp_timing(value))
                continue
            if kind == "release_failed":
                raise DeploymentError(f"Actuator release failed: {value}")
            if kind == "warm_start_invalidated":
                raise HandFeedbackReplan(value)
            if kind == "close_failed":
                raise DeploymentError(f"Actuator resource cleanup failed: {value}")
            if kind == "stopped":
                self._stopped_acknowledged = True
                if self._immediate_release_requested.is_set():
                    raise ImmediateControlEvent("release")
                raise DeploymentError("Actuator stopped before the requested operation completed")
            if kind in {"rtc_completed", "rtc_underrun", "rtc_rejected"} and expected != "urgent_holding":
                event = self._accept_rtc_terminal_status(kind, value)
                if event is not None:
                    raise RtcTerminalEvent(*event)
                continue
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
            timeout_s = ARM_GRAVITY_RAMP_S + ARM_TAKEOVER_SETTLE_TIMEOUT_S + 2.0
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
                elif kind in {"rtc_completed", "rtc_underrun", "rtc_rejected"}:
                    self._accept_rtc_terminal_status(kind, value)
                elif kind == "replan_required":
                    detail = self._accept_replan_status(value)
                    if detail is not None:
                        self._rtc_terminal = ("replan", detail)
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

    def warmup_pose(self, spec: InitializationSpec) -> None:
        """Move to one guarded non-policy pose and remain in powered HOLD."""

        if not self._initialized or not self._holding or self._chunk_in_flight:
            raise DeploymentError(
                "Guarded warmup pose requires initialized HOLD with no chunk in flight"
            )
        validate_initialization_spec(spec)
        self._wait_for_hand_feedback()
        self.heartbeat()
        try:
            self._command_queue.put(("warmup_pose", time.monotonic(), spec), timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing guarded warmup pose") from exc
        self._wait_status(
            "warmup_pose_completed",
            timeout_s=(
                INITIALIZATION_START_TIMEOUT_S
                + INITIALIZATION_MAX_DURATION_S
                + INITIALIZATION_CONVERGENCE_TIMEOUT_S
                + 3.0
            ),
            payload=spec.label,
        )
        self._holding = True

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
        pause_generation = (
            self.hand_pause_generation
            if chunk.hand_pause_generation is None
            else int(chunk.hand_pause_generation)
        )
        self.heartbeat()
        try:
            self._command_queue.put(
                ("warm_start", time.monotonic(), pause_generation, target),
                timeout=0.2,
            )
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
        pause_generation = (
            self.hand_pause_generation
            if chunk.hand_pause_generation is None
            else int(chunk.hand_pause_generation)
        )
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
                pause_generation,
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
        pause_generation = (
            self.hand_pause_generation
            if plan.hand_pause_generation is None
            else int(plan.hand_pause_generation)
        )
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
                pause_generation,
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
        self._sequence = next_sequence
        self._chunk_in_flight = True
        self._pending_sequence = next_sequence
        self._rtc_active = True
        self._rtc_terminal = None
        self._holding = False
        try:
            self._wait_status("rtc_started", timeout_s=1.0, payload=next_sequence)
        except RtcTerminalEvent as exc:
            if exc.outcome == "replan":
                raise HandFeedbackReplan(exc.detail) from exc
            raise
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
                    if kind == "hand_state_operator_hold":
                        event = ("hold", value)
                    continue
                if kind == "fault":
                    raise DeploymentError(f"Actuator fault: {value}")
                if kind == "replan_required":
                    detail = self._accept_replan_status(value)
                    if detail is not None:
                        event = ("replan", detail)
                    continue
                if kind == "release_failed":
                    raise DeploymentError(f"Actuator release failed: {value}")
                if kind == "close_failed":
                    raise DeploymentError(f"Actuator resource cleanup failed: {value}")
                if kind == "stopped":
                    self._stopped_acknowledged = True
                    if self._immediate_release_requested.is_set():
                        raise ImmediateControlEvent("release")
                    raise DeploymentError("Actuator stopped during RTC")
                if kind in {"rtc_completed", "rtc_underrun", "rtc_rejected"}:
                    accepted = self._accept_rtc_terminal_status(kind, value)
                    if accepted is not None:
                        event = accepted
        except queue.Empty:
            pass
        if event is not None:
            # poll_rtc_event() is the consumer of this terminal event. Do not
            # leave the helper's cache armed and return the same event twice.
            self._rtc_terminal = None
            self._chunk_in_flight = False
            self._pending_sequence = None
            self._rtc_active = False
            self._holding = event[0] != "replan"
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
        if getattr(self, "_hand_state_paused", False):
            deadline += ACTUATOR_HAND_RECOVERY_WAIT_GRACE_S
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
                if kind == "hand_state_operator_hold":
                    return "hold"
                if kind == "hand_state_pause":
                    deadline += ACTUATOR_HAND_RECOVERY_WAIT_GRACE_S
                continue
            if kind == "fault":
                raise DeploymentError(f"Actuator fault: {value}")
            if kind == "replan_required":
                detail = self._accept_replan_status(value, expected_sequence=sequence)
                if detail is not None:
                    return "replan"
                continue
            if kind == "warm_start_invalidated":
                raise HandFeedbackReplan(value)
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
                # The 1.25 s feedback boundary enters operator HOLD
                # immediately. A later explicit goal remains independently
                # blocked until the three-reading recovery gate completes.
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
                event = self._accept_rtc_terminal_status(kind, value)
                if event is not None:
                    raise RtcTerminalEvent(*event)
                continue
            if kind == "rtc_completed":
                event = self._accept_rtc_terminal_status(kind, value)
                if event is not None:
                    raise RtcTerminalEvent(*event)
                continue
        raise TimeoutError(f"Timed out waiting for action chunk {sequence}")

    def hold(self) -> None:
        """Hold measured arm q and the last commanded grip under watchdog control."""

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
