"""Guarded Unitree G1 Inspire DFX state and command adapters.

The official DFX bridge publishes both hands in one ``MotorStates_`` message:
right motors at wire indices 0..5 and left motors at 6..11.  DDS callback
freshness alone is insufficient because the bridge republishes cached positions
when an internal hand read fails.  Its per-motor ``lost`` counters are therefore
used to advance each hand's accepted timestamp independently.

The command adapter is intentionally small: it validates both canonical hands,
maps them to DFX's right-first wire order, and performs one combined DDS write.
It never writes during construction and has no synthetic motor-stop command.
"""

from __future__ import annotations

import contextlib
import operator
import threading
import time
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.g1_end_effectors import (
    UNQUALIFIED_INSPIRE_DFX_COMMAND_MAX_STEP,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ARM_DOF, validate_measured_state
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import RobotState, STATE_MAX_AGE_S


DFX_MOTORS_PER_HAND = 6
DFX_TOTAL_MOTORS = 12
DFX_RIGHT_SLICE = slice(0, 6)
DFX_LEFT_SLICE = slice(6, 12)
UINT32_MAX = (1 << 32) - 1
# Compatibility alias for the first shadow-client API. The canonical name
# above makes explicit that 0.2 is teleop-derived, not a manufacturer limit.
INSPIRE_DFX_COMMAND_MAX_STEP = UNQUALIFIED_INSPIRE_DFX_COMMAND_MAX_STEP
INSPIRE_DFX_WRITE_TIMEOUT_S = 0.5


def _motor_sequence(message: Any) -> Any:
    try:
        # Official unitree_go::msg::dds_::MotorStates_ names this sequence
        # ``states`` (not LowState_/HandState_'s ``motor_state``).
        motors = message.states
        if len(motors) != DFX_TOTAL_MOTORS:
            raise ValueError(f"expected {DFX_TOTAL_MOTORS} motors, got {len(motors)}")
        return motors
    except (AttributeError, TypeError) as exc:
        raise ValueError("message has no sized states sequence") from exc


def _side_values(motors: Any, indices: slice) -> tuple[np.ndarray, tuple[int, ...]]:
    q_values: list[float] = []
    lost_values: list[int] = []
    for motor in motors[indices]:
        try:
            q = float(motor.q)
            raw_lost = motor.lost
            if isinstance(raw_lost, bool):
                raise TypeError("boolean lost counter")
            lost = operator.index(raw_lost)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("motor q/lost fields are malformed") from exc
        if not np.isfinite(q) or not 0 <= lost <= UINT32_MAX:
            raise ValueError("motor q is non-finite or lost is outside uint32")
        q_values.append(q)
        lost_values.append(lost)
    return np.asarray(q_values, dtype=np.float64), tuple(lost_values)


def _command_values(values: Any, *, side: str) -> np.ndarray:
    """Return one exact normalized six-axis command without coercing strings/bools."""

    raw = np.asarray(values)
    if raw.shape != (DFX_MOTORS_PER_HAND,) or raw.dtype.kind not in "iuf":
        raise DeploymentError(
            f"Inspire DFX {side} command must be a numeric ({DFX_MOTORS_PER_HAND},) vector"
        )
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise DeploymentError(f"Inspire DFX {side} command contains NaN or infinity")
    bad = np.flatnonzero((result < 0.0) | (result > 1.0))
    if bad.size:
        joint = int(bad[0])
        raise DeploymentError(
            f"Inspire DFX {side} command joint {joint} is {result[joint]:.6f}; "
            "normalized_open_fraction must remain inside [0, 1]"
        )
    return result


class InspireDfxCommandWriter:
    """One atomic combined DFX writer with a teleop-derived 0.2 backstop."""

    def __init__(self, initial_left: Any, initial_right: Any) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        self._last_left = _command_values(initial_left, side="left").copy()
        self._last_right = _command_values(initial_right, side="right").copy()
        publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        try:
            publisher.Init()
            message = MotorCmds_()
            message.cmds = [
                unitree_go_msg_dds__MotorCmd_() for _ in range(DFX_TOTAL_MOTORS)
            ]
        except BaseException:
            with contextlib.suppress(Exception):
                publisher.Close()
            raise
        self._publisher = publisher
        self._message = message
        self._closed = False
        self._has_written = False

    @property
    def has_written(self) -> bool:
        """Whether this writer has successfully acquired the DFX command lease."""

        return self._has_written

    def reseed_before_first_write(self, left: Any, right: Any) -> None:
        """Move the step-check origin to the final measured pre-arm hold."""

        if self._closed:
            raise DeploymentError("Inspire DFX command writer is closed")
        if self._has_written:
            raise DeploymentError("Inspire DFX command writer cannot be reseeded after a write")
        candidate_left = _command_values(left, side="left")
        candidate_right = _command_values(right, side="right")
        self._last_left = candidate_left.copy()
        self._last_right = candidate_right.copy()

    def write(self, left: Any, right: Any) -> tuple[float, np.ndarray, np.ndarray]:
        """Publish both hands once and return the shared completion timestamp."""

        if self._closed:
            raise DeploymentError("Inspire DFX command writer is closed")
        candidate_left = _command_values(left, side="left")
        candidate_right = _command_values(right, side="right")
        for side, candidate, previous in (
            ("left", candidate_left, self._last_left),
            ("right", candidate_right, self._last_right),
        ):
            bad = np.flatnonzero(np.abs(candidate - previous) > INSPIRE_DFX_COMMAND_MAX_STEP + 1e-12)
            if bad.size:
                joint = int(bad[0])
                raise DeploymentError(
                    f"Inspire DFX {side} command step at joint {joint} is "
                    f"{candidate[joint] - previous[joint]:+.6f}; "
                    f"INSPIRE_DFX_COMMAND_MAX_STEP={INSPIRE_DFX_COMMAND_MAX_STEP:.3f}"
                )

        # Allocate every array needed by our post-Write state transition before
        # touching DDS. Once Write returns True, only non-allocating reference
        # assignments mark the physically accepted command as authoritative.
        cached_left = candidate_left.copy()
        cached_right = candidate_right.copy()
        returned_left = candidate_left.copy()
        returned_right = candidate_right.copy()

        # DFX wire order is right 0..5 followed by left 6..11. Both candidates
        # have already passed every check before the reusable message changes.
        for index, value in enumerate(candidate_right):
            self._message.cmds[index].q = float(value)
        for offset, value in enumerate(candidate_left, start=DFX_MOTORS_PER_HAND):
            self._message.cmds[offset].q = float(value)
        write_ok = self._publisher.Write(self._message, timeout=INSPIRE_DFX_WRITE_TIMEOUT_S)
        if write_ok is not True:
            raise DeploymentError(
                "Inspire DFX combined DDS Write failed; "
                f"INSPIRE_DFX_WRITE_TIMEOUT_S={INSPIRE_DFX_WRITE_TIMEOUT_S:.3f}s"
            )
        self._last_left = cached_left
        self._last_right = cached_right
        self._has_written = True
        completed_at = time.monotonic()
        return completed_at, returned_left, returned_right

    def refresh_last_successful(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Refresh only the command that DDS most recently accepted.

        Release uses this instead of the backend's current target. If a newer
        policy target failed before its hand Write, cleanup must not turn that
        failed target into a new movement while authority is being released.
        """

        if self._closed:
            raise DeploymentError("Inspire DFX command writer is closed")
        if not self._has_written:
            raise DeploymentError("Inspire DFX command lease has not been acquired")
        returned_left = self._last_left.copy()
        returned_right = self._last_right.copy()
        for index, value in enumerate(self._last_right):
            self._message.cmds[index].q = float(value)
        for offset, value in enumerate(self._last_left, start=DFX_MOTORS_PER_HAND):
            self._message.cmds[offset].q = float(value)
        write_ok = self._publisher.Write(self._message, timeout=INSPIRE_DFX_WRITE_TIMEOUT_S)
        if write_ok is not True:
            raise DeploymentError(
                "Inspire DFX lease-refresh DDS Write failed; "
                f"INSPIRE_DFX_WRITE_TIMEOUT_S={INSPIRE_DFX_WRITE_TIMEOUT_S:.3f}s"
            )
        completed_at = time.monotonic()
        return completed_at, returned_left, returned_right

    def close(self) -> None:
        if self._closed:
            return
        self._publisher.Close()
        self._closed = True


class G1InspireDfxStateReader:
    """Subscribe to G1 arm and combined Inspire DFX state without publishers."""

    end_effector = "inspire-dfx"

    def __init__(
        self,
        simulation: bool = False,
        max_age_s: float = STATE_MAX_AGE_S,
        hand_max_age_s: float | None = None,
        *,
        max_age_constant: str = "STATE_MAX_AGE_S",
        hand_max_age_constant: str | None = None,
    ):
        if simulation:
            raise DeploymentError("Inspire DFX simulation is not qualified in this guarded client")
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self._arm_max_age_s = float(max_age_s)
        self._hand_max_age_s = float(max_age_s if hand_max_age_s is None else hand_max_age_s)
        if self._arm_max_age_s <= 0.0 or self._hand_max_age_s <= 0.0:
            raise ValueError("State freshness limits must be positive")
        self._arm_max_age_constant = max_age_constant
        self._hand_max_age_constant = max_age_constant if hand_max_age_constant is None else hand_max_age_constant
        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        if len(self._arm_indices) != ARM_DOF:
            raise DeploymentError(f"Expected {ARM_DOF} G1 arm indices, got {len(self._arm_indices)}")
        self._lock = threading.Lock()
        self._arm_message: Any | None = None
        self._arm_updated_at = 0.0
        self._hand_q: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._hand_updated_at = {"left": 0.0, "right": 0.0}
        self._lost_baseline: dict[str, tuple[int, ...] | None] = {"left": None, "right": None}
        self._diagnostics = {
            "callbacks": 0,
            "malformed_messages": 0,
            "left": {"baselines": 0, "accepted": 0, "drop_events": 0, "lost_increments": 0, "resets": 0},
            "right": {"baselines": 0, "accepted": 0, "drop_events": 0, "lost_increments": 0, "resets": 0},
        }
        self._subscribers: dict[str, Any] = {}
        try:
            self._subscribers["arm"] = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscribers["hands"] = ChannelSubscriber(
                "rt/inspire/state", MotorStates_
            )
            self._subscribers["arm"].Init(handler=self._on_arm)
            self._subscribers["hands"].Init(handler=self._on_hands)
        except BaseException:
            for subscriber in self._subscribers.values():
                with contextlib.suppress(Exception):
                    subscriber.Close()
            raise

    def _on_arm(self, message: Any) -> None:
        if message is None:
            return
        with self._lock:
            self._arm_message = message
            self._arm_updated_at = time.monotonic()

    def _on_hands(self, message: Any) -> None:
        if message is None:
            return
        received_at = time.monotonic()
        try:
            motors = _motor_sequence(message)
            decoded = {
                "right": _side_values(motors, DFX_RIGHT_SLICE),
                "left": _side_values(motors, DFX_LEFT_SLICE),
            }
        except ValueError:
            with self._lock:
                self._diagnostics["callbacks"] += 1
                self._diagnostics["malformed_messages"] += 1
            return

        with self._lock:
            self._diagnostics["callbacks"] += 1
            for side, (q, lost) in decoded.items():
                side_diagnostics = self._diagnostics[side]
                # The DFX service increments all six counters together for a
                # failed side read.  Divergence means the state is not one
                # coherent side sample; require a new baseline and clean pair.
                if len(set(lost)) != 1:
                    self._lost_baseline[side] = None
                    self._hand_q[side] = None
                    self._hand_updated_at[side] = 0.0
                    side_diagnostics["resets"] += 1
                    continue
                baseline = self._lost_baseline[side]
                if baseline is None:
                    self._lost_baseline[side] = lost
                    side_diagnostics["baselines"] += 1
                    continue
                if any(current < previous for current, previous in zip(lost, baseline, strict=True)):
                    self._lost_baseline[side] = None
                    self._hand_q[side] = None
                    self._hand_updated_at[side] = 0.0
                    side_diagnostics["resets"] += 1
                    continue
                deltas = tuple(current - previous for current, previous in zip(lost, baseline, strict=True))
                self._lost_baseline[side] = lost
                if any(deltas):
                    side_diagnostics["drop_events"] += 1
                    # All six counters are copies of one failed side-read
                    # count; record the common/max delta, not six times it.
                    side_diagnostics["lost_increments"] += max(deltas)
                    continue
                self._hand_q[side] = q
                self._hand_updated_at[side] = received_at
                side_diagnostics["accepted"] += 1

    def latest(self) -> RobotState:
        now = time.monotonic()
        with self._lock:
            arm_message = self._arm_message
            arm_updated_at = self._arm_updated_at
            hand_q = {side: None if values is None else values.copy() for side, values in self._hand_q.items()}
            hand_updated_at = dict(self._hand_updated_at)
            lost = dict(self._lost_baseline)
            diagnostics = {
                "malformed_messages": self._diagnostics["malformed_messages"],
                "left": dict(self._diagnostics["left"]),
                "right": dict(self._diagnostics["right"]),
            }
        stale: list[str] = []
        if arm_message is None:
            stale.append(f"arm (missing; {self._arm_max_age_constant}={self._arm_max_age_s:.3f}s)")
        elif now - arm_updated_at > self._arm_max_age_s:
            stale.append(
                f"arm (age {now - arm_updated_at:.3f}s > {self._arm_max_age_s:.3f}s; "
                f"{self._arm_max_age_constant}={self._arm_max_age_s:.3f}s)"
            )
        for side in ("left", "right"):
            if hand_q[side] is None:
                stale.append(
                    f"{side} (no two consecutive clean DFX samples; "
                    f"drops={diagnostics[side]['drop_events']}, "
                    f"malformed={diagnostics['malformed_messages']})"
                )
            elif now - hand_updated_at[side] > self._hand_max_age_s:
                stale.append(
                    f"{side} (accepted age {now - hand_updated_at[side]:.3f}s > "
                    f"{self._hand_max_age_s:.3f}s; drops={diagnostics[side]['drop_events']}, "
                    f"lost_increments={diagnostics[side]['lost_increments']})"
                )
        if stale:
            raise TimeoutError(f"Stale Unitree/Inspire DFX state: {', '.join(stale)}")

        assert arm_message is not None and hand_q["left"] is not None and hand_q["right"] is not None
        arm = np.asarray([arm_message.motor_state[index].q for index in self._arm_indices], dtype=np.float64)
        arm_dq = np.asarray([arm_message.motor_state[index].dq for index in self._arm_indices], dtype=np.float64)
        validate_measured_state(
            arm,
            arm_dq,
            hand_q["left"],
            hand_q["right"],
            end_effector="inspire-dfx",
        )
        return RobotState(
            captured_at=min(arm_updated_at, hand_updated_at["left"], hand_updated_at["right"]),
            mode_machine=int(getattr(arm_message, "mode_machine", 0)),
            arm=arm,
            arm_dq=arm_dq,
            left_hand=hand_q["left"],
            right_hand=hand_q["right"],
            left_hand_received_at=hand_updated_at["left"],
            right_hand_received_at=hand_updated_at["right"],
            arm_received_at=arm_updated_at,
            left_hand_lost=lost["left"],
            right_hand_lost=lost["right"],
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
        raise TimeoutError(f"Timed out waiting for fresh Unitree/Inspire DFX state ({last_error})")

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "callbacks": int(self._diagnostics["callbacks"]),
                "malformed_messages": int(self._diagnostics["malformed_messages"]),
                "left": dict(self._diagnostics["left"]),
                "right": dict(self._diagnostics["right"]),
                "left_lost": self._lost_baseline["left"],
                "right_lost": self._lost_baseline["right"],
            }

    def close(self) -> None:
        for subscriber in self._subscribers.values():
            with contextlib.suppress(Exception):
                subscriber.Close()
