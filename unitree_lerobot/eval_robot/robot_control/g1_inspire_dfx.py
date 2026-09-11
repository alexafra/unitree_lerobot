"""Read-only Unitree G1 Inspire DFX state adapter.

The official DFX bridge publishes both hands in one ``MotorStates_`` message:
right motors at wire indices 0..5 and left motors at 6..11.  DDS callback
freshness alone is insufficient because the bridge republishes cached positions
when an internal hand read fails.  Its per-motor ``lost`` counters are therefore
used to advance each hand's accepted timestamp independently.

This module intentionally contains no command publisher.  Guarded Inspire DFX
actuation is not qualified yet.
"""

from __future__ import annotations

import contextlib
import operator
import threading
import time
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ARM_DOF, validate_measured_state
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import RobotState, STATE_MAX_AGE_S


DFX_MOTORS_PER_HAND = 6
DFX_TOTAL_MOTORS = 12
DFX_RIGHT_SLICE = slice(0, 6)
DFX_LEFT_SLICE = slice(6, 12)
UINT32_MAX = (1 << 32) - 1


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
        self._subscribers = {
            "arm": ChannelSubscriber("rt/lowstate", LowState_),
            "hands": ChannelSubscriber("rt/inspire/state", MotorStates_),
        }
        self._subscribers["arm"].Init(handler=self._on_arm)
        self._subscribers["hands"].Init(handler=self._on_hands)

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
