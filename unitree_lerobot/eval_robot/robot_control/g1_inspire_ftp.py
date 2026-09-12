"""Guarded Unitree G1 Inspire RH56E2/FTP state and command adapters.

This is the transport used by ``xr_teleoperate --ee inspire_ftp``.  Each hand
has its own DDS state and command topic.  The vendor wire representation is six
dimensionless angle codes in ``[0, 1000]``; the policy/data representation is
the same six normalized-open fractions in ``[0, 1]`` used by teleoperation.

Construction subscribes and creates publishers but never writes.  State
freshness is tracked independently for the two hands, and every command is
checked again against the teleop-derived 0.2 normalized per-write backstop.
Successful DDS ``Write`` return values are tracked as transport completions;
they are not bridge or physical-device acknowledgements.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import threading
import time
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.g1_end_effectors import (
    INSPIRE_FTP_PROFILE,
    UNQUALIFIED_INSPIRE_COMMAND_MAX_STEP,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ARM_DOF, validate_measured_state
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import RobotState, STATE_MAX_AGE_S


FTP_MOTORS_PER_HAND = 6
FTP_WIRE_SCALE = 1000
INSPIRE_FTP_COMMAND_MAX_STEP = UNQUALIFIED_INSPIRE_COMMAND_MAX_STEP
INSPIRE_FTP_WRITE_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class InspireFtpSdk:
    hand_control_type: Any
    hand_state_type: Any
    control_factory: Any
    channel_publisher: Any
    channel_subscriber: Any
    low_state_type: Any


def require_inspire_ftp_sdk() -> InspireFtpSdk:
    """Resolve every FTP runtime dependency without initializing DDS."""

    try:
        from inspire_sdkpy import inspire_dds
        from inspire_sdkpy.inspire_hand_defaut import get_inspire_hand_ctrl
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError as exc:
        raise DeploymentError(
            "Inspire FTP requires the vendor inspire_sdkpy package used by xr_teleoperate"
        ) from exc
    try:
        hand_control_type = inspire_dds.inspire_hand_ctrl
        hand_state_type = inspire_dds.inspire_hand_state
    except AttributeError as exc:
        raise DeploymentError(
            "Installed inspire_sdkpy is incompatible: inspire_hand_ctrl/state IDL types are missing"
        ) from exc
    callables = {
        "get_inspire_hand_ctrl": get_inspire_hand_ctrl,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "LowState_": LowState_,
        "inspire_hand_ctrl": hand_control_type,
        "inspire_hand_state": hand_state_type,
    }
    invalid = [name for name, value in callables.items() if not callable(value)]
    if invalid:
        raise DeploymentError(
            "Installed Inspire FTP SDK has non-callable runtime types: " + ", ".join(invalid)
        )
    try:
        control_probe = get_inspire_hand_ctrl()
    except Exception as exc:
        raise DeploymentError(
            "Installed Inspire FTP SDK could not construct a default control message"
        ) from exc
    missing_fields = [
        name for name in ("angle_set", "mode") if not hasattr(control_probe, name)
    ]
    if missing_fields:
        raise DeploymentError(
            "Installed Inspire FTP control message is missing fields: "
            + ", ".join(missing_fields)
        )
    return InspireFtpSdk(
        hand_control_type=hand_control_type,
        hand_state_type=hand_state_type,
        control_factory=get_inspire_hand_ctrl,
        channel_publisher=ChannelPublisher,
        channel_subscriber=ChannelSubscriber,
        low_state_type=LowState_,
    )


def _normalized_values(values: Any, *, side: str, source: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (FTP_MOTORS_PER_HAND,) or raw.dtype.kind not in "iuf":
        raise DeploymentError(
            f"Inspire FTP {side} {source} must be a numeric "
            f"({FTP_MOTORS_PER_HAND},) vector"
        )
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise DeploymentError(f"Inspire FTP {side} {source} contains NaN or infinity")
    bad = np.flatnonzero((result < 0.0) | (result > 1.0))
    if bad.size:
        joint = int(bad[0])
        raise DeploymentError(
            f"Inspire FTP {side} {source} joint {joint} is {result[joint]:.6f}; "
            "normalized_open_fraction must remain inside [0, 1]"
        )
    return result


def _decode_state(message: Any, *, side: str) -> np.ndarray:
    try:
        raw = np.asarray(message.angle_act)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("message has no usable angle_act sequence") from exc
    if raw.shape != (FTP_MOTORS_PER_HAND,) or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"angle_act must be a numeric ({FTP_MOTORS_PER_HAND},) vector, got {raw.shape}"
        )
    scaled = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(scaled)) or np.any(scaled < 0.0) or np.any(scaled > FTP_WIRE_SCALE):
        raise ValueError("angle_act contains a non-finite value or is outside [0, 1000]")
    return scaled / FTP_WIRE_SCALE


def _wire_values(values: np.ndarray) -> list[int]:
    # Match xr_teleoperate exactly: int(clip(normalized * 1000, 0, 1000)).
    # Inputs are non-negative, so Python's int truncation is floor.
    return [int(np.clip(value * FTP_WIRE_SCALE, 0, FTP_WIRE_SCALE)) for value in values]


@dataclass(frozen=True)
class InspireFtpWriteResult:
    left_completed_at: float
    right_completed_at: float
    left: np.ndarray
    right: np.ndarray


class InspireFtpPartialWriteError(DeploymentError):
    """The left DDS Write completed before the right-hand DDS Write failed."""

    def __init__(self, message: str, *, left_completed_at: float, left: np.ndarray) -> None:
        super().__init__(message)
        self.left_completed_at = left_completed_at
        self.left = left


class InspireFtpCommandWriter:
    """Two-topic RH56E2 writer with normalized-to-vendor scaling."""

    def __init__(self, initial_left: Any, initial_right: Any) -> None:
        sdk = require_inspire_ftp_sdk()

        self._last_left = _normalized_values(
            initial_left, side="left", source="initial command"
        ).copy()
        self._last_right = _normalized_values(
            initial_right, side="right", source="initial command"
        ).copy()
        self._left_publisher: Any | None = None
        self._right_publisher: Any | None = None
        try:
            self._left_publisher = sdk.channel_publisher(
                INSPIRE_FTP_PROFILE.left_command_topic,
                sdk.hand_control_type,
            )
            self._left_publisher.Init()
            self._right_publisher = sdk.channel_publisher(
                INSPIRE_FTP_PROFILE.right_command_topic,
                sdk.hand_control_type,
            )
            self._right_publisher.Init()
            self._left_message = sdk.control_factory()
            self._right_message = sdk.control_factory()
        except BaseException:
            self._close_publishers(suppress_errors=True)
            raise
        self._closed = False
        self._left_has_written = False
        self._right_has_written = False

    @property
    def has_written(self) -> bool:
        return self._left_has_written or self._right_has_written

    @property
    def left_has_written(self) -> bool:
        return self._left_has_written

    @property
    def right_has_written(self) -> bool:
        return self._right_has_written

    def reseed_before_first_write(self, left: Any, right: Any) -> None:
        if self._closed:
            raise DeploymentError("Inspire FTP command writer is closed")
        if self.has_written:
            raise DeploymentError("Inspire FTP command writer cannot be reseeded after a write")
        self._last_left = _normalized_values(
            left, side="left", source="command"
        ).copy()
        self._last_right = _normalized_values(
            right, side="right", source="command"
        ).copy()

    def _validated_candidates(self, left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
        candidate_left = _normalized_values(left, side="left", source="command")
        candidate_right = _normalized_values(right, side="right", source="command")
        for side, candidate, previous in (
            ("left", candidate_left, self._last_left),
            ("right", candidate_right, self._last_right),
        ):
            bad = np.flatnonzero(
                np.abs(candidate - previous) > INSPIRE_FTP_COMMAND_MAX_STEP + 1e-12
            )
            if bad.size:
                joint = int(bad[0])
                raise DeploymentError(
                    f"Inspire FTP {side} command step at joint {joint} is "
                    f"{candidate[joint] - previous[joint]:+.6f}; "
                    f"INSPIRE_FTP_COMMAND_MAX_STEP={INSPIRE_FTP_COMMAND_MAX_STEP:.3f}"
                )
        return candidate_left, candidate_right

    @staticmethod
    def _prepare_message(message: Any, values: np.ndarray) -> np.ndarray:
        wire = _wire_values(values)
        message.angle_set = wire
        message.mode = 1
        # History and the next slew origin describe the exact integer command
        # passed to the last successful DDS Write call, not an unrepresentable
        # between-code target. This is not a bridge/device acknowledgement.
        return np.asarray(wire, dtype=np.float64) / FTP_WIRE_SCALE

    @staticmethod
    def _write_side(publisher: Any, message: Any, *, side: str) -> float:
        try:
            write_ok = publisher.Write(message, timeout=INSPIRE_FTP_WRITE_TIMEOUT_S)
        except Exception as exc:
            raise DeploymentError(
                f"Inspire FTP {side} DDS Write raised {type(exc).__name__}; "
                f"INSPIRE_FTP_WRITE_TIMEOUT_S={INSPIRE_FTP_WRITE_TIMEOUT_S:.3f}s"
            ) from exc
        if write_ok is not True:
            raise DeploymentError(
                f"Inspire FTP {side} DDS Write failed; "
                f"INSPIRE_FTP_WRITE_TIMEOUT_S={INSPIRE_FTP_WRITE_TIMEOUT_S:.3f}s"
            )
        return time.monotonic()

    def write(self, left: Any, right: Any) -> InspireFtpWriteResult:
        if self._closed:
            raise DeploymentError("Inspire FTP command writer is closed")
        candidate_left, candidate_right = self._validated_candidates(left, right)
        written_left = self._prepare_message(self._left_message, candidate_left)
        written_right = self._prepare_message(self._right_message, candidate_right)

        assert self._left_publisher is not None and self._right_publisher is not None
        left_completed_at = self._write_side(
            self._left_publisher, self._left_message, side="left"
        )
        # Preserve the exact left target whose DDS Write returned True even if
        # the subsequent right-hand Write fails. Neither return value proves
        # bridge receipt or physical hand execution.
        self._last_left = written_left
        self._left_has_written = True
        try:
            right_completed_at = self._write_side(
                self._right_publisher, self._right_message, side="right"
            )
        except DeploymentError as exc:
            raise InspireFtpPartialWriteError(
                str(exc),
                left_completed_at=left_completed_at,
                left=written_left.copy(),
            ) from exc
        self._last_right = written_right
        self._right_has_written = True
        return InspireFtpWriteResult(
            left_completed_at,
            right_completed_at,
            written_left.copy(),
            written_right.copy(),
        )

    def _close_publishers(self, *, suppress_errors: bool) -> None:
        failures: list[str] = []
        for side, publisher in (
            ("right", self._right_publisher),
            ("left", self._left_publisher),
        ):
            if publisher is None:
                continue
            try:
                publisher.Close()
            except BaseException as exc:
                failures.append(f"{side}: {type(exc).__name__}: {exc}")
        self._right_publisher = None
        self._left_publisher = None
        if failures and not suppress_errors:
            raise DeploymentError("Inspire FTP publisher cleanup failed: " + "; ".join(failures))

    def close(self) -> None:
        if self._closed:
            return
        self._close_publishers(suppress_errors=False)
        self._closed = True


class G1InspireFtpStateReader:
    """Subscribe to G1 arm and independent Inspire FTP hand state topics.

    The vendor state IDL has no device-read timestamp, sequence, or lost counter.
    Hand ages therefore measure DDS callback receipt only and cannot distinguish
    a new physical read from a bridge that republishes a cached valid sample.
    """

    end_effector = "inspire-ftp"

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
            raise DeploymentError("Inspire FTP simulation is not qualified in this guarded client")
        sdk = require_inspire_ftp_sdk()
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex

        self._arm_max_age_s = float(max_age_s)
        self._hand_max_age_s = float(max_age_s if hand_max_age_s is None else hand_max_age_s)
        if self._arm_max_age_s <= 0.0 or self._hand_max_age_s <= 0.0:
            raise ValueError("State freshness limits must be positive")
        self._arm_max_age_constant = max_age_constant
        self._hand_max_age_constant = (
            max_age_constant if hand_max_age_constant is None else hand_max_age_constant
        )
        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        if len(self._arm_indices) != ARM_DOF:
            raise DeploymentError(f"Expected {ARM_DOF} G1 arm indices, got {len(self._arm_indices)}")
        self._lock = threading.Lock()
        self._arm_message: Any | None = None
        self._arm_updated_at = 0.0
        self._hand_q: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._hand_updated_at = {"left": 0.0, "right": 0.0}
        self._diagnostics = {
            "arm_callbacks": 0,
            "left": {"callbacks": 0, "accepted": 0, "malformed_messages": 0},
            "right": {"callbacks": 0, "accepted": 0, "malformed_messages": 0},
        }
        self._last_rejection: dict[str, str | None] = {"left": None, "right": None}
        self._subscribers: dict[str, Any] = {}
        try:
            self._subscribers["arm"] = sdk.channel_subscriber("rt/lowstate", sdk.low_state_type)
            self._subscribers["left"] = sdk.channel_subscriber(
                INSPIRE_FTP_PROFILE.left_state_topic,
                sdk.hand_state_type,
            )
            self._subscribers["right"] = sdk.channel_subscriber(
                INSPIRE_FTP_PROFILE.right_state_topic,
                sdk.hand_state_type,
            )
            self._subscribers["arm"].Init(handler=self._on_arm)
            self._subscribers["left"].Init(handler=lambda message: self._on_hand("left", message))
            self._subscribers["right"].Init(handler=lambda message: self._on_hand("right", message))
        except BaseException:
            self.close()
            raise

    def _on_arm(self, message: Any) -> None:
        if message is None:
            return
        with self._lock:
            self._arm_message = message
            self._arm_updated_at = time.monotonic()
            self._diagnostics["arm_callbacks"] += 1

    def _on_hand(self, side: str, message: Any) -> None:
        if message is None:
            return
        received_at = time.monotonic()
        try:
            q = _decode_state(message, side=side)
        except ValueError as exc:
            with self._lock:
                self._diagnostics[side]["callbacks"] += 1
                self._diagnostics[side]["malformed_messages"] += 1
                self._last_rejection[side] = str(exc)
            return
        with self._lock:
            self._diagnostics[side]["callbacks"] += 1
            self._diagnostics[side]["accepted"] += 1
            self._last_rejection[side] = None
            self._hand_q[side] = q
            self._hand_updated_at[side] = received_at

    def latest(self) -> RobotState:
        now = time.monotonic()
        with self._lock:
            arm_message = self._arm_message
            arm_updated_at = self._arm_updated_at
            hand_q = {
                side: None if values is None else values.copy()
                for side, values in self._hand_q.items()
            }
            hand_updated_at = dict(self._hand_updated_at)
            diagnostics = {
                "left": dict(self._diagnostics["left"]),
                "right": dict(self._diagnostics["right"]),
            }
            last_rejection = dict(self._last_rejection)
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
                    f"{side} (missing valid FTP angle_act; "
                    f"malformed={diagnostics[side]['malformed_messages']}, "
                    f"last_rejection={last_rejection[side]!r})"
                )
            elif now - hand_updated_at[side] > self._hand_max_age_s:
                stale.append(
                    f"{side} (age {now - hand_updated_at[side]:.3f}s > "
                    f"{self._hand_max_age_s:.3f}s; "
                    f"{self._hand_max_age_constant}={self._hand_max_age_s:.3f}s)"
                )
        if stale:
            raise TimeoutError(f"Stale Unitree/Inspire FTP state: {', '.join(stale)}")

        assert arm_message is not None and hand_q["left"] is not None and hand_q["right"] is not None
        arm = np.asarray(
            [arm_message.motor_state[index].q for index in self._arm_indices],
            dtype=np.float64,
        )
        arm_dq = np.asarray(
            [arm_message.motor_state[index].dq for index in self._arm_indices],
            dtype=np.float64,
        )
        validate_measured_state(
            arm,
            arm_dq,
            hand_q["left"],
            hand_q["right"],
            end_effector=self.end_effector,
        )
        return RobotState(
            captured_at=min(
                arm_updated_at,
                hand_updated_at["left"],
                hand_updated_at["right"],
            ),
            mode_machine=int(getattr(arm_message, "mode_machine", 0)),
            arm=arm,
            arm_dq=arm_dq,
            left_hand=hand_q["left"],
            right_hand=hand_q["right"],
            left_hand_received_at=hand_updated_at["left"],
            right_hand_received_at=hand_updated_at["right"],
            arm_received_at=arm_updated_at,
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
        raise TimeoutError(f"Timed out waiting for fresh Unitree/Inspire FTP state ({last_error})")

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "arm_callbacks": int(self._diagnostics["arm_callbacks"]),
                "left": dict(self._diagnostics["left"]),
                "right": dict(self._diagnostics["right"]),
                "last_rejection": dict(self._last_rejection),
            }

    def close(self) -> None:
        for subscriber in getattr(self, "_subscribers", {}).values():
            with contextlib.suppress(Exception):
                subscriber.Close()
