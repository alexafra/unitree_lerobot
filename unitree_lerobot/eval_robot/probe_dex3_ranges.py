#!/usr/bin/env python3
"""Subscriber-only Dex3 or Inspire FTP hand-state/range probe.

This tool creates only the two selected hand-state subscribers. It never
constructs a command publisher and never contacts the camera or policy server.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unitree_lerobot.eval_robot.g1_end_effectors import (
    DEX3_PROFILE,
    INSPIRE_FTP_PROFILE,
    EndEffectorProfile,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import initialize_dds


FRESHNESS_REFERENCE_S = 0.075
PROBE_PROFILES = {
    DEX3_PROFILE.name: DEX3_PROFILE,
    INSPIRE_FTP_PROFILE.name: INSPIRE_FTP_PROFILE,
}


def _decode_dex3_state(message: Any) -> np.ndarray | None:
    motor_state = getattr(message, "motor_state", None)
    if motor_state is None or len(motor_state) < DEX3_PROFILE.hand_dof:
        return None
    try:
        values = np.asarray(
            [motor_state[index].q for index in range(DEX3_PROFILE.hand_dof)],
            dtype=np.float64,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if values.shape != (DEX3_PROFILE.hand_dof,) or not np.all(np.isfinite(values)):
        return None
    return values


def _decode_inspire_ftp_state(message: Any) -> np.ndarray | None:
    try:
        raw = np.asarray(message.angle_act)
    except (AttributeError, TypeError, ValueError):
        return None
    if raw.shape != (INSPIRE_FTP_PROFILE.hand_dof,) or raw.dtype.kind not in "iuf":
        return None
    values = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1000.0):
        return None
    return values / 1000.0


def _profile_and_decoder(end_effector: str):
    try:
        profile = PROBE_PROFILES[end_effector]
    except KeyError as exc:
        choices = ", ".join(PROBE_PROFILES)
        raise ValueError(f"Unsupported end effector {end_effector!r}; expected one of: {choices}") from exc
    decoder = _decode_dex3_state if profile is DEX3_PROFILE else _decode_inspire_ftp_state
    return profile, decoder


def _state_message_type(end_effector: str):
    if end_effector == DEX3_PROFILE.name:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

        return HandState_
    if end_effector == INSPIRE_FTP_PROFILE.name:
        try:
            from inspire_sdkpy import inspire_dds
        except ImportError as exc:
            raise RuntimeError(
                "Inspire FTP probing requires the vendor inspire_sdkpy package"
            ) from exc
        try:
            return inspire_dds.inspire_hand_state
        except AttributeError as exc:
            raise RuntimeError(
                "Installed inspire_sdkpy is missing the inspire_hand_state DDS type"
            ) from exc
    raise ValueError(f"Unsupported end effector {end_effector!r}")


@dataclass
class HandRange:
    name: str
    hand_dof: int
    callback_count: int = 0
    count: int = 0
    rejected_count: int = 0
    first_received_at: float | None = None
    received_at: float | None = None
    previous_received_at: float | None = None
    maximum_gap_s: float = 0.0
    gaps_over_freshness: list[float] = field(default_factory=list)
    current: np.ndarray = field(init=False)
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.hand_dof <= 0:
            raise ValueError("hand_dof must be positive")
        self.current = np.full(self.hand_dof, np.nan)
        self.minimum = np.full(self.hand_dof, np.inf)
        self.maximum = np.full(self.hand_dof, -np.inf)

    def update(self, values: np.ndarray | None, received_at: float) -> None:
        self.callback_count += 1
        if values is None:
            self.rejected_count += 1
            return
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.hand_dof,) or not np.all(np.isfinite(values)):
            self.rejected_count += 1
            return
        if self.received_at is not None:
            gap = received_at - self.received_at
            self.maximum_gap_s = max(self.maximum_gap_s, gap)
            if gap > FRESHNESS_REFERENCE_S:
                self.gaps_over_freshness.append(gap)
        if self.first_received_at is None:
            self.first_received_at = received_at
        self.previous_received_at = self.received_at
        self.received_at = received_at
        self.count += 1
        self.current = values
        self.minimum = np.minimum(self.minimum, values)
        self.maximum = np.maximum(self.maximum, values)

    def copy(self) -> HandRange:
        result = HandRange(self.name, self.hand_dof)
        result.callback_count = self.callback_count
        result.count = self.count
        result.rejected_count = self.rejected_count
        result.first_received_at = self.first_received_at
        result.received_at = self.received_at
        result.previous_received_at = self.previous_received_at
        result.maximum_gap_s = self.maximum_gap_s
        result.gaps_over_freshness = list(self.gaps_over_freshness)
        result.current = self.current.copy()
        result.minimum = self.minimum.copy()
        result.maximum = self.maximum.copy()
        return result

    @property
    def mean_rate_hz(self) -> float:
        if self.count < 2 or self.first_received_at is None or self.received_at is None:
            return math.nan
        elapsed = self.received_at - self.first_received_at
        return (self.count - 1) / elapsed if elapsed > 0.0 else math.nan

    @property
    def latest_rate_hz(self) -> float:
        if self.previous_received_at is None or self.received_at is None:
            return math.nan
        elapsed = self.received_at - self.previous_received_at
        return 1.0 / elapsed if elapsed > 0.0 else math.nan


class Dex3RangeProbe:
    """Historical name retained for existing imports; supports both probe profiles."""

    def __init__(self, network_interface: str, end_effector: str = "dex3"):
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self.profile, self._decoder = _profile_and_decoder(end_effector)
        message_type = _state_message_type(end_effector)
        initialize_dds(False, network_interface)
        self._lock = threading.Lock()
        self.hands = {
            "left": HandRange("left", self.profile.hand_dof),
            "right": HandRange("right", self.profile.hand_dof),
        }
        self._subscribers = {
            "left": ChannelSubscriber(self.profile.left_state_topic, message_type),
            "right": ChannelSubscriber(self.profile.right_state_topic, message_type),
        }
        for name, subscriber in self._subscribers.items():
            subscriber.Init(handler=self._handler(name))

    def _handler(self, name: str):
        def receive(message: Any) -> None:
            if message is None:
                return
            received_at = time.monotonic()
            values = self._decoder(message)
            with self._lock:
                self.hands[name].update(values, received_at)

        return receive

    def snapshot(self) -> dict[str, HandRange]:
        with self._lock:
            return {name: hand.copy() for name, hand in self.hands.items()}

    def close(self) -> None:
        for subscriber in self._subscribers.values():
            with contextlib.suppress(Exception):
                subscriber.Close()


def _format_live(hand: HandRange, now: float) -> str:
    if hand.received_at is None:
        return f"{hand.name:>5}: waiting for first sample"
    age = now - hand.received_at
    values = " ".join(f"{value:+.4f}" for value in hand.current)
    return (
        f"{hand.name:>5}: n={hand.count:7d} rate={hand.mean_rate_hz:7.2f}Hz "
        f"latest={hand.latest_rate_hz:7.2f}Hz age={age:6.3f}s "
        f"max_gap={hand.maximum_gap_s:6.3f}s state=[{values}]"
    )


def _print_mapping(profile: EndEffectorProfile) -> None:
    print(
        f"End effector: {profile.name}; value unit: {profile.value_unit}; "
        "timing: accepted DDS callback receipt"
    )
    print(f"State topics: left={profile.left_state_topic} right={profile.right_state_topic}")
    print("Numeric motor IDs and repository labels (verify the physical finger independently):")
    for side, names in (("left", profile.left_joint_names), ("right", profile.right_joint_names)):
        print(f"  {side}: " + ", ".join(f"{index}={name}" for index, name in enumerate(names)))


def _print_summary(
    hand: HandRange,
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    value_unit: str,
) -> None:
    print(f"\n{hand.name.upper()} HAND SUMMARY")
    print(
        f"callbacks={hand.callback_count} accepted={hand.count} rejected={hand.rejected_count} "
        f"mean_rate={hand.mean_rate_hz:.3f}Hz max_gap={hand.maximum_gap_s:.6f}s "
        f"gaps_over_{FRESHNESS_REFERENCE_S:.3f}s={len(hand.gaps_over_freshness)}"
    )
    print(f"values are {value_unit}")
    print(" id  label                         minimum    maximum      span    nominal_lower nominal_upper")
    for index, name in enumerate(names):
        if hand.count:
            minimum = hand.minimum[index]
            maximum = hand.maximum[index]
            span = maximum - minimum
        else:
            minimum = maximum = span = math.nan
        print(
            f" {index:>2d}  {name:<28} {minimum:+10.5f} {maximum:+10.5f} "
            f"{span:10.5f} {lower[index]:+13.5f} {upper[index]:+13.5f}"
        )
    if hand.gaps_over_freshness:
        gaps = ", ".join(f"{gap:.4f}" for gap in hand.gaps_over_freshness)
        print(f"gaps over freshness limit (s): {gaps}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True, help="Robot DDS interface, for example enp0s1")
    parser.add_argument(
        "--end-effector",
        choices=tuple(PROBE_PROFILES),
        default=DEX3_PROFILE.name,
        help="Hand DDS transport to probe (default: dex3)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to sample; 0 waits until Ctrl-C (default: 0)",
    )
    parser.add_argument("--print-hz", type=float, default=2.0, help="Live display rate (default: 2)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not np.isfinite(args.duration) or args.duration < 0.0:
        raise SystemExit("--duration must be finite and >= 0")
    if not np.isfinite(args.print_hz) or args.print_hz <= 0.0:
        raise SystemExit("--print-hz must be finite and > 0")

    print("SUBSCRIBER ONLY: no DDS command publishers will be created.")
    print("Do not manually force a powered joint; use only a supported passive/test procedure.")
    profile, _ = _profile_and_decoder(args.end_effector)
    _print_mapping(profile)
    probe = Dex3RangeProbe(args.network_interface, args.end_effector)
    started = time.monotonic()
    try:
        while args.duration == 0.0 or time.monotonic() - started < args.duration:
            now = time.monotonic()
            snapshot = probe.snapshot()
            print(_format_live(snapshot["left"], now))
            print(_format_live(snapshot["right"], now), flush=True)
            time.sleep(1.0 / args.print_hz)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        snapshot = probe.snapshot()
        probe.close()

    _print_summary(
        snapshot["left"],
        profile.left_joint_names,
        profile.left_lower,
        profile.left_upper,
        profile.value_unit,
    )
    _print_summary(
        snapshot["right"],
        profile.right_joint_names,
        profile.right_lower,
        profile.right_upper,
        profile.value_unit,
    )


if __name__ == "__main__":
    main()
