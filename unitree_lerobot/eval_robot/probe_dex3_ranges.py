#!/usr/bin/env python3
"""Subscriber-only Dex3 joint-position/range probe.

This tool creates only the two Dex3 state subscribers. It never constructs a
command publisher and never contacts the camera or policy server.
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

from unitree_lerobot.eval_robot.groot_contract import (
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_LOWER,
    LEFT_HAND_UPPER,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_HAND_LOWER,
    RIGHT_HAND_UPPER,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import initialize_dds


HAND_DOF = 7
FRESHNESS_REFERENCE_S = 0.075


@dataclass
class HandRange:
    name: str
    count: int = 0
    received_at: float | None = None
    previous_received_at: float | None = None
    maximum_gap_s: float = 0.0
    gaps_over_freshness: list[float] = field(default_factory=list)
    current: np.ndarray = field(default_factory=lambda: np.full(HAND_DOF, np.nan))
    minimum: np.ndarray = field(default_factory=lambda: np.full(HAND_DOF, np.inf))
    maximum: np.ndarray = field(default_factory=lambda: np.full(HAND_DOF, -np.inf))

    def update(self, message: Any, received_at: float) -> None:
        motor_state = getattr(message, "motor_state", None)
        if motor_state is None or len(motor_state) < HAND_DOF:
            return
        values = np.asarray([motor_state[index].q for index in range(HAND_DOF)], dtype=np.float64)
        if values.shape != (HAND_DOF,) or not np.all(np.isfinite(values)):
            return
        if self.received_at is not None:
            gap = received_at - self.received_at
            self.maximum_gap_s = max(self.maximum_gap_s, gap)
            if gap > FRESHNESS_REFERENCE_S:
                self.gaps_over_freshness.append(gap)
        self.previous_received_at = self.received_at
        self.received_at = received_at
        self.count += 1
        self.current = values
        self.minimum = np.minimum(self.minimum, values)
        self.maximum = np.maximum(self.maximum, values)


class Dex3RangeProbe:
    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

        initialize_dds(False, network_interface)
        self._lock = threading.Lock()
        self.hands = {"left": HandRange("left"), "right": HandRange("right")}
        self._subscribers = {
            "left": ChannelSubscriber("rt/dex3/left/state", HandState_),
            "right": ChannelSubscriber("rt/dex3/right/state", HandState_),
        }
        for name, subscriber in self._subscribers.items():
            subscriber.Init(handler=self._handler(name))

    def _handler(self, name: str):
        def receive(message: Any) -> None:
            if message is None:
                return
            with self._lock:
                self.hands[name].update(message, time.monotonic())

        return receive

    def snapshot(self) -> dict[str, HandRange]:
        with self._lock:
            result = {}
            for name, hand in self.hands.items():
                result[name] = HandRange(
                    name=hand.name,
                    count=hand.count,
                    received_at=hand.received_at,
                    previous_received_at=hand.previous_received_at,
                    maximum_gap_s=hand.maximum_gap_s,
                    gaps_over_freshness=list(hand.gaps_over_freshness),
                    current=hand.current.copy(),
                    minimum=hand.minimum.copy(),
                    maximum=hand.maximum.copy(),
                )
            return result

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
        f"{hand.name:>5}: n={hand.count:7d} age={age:6.3f}s "
        f"max_gap={hand.maximum_gap_s:6.3f}s q=[{values}]"
    )


def _print_mapping() -> None:
    print("Numeric motor IDs and repository labels (verify the physical finger independently):")
    for side, names in (("left", LEFT_HAND_JOINT_NAMES), ("right", RIGHT_HAND_JOINT_NAMES)):
        print(f"  {side}: " + ", ".join(f"{index}={name}" for index, name in enumerate(names)))


def _print_summary(hand: HandRange, names: tuple[str, ...], lower: np.ndarray, upper: np.ndarray) -> None:
    print(f"\n{hand.name.upper()} HAND SUMMARY")
    print(
        f"samples={hand.count} max_gap={hand.maximum_gap_s:.6f}s "
        f"gaps_over_{FRESHNESS_REFERENCE_S:.3f}s={len(hand.gaps_over_freshness)}"
    )
    print(" id  label                         min_q      max_q       span    nominal_lower nominal_upper")
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
    _print_mapping()
    probe = Dex3RangeProbe(args.network_interface)
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

    _print_summary(snapshot["left"], LEFT_HAND_JOINT_NAMES, LEFT_HAND_LOWER, LEFT_HAND_UPPER)
    _print_summary(snapshot["right"], RIGHT_HAND_JOINT_NAMES, RIGHT_HAND_LOWER, RIGHT_HAND_UPPER)


if __name__ == "__main__":
    main()
