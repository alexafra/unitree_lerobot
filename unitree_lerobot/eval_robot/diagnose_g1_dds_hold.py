"""Measure real G1 DDS timing while holding the freshly measured pose.

This diagnostic uses the production actuator child and its existing watchdog
and release path.  It acquires arm authority, remains in pre-initialization
measured HOLD, and records state/read/write timing.  It has no camera, GR00T,
initialization, warm-start, or policy-action path.
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import signal
import sys
import termios
import time
from typing import Any

from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import SafeG1Dex3Actuator


LOGGER = logging.getLogger("diagnose_g1_dds_hold")
MIN_DURATION_S = 1.0
MAX_DURATION_S = 300.0
DIAGNOSTIC_ARM_TIMEOUT_S = 6.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire real G1 command authority, hold the freshly measured pose, "
            "and attribute timing stalls to state lookup, VM scheduling, or an "
            "individual arm/left/right DDS Write. No camera or policy code is used."
        )
    )
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--allow-unqualified-real",
        action="store_true",
        help="Required acknowledgement that this is an unqualified real-hardware diagnostic.",
    )
    args = parser.parse_args()
    if not args.allow_unqualified_real:
        parser.error("--allow-unqualified-real is required")
    if not MIN_DURATION_S <= args.duration <= MAX_DURATION_S:
        parser.error(f"--duration must be between {MIN_DURATION_S:g} and {MAX_DURATION_S:g} seconds")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def _confirm(duration_s: float) -> None:
    token = f"HOLD-TIMING-{secrets.randbelow(900000) + 100000}"
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    print("\nREAL MEASURED-HOLD DDS TIMING DIAGNOSTIC.", flush=True)
    print("This creates arm and Dex3 command publishers and acquires arm authority.", flush=True)
    print(f"It remains in measured HOLD for {duration_s:g}s, then releases.", flush=True)
    print("It cannot initialize, contact the camera/GR00T, warm-start, or submit actions.", flush=True)
    print("Use Regular mode, clear/support the workspace, hold the physical E-stop,", flush=True)
    print("and stop every other arm/hand writer. Never press Ctrl-Z.", flush=True)
    print(f"Type exactly {token} then Enter; anything else cancels: ", end="", flush=True)
    if sys.stdin.readline().strip() != token:
        raise SystemExit("Cancelled before command-publisher construction")


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def _format_metric(name: str, metrics: dict[str, Any]) -> str:
    values = metrics.get(name, {})
    if not isinstance(values, dict) or not values.get("count"):
        return f"{name}: no samples"
    return (
        f"{name}: mean={values['mean']:.3f} p50={values['p50']:.3f} "
        f"p95={values['p95']:.3f} p99={values['p99']:.3f} max={values['max']:.3f} ms"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    LOGGER.warning(
        "DDS_HOLD_COMPLETE samples=%d elapsed=%.3fs achieved=%.2fHz pauses=%d recoveries=%d max_pause=%.3fs",
        int(summary.get("samples", 0)),
        float(summary.get("elapsed_s", 0.0)),
        float(summary.get("achieved_hz", 0.0)),
        int(summary.get("hand_pause_count", 0)),
        int(summary.get("hand_recovery_count", 0)),
        float(summary.get("max_hand_pause_s", 0.0)),
    )
    for name in (
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
    ):
        LOGGER.warning("DDS_HOLD_METRIC %s", _format_metric(name, metrics))
    LOGGER.warning("DDS_HOLD_OVERRUNS %s", json.dumps(summary.get("overruns", {}), sort_keys=True))
    for record in summary.get("slowest", []):
        LOGGER.warning("DDS_HOLD_SLOW %s", json.dumps(record, sort_keys=True))
    print("DDS_HOLD_TIMING_JSON=" + json.dumps(summary, sort_keys=True), flush=True)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for signal_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, _raise_interrupt)

    _confirm(args.duration)
    actuator = SafeG1Dex3Actuator(
        simulation=False,
        network_interface=args.network_interface,
        command_conditioning="none",
        dds_hold_diagnostics=True,
    )
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        actuator.start()
        LOGGER.warning("DDS_HOLD_READY publishers exist; no DDS command has been written yet")
        actuator.arm(timeout_s=DIAGNOSTIC_ARM_TIMEOUT_S)
        LOGGER.warning("DDS_HOLD_ARMED pre-initialization measured HOLD active; timing for %.1fs", args.duration)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            actuator.heartbeat()
            actuator.assert_healthy()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            actuator.close()
            LOGGER.warning("DDS_HOLD_LOCAL_STOP_ACKNOWLEDGED")
        except BaseException as exc:
            cleanup_error = exc

    summary = actuator.last_dds_hold_timing
    if summary is not None:
        _print_summary(summary)
    else:
        LOGGER.error("DDS_HOLD_NO_TIMING_SUMMARY")
    if primary_error is not None:
        LOGGER.error("DDS_HOLD_FAILED %s: %s", type(primary_error).__name__, primary_error)
    if cleanup_error is not None:
        LOGGER.error("DDS_HOLD_CLEANUP_FAILED %s: %s", type(cleanup_error).__name__, cleanup_error)
    return 1 if primary_error is not None or cleanup_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
