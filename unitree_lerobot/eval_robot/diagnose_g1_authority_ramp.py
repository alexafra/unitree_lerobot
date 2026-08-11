"""Measure the real G1 arm-authority ramp without camera, policy, or actions.

This diagnostic deliberately stops immediately after the existing guarded
actuator reports ``armed``.  It never calls initialization, warm-start, policy
inference, or action submission.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import signal
import sys
import termios

from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import SafeG1Dex3Actuator


LOGGER = logging.getLogger("diagnose_g1_authority_ramp")
DIAGNOSTIC_ARM_TIMEOUT_S = 6.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the existing measured-pose arm-authority ramp, record its timing, "
            "then immediately release. No camera or policy code is used."
        )
    )
    parser.add_argument("--network-interface", required=True)
    parser.add_argument(
        "--allow-unqualified-real",
        action="store_true",
        help="Required acknowledgement that this is an unqualified real-hardware diagnostic.",
    )
    args = parser.parse_args()
    if not args.allow_unqualified_real:
        parser.error("--allow-unqualified-real is required")
    if not sys.stdin.isatty():
        parser.error("an interactive terminal is required")
    return args


def _confirm() -> None:
    token = f"RAMP-ONLY-{secrets.randbelow(900000) + 100000}"
    # ``tee`` redirects stdout but leaves stdin attached to the terminal.  Use
    # that existing descriptor: some PTYs reject a text-mode ``r+`` wrapper on
    # /dev/tty as non-seekable.
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    print("\nREAL RAMP-ONLY DIAGNOSTIC.", flush=True)
    print("This creates arm and Dex3 command publishers and can move the robot.", flush=True)
    print("It cannot initialize, warm-start, contact GR00T, or submit policy actions.", flush=True)
    print("Use Regular mode, clear/support the workspace, hold the physical E-stop,", flush=True)
    print("and stop every other arm/hand writer. Never press Ctrl-Z.", flush=True)
    print(f"Type exactly {token} then Enter; anything else cancels: ", end="", flush=True)
    if sys.stdin.readline().strip() != token:
        raise SystemExit("Cancelled before command-publisher construction")


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for signal_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, _raise_interrupt)

    _confirm()
    actuator = SafeG1Dex3Actuator(
        simulation=False,
        network_interface=args.network_interface,
        command_conditioning="none",
        authority_ramp_diagnostics=True,
    )
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        actuator.start()
        LOGGER.warning("RAMP_ONLY_READY publishers exist; no DDS command has been written yet")
        actuator.arm(timeout_s=DIAGNOSTIC_ARM_TIMEOUT_S)
        LOGGER.warning("RAMP_ONLY_ARMED ramp completed; releasing immediately")
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            actuator.close()
            LOGGER.warning("RAMP_ONLY_LOCAL_STOP_ACKNOWLEDGED")
        except BaseException as exc:
            cleanup_error = exc

    if primary_error is not None:
        LOGGER.error("RAMP_ONLY_FAILED %s: %s", type(primary_error).__name__, primary_error)
    if cleanup_error is not None:
        LOGGER.error("RAMP_ONLY_CLEANUP_FAILED %s: %s", type(cleanup_error).__name__, cleanup_error)
    return 1 if primary_error is not None or cleanup_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
