"""Move the real G1/Dex3 slowly to one frozen demonstration start pose.

Run this on the GPU PC whose Ethernet cable is connected directly to the
robot. Do not run it on the robot or on the TeleImager/camera PC. It does not
use GR00T, a policy server, or the camera server.

Command (from the unitree_lerobot environment on the GPU PC)::

    conda activate unitree_lerobot
    cd /home/alex/Development/unitree_lerobot
    python -m unitree_lerobot.eval_robot.zero_state_test \
      --network-interface enp132s0 \
      --allow-unqualified-real

This is deliberately separate from GR00T inference.  It commands the measured
``observation.state`` from frame 0 of training episode 0, holds that exact
joint-position target, and keeps the existing immediate-q orderly-release path
live for the entire period in which command authority exists.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    OperatorRelease,
    _OperatorTerminal,
    _confirm_before_authority,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import InitializationSpec
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    ARM_RELEASE_RAMP_S,
    INITIALIZATION_ARM_SPEED_RAD_S,
    INITIALIZATION_HAND_SPEED_RAD_S,
    ImmediateControlEvent,
    SafeG1Dex3Actuator,
)
from unitree_lerobot.eval_robot.run_logging import (
    configure_process_logging,
    create_run_directory,
    write_json,
)
from unitree_lerobot.eval_robot.training_start_pose import (
    TRAINING_START_JOINTS_RAD,
    TRAINING_START_SOURCE,
    training_start_spec,
)


LOGGER = logging.getLogger("zero_state_test")

# Backward-compatible names retained for the standalone script and its tests.
ZERO_STATE_SOURCE = TRAINING_START_SOURCE
ZERO_STATE_JOINTS_RAD = TRAINING_START_JOINTS_RAD

MAX_ZERO_STATE_ARM_SPEED_RAD_S = 0.25
MAX_ZERO_STATE_HAND_SPEED_RAD_S = 0.50
REQUIRED_RELEASE_RAMP_S = 3.0
HOLD_POLL_S = 0.02


class ZeroStateActuator(SafeG1Dex3Actuator):
    """Make any Dex3 feedback pause sticky for this fixed-target test.

    The general policy actuator deliberately captures measured arm q after a
    short hand-feedback pause.  That is a safe recovery for policy execution,
    but it means a fixed 28-joint test no longer holds its stated arm target.
    This wrapper detects the parent status even if pause and recovery are
    drained together, so this test releases instead of silently changing pose.
    """

    def __init__(self, *args: object, **kwargs: object):
        self._zero_state_hand_pause: object | None = None
        super().__init__(*args, **kwargs)

    def _record_auxiliary_status(self, kind: str, value: object) -> bool:
        handled = super()._record_auxiliary_status(kind, value)
        if kind == "hand_state_pause" and self._zero_state_hand_pause is None:
            self._zero_state_hand_pause = value
        return handled

    def assert_fixed_target_healthy(self) -> None:
        self.assert_healthy()
        if self._zero_state_hand_pause is not None:
            raise DeploymentError(
                "Dex3 feedback paused during the zero state test; releasing because the "
                "shared actuator recovery may replace the fixed arm target: "
                f"{self._zero_state_hand_pause}"
            )


def zero_state_spec() -> InitializationSpec:
    """Return a fresh validated target so callers cannot mutate shared state."""

    return training_start_spec()


def _validate_runtime(args: argparse.Namespace) -> None:
    if not sys.stdin.isatty():
        raise DeploymentError(
            "Zero state test requires an interactive TTY so q is always available"
        )
    if not args.network_interface:
        raise DeploymentError("Zero state test requires an explicit --network-interface")
    if not args.allow_unqualified_real:
        raise DeploymentError(
            "Real actuation remains unqualified for DDS publisher-loss and Dex3 failover; "
            "pass --allow-unqualified-real only with the physical emergency stop in hand"
        )
    if INITIALIZATION_ARM_SPEED_RAD_S > MAX_ZERO_STATE_ARM_SPEED_RAD_S:
        raise DeploymentError(
            f"Configured initialization arm speed {INITIALIZATION_ARM_SPEED_RAD_S:.3f} rad/s "
            "exceeds "
            f"the zero-state-test ceiling {MAX_ZERO_STATE_ARM_SPEED_RAD_S:.3f} rad/s"
        )
    if INITIALIZATION_HAND_SPEED_RAD_S > MAX_ZERO_STATE_HAND_SPEED_RAD_S:
        raise DeploymentError(
            f"Configured initialization hand speed {INITIALIZATION_HAND_SPEED_RAD_S:.3f} rad/s "
            "exceeds "
            f"the zero-state-test ceiling {MAX_ZERO_STATE_HAND_SPEED_RAD_S:.3f} rad/s"
        )
    if not math.isclose(
        ARM_RELEASE_RAMP_S,
        REQUIRED_RELEASE_RAMP_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DeploymentError(
            f"Zero state test requires the reviewed {REQUIRED_RELEASE_RAMP_S:.1f}s release ramp; "
            f"the actuator is configured for {ARM_RELEASE_RAMP_S:.3f}s"
        )


def _raise_if_release(terminal: _OperatorTerminal) -> None:
    if terminal.poll_control() == "release":
        raise OperatorRelease


def _run_interruptible(
    terminal: _OperatorTerminal,
    operation: Callable[[], None],
    verify: Callable[[], None] | None = None,
) -> None:
    _raise_if_release(terminal)
    try:
        operation()
    except ImmediateControlEvent as exc:
        if exc.action == "release":
            raise OperatorRelease from exc
        raise
    _raise_if_release(terminal)
    if verify is not None:
        try:
            verify()
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                raise OperatorRelease from exc
            raise


def _hold_until_release(
    actuator: SafeG1Dex3Actuator,
    terminal: _OperatorTerminal,
) -> None:
    while True:
        _raise_if_release(terminal)
        try:
            actuator.heartbeat()
            fixed_target_check = getattr(actuator, "assert_fixed_target_healthy", None)
            if callable(fixed_target_check):
                fixed_target_check()
            else:
                actuator.assert_healthy()
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                raise OperatorRelease from exc
            raise
        time.sleep(HOLD_POLL_S)


def run(args: argparse.Namespace) -> None:
    """Acquire authority, move once, and hold until q or a fault releases it."""

    _validate_runtime(args)
    target = zero_state_spec()
    print(
        "\nZERO STATE TEST WILL COMMAND A RECORDED 28-JOINT POSE.\n"
        f"Source: {ZERO_STATE_SOURCE['dataset_path']}, episode 0, frame 0.\n"
        f"Maximum interpolation rates: arms {INITIALIZATION_ARM_SPEED_RAD_S:.2f} rad/s; "
        f"hands {INITIALIZATION_HAND_SPEED_RAD_S:.2f} rad/s.\n"
        "The robot must already be in Motion mode with mode_machine=6 (29-DoF "
        "locked-waist with hands). This script does not change modes and the actuator rejects "
        "any other mode before motion. "
        "These limits slow the position target; they do not reduce controller gains or torque. "
        "The path is joint-space only and is not collision-aware. A recalibrated robot may "
        "place the same numeric joint coordinates at a different physical pose. Keep the "
        "workspace clear and hold the physical emergency stop. Stop every other arm/hand "
        "publisher.\n"
        "q/Q requests orderly release immediately. Startup takes full arm authority at the "
        "measured pose with zero feed-forward torque, then blends gravity compensation over "
        "1.5 seconds. A full-authority release ramps down over 3 seconds; the hands retain "
        "their target until stopMotors afterward, "
        "and a blocked DDS write can extend the total time. This is not an electrical "
        "emergency stop."
    )
    response = _confirm_before_authority(
        "Press r to create publishers and move to ZERO STATE (no Enter); s/q cancels: "
    )
    if response != "continue":
        raise OperatorRelease

    actuator_kwargs = {
        "run_log_dir": (
            None
            if getattr(args, "_run_log_dir", None) is None
            else str(args._run_log_dir)
        ),
    }
    if hasattr(args, "gravity_feedforward"):
        actuator_kwargs["gravity_feedforward"] = args.gravity_feedforward
    actuator = ZeroStateActuator(
        False,
        args.network_interface,
        "none",
        **actuator_kwargs,
    )
    try:
        # The same terminal monitor remains alive across process startup,
        # authority acquisition, interpolation, and the final powered hold.
        with _OperatorTerminal(actuator, stop_enabled=False) as terminal:
            _run_interruptible(
                terminal,
                actuator.start,
                actuator.assert_fixed_target_healthy,
            )
            _run_interruptible(
                terminal,
                actuator.arm,
                actuator.assert_fixed_target_healthy,
            )
            LOGGER.warning("REAL ROBOT COMMAND MODE ARMED FOR ZERO STATE TEST")
            _run_interruptible(
                terminal,
                lambda: actuator.initialize(target),
                actuator.assert_fixed_target_healthy,
            )
            LOGGER.warning(
                "Zero state target is being held. The commanded target is exact; measured "
                "joints remain subject to the actuator's configured convergence tolerances. "
                "Press q to release."
            )
            _hold_until_release(actuator, terminal)
    finally:
        actuator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slowly command training episode 0 frame 0 and hold until q releases authority",
    )
    parser.add_argument(
        "--network-interface",
        required=True,
        help="Explicit CycloneDDS Ethernet interface connected to the real G1",
    )
    parser.add_argument(
        "--allow-unqualified-real",
        action="store_true",
        help="Required expert acknowledgment of unqualified DDS/Dex3 failover behavior",
    )
    parser.add_argument(
        "--gravity-feedforward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Publish pose-dependent arm gravity feed-forward torque (default: enabled); "
            "--no-gravity-feedforward publishes zero arm tau"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory for the zero-state-test run log",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_log_dir = create_run_directory(args.log_dir, prefix="zero_state_test")
        configure_process_logging(run_log_dir / "parent.log")
        args._run_log_dir = run_log_dir
        write_json(
            run_log_dir / "run.json",
            {
                "schema_version": 1,
                "argv": list(sys.argv),
                "network_interface": args.network_interface,
                "gravity_feedforward": getattr(args, "gravity_feedforward", True),
                "source": ZERO_STATE_SOURCE,
                "target_rad": ZERO_STATE_JOINTS_RAD.tolist(),
                "run_log_dir": str(run_log_dir),
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Could not create the zero state test log: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    LOGGER.info("RUN_LOG_DIR=%s", run_log_dir)

    def stop_on_signal(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signal_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, stop_on_signal)
    try:
        run(args)
    except OperatorRelease:
        LOGGER.warning("Operator requested orderly command-authority release")
    except KeyboardInterrupt:
        LOGGER.warning("Stop requested; releasing command authority")
    except (DeploymentError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
