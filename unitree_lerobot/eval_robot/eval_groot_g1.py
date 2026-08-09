#!/usr/bin/env python3
"""Run a colour-only GR00T policy on a G1-29 with Dex3 hands.

The default is read-only shadow mode.  ``--actuate`` creates a separate,
watchdog-owning DDS actuator process only after model, camera, state, and action
preflight checks and an interactive confirmation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError, Gr00tClient
from unitree_lerobot.eval_robot.groot_contract import (
    CONTROL_HZ,
    MAX_EXECUTION_HORIZON,
    TASKS,
    ActionChunk,
    make_observation,
    parse_action_chunk,
    validate_model_contract,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    G1Dex3StateReader,
    SafeG1Dex3Actuator,
    TeleimagerColourCamera,
    initialize_dds,
)


LOGGER = logging.getLogger("eval_groot_g1")
LOCAL_POLICY_HOSTS = {"127.0.0.1", "localhost"}


def select_instruction(task_name: str | None) -> tuple[str, str]:
    if task_name is not None:
        return task_name, TASKS[task_name]
    print("Select a trained task:")
    task_names = list(TASKS)
    for index, name in enumerate(task_names, start=1):
        print(f"  {index}. {name:16s}  {TASKS[name]}")
    try:
        selected = int(input("Task number: ").strip())
        name = task_names[selected - 1]
    except (EOFError, ValueError, IndexError) as exc:
        raise DeploymentError("A valid trained task must be selected") from exc
    return name, TASKS[name]


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.execution_horizon <= MAX_EXECUTION_HORIZON:
        raise DeploymentError(f"--execution-horizon must be between 1 and {MAX_EXECUTION_HORIZON}")
    if args.max_chunks < 1:
        raise DeploymentError("--max-chunks must be finite and at least 1")
    if args.actuate and not args.sim and not args.network_interface:
        raise DeploymentError("Real actuation requires an explicit --network-interface")
    if args.actuate and args.sim and args.network_interface is not None:
        raise DeploymentError(
            "The stock Unitree IsaacLab DDS process auto-selects its interface; an explicit "
            "runner --network-interface may not match it. Omit the option and isolate the simulator host."
        )
    if args.actuate and args.sim and not args.confirm_sim_network_isolated:
        raise DeploymentError(
            "IsaacLab DDS domain 1 is not a physical safety boundary. Disconnect/isolate every "
            "physical robot network, then pass --confirm-sim-network-isolated."
        )
    if args.actuate and not args.sim and not args.allow_unqualified_real:
        raise DeploymentError(
            "Real actuation is fail-closed until robot-side publisher-loss and Dex3 stop "
            "behavior are qualified. Use shadow/IsaacLab first; --allow-unqualified-real is "
            "an expert override, not a safety guarantee."
        )
    if args.actuate and args.policy_host not in LOCAL_POLICY_HOSTS:
        raise DeploymentError(
            "Live mode requires a loopback GR00T server. The wire protocol is not authenticated; "
            "use a shadow run for remote-server testing."
        )


def confirm_actuation(simulation: bool, task_name: str, instruction: str) -> None:
    if simulation:
        print(
            f"\nSIMULATION DDS COMMANDS ENABLED for {task_name!r}: {instruction}\n"
            "DDS domain 1 reuses robot command topic names. Confirm this host is isolated "
            "from every physical robot network."
        )
        required = "SIMULATE"
    else:
        print(
            "\nREAL ROBOT ACTUATION ENABLED. Confirm Regular motion mode, a clear and "
            "supported workspace, correct DDS interface, and an operator holding the "
            "physical emergency stop. Stop XR teleoperation and every other arm/hand publisher."
        )
        print(
            "WARNING: DDS Write and robot-side Dex3/publisher-loss failover are not proven "
            "time-bounded by this client. This is an explicitly unqualified hardware test."
        )
        print(f"Task {task_name!r}: {instruction}")
        required = "ACTUATE"
    if input(f"Type {required} to create command publishers: ").strip() != required:
        raise DeploymentError("Actuation was not confirmed")


def infer_chunk(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerColourCamera,
    instruction: str,
    model_horizon: int,
    execution_horizon: int,
    actuator: SafeG1Dex3Actuator | None = None,
    camera_timeout_s: float = 0.5,
) -> tuple[ActionChunk, float]:
    if actuator is not None:
        actuator.heartbeat()
    state = state_reader.read(timeout_s=0.5)
    if actuator is not None:
        actuator.heartbeat()
    rgb = camera.read_rgb(timeout_s=camera_timeout_s)
    if actuator is not None:
        actuator.heartbeat()

    observation = make_observation(rgb, state.arm, state.left_hand, state.right_hand, instruction)
    started = time.monotonic()
    action = policy.get_action(observation)
    inference_s = time.monotonic() - started
    if actuator is not None:
        actuator.assert_healthy()
    chunk = parse_action_chunk(
        action,
        model_horizon=model_horizon,
        execution_horizon=execution_horizon,
        current_arm=state.arm,
        current_left=state.left_hand,
        current_right=state.right_hand,
    )
    return chunk, inference_s


def chunk_delta_summary(chunk: ActionChunk, state_reader: G1Dex3StateReader) -> str:
    state = state_reader.read(timeout_s=0.5)
    arm_delta = float(np.max(np.abs(chunk.arm[0] - state.arm)))
    hand_delta = float(
        max(
            np.max(np.abs(chunk.left_hand[0] - state.left_hand)),
            np.max(np.abs(chunk.right_hand[0] - state.right_hand)),
        )
    )
    return f"first-step delta arm={arm_delta:.4f} rad, hand={hand_delta:.4f} rad"


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    repository_root = Path(__file__).resolve().parents[2]
    os.chdir(repository_root)
    task_name, instruction = select_instruction(args.task)
    image_host = args.image_host or ("127.0.0.1" if args.sim else "192.168.123.164")

    policy: Gr00tClient | None = None
    camera: TeleimagerColourCamera | None = None
    state_reader: G1Dex3StateReader | None = None
    actuator: SafeG1Dex3Actuator | None = None
    cleanup_error: Exception | None = None
    try:
        policy = Gr00tClient(args.policy_host, args.policy_port)
        if not policy.ping():
            raise DeploymentError(f"GR00T server at {args.policy_host}:{args.policy_port} did not answer ping")
        validate_policy_metadata(policy.get_policy_metadata())
        contract = validate_model_contract(policy.get_modality_config())
        if args.execution_horizon > contract.action_horizon:
            raise DeploymentError(
                f"Checkpoint action horizon is only {contract.action_horizon}, but "
                f"{args.execution_horizon} steps were requested"
            )
        LOGGER.info(
            "GR00T contract verified: ego_view + four G1/Dex3 state/action keys, horizon %d",
            contract.action_horizon,
        )

        initialize_dds(args.sim, args.network_interface)
        state_reader = G1Dex3StateReader(simulation=args.sim)
        camera = TeleimagerColourCamera(image_host)
        head = camera.config["head_camera"]
        LOGGER.info(
            "TeleImager config: host=%s type=%s shape=%s binocular=%s fps=%s",
            image_host,
            head.get("type"),
            head.get("image_shape"),
            head.get("binocular"),
            head.get("fps"),
        )

        # Complete one observation -> server -> validated action pass while no
        # command publisher exists.  This output is deliberately discarded.
        policy.reset()
        preflight, inference_s = infer_chunk(
            policy,
            state_reader,
            camera,
            instruction,
            contract.action_horizon,
            args.execution_horizon,
            camera_timeout_s=3.0,
        )
        LOGGER.info(
            "Publisher-free preflight passed in %.3fs: %s",
            inference_s,
            chunk_delta_summary(preflight, state_reader),
        )

        if not args.actuate:
            LOGGER.info("SHADOW MODE: no command publishers were created")
            LOGGER.info(
                "Shadow chunk 1/%d: inference %.3fs, %s",
                args.max_chunks,
                inference_s,
                chunk_delta_summary(preflight, state_reader),
            )
            for chunk_number in range(2, args.max_chunks + 1):
                time.sleep(args.execution_horizon / CONTROL_HZ)
                chunk, inference_s = infer_chunk(
                    policy,
                    state_reader,
                    camera,
                    instruction,
                    contract.action_horizon,
                    args.execution_horizon,
                )
                LOGGER.info(
                    "Shadow chunk %d/%d: inference %.3fs, %s",
                    chunk_number,
                    args.max_chunks,
                    inference_s,
                    chunk_delta_summary(chunk, state_reader),
                )
            return

        confirm_actuation(args.sim, task_name, instruction)
        actuator = SafeG1Dex3Actuator(args.sim, args.network_interface)
        actuator.start()
        actuator.arm()
        LOGGER.warning("%s COMMAND MODE ARMED", "SIMULATION" if args.sim else "REAL ROBOT")

        # Authority ramping takes time, so discard preflight and reset/re-observe.
        policy.reset()
        for chunk_number in range(1, args.max_chunks + 1):
            chunk, inference_s = infer_chunk(
                policy,
                state_reader,
                camera,
                instruction,
                contract.action_horizon,
                args.execution_horizon,
                actuator,
            )
            sequence = actuator.submit(chunk)
            actuator.wait_completed(sequence, timeout_s=args.execution_horizon / CONTROL_HZ + 1.0)
            LOGGER.info(
                "Completed live chunk %d/%d (%d actions, inference %.3fs)",
                chunk_number,
                args.max_chunks,
                chunk.length,
                inference_s,
            )
    finally:
        active_error = sys.exc_info()[0] is not None
        if actuator is not None:
            try:
                actuator.close()
            except Exception as exc:
                LOGGER.exception("Actuator cleanup failed")
                cleanup_error = exc
        if camera is not None:
            try:
                camera.close()
            except Exception:
                LOGGER.exception("Camera cleanup failed")
        if state_reader is not None:
            state_reader.close()
        if policy is not None:
            policy.close()
        if cleanup_error is not None and not active_error:
            raise DeploymentError(f"Actuator cleanup failed: {cleanup_error}") from cleanup_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colour-only GR00T runner for Unitree G1-29 + Dex3")
    parser.add_argument("--task", choices=tuple(TASKS), help="Trained task ID; omit for a menu")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument(
        "--image-host",
        help="TeleImager server host (default: robot PC2, or 127.0.0.1 with --sim)",
    )
    parser.add_argument(
        "--network-interface",
        help="CycloneDDS interface (real: explicit robot NIC; IsaacLab actuation: omit and isolate host)",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use IsaacLab DDS domain 1 and rt/lowcmd (never real rt/arm_sdk)",
    )
    parser.add_argument(
        "--actuate",
        action="store_true",
        help="Create command publishers after preflight and confirmation",
    )
    parser.add_argument(
        "--allow-unqualified-real",
        action="store_true",
        help="Expert override for unqualified real DDS/Dex3 failover; has no effect with --sim",
    )
    parser.add_argument(
        "--confirm-sim-network-isolated",
        action="store_true",
        help="Assert that no physical robot network is reachable during IsaacLab actuation",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    def stop_on_signal(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signal_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, stop_on_signal)
    try:
        run(build_parser().parse_args())
    except KeyboardInterrupt:
        LOGGER.warning("Stop requested")
    except (DeploymentError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
