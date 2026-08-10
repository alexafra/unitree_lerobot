#!/usr/bin/env python3
"""Run a supported colour or colour+aligned-depth GR00T policy on G1/Dex3.

The default is read-only shadow mode.  ``--actuate`` creates a separate,
watchdog-owning DDS actuator process only after model, camera, state, and action
preflight checks and an interactive confirmation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import select
import signal
import sys
import time

import cv2
import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError, Gr00tClient
from unitree_lerobot.eval_robot.groot_contract import (
    CONTROL_HZ,
    INITIALIZATION_MODES,
    MAX_EXECUTION_HORIZON,
    TASKS,
    ActionChunk,
    InitializationSpec,
    ModelContract,
    load_initialization_spec,
    make_observation,
    parse_action_chunk,
    validate_model_contract,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    G1Dex3StateReader,
    SafeG1Dex3Actuator,
    TeleimagerCamera,
    initialize_dds,
)


LOGGER = logging.getLogger("eval_groot_g1")
LOCAL_POLICY_HOSTS = {"127.0.0.1", "localhost"}
OPERATOR_CONFIRMATION_TIMEOUT_S = 60.0
PREVIEW_WINDOWS = ("GR00T input: ego_view", "GR00T input: depth_gray_view")
HOLD_COMMANDS = {"hold", "h"}
EXIT_COMMANDS = {"quit", "q"}


class OperatorRelease(Exception):
    """Internal control flow for an orderly operator-requested release."""


def select_instruction(task_name: str | None, custom_goal: str | None = None) -> tuple[str, str]:
    if custom_goal is not None:
        if task_name is not None:
            raise DeploymentError("--task and --custom-goal are mutually exclusive")
        if not isinstance(custom_goal, str):
            raise DeploymentError("--custom-goal must be text")
        goal = custom_goal.strip()
        if not goal or len(goal) > 256 or any(ord(character) < 32 for character in goal):
            raise DeploymentError("--custom-goal must be 1..256 printable characters")
        return "custom-goal", goal
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


def confirm_custom_goal(instruction: str) -> None:
    print(
        f"\nCUSTOM GOAL IS NOT AN EXACT TRAINING INSTRUCTION:\n  {instruction}\n"
        "Behavior may be outside the fine-tuning distribution. All normal motion checks remain active."
    )
    try:
        response = input("Type YES to send this custom goal to GR00T: ")
    except EOFError as exc:
        raise DeploymentError("Custom goal confirmation requires interactive input") from exc
    if response.strip() != "YES":
        raise DeploymentError("Custom goal was not confirmed")


def resolve_runtime_goal(response: str) -> tuple[str, str]:
    """Resolve a held-session goal without weakening custom-language confirmation."""

    value = response.strip()
    if value in TASKS:
        return value, TASKS[value]
    if value.isdigit():
        task_names = list(TASKS)
        index = int(value)
        if 1 <= index <= len(task_names):
            task_name = task_names[index - 1]
            return task_name, TASKS[task_name]
    for task_name, instruction in TASKS.items():
        if value == instruction:
            return task_name, instruction
    return select_instruction(None, value)


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
    initialization = getattr(args, "initialization", "measured")
    initial_pose_file = getattr(args, "initial_pose_file", None)
    if initialization not in INITIALIZATION_MODES:
        raise DeploymentError(f"--initialization must be one of {INITIALIZATION_MODES}")
    if initialization == "pose-file" and not initial_pose_file:
        raise DeploymentError("--initialization pose-file requires --initial-pose-file")
    if initialization != "pose-file" and initial_pose_file:
        raise DeploymentError("--initial-pose-file is valid only with --initialization pose-file")
    if not args.actuate and initialization != "measured":
        raise DeploymentError("Moving initialization modes require --actuate")
    if getattr(args, "custom_goal", None) is not None and initialization == "pose-file":
        raise DeploymentError("--custom-goal cannot use a task-bound --initialization pose-file")


def confirm_actuation(simulation: bool, task_name: str, instruction: str) -> None:
    if simulation:
        print(
            f"\nSIMULATION DDS COMMANDS ENABLED for {task_name!r}: {instruction}\n"
            "DDS domain 1 reuses robot command topic names. Confirm this host is isolated "
            "from every physical robot network."
        )
        required = "SIMULATE"
    else:
        if not sys.stdin.isatty():
            raise DeploymentError("Real actuation requires an interactive terminal; piped confirmations are refused")
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


def _readline_while_armed(
    actuator: SafeG1Dex3Actuator,
    prompt: str,
    *,
    timeout_s: float | None,
) -> str:
    """Read a terminal line while continuously servicing the actuator watchdog."""

    print(prompt, end="", flush=True)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        actuator.heartbeat()
        actuator.assert_healthy()
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0.0:
            print()
            raise DeploymentError("Timed out waiting for armed operator input; releasing command authority")
        try:
            wait_s = 0.1 if remaining is None else min(0.1, remaining)
            readable, _, _ = select.select([sys.stdin], [], [], wait_s)
        except (OSError, TypeError, ValueError) as exc:
            print()
            raise DeploymentError("Armed input requires an interactive stdin terminal") from exc
        if not readable:
            continue
        response = sys.stdin.readline()
        if response == "":
            raise DeploymentError("stdin closed while command authority was active")
        return response.strip()


def _confirm_while_armed(
    actuator: SafeG1Dex3Actuator,
    message: str,
    required: str,
) -> None:
    """Wait for bounded operator confirmation without starving the heartbeat."""

    print(message)
    while True:
        try:
            response = _readline_while_armed(
                actuator,
                f"Type {required} to continue: ",
                timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
            )
        except DeploymentError as exc:
            if str(exc).startswith("Timed out waiting for armed operator input"):
                raise DeploymentError(f"Timed out waiting for {required}; releasing command authority") from exc
            raise
        if response == required:
            return
        lowered = response.lower()
        if lowered in EXIT_COMMANDS:
            raise OperatorRelease
        if lowered in HOLD_COMMANDS:
            LOGGER.info("Command remains held; still waiting for exact %s", required)
            continue
        raise DeploymentError(f"Expected {required}; releasing command authority")


def _poll_active_command() -> str | None:
    """Return one completed terminal line without delaying the policy loop."""

    if not sys.stdin.isatty():
        return None
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (OSError, TypeError, ValueError) as exc:
        raise DeploymentError("Could not poll the active command terminal") from exc
    if not readable:
        return None
    response = sys.stdin.readline()
    if response == "":
        raise DeploymentError("stdin closed while command authority was active")
    return response.strip()


def _select_next_goal_while_holding(
    actuator: SafeG1Dex3Actuator,
) -> tuple[str, str] | None:
    """Wait in powered hold until a new goal or an orderly release request."""

    print(
        "\nHOLDING CURRENT MEASURED POSE. GR00T requests are stopped.\n"
        "Enter a trained task ID/number/exact instruction, or arbitrary custom goal text.\n"
        "Type quit or q to release command authority; Ctrl-C also releases."
    )
    for index, (task_name, instruction) in enumerate(TASKS.items(), start=1):
        print(f"  {index}. {task_name:16s}  {instruction}")
    while True:
        response = _readline_while_armed(actuator, "Next goal> ", timeout_s=None)
        lowered = response.lower()
        if lowered in EXIT_COMMANDS:
            return None
        if lowered in HOLD_COMMANDS or not response:
            LOGGER.info("Already holding; no GR00T request was sent")
            continue
        task_name, instruction = resolve_runtime_goal(response)
        if task_name != "custom-goal":
            return task_name, instruction

        LOGGER.warning(
            "Proposed next goal is not an exact trained instruction: %r",
            instruction,
        )
        print(
            f"\nCUSTOM GOAL IS NOT AN EXACT TRAINING INSTRUCTION:\n  {instruction}\n"
            "Behavior may be outside the fine-tuning distribution. All normal motion checks remain active."
        )
        confirmation = _readline_while_armed(
            actuator,
            "Type YES to send this custom goal to GR00T (anything else keeps HOLD): ",
            timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
        )
        if confirmation == "YES":
            return task_name, instruction
        if confirmation.lower() in EXIT_COMMANDS:
            return None
        LOGGER.warning("Custom goal rejected; remaining in powered hold")


def confirm_initialization(actuator: SafeG1Dex3Actuator, spec: InitializationSpec) -> None:
    if not spec.moves:
        return
    warning = (
        f"\nINITIALIZATION WILL MOVE THE ROBOT using {spec.label!r}. "
        "The path is a slow bounded joint-space interpolation, not collision-aware planning. "
        "Keep the workspace clear and remain on the emergency stop."
    )
    if spec.mode == "xr-home":
        warning += " XR-home targets all 14 arm joints and both 7-joint Dex3 hands to zero; both hands must be empty."
    elif spec.moves_hands:
        warning += " This pose explicitly moves one or both hands; verify their contents."
    _confirm_while_armed(actuator, warning, "INITIALIZE")


def confirm_policy_start(actuator: SafeG1Dex3Actuator, spec: InitializationSpec) -> None:
    _confirm_while_armed(
        actuator,
        f"\nInitialization {spec.label!r} converged. Visually verify robot and scene state. "
        "The discarded preflight will not be reused; RUN starts with policy reset and a fresh observation.",
        "RUN",
    )


def _confirm_goal_transition(
    actuator: SafeG1Dex3Actuator,
    message: str,
    required: str,
) -> str:
    """Confirm a goal transition while honoring global HOLD/release controls."""

    print(message)
    response = _readline_while_armed(
        actuator,
        f"Type {required} to continue (hold/h or quit/q): ",
        timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
    )
    if response == required:
        return "continue"
    lowered = response.lower()
    if lowered in HOLD_COMMANDS:
        actuator.hold()
        LOGGER.warning("Goal transition cancelled; remaining in powered HOLD")
        return "hold"
    if lowered in EXIT_COMMANDS:
        return "release"
    raise DeploymentError(f"Expected {required}; releasing command authority")


def confirm_policy_warm_start(actuator: SafeG1Dex3Actuator, delta_summary: str) -> str:
    return _confirm_goal_transition(
        actuator,
        "\nPOLICY WARM-START WILL MOVE THE ROBOT to the first target from a fresh policy "
        f"inference ({delta_summary}). The transition is rate-bounded but is joint-space only "
        "and not collision-aware. The inferred chunk will be discarded afterward.",
        "WARMSTART",
    )


def confirm_policy_resume(actuator: SafeG1Dex3Actuator) -> str:
    return _confirm_goal_transition(
        actuator,
        "\nPolicy warm-start converged. Visually verify the robot and scene. RESUME resets "
        "GR00T, captures a fresh observation, and restores normal action-step limits.",
        "RESUME",
    )


def show_camera_preview(rgb: np.ndarray, depth_gray: np.ndarray | None) -> None:
    """Display exactly the decoded image arrays being placed in the observation."""

    try:
        cv2.imshow(PREVIEW_WINDOWS[0], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if depth_gray is not None:
            cv2.imshow(PREVIEW_WINDOWS[1], cv2.cvtColor(depth_gray, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise DeploymentError("Camera preview closed by user")
    except cv2.error as exc:
        raise DeploymentError("Camera preview could not open; check DISPLAY/desktop access") from exc


def close_camera_preview() -> None:
    for window in PREVIEW_WINDOWS:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def infer_chunk(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    instruction: str,
    model_contract: ModelContract,
    execution_horizon: int,
    actuator: SafeG1Dex3Actuator | None = None,
    camera_timeout_s: float = 0.5,
    validate_initial_step: bool = True,
    show_camera: bool = False,
    allow_custom_instruction: bool = False,
) -> tuple[ActionChunk, float]:
    if actuator is not None:
        actuator.heartbeat()
    state = state_reader.read(timeout_s=0.5)
    if actuator is not None:
        actuator.heartbeat()
    images = camera.read(timeout_s=camera_timeout_s)
    if show_camera:
        show_camera_preview(images.rgb, images.depth_gray)
    if actuator is not None:
        actuator.heartbeat()

    observation = make_observation(
        images.rgb,
        state.arm,
        state.left_hand,
        state.right_hand,
        instruction,
        video_keys=model_contract.video_keys,
        depth_gray=images.depth_gray,
        allow_custom_instruction=allow_custom_instruction,
    )
    started = time.monotonic()
    action = policy.get_action(observation)
    inference_s = time.monotonic() - started
    if actuator is not None:
        actuator.assert_healthy()
    chunk = parse_action_chunk(
        action,
        model_horizon=model_contract.action_horizon,
        execution_horizon=execution_horizon,
        current_arm=state.arm,
        current_left=state.left_hand,
        current_right=state.right_hand,
        validate_initial_step=validate_initial_step,
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


def _prepare_policy_goal(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    actuator: SafeG1Dex3Actuator,
    instruction: str,
    contract: ModelContract,
    args: argparse.Namespace,
    *,
    allow_custom_instruction: bool,
) -> str:
    """Reset a goal session and optionally perform its guarded warm-start."""

    policy.reset()
    if not getattr(args, "policy_warm_start", False):
        return "ready"

    warm_start_chunk, inference_s = infer_chunk(
        policy,
        state_reader,
        camera,
        instruction,
        contract,
        args.execution_horizon,
        actuator,
        validate_initial_step=False,
        show_camera=getattr(args, "show_camera", False),
        allow_custom_instruction=allow_custom_instruction,
    )
    delta_summary = chunk_delta_summary(warm_start_chunk, state_reader)
    LOGGER.warning(
        "Fresh policy warm-start target inferred in %.3fs: %s",
        inference_s,
        delta_summary,
    )
    decision = confirm_policy_warm_start(actuator, delta_summary)
    if decision in {"hold", "release"}:
        return decision
    actuator.warm_start(warm_start_chunk)
    LOGGER.warning("Policy warm-start reached the first target; inferred chunk discarded")
    decision = confirm_policy_resume(actuator)
    if decision in {"hold", "release"}:
        return decision
    policy.reset()
    return "ready"


def _active_command_action(command: str | None) -> str | None:
    if command is None or not command:
        return None
    lowered = command.lower()
    if lowered in HOLD_COMMANDS:
        return "hold"
    if lowered in EXIT_COMMANDS:
        return "release"
    LOGGER.warning(
        "Ignoring active-run terminal input %r; use hold/h or quit/q",
        command,
    )
    return None


def _run_active_goal(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    actuator: SafeG1Dex3Actuator,
    task_name: str,
    instruction: str,
    contract: ModelContract,
    args: argparse.Namespace,
    *,
    allow_custom_instruction: bool,
) -> str:
    """Run one finite goal; return ``hold``, ``release``, or ``complete``."""

    for chunk_number in range(1, args.max_chunks + 1):
        action = _active_command_action(_poll_active_command())
        if action == "hold":
            actuator.hold()
            LOGGER.warning("Goal %r entered powered HOLD", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release")
            return "release"

        chunk, inference_s = infer_chunk(
            policy,
            state_reader,
            camera,
            instruction,
            contract,
            args.execution_horizon,
            actuator,
            show_camera=getattr(args, "show_camera", False),
            allow_custom_instruction=allow_custom_instruction,
        )

        # A command typed during synchronous inference is honored before the
        # newly returned chunk can be submitted. The request cannot be
        # cancelled, but its result is discarded and no further request is made.
        action = _active_command_action(_poll_active_command())
        if action == "hold":
            actuator.hold()
            LOGGER.warning("Goal %r inferred chunk discarded; entered powered HOLD", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release; inferred chunk discarded")
            return "release"

        sequence = actuator.submit(chunk)
        actuator.wait_completed(sequence, timeout_s=args.execution_horizon / CONTROL_HZ + 1.0)
        LOGGER.info(
            "Completed live chunk %d/%d for %r (%d actions, inference %.3fs)",
            chunk_number,
            args.max_chunks,
            task_name,
            chunk.length,
            inference_s,
        )
        action = _active_command_action(_poll_active_command())
        if action == "hold":
            actuator.hold()
            LOGGER.warning("Goal %r entered powered HOLD", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release")
            return "release"
    return "complete"


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    repository_root = Path(__file__).resolve().parents[2]
    os.chdir(repository_root)
    task_name, instruction = select_instruction(args.task, getattr(args, "custom_goal", None))
    allow_custom_instruction = task_name == "custom-goal"
    if allow_custom_instruction and instruction not in TASKS.values():
        LOGGER.warning(
            "Using custom goal text that was not selected from the exact trained-task allowlist: %r",
            instruction,
        )
        confirm_custom_goal(instruction)
    initialization = load_initialization_spec(
        getattr(args, "initialization", "measured"),
        task_name=task_name,
        pose_file=getattr(args, "initial_pose_file", None),
    )
    image_host = args.image_host or ("127.0.0.1" if args.sim else "192.168.123.164")

    policy: Gr00tClient | None = None
    camera: TeleimagerCamera | None = None
    state_reader: G1Dex3StateReader | None = None
    actuator: SafeG1Dex3Actuator | None = None
    cleanup_error: Exception | None = None
    try:
        policy = Gr00tClient(args.policy_host, args.policy_port)
        if not policy.ping():
            raise DeploymentError(f"GR00T server at {args.policy_host}:{args.policy_port} did not answer ping")
        contract = validate_model_contract(policy.get_modality_config())
        depth_encoding = validate_policy_metadata(
            policy.get_policy_metadata(),
            requires_depth=contract.requires_depth,
        )
        if args.execution_horizon > contract.action_horizon:
            raise DeploymentError(
                f"Checkpoint action horizon is only {contract.action_horizon}, but "
                f"{args.execution_horizon} steps were requested"
            )
        LOGGER.info(
            "GR00T contract verified: video=%s + four G1/Dex3 state/action keys, horizon %d",
            ",".join(contract.video_keys),
            contract.action_horizon,
        )

        initialize_dds(args.sim, args.network_interface)
        state_reader = G1Dex3StateReader(simulation=args.sim)
        camera = TeleimagerCamera(image_host, depth_encoding=depth_encoding)
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
            contract,
            args.execution_horizon,
            camera_timeout_s=3.0,
            show_camera=getattr(args, "show_camera", False),
            allow_custom_instruction=allow_custom_instruction,
            validate_initial_step=not (
                args.actuate and (initialization.moves or getattr(args, "policy_warm_start", False))
            ),
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
                    contract,
                    args.execution_horizon,
                    show_camera=getattr(args, "show_camera", False),
                    allow_custom_instruction=allow_custom_instruction,
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

        confirm_initialization(actuator, initialization)
        actuator.initialize(initialization)
        LOGGER.warning("Initialization completed: %s", initialization.label)
        confirm_policy_start(actuator, initialization)

        # The publisher-free result predates initialization and is deliberately
        # discarded. Every initial or replacement goal resets, re-observes,
        # warm-starts from the held pose, discards that chunk, then resets again.
        if sys.stdin.isatty():
            print(
                "\nACTIVE GOAL CONTROLS: hold or h enters powered HOLD at a chunk boundary; "
                "quit or q releases authority and exits. Ctrl-C also releases from any state."
            )

        while True:
            preparation = _prepare_policy_goal(
                policy,
                state_reader,
                camera,
                actuator,
                instruction,
                contract,
                args,
                allow_custom_instruction=allow_custom_instruction,
            )
            if preparation == "release":
                LOGGER.warning("Operator requested orderly authority release during goal transition")
                break
            if preparation == "hold":
                outcome = "hold"
            else:
                outcome = _run_active_goal(
                    policy,
                    state_reader,
                    camera,
                    actuator,
                    task_name,
                    instruction,
                    contract,
                    args,
                    allow_custom_instruction=allow_custom_instruction,
                )
            if outcome != "hold":
                break

            next_goal = _select_next_goal_while_holding(actuator)
            if next_goal is None:
                LOGGER.warning("Operator requested orderly authority release from HOLD")
                break
            task_name, instruction = next_goal
            allow_custom_instruction = task_name == "custom-goal"
            LOGGER.warning("Next goal accepted while holding: %r — %s", task_name, instruction)
    finally:
        active_exception = sys.exc_info()[1]
        active_error = active_exception is not None
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
        if getattr(args, "show_camera", False):
            close_camera_preview()
        if cleanup_error is not None and (
            not active_error or isinstance(active_exception, (KeyboardInterrupt, OperatorRelease))
        ):
            raise DeploymentError(f"Actuator cleanup failed: {cleanup_error}") from cleanup_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colour/RGBD GR00T runner for Unitree G1-29 + Dex3")
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument("--task", choices=tuple(TASKS), help="Trained task ID; omit for a menu")
    goal_group.add_argument(
        "--custom-goal",
        metavar="TEXT",
        help="Send custom language goal text instead of one of the exact trained task strings",
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument(
        "--image-host",
        help="TeleImager server host (default: robot PC2, or 127.0.0.1 with --sim)",
    )
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="Show each decoded RGB/depth frame actually placed in the GR00T observation",
    )
    parser.add_argument(
        "--network-interface",
        help="CycloneDDS interface (real: explicit robot NIC; IsaacLab actuation: omit and isolate host)",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_MODES,
        default="measured",
        help=(
            "Pose before policy execution: preserve measured q (default), target XR's arm+hand "
            "joint-zero home with guarded motion, or load an experimental reviewed task-bound JSON pose"
        ),
    )
    parser.add_argument(
        "--initial-pose-file",
        help="Reviewed initialization JSON; required only with --initialization pose-file",
    )
    parser.add_argument(
        "--policy-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For actuated runs, smoothly reach one fresh first policy target after RUN, "
            "discard that chunk, reset/re-observe, then enforce normal limits (default: enabled)"
        ),
    )
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
    except OperatorRelease:
        LOGGER.warning("Operator requested orderly command-authority release")
    except KeyboardInterrupt:
        LOGGER.warning("Stop requested")
    except (DeploymentError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
