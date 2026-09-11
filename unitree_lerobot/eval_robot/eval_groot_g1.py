#!/usr/bin/env python3
"""Run a supported colour or colour+aligned-depth-derived GR00T policy on G1.

The default is read-only shadow mode.  ``--actuate`` creates a separate,
watchdog-owning DDS actuator process only after model, camera, state, and action
preflight checks and an interactive confirmation.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import select
import signal
import sys
import termios
import threading
import time
from typing import Callable

import cv2
import numpy as np

from unitree_lerobot.eval_robot.groot_client import DeploymentError, Gr00tClient
from unitree_lerobot.eval_robot.g1_end_effectors import get_end_effector_profile
from unitree_lerobot.eval_robot.groot_contract import (
    CONTROL_HZ,
    INITIALIZATION_MODES,
    TASKS,
    ActionChunk,
    DepthEncodingContract,
    InitializationSpec,
    ModelContract,
    SurfaceNormalEncodingContract,
    load_initialization_spec,
    make_observation,
    parse_action_chunk,
    parse_action_plan,
    validate_action_chunk,
    validate_model_contract,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.run_logging import (
    configure_process_logging,
    create_run_directory,
    write_json,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
    G1Dex3StateReader,
    HandFeedbackOperatorHold,
    HandFeedbackReplan,
    ImmediateControlEvent,
    RtcTerminalEvent,
    SafeG1Dex3Actuator,
    TeleimagerCamera,
    initialize_dds,
)
from unitree_lerobot.eval_robot.robot_control.g1_inspire_dfx import G1InspireDfxStateReader
from unitree_lerobot.eval_robot.training_start_pose import (
    TRAINING_START_SOURCE,
    training_start_spec,
)
from unitree_lerobot.eval_robot.voice_command_server import (
    DEFAULT_VOICE_PORT,
    VoiceCommandServer,
    load_voice_session_token,
)
from unitree_lerobot.utils.depth_encoding import DEPTH_OUTPUT_KEY
from unitree_lerobot.utils.surface_normal_encoding import SURFACE_NORMAL_OUTPUT_KEY


LOGGER = logging.getLogger("eval_groot_g1")
LOCAL_POLICY_HOSTS = {"127.0.0.1", "localhost"}
OPERATOR_CONFIRMATION_TIMEOUT_S = 180.0
ALT_ESCAPE_WINDOW_S = 0.1
GOAL_MODE_TOGGLE = "\t"
RETURN_TO_START = "\x1b[Z"
PREVIEW_WINDOWS = (
    "GR00T input: ego_view",
    f"GR00T input: {DEPTH_OUTPUT_KEY}",
    f"GR00T input: {SURFACE_NORMAL_OUTPUT_KEY}",
)
STOP_COMMANDS = {"stop", "s"}
EXIT_COMMANDS = {"quit", "q"}
VOICE_STOP_COMMANDS = {"stop", "pause"}
VOICE_EXIT_COMMANDS = {"quit"}
VOICE_RETURN_TO_START_COMMAND = "return to start"
VOICE_INPUT_AVAILABLE = "\x00voice-input-available"
VOICE_STAY_HOLDING = "\x00voice-stay-holding"
VOICE_RETURN_TO_START_UNAVAILABLE = "\x00voice-return-to-start-unavailable"
RTC_MIN_MODEL_HORIZON = 32
RTC_DELAY_HISTORY = 8


class OperatorRelease(Exception):
    """Internal control flow for an orderly operator-requested release."""


class OperatorStop(Exception):
    """Internal control flow for an immediate powered operator STOP."""


class _OperatorTerminal:
    """Capture active-motion ``s``/``q`` keys without requiring Enter.

    Raw key capture is enabled only while a policy goal is actively executing.
    Armed line prompts use their own explicit q/Alt-Q/uppercase-S mapping.
    The reader thread performs only thread-safe event signalling; it never
    consumes actuator status acknowledgements.
    """

    def __init__(self, actuator: SafeG1Dex3Actuator, *, stop_enabled: bool = True):
        self._actuator = actuator
        self._stop_enabled = stop_enabled
        self._commands: queue.Queue[str | BaseException] = queue.Queue()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved_attributes: list[object] | None = None

    @property
    def enabled(self) -> bool:
        return self._thread is not None

    def __enter__(self) -> "_OperatorTerminal":
        if not sys.stdin.isatty():
            return self
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            active = saved.copy()
            active[6] = saved[6][:]
            active[0] &= ~termios.IXON
            active[3] &= ~(termios.ICANON | termios.ECHO)
            active[6][termios.VMIN] = 1
            active[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, active)
        except (AttributeError, OSError, TypeError, ValueError, termios.error) as exc:
            raise DeploymentError("Immediate operator keys require an interactive POSIX terminal") from exc
        self._fd = fd
        self._saved_attributes = saved
        stop_sent = False
        release_sent = False
        # Drain keys that landed in the tiny transition from the preceding
        # prompt before starting the reader thread. This ensures a rapid rq
        # cannot launch a motion before q becomes visible to the operator
        # control path.
        try:
            while True:
                readable, _, _ = select.select([fd], [], [], 0.0)
                if not readable:
                    break
                value = os.read(fd, 1)
                if value == b"":
                    raise DeploymentError("stdin closed while command authority was active")
                stop_sent = self._handle_key(value, stop_sent)
                if value in {b"q", b"Q", b"\x11"}:
                    release_sent = True
                    break
        except (OSError, ValueError, DeploymentError) as exc:
            termios.tcsetattr(fd, termios.TCSANOW, saved)
            self._actuator.request_immediate_release()
            raise DeploymentError("Could not poll buffered operator keys") from exc
        if release_sent:
            return self
        self._thread = threading.Thread(
            target=self._read_keys,
            name="groot-operator-keys",
            daemon=True,
            args=(stop_sent,),
        )
        self._thread.start()
        return self

    def _read_keys(self, stop_sent: bool = False) -> None:
        assert self._fd is not None
        try:
            while not self._stopping.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.05)
                if not readable:
                    continue
                value = os.read(self._fd, 1)
                if value == b"":
                    raise DeploymentError("stdin closed while command authority was active")
                stop_sent = self._handle_key(value, stop_sent)
                if value in {b"q", b"Q", b"\x11"}:
                    return
        except BaseException as exc:
            # Input failure is fail-closed: begin release even if the main
            # thread is blocked in camera or policy I/O.
            try:
                self._actuator.request_immediate_release()
            finally:
                self._commands.put(exc)

    def _handle_key(self, value: bytes, stop_sent: bool) -> bool:
        if self._stop_enabled and value in {b"s", b"S"} and not stop_sent:
            self._actuator.request_immediate_hold()
            self._commands.put("hold")
            return True
        if value in {b"q", b"Q", b"\x11"}:
            self._actuator.request_immediate_release()
            self._commands.put("release")
        return stop_sent

    def poll_control(self) -> str | None:
        result: str | None = None
        while True:
            try:
                value = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(value, BaseException):
                if isinstance(value, DeploymentError):
                    raise value
                raise DeploymentError(f"Immediate operator-key reader failed: {value}") from value
            if value == "release":
                result = "release"
            elif result is None:
                result = "hold"
        return result

    def __exit__(self, *_: object) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        # Close the tiny boundary between the active runner's final poll and
        # terminal restoration. If s/q was already buffered, honor it before
        # command authority can transition to the next state.
        if self._fd is not None and (self._thread is None or not self._thread.is_alive()):
            stop_sent = False
            while True:
                try:
                    readable, _, _ = select.select([self._fd], [], [], 0.0)
                except (OSError, ValueError):
                    break
                if not readable:
                    break
                value = os.read(self._fd, 1)
                if value == b"":
                    break
                stop_sent = self._handle_key(value, stop_sent)
        if self._fd is not None and self._saved_attributes is not None:
            try:
                # Preserve a q/Q that lands just after the final drain so the
                # next armed phase can still honor it without Enter.
                termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_attributes)
            except (OSError, ValueError, termios.error) as exc:
                self._actuator.request_immediate_release()
                raise DeploymentError("Could not restore the operator terminal; releasing authority") from exc


def _readline_before_authority(
    prompt: str,
    *,
    confirmation_mode: bool = False,
    external_pending: Callable[[], bool] | None = None,
) -> str:
    """Read unarmed text or an r/s/q single-key confirmation."""

    if not sys.stdin.isatty():
        print(prompt, end="", flush=True)
        if external_pending is not None:
            while True:
                if external_pending():
                    print()
                    return VOICE_INPUT_AVAILABLE
                try:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                except (OSError, TypeError, ValueError) as exc:
                    raise DeploymentError("Could not poll terminal input") from exc
                if readable:
                    break
        response = sys.stdin.readline()
        if response == "":
            raise DeploymentError("stdin closed while waiting for operator input")
        response = response.strip()
        if not confirmation_mode:
            return response
        lowered = response.lower()
        if lowered == "r":
            return "continue"
        if lowered == "s":
            return "stop"
        if lowered == "q":
            raise OperatorRelease
        raise DeploymentError("Expected single-key r to continue, s to cancel, or q to quit")
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        active = saved.copy()
        active[6] = saved[6][:]
        active[0] &= ~termios.IXON
        active[3] &= ~(termios.ICANON | termios.ECHO)
        active[6][termios.VMIN] = 1
        active[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, active)
    except (AttributeError, OSError, TypeError, ValueError, termios.error) as exc:
        print()
        raise DeploymentError("Operator input requires an interactive POSIX terminal") from exc

    print(prompt, end="", flush=True)
    entered = bytearray()
    try:
        while True:
            if external_pending is not None and external_pending():
                print()
                return VOICE_INPUT_AVAILABLE
            try:
                readable, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, TypeError, ValueError) as exc:
                raise DeploymentError("Could not poll operator input") from exc
            if not readable:
                continue
            value = os.read(fd, 1)
            if value == b"":
                raise DeploymentError("stdin closed while waiting for operator input")
            if value in {b"q", b"Q", b"\x11"}:
                print()
                raise OperatorRelease
            if confirmation_mode:
                if value in {b"r", b"R"}:
                    print()
                    return "continue"
                if value in {b"s", b"S"}:
                    print()
                    return "stop"
                # Confirmation prompts deliberately ignore every other key;
                # no line or Enter key is required.
                continue
            if value in {b"\r", b"\n"}:
                print()
                try:
                    return entered.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise DeploymentError("Operator input is not valid UTF-8") from exc
            if value in {b"\x08", b"\x7f"}:
                if entered:
                    entered.pop()
                    print("\b \b", end="", flush=True)
                continue
            if value >= b" " and value != b"\x7f":
                entered.extend(value)
                print(value.decode("utf-8", errors="ignore"), end="", flush=True)
    finally:
        # Do not discard a q that arrived immediately after r. The next
        # operator-key monitor must be able to release before any motion starts.
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def _confirm_before_authority(prompt: str) -> str:
    """Return ``continue``/``stop`` from one r/s key; q raises release."""

    return _readline_before_authority(prompt, confirmation_mode=True)


def _run_blocking_motion_with_immediate_release(
    actuator: SafeG1Dex3Actuator,
    operation: Callable[[], None],
) -> None:
    """Keep q/Q live while an initialization-style motion blocks the main thread."""

    with _OperatorTerminal(actuator, stop_enabled=False) as terminal:
        if terminal.poll_control() == "release":
            raise OperatorRelease
        try:
            operation()
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                raise OperatorRelease from exc
            raise
    checker = getattr(actuator, "immediate_control_requested", None)
    if callable(checker) and checker() == "release":
        raise OperatorRelease


@dataclass(frozen=True)
class _RtcRequest:
    generation: int
    observation: dict[str, object]
    options: dict[str, object]


@dataclass(frozen=True)
class _RtcResponse:
    generation: int
    action: dict[str, object] | None
    inference_s: float
    error: BaseException | None


class _RtcInferenceWorker:
    """Daemon whose ZeroMQ client is created and used only in its own thread."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._requests: queue.Queue[_RtcRequest | None] = queue.Queue(maxsize=1)
        self._responses: queue.Queue[_RtcResponse] = queue.Queue()
        self._lock = threading.Lock()
        self._busy = False
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="groot-rtc-inference", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        client: Gr00tClient | None = None
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                started = time.monotonic()
                try:
                    if client is None:
                        client = Gr00tClient(self._host, self._port)
                    action = client.get_action(request.observation, request.options)
                except BaseException as exc:
                    response = _RtcResponse(request.generation, None, time.monotonic() - started, exc)
                else:
                    response = _RtcResponse(request.generation, action, time.monotonic() - started, None)
                self._responses.put(response)
                with self._lock:
                    self._busy = False
                if self._closing.is_set():
                    return
        finally:
            if client is not None:
                client.close()

    def submit(self, request: _RtcRequest) -> None:
        with self._lock:
            if self._busy:
                raise DeploymentError("An RTC inference request is already active")
            self._busy = True
        try:
            self._requests.put_nowait(request)
        except queue.Full as exc:
            with self._lock:
                self._busy = False
            raise DeploymentError("RTC inference request queue is full") from exc

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def poll(self) -> _RtcResponse | None:
        try:
            return self._responses.get_nowait()
        except queue.Empty:
            return None

    def close(self, *, wait: bool) -> None:
        self._closing.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            # A bounded request is in flight. The daemon may finish after an
            # immediate q/Ctrl-C release; it cannot keep the process alive.
            pass
        if wait:
            self._thread.join(timeout=6.0)


def select_instruction(
    task_name: str | None,
    custom_goal: str | None = None,
    *,
    voice_server: VoiceCommandServer | None = None,
    confirm_voice_text: bool = False,
) -> tuple[str, str]:
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
    while True:
        if voice_server is not None:
            voice_goal = _take_initial_voice_command(voice_server, confirm_voice_text=confirm_voice_text)
            if voice_goal == VOICE_STAY_HOLDING:
                continue
            if isinstance(voice_goal, tuple):
                return voice_goal
        response = _readline_before_authority(
            "Task number: ",
            external_pending=(None if voice_server is None else lambda: voice_server.command_pending),
        )
        if response == VOICE_INPUT_AVAILABLE:
            continue
        try:
            selected = int(response)
            name = task_names[selected - 1]
        except (EOFError, ValueError, IndexError) as exc:
            raise DeploymentError("A valid trained task must be selected") from exc
        return name, TASKS[name]


def confirm_custom_goal(instruction: str) -> None:
    print(
        f"\nCUSTOM GOAL IS NOT AN EXACT TRAINING INSTRUCTION:\n  {instruction}\n"
        "Behavior may be outside the fine-tuning distribution. All normal motion checks remain active."
    )
    response = _readline_before_authority("Type YES to send this custom goal to GR00T: ")
    if response != "YES":
        raise DeploymentError("Custom goal was not confirmed")


def resolve_runtime_goal(response: str) -> tuple[str, str] | None:
    """Resolve only an allowlisted goal from the powered-HOLD task menu."""

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
    return None


def _voice_warning(goal: str) -> str:
    lowered = goal.casefold()
    if lowered in VOICE_STOP_COMMANDS:
        return "Confirmed STOP/PAUSE will keep the robot in powered HOLD."
    if lowered in VOICE_EXIT_COMMANDS:
        return "Confirmed QUIT will release command authority and exit."
    if lowered == VOICE_RETURN_TO_START_COMMAND:
        return "Confirmed RETURN TO START will request the configured guarded startup target when enabled."
    return (
        "The robot is entering powered HOLD. Confirm only after reviewing the complete goal; "
        "custom language may be outside the fine-tuning distribution."
    )


def _start_voice_server(
    args: argparse.Namespace,
    *,
    on_proposal: Callable[[], None] | None = None,
) -> VoiceCommandServer:
    token_path = getattr(args, "voice_session_token_file", None)
    try:
        token = load_voice_session_token(token_path)
        server = VoiceCommandServer(
            getattr(args, "voice_listen_host", "0.0.0.0"),
            int(getattr(args, "voice_port", DEFAULT_VOICE_PORT)),
            token,
            warning_factory=_voice_warning,
            on_proposal=on_proposal,
        )
        server.start()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DeploymentError(f"Could not start voice command server: {exc}") from exc
    host, port = server.address
    LOGGER.info("Voice command server listening on TCP %s:%d", host, port)
    return server


def _resolve_confirmed_voice_goal(
    goal: str,
    *,
    allow_return_to_start: bool = False,
) -> tuple[str, str] | str:
    lowered = goal.casefold()
    if lowered in VOICE_STOP_COMMANDS:
        return VOICE_STAY_HOLDING
    if lowered in VOICE_EXIT_COMMANDS:
        return "release"
    if lowered == VOICE_RETURN_TO_START_COMMAND:
        return RETURN_TO_START if allow_return_to_start else VOICE_RETURN_TO_START_UNAVAILABLE
    trained = resolve_runtime_goal(goal)
    if trained is not None:
        return trained
    return select_instruction(None, goal)


def _take_initial_voice_command(
    voice_server: VoiceCommandServer,
    *,
    confirm_voice_text: bool,
) -> tuple[str, str] | str | None:
    command = voice_server.poll_command()
    if command is None:
        return None
    if confirm_voice_text:
        response = _readline_before_authority(
            f"Voice text: {command.goal!r}. Type YES to accept locally: "
        )
        if response != "YES":
            voice_server.reject_command(command, "local_confirmation_rejected", "Local text confirmation rejected")
            return VOICE_STAY_HOLDING
    resolved = _resolve_confirmed_voice_goal(command.goal)
    if resolved == VOICE_RETURN_TO_START_UNAVAILABLE:
        voice_server.reject_command(
            command,
            "return_to_start_disabled",
            "Return to Start is not enabled for this run",
        )
        return VOICE_STAY_HOLDING
    if not voice_server.accept_command(command):
        LOGGER.warning("Voice proposal disconnected before it could be accepted")
        return VOICE_STAY_HOLDING
    if resolved == "release":
        raise OperatorRelease
    if resolved == VOICE_STAY_HOLDING:
        LOGGER.info("Voice STOP/PAUSE accepted before command authority; remaining stationary")
    return resolved


def validate_args(args: argparse.Namespace) -> None:
    end_effector = getattr(args, "end_effector", "dex3")
    if end_effector not in {"dex3", "inspire-dfx"}:
        raise DeploymentError("--end-effector must be dex3 or inspire-dfx")
    if end_effector == "inspire-dfx":
        if args.actuate:
            raise DeploymentError(
                "Inspire DFX is currently shadow/read-only only: live actuation is fail-closed "
                "until its motion envelope, gravity payload, and command-loss behavior are qualified"
            )
        if args.sim:
            raise DeploymentError("Inspire DFX simulation is not qualified in this guarded client")
        if getattr(args, "gravity_feedforward", True):
            raise DeploymentError(
                "Inspire DFX cannot use the Dex3-payload gravity model; pass --no-gravity-feedforward"
            )
        if bool(getattr(args, "warmup1", False)):
            raise DeploymentError(
                "Inspire DFX has no reviewed training-frame Warmup1 pose; pass --no-warmup1"
            )
    if args.execution_horizon < 1:
        raise DeploymentError("--execution-horizon must be at least 1")
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
    return_to_start = bool(getattr(args, "return_to_start", False))
    warmup1_enabled = bool(getattr(args, "warmup1", False))
    if return_to_start and not args.actuate:
        raise DeploymentError("--return-to-start requires --actuate")
    if return_to_start and not warmup1_enabled and initialization == "measured":
        raise DeploymentError(
            "--return-to-start has no fixed target with --no-warmup1 and "
            "--initialization measured; enable Warmup1 or select xr-home/pose-file"
        )
    inference_mode = getattr(args, "inference_mode", "synchronous")
    if inference_mode not in {"synchronous", "rtc"}:
        raise DeploymentError("--inference-mode must be synchronous or rtc")
    rtc_frozen_steps = getattr(args, "rtc_frozen_steps", None)
    if rtc_frozen_steps is not None and rtc_frozen_steps < 1:
        raise DeploymentError("--rtc-frozen-steps must be at least 1")
    rtc_ramp_rate = getattr(args, "rtc_ramp_rate", None)
    if rtc_ramp_rate is not None and (not np.isfinite(rtc_ramp_rate) or rtc_ramp_rate <= 0.0):
        raise DeploymentError("--rtc-ramp-rate must be a finite positive number")
    if inference_mode != "rtc" and (rtc_frozen_steps is not None or rtc_ramp_rate is not None):
        raise DeploymentError("RTC tuning options require --inference-mode rtc")
    command_conditioning = getattr(args, "command_conditioning", "xr")
    if command_conditioning not in {"none", "xr"}:
        raise DeploymentError("--command-conditioning must be none or xr")
    voice_enabled = bool(getattr(args, "voice", False))
    if getattr(args, "confirm_text", False) and not voice_enabled:
        raise DeploymentError("--confirm-text requires --voice")
    voice_port = int(getattr(args, "voice_port", DEFAULT_VOICE_PORT))
    if not 1 <= voice_port <= 65535:
        raise DeploymentError("--voice-port must be in 1..65535")
    if voice_enabled and not str(getattr(args, "voice_listen_host", "")).strip():
        raise DeploymentError("--voice-listen-host cannot be empty")


def confirm_actuation(simulation: bool, task_name: str, instruction: str) -> None:
    if not sys.stdin.isatty():
        raise DeploymentError("Actuation requires an interactive TTY so immediate s/q operator keys are available")
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
    response = _confirm_before_authority(
        f"Press r to confirm {required} and create command publishers (no Enter); s/q cancels: "
    )
    if response != "continue":
        raise OperatorRelease


def _readline_while_armed(
    actuator: SafeG1Dex3Actuator,
    prompt: str,
    *,
    timeout_s: float | None,
    confirmation_mode: bool = False,
    goal_mode_toggle: bool = False,
    return_to_start: bool = False,
    external_pending: Callable[[], bool] | None = None,
) -> str:
    """Read a terminal line while continuously servicing the actuator watchdog."""

    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    if sys.stdin.isatty():
        return _readline_with_immediate_prompt_controls(
            actuator,
            prompt,
            deadline,
            confirmation_mode=confirmation_mode,
            goal_mode_toggle=goal_mode_toggle,
            return_to_start=return_to_start,
            external_pending=external_pending,
        )
    LOGGER.warning("stdin is not a TTY; armed input is line-buffered and immediate keys are unavailable")
    print(prompt, end="", flush=True)
    while True:
        actuator.heartbeat()
        actuator.assert_healthy()
        if external_pending is not None and external_pending():
            print()
            return VOICE_INPUT_AVAILABLE
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
        raw_response = sys.stdin.readline()
        if raw_response == "":
            raise DeploymentError("stdin closed while command authority was active")
        if goal_mode_toggle and raw_response.rstrip("\r\n") == GOAL_MODE_TOGGLE:
            return GOAL_MODE_TOGGLE
        if return_to_start and raw_response.rstrip("\r\n") == RETURN_TO_START:
            return RETURN_TO_START
        response = raw_response.strip()
        if not confirmation_mode:
            return response
        lowered = response.lower()
        if lowered == "r":
            return "continue"
        if lowered == "s":
            return "stop"
        if lowered == "q":
            return "q"
        raise DeploymentError("Expected single-key r to continue, s to STOP, or q to release")


def _readline_with_immediate_prompt_controls(
    actuator: SafeG1Dex3Actuator,
    prompt: str,
    deadline: float | None,
    *,
    confirmation_mode: bool = False,
    goal_mode_toggle: bool = False,
    return_to_start: bool = False,
    external_pending: Callable[[], bool] | None = None,
) -> str:
    """Read an armed line while keeping the physical Q key fail-safe.

    Either q/Q releases immediately. Alt-q inserts a literal lowercase q and
    Alt-Shift-q inserts a literal capital Q into custom text. Lowercase s
    remains ordinary prompt text, while uppercase S requests the powered STOP
    state. Both common Backspace encodings remain available for editing.
    """

    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        active = saved.copy()
        active[6] = saved[6][:]
        active[0] &= ~termios.IXON
        active[3] &= ~(termios.ICANON | termios.ECHO)
        active[6][termios.VMIN] = 1
        active[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, active)
    except (AttributeError, OSError, TypeError, ValueError, termios.error) as exc:
        print()
        raise DeploymentError("Armed input requires an interactive POSIX terminal") from exc

    # Display the prompt only after ICANON is disabled. Otherwise a fast key
    # pressed as the prompt appears can enter the old canonical input buffer
    # and remain unavailable until Enter is pressed.
    print(prompt, end="", flush=True)
    entered = bytearray()
    escape_prefix: bytes | None = None
    escape_prefix_at: float | None = None
    try:
        while True:
            actuator.heartbeat()
            actuator.assert_healthy()
            if external_pending is not None and external_pending():
                print()
                return VOICE_INPUT_AVAILABLE
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0.0:
                print()
                raise DeploymentError("Timed out waiting for armed operator input; releasing command authority")
            wait_s = 0.1 if remaining is None else min(0.1, remaining)
            try:
                readable, _, _ = select.select([fd], [], [], wait_s)
            except (OSError, TypeError, ValueError) as exc:
                print()
                raise DeploymentError("Could not poll the armed operator terminal") from exc
            if not readable:
                continue
            value = os.read(fd, 1)
            if value == b"":
                raise DeploymentError("stdin closed while command authority was active")
            now = time.monotonic()
            if value == b"\x1b":
                # Most POSIX terminals encode an Alt chord as ESC followed by
                # the modified character. Keep the escape window deliberately
                # short so a standalone Escape cannot suppress a later q.
                escape_prefix = value
                escape_prefix_at = now
                continue
            if escape_prefix is not None:
                prefix_is_fresh = (
                    escape_prefix_at is not None
                    and now - escape_prefix_at <= ALT_ESCAPE_WINDOW_S
                )
                if prefix_is_fresh and escape_prefix == b"\x1b" and value in {b"q", b"Q"}:
                    escape_prefix = None
                    escape_prefix_at = None
                    entered.extend(value)
                    print(value.decode("ascii"), end="", flush=True)
                    continue
                if prefix_is_fresh and return_to_start and escape_prefix == b"\x1b" and value == b"[":
                    escape_prefix = b"\x1b["
                    continue
                is_return_to_start = (
                    prefix_is_fresh
                    and return_to_start
                    and escape_prefix == b"\x1b["
                    and value == b"Z"
                )
                escape_prefix = None
                escape_prefix_at = None
                if is_return_to_start:
                    # Shift+Tab is the conventional POSIX terminal sequence
                    # ESC [ Z. Return after its final byte so a following q is
                    # left buffered for the next armed prompt.
                    print()
                    return RETURN_TO_START
            if value in {b"q", b"Q", b"\x11"}:
                actuator.request_immediate_release()
                print()
                return "q"
            if goal_mode_toggle and value == b"\t":
                # Tab cannot occur in a validated custom goal. Treat it as an
                # immediate mode switch and discard any partially typed line.
                print()
                return GOAL_MODE_TOGGLE
            if confirmation_mode and value in {b"r", b"R"}:
                print()
                return "continue"
            if value == b"S" or (confirmation_mode and value == b"s"):
                try:
                    actuator.request_immediate_hold()
                except DeploymentError:
                    # Before initialization the child is already publishing its
                    # measured target, so STOP is an idempotent prompt response.
                    pass
                print()
                return "stop"
            if confirmation_mode:
                # The three-key confirmation protocol needs no line editing
                # and deliberately ignores every unrelated key, including
                # Enter. Only r/s/q can change state.
                continue
            if value in {b"\r", b"\n"}:
                print()
                try:
                    return entered.decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    raise DeploymentError("Armed operator input is not valid UTF-8") from exc
            if value in {b"\x08", b"\x7f"}:
                if entered:
                    entered.pop()
                    print("\b \b", end="", flush=True)
                continue
            if value >= b" " and value != b"\x7f":
                entered.extend(value)
                print(value.decode("utf-8", errors="ignore"), end="", flush=True)
    finally:
        try:
            # Preserve any key that arrived just after an uppercase-S STOP.
            # In particular, a following q/Q must remain available to win at
            # the next armed read rather than being discarded during the
            # STOP-to-HOLD transition.
            termios.tcsetattr(fd, termios.TCSANOW, saved)
        except (OSError, ValueError, termios.error) as exc:
            actuator.request_immediate_release()
            raise DeploymentError("Could not restore the armed terminal; releasing authority") from exc


def _confirm_while_armed(
    actuator: SafeG1Dex3Actuator,
    message: str,
    required: str,
    *,
    stop_is_already_held: bool = False,
) -> None:
    """Wait for bounded operator confirmation without starving the heartbeat."""

    print(message)
    while True:
        try:
            response = _readline_while_armed(
                actuator,
                f"Press r to {required} (no Enter); s STOP; q release: ",
                timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
                confirmation_mode=True,
            )
        except DeploymentError as exc:
            if str(exc).startswith("Timed out waiting for armed operator input"):
                raise DeploymentError(
                    f"Timed out waiting for {required}; releasing command authority; "
                    f"OPERATOR_CONFIRMATION_TIMEOUT_S={OPERATOR_CONFIRMATION_TIMEOUT_S}"
                ) from exc
            raise
        if response == "continue":
            return
        lowered = response.lower()
        if lowered in EXIT_COMMANDS:
            raise OperatorRelease
        if lowered in STOP_COMMANDS:
            if not stop_is_already_held:
                _finish_operator_stop(actuator)
            LOGGER.info("Command remains stopped; press r to %s when ready", required)
            continue
        raise DeploymentError(f"Expected r to {required}; releasing command authority")


def _poll_active_command(terminal: _OperatorTerminal | None = None) -> str | None:
    """Return one active control without delaying the policy loop.

    Live active execution supplies ``terminal`` and receives a single raw key.
    The line-based fallback is retained for non-live callers and unit tests.
    """

    if terminal is not None:
        control = terminal.poll_control()
        if control == "hold":
            return "s"
        if control == "release":
            return "q"
        return None

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


def _take_held_voice_command(
    actuator: SafeG1Dex3Actuator,
    voice_server: VoiceCommandServer,
    *,
    confirm_voice_text: bool,
    return_to_start: bool = False,
) -> tuple[str, str] | str:
    command = voice_server.poll_command()
    if command is None:
        return VOICE_INPUT_AVAILABLE
    if confirm_voice_text:
        response = _readline_while_armed(
            actuator,
            f"Voice text: {command.goal!r}. Type YES to accept locally; S holds; q releases: ",
            timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
        )
        if response != "YES":
            voice_server.reject_command(command, "local_confirmation_rejected", "Local text confirmation rejected")
            lowered = response.casefold()
            if lowered in EXIT_COMMANDS:
                return "release"
            if lowered in STOP_COMMANDS:
                _finish_operator_stop(actuator)
            return VOICE_STAY_HOLDING

    resolved = _resolve_confirmed_voice_goal(
        command.goal,
        allow_return_to_start=return_to_start,
    )
    if resolved == VOICE_RETURN_TO_START_UNAVAILABLE:
        voice_server.reject_command(
            command,
            "return_to_start_disabled",
            "Return to Start is not enabled for this run",
        )
        LOGGER.warning("Voice RETURN TO START rejected because --return-to-start is disabled")
        return VOICE_STAY_HOLDING
    if not voice_server.accept_command(
        command,
        service_callback=lambda: (actuator.heartbeat(), actuator.assert_healthy()),
    ):
        LOGGER.warning("Voice connection closed before the confirmed command could be accepted")
        return VOICE_STAY_HOLDING
    if resolved == VOICE_STAY_HOLDING:
        _finish_operator_stop(actuator)
        LOGGER.warning("Confirmed voice STOP/PAUSE: remaining in powered HOLD")
    elif resolved == "release":
        LOGGER.warning("Confirmed voice QUIT: releasing command authority")
    else:
        LOGGER.warning("Confirmed voice goal accepted: %r", command.goal)
    return resolved


def _select_next_goal_while_holding(
    actuator: SafeG1Dex3Actuator,
    *,
    custom_goal_mode: bool = False,
    mode_state: dict[str, bool] | None = None,
    return_to_start: bool = False,
    voice_server: VoiceCommandServer | None = None,
    confirm_voice_text: bool = False,
) -> tuple[str, str] | str | None:
    """Wait in powered hold, with an explicit trained/custom goal-mode toggle."""

    print(
        "\nSTOPPED IN A POWERED POSITION HOLD. GR00T requests are stopped.\n"
        "Press q or Q to release immediately (no Enter); Alt-q types q and "
        "Alt-Shift-q types Q. Press uppercase S to remain stopped. Ctrl-C also releases."
    )

    def show_mode() -> None:
        if return_to_start:
            print("Press Shift+Tab to move back to the configured startup target.")
        if custom_goal_mode:
            print(
                "\nCUSTOM GOAL MODE (outside the exact trained-task allowlist).\n"
                "Enter custom goal text, then explicitly confirm it. "
                "Press Tab to return to trained-task options."
            )
            return
        print("\nTRAINED TASK MODE. Press Tab to enable custom-goal entry.")
        for index, (task_name, instruction) in enumerate(TASKS.items(), start=1):
            print(f"  {index}. {task_name:16s}  {instruction}")

    show_mode()
    while True:
        if voice_server is not None and voice_server.command_pending:
            voice_result = _take_held_voice_command(
                actuator,
                voice_server,
                confirm_voice_text=confirm_voice_text,
                return_to_start=return_to_start,
            )
            if voice_result in {VOICE_INPUT_AVAILABLE, VOICE_STAY_HOLDING}:
                continue
            if voice_result == "release":
                return None
            if voice_result == RETURN_TO_START:
                return RETURN_TO_START
            assert isinstance(voice_result, tuple)
            return voice_result
        prompt = "Custom goal> " if custom_goal_mode else "Trained task> "
        response = _readline_while_armed(
            actuator,
            prompt,
            timeout_s=None,
            goal_mode_toggle=True,
            return_to_start=return_to_start,
            external_pending=(None if voice_server is None else lambda: voice_server.command_pending),
        )
        if response == VOICE_INPUT_AVAILABLE:
            continue
        if response == RETURN_TO_START:
            return RETURN_TO_START
        if response == GOAL_MODE_TOGGLE:
            custom_goal_mode = not custom_goal_mode
            if mode_state is not None:
                mode_state["custom_goal_mode"] = custom_goal_mode
            show_mode()
            continue
        lowered = response.lower()
        if lowered in EXIT_COMMANDS:
            return None
        if lowered in STOP_COMMANDS or not response:
            if lowered in STOP_COMMANDS:
                _finish_operator_stop(actuator)
            LOGGER.info("Already stopped; no GR00T request was sent")
            continue

        if not custom_goal_mode:
            resolved = resolve_runtime_goal(response)
            if resolved is not None:
                return resolved
            LOGGER.warning(
                "Input %r is not a trained task; remaining in HOLD. Press Tab to enable custom goals.",
                response,
            )
            continue

        try:
            task_name, instruction = select_instruction(None, response)
        except DeploymentError as exc:
            LOGGER.warning("Invalid custom goal; remaining in powered HOLD: %s", exc)
            continue

        LOGGER.warning(
            "Proposed next goal is not an exact trained instruction: %r",
            instruction,
        )
        print(
            f"\nCUSTOM GOAL IS NOT AN EXACT TRAINING INSTRUCTION:\n  {instruction}\n"
            "Behavior may be outside the fine-tuning distribution. All normal motion checks remain active."
        )
        try:
            confirmation = _readline_while_armed(
                actuator,
                "Type YES to send this custom goal, or press Tab for trained tasks: ",
                timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
                goal_mode_toggle=True,
                return_to_start=return_to_start,
            )
        except DeploymentError as exc:
            if str(exc).startswith("Timed out waiting for armed operator input"):
                raise DeploymentError(
                    "Timed out waiting for custom-goal confirmation; releasing command authority; "
                    f"OPERATOR_CONFIRMATION_TIMEOUT_S={OPERATOR_CONFIRMATION_TIMEOUT_S}"
                ) from exc
            raise
        if confirmation == "YES":
            return task_name, instruction
        if confirmation == GOAL_MODE_TOGGLE:
            custom_goal_mode = False
            if mode_state is not None:
                mode_state["custom_goal_mode"] = False
            show_mode()
            continue
        if confirmation == RETURN_TO_START:
            return RETURN_TO_START
        if confirmation.lower() in EXIT_COMMANDS:
            return None
        if confirmation.lower() in STOP_COMMANDS:
            _finish_operator_stop(actuator)
            LOGGER.info("Custom goal was not sent; remaining STOPPED")
            continue
        LOGGER.warning("Custom goal rejected; remaining in powered hold")


def confirm_return_to_start(
    actuator: SafeG1Dex3Actuator,
    spec: InitializationSpec,
) -> str:
    """Confirm one repeatable guarded move back to the selected startup target."""

    warning = (
        f"\nRETURN TO START WILL MOVE THE ROBOT toward {spec.label!r}. "
        "This is a slow bounded joint-space path, not collision-aware planning. "
        "It reuses the startup target without rerunning authority acquisition or the "
        "one-time initialization/Warmup1 protocol stages. Keep the workspace clear and "
        "remain on the emergency stop."
    )
    if spec.moves_hands:
        warning += " This target explicitly moves both Dex3 hands; verify their contents."
    return _confirm_goal_transition(actuator, warning, "RETURN TO START")


def _run_return_to_start_from_hold(
    actuator: SafeG1Dex3Actuator,
    spec: InitializationSpec,
) -> str:
    """Run a confirmed return transition while preserving feedback-triggered HOLD."""

    try:
        decision = confirm_return_to_start(actuator, spec)
        if decision != "continue":
            return decision
        # Recheck after confirmation but before enqueueing any motion. A
        # feedback-HOLD may have arrived while the prompt was open.
        _wait_for_hand_feedback(actuator)
    except HandFeedbackOperatorHold as exc:
        LOGGER.warning(
            "Dex3 feedback crossed %.2f s during Return-to-Start; transition cancelled "
            "and robot remains in powered HOLD: %s",
            ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
            exc.detail,
            extra={"terminal_yellow": True},
        )
        return "hold"
    # Do not catch feedback-HOLD after the child has accepted this operation:
    # initialization motion may be paused internally and later resume. An
    # exception from that phase must retain the fail-closed release behavior.
    _run_blocking_motion_with_immediate_release(
        actuator,
        lambda: actuator.warmup_pose(spec),
    )
    LOGGER.warning(
        "Returned to startup target %r; remaining in powered HOLD",
        spec.label,
    )
    return "continue"


def confirm_initialization(
    actuator: SafeG1Dex3Actuator,
    spec: InitializationSpec,
    *,
    stage: str = "INITIALIZE",
) -> None:
    if not spec.moves:
        return
    warning = (
        f"\n{stage} WILL MOVE THE ROBOT using {spec.label!r}. "
        "The path is a slow bounded joint-space interpolation, not collision-aware planning. "
        "Keep the workspace clear and remain on the emergency stop."
    )
    if spec.mode == "xr-home":
        warning += " XR-home targets all 14 arm joints and both 7-joint Dex3 hands to zero; both hands must be empty."
    elif stage == "WARMUP1":
        warning += (
            " Warmup1 commands all 14 arms and both 7-joint hands from one recorded "
            "cereal-box-pick frame. Both hands must be empty. It does not reproduce the "
            "recorded legs, waist, pelvis height, or world pose."
        )
    elif spec.moves_hands:
        warning += " This pose explicitly moves one or both hands; verify their contents."
    _confirm_while_armed(
        actuator,
        warning,
        stage,
        stop_is_already_held=True,
    )


def confirm_policy_start(actuator: SafeG1Dex3Actuator, spec: InitializationSpec) -> None:
    _confirm_while_armed(
        actuator,
        f"\nStartup pose {spec.label!r} converged. Visually verify robot and scene state. "
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
    try:
        response = _readline_while_armed(
            actuator,
            f"Press r to {required} (no Enter); s STOP; q release: ",
            timeout_s=OPERATOR_CONFIRMATION_TIMEOUT_S,
            confirmation_mode=True,
        )
    except DeploymentError as exc:
        if str(exc).startswith("Timed out waiting for armed operator input"):
            raise DeploymentError(
                f"Timed out waiting for {required}; releasing command authority; "
                f"OPERATOR_CONFIRMATION_TIMEOUT_S={OPERATOR_CONFIRMATION_TIMEOUT_S}"
            ) from exc
        raise
    if response == "continue":
        return "continue"
    lowered = response.lower()
    if lowered in STOP_COMMANDS:
        _finish_operator_stop(actuator)
        LOGGER.warning("Goal transition cancelled; remaining STOPPED in powered position hold")
        return "hold"
    if lowered in EXIT_COMMANDS:
        return "release"
    raise DeploymentError(f"Expected r to {required}; releasing command authority")


def confirm_policy_warm_start(actuator: SafeG1Dex3Actuator, delta_summary: str) -> str:
    return _confirm_goal_transition(
        actuator,
        "\nWARMUP2 WILL MOVE THE ROBOT to the first target from a fresh policy "
        f"inference ({delta_summary}). The transition is rate-bounded but is joint-space only "
        "and not collision-aware. The inferred chunk will be discarded afterward.",
        "WARMUP2",
    )


def confirm_policy_continue(actuator: SafeG1Dex3Actuator) -> str:
    return _confirm_goal_transition(
        actuator,
        "\nWarmup2 converged. Visually verify the robot and scene. CONTINUE resets "
        "GR00T, captures a fresh observation, and restores normal action-step limits.",
        "CONTINUE",
    )


def show_camera_preview(
    rgb: np.ndarray,
    geometry: np.ndarray | None,
    geometry_key: str = DEPTH_OUTPUT_KEY,
) -> None:
    """Display exactly the decoded image arrays being placed in the observation."""

    try:
        cv2.imshow(PREVIEW_WINDOWS[0], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if geometry is not None:
            if geometry_key not in (DEPTH_OUTPUT_KEY, SURFACE_NORMAL_OUTPUT_KEY):
                raise DeploymentError(f"Cannot preview unsupported geometry view {geometry_key!r}")
            cv2.imshow(f"GR00T input: {geometry_key}", cv2.cvtColor(geometry, cv2.COLOR_RGB2BGR))
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


def _raise_if_immediate_control(actuator: SafeG1Dex3Actuator | None) -> None:
    if actuator is None:
        return
    checker = getattr(actuator, "immediate_control_requested", None)
    if not callable(checker):
        return
    requested = checker()
    if requested == "release":
        raise OperatorRelease
    if requested == "hold":
        raise OperatorStop


def _hand_pause_generation(actuator: SafeG1Dex3Actuator | None) -> int | None:
    if actuator is None:
        return None
    value = getattr(actuator, "hand_pause_generation", None)
    return None if value is None else int(value)


def _wait_for_hand_feedback(actuator: SafeG1Dex3Actuator | None) -> None:
    if actuator is None:
        return
    waiter = getattr(actuator, "wait_for_hand_feedback", None)
    if callable(waiter):
        waiter()
        return
    checker = getattr(actuator, "assert_healthy", None)
    if callable(checker):
        checker()


def capture_policy_observation(
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    instruction: str,
    model_contract: ModelContract,
    actuator: SafeG1Dex3Actuator | None = None,
    camera_timeout_s: float = 0.5,
    show_camera: bool = False,
    allow_custom_instruction: bool = False,
) -> tuple[dict[str, object], object]:
    if actuator is not None:
        _raise_if_immediate_control(actuator)
        waiter = getattr(actuator, "wait_for_hand_feedback", None)
        if callable(waiter):
            waiter()
        actuator.heartbeat()
    state = state_reader.read(timeout_s=0.5)
    if actuator is not None:
        _raise_if_immediate_control(actuator)
        actuator.heartbeat()
    images = camera.read(timeout_s=camera_timeout_s)
    if actuator is not None:
        waiter = getattr(actuator, "wait_for_hand_feedback", None)
        if callable(waiter):
            waiter()
        actuator.heartbeat()
    depth_gray = getattr(images, "depth_gray", None)
    surface_normals = getattr(images, "surface_normals", None)
    if show_camera:
        geometry = depth_gray if depth_gray is not None else surface_normals
        geometry_key = DEPTH_OUTPUT_KEY if depth_gray is not None else SURFACE_NORMAL_OUTPUT_KEY
        show_camera_preview(images.rgb, geometry, geometry_key)
    if actuator is not None:
        _raise_if_immediate_control(actuator)
        actuator.heartbeat()

    observation = make_observation(
        images.rgb,
        state.arm,
        state.left_hand,
        state.right_hand,
        instruction,
        video_keys=model_contract.video_keys,
        depth_gray=depth_gray,
        surface_normals=surface_normals,
        allow_custom_instruction=allow_custom_instruction,
        end_effector=getattr(model_contract, "end_effector", "dex3"),
    )
    return observation, state


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
    command_conditioning: str = "none",
    show_camera: bool = False,
    allow_custom_instruction: bool = False,
) -> tuple[ActionChunk, float]:
    while True:
        pause_generation = _hand_pause_generation(actuator)
        observation, state = capture_policy_observation(
            state_reader,
            camera,
            instruction,
            model_contract,
            actuator,
            camera_timeout_s,
            show_camera,
            allow_custom_instruction,
        )
        if actuator is not None and _hand_pause_generation(actuator) != pause_generation:
            LOGGER.warning("Discarded observation captured across a Dex3 feedback pause; recapturing")
            continue
        started = time.monotonic()
        try:
            action = policy.get_action(observation)
        except BaseException:
            _raise_if_immediate_control(actuator)
            raise
        inference_s = time.monotonic() - started
        if actuator is not None:
            _raise_if_immediate_control(actuator)
            _wait_for_hand_feedback(actuator)
            if _hand_pause_generation(actuator) != pause_generation:
                LOGGER.warning(
                    "Discarded policy response computed across a Dex3 feedback pause; "
                    "resetting and inferring from a fresh observation"
                )
                policy.reset()
                continue
        chunk = parse_action_chunk(
            action,
            model_horizon=model_contract.action_horizon,
            execution_horizon=execution_horizon,
            current_arm=state.arm,
            current_left=state.left_hand,
            current_right=state.right_hand,
            validate_initial_step=validate_initial_step,
            validate_target_steps=command_conditioning == "none",
            end_effector=getattr(model_contract, "end_effector", "dex3"),
        )
        if pause_generation is not None:
            chunk = ActionChunk(
                arm=chunk.arm,
                left_hand=chunk.left_hand,
                right_hand=chunk.right_hand,
                hand_pause_generation=pause_generation,
                end_effector=chunk.end_effector,
            )
        return chunk, inference_s


def infer_plan(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    instruction: str,
    model_contract: ModelContract,
    actuator: SafeG1Dex3Actuator | None = None,
    camera_timeout_s: float = 0.5,
    validate_initial_step: bool = True,
    command_conditioning: str = "none",
    show_camera: bool = False,
    allow_custom_instruction: bool = False,
) -> tuple[ActionChunk, float]:
    while True:
        pause_generation = _hand_pause_generation(actuator)
        observation, state = capture_policy_observation(
            state_reader,
            camera,
            instruction,
            model_contract,
            actuator,
            camera_timeout_s,
            show_camera,
            allow_custom_instruction,
        )
        if actuator is not None and _hand_pause_generation(actuator) != pause_generation:
            LOGGER.warning("Discarded observation captured across a Dex3 feedback pause; recapturing")
            continue
        started = time.monotonic()
        try:
            action = policy.get_action(observation)
        except BaseException:
            _raise_if_immediate_control(actuator)
            raise
        inference_s = time.monotonic() - started
        if actuator is not None:
            _raise_if_immediate_control(actuator)
            _wait_for_hand_feedback(actuator)
            if _hand_pause_generation(actuator) != pause_generation:
                LOGGER.warning(
                    "Discarded policy response computed across a Dex3 feedback pause; "
                    "resetting and inferring from a fresh observation"
                )
                policy.reset()
                continue
        plan = parse_action_plan(
            action,
            model_horizon=model_contract.action_horizon,
            current_arm=state.arm,
            current_left=state.left_hand,
            current_right=state.right_hand,
            validate_initial_step=validate_initial_step,
            validate_target_steps=command_conditioning == "none",
            end_effector=getattr(model_contract, "end_effector", "dex3"),
        )
        if pause_generation is not None:
            plan = ActionChunk(
                arm=plan.arm,
                left_hand=plan.left_hand,
                right_hand=plan.right_hand,
                hand_pause_generation=pause_generation,
                end_effector=plan.end_effector,
            )
        return plan, inference_s


def chunk_delta_summary(chunk: ActionChunk, state_reader: G1Dex3StateReader) -> str:
    state = state_reader.read(timeout_s=0.5)
    arm_delta = float(np.max(np.abs(chunk.arm[0] - state.arm)))
    hand_delta = float(
        max(
            np.max(np.abs(chunk.left_hand[0] - state.left_hand)),
            np.max(np.abs(chunk.right_hand[0] - state.right_hand)),
        )
    )
    hand_unit = get_end_effector_profile(chunk.end_effector).value_unit
    return f"first-target pose gap arm={arm_delta:.4f} rad, hand={hand_delta:.4f} {hand_unit}"


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
    warmup2_enabled: bool | None = None,
) -> str:
    """Reset a goal session and optionally perform its guarded warm-start."""

    try:
        # A goal selected from the 1.25 s operator-HOLD terminal is explicit
        # consent to try again, but no policy-server command may be sent until
        # three distinct paired Dex3 readings have re-established feedback.
        _wait_for_hand_feedback(actuator)
        _run_with_immediate_operator_keys(actuator, policy.reset)
        if warmup2_enabled is None:
            warmup2_enabled = bool(getattr(args, "policy_warm_start", True))
        if not warmup2_enabled:
            return "ready"

        warm_start_chunk, inference_s = _run_with_immediate_operator_keys(
            actuator,
            lambda: infer_chunk(
                policy,
                state_reader,
                camera,
                instruction,
                contract,
                args.execution_horizon,
                actuator,
                validate_initial_step=False,
                command_conditioning=getattr(args, "command_conditioning", "xr"),
                show_camera=getattr(args, "show_camera", False),
                allow_custom_instruction=allow_custom_instruction,
            ),
        )
        delta_summary = _run_with_immediate_operator_keys(
            actuator,
            lambda: chunk_delta_summary(warm_start_chunk, state_reader),
        )
    except OperatorStop:
        return "hold"
    except OperatorRelease:
        return "release"
    except HandFeedbackOperatorHold as exc:
        LOGGER.warning(
            "Dex3 feedback crossed %.2f s during goal preparation; "
            "remaining in powered HOLD: %s",
            ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
            exc.detail,
            extra={"terminal_yellow": True},
        )
        return "hold"
    LOGGER.warning(
        "Warmup2 first target inferred in %.3fs: %s",
        inference_s,
        delta_summary,
    )
    decision = confirm_policy_warm_start(actuator, delta_summary)
    if decision in {"hold", "release"}:
        return decision
    try:
        # Catch an already-pending operator HOLD before any Warmup2 motion is
        # queued. A failure after enqueue must still propagate so the child
        # cannot resume a paused transition while the parent is back at a menu.
        _wait_for_hand_feedback(actuator)
    except HandFeedbackOperatorHold as exc:
        LOGGER.warning(
            "Dex3 feedback crossed %.2f s before Warmup2 motion; "
            "remaining in powered HOLD: %s",
            ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
            exc.detail,
            extra={"terminal_yellow": True},
        )
        return "hold"
    try:
        _run_blocking_motion_with_immediate_release(
            actuator,
            lambda: actuator.warm_start(warm_start_chunk),
        )
    except HandFeedbackReplan as exc:
        LOGGER.warning(
            "Warmup2 policy target was invalidated by a Dex3 feedback pause; "
            "re-observing the same goal: %s",
            exc.detail,
            extra={"terminal_yellow": True},
        )
        return _prepare_policy_goal(
            policy,
            state_reader,
            camera,
            actuator,
            instruction,
            contract,
            args,
            allow_custom_instruction=allow_custom_instruction,
            warmup2_enabled=warmup2_enabled,
        )
    LOGGER.warning("Warmup2 reached target 0; the inferred chunk was discarded")
    decision = confirm_policy_continue(actuator)
    if decision in {"hold", "release"}:
        return decision
    try:
        _run_with_immediate_operator_keys(actuator, policy.reset)
    except OperatorStop:
        return "hold"
    except OperatorRelease:
        return "release"
    return "ready"


def _active_command_action(command: str | None) -> str | None:
    if command is None or not command:
        return None
    lowered = command.lower()
    if lowered in STOP_COMMANDS:
        return "hold"
    if lowered in EXIT_COMMANDS:
        return "release"
    LOGGER.warning(
        "Ignoring active-run terminal input %r; press s to STOP or q to release",
        command,
    )
    return None


def _finish_operator_stop(actuator: SafeG1Dex3Actuator) -> None:
    checker = getattr(actuator, "immediate_control_requested", None)
    requested = checker() if callable(checker) else None
    if requested == "release":
        raise OperatorRelease
    if requested == "hold":
        if actuator.finish_immediate_hold() == "release":
            raise OperatorRelease
    else:
        actuator.hold()


def _run_with_immediate_operator_keys(
    actuator: SafeG1Dex3Actuator,
    operation: Callable[[], object],
) -> object:
    """Run one blocking policy/client call while s/q remain immediate."""

    with _OperatorTerminal(actuator):
        try:
            result = operation()
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                raise OperatorRelease from exc
            _finish_operator_stop(actuator)
            raise OperatorStop from exc
        except OperatorStop:
            _finish_operator_stop(actuator)
            raise
    checker = getattr(actuator, "immediate_control_requested", None)
    requested = checker() if callable(checker) else None
    if requested == "release":
        raise OperatorRelease
    if requested == "hold":
        _finish_operator_stop(actuator)
        raise OperatorStop
    return result


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
    with _OperatorTerminal(actuator) as terminal:
        try:
            outcome = _run_active_goal_controlled(
                policy,
                state_reader,
                camera,
                actuator,
                task_name,
                instruction,
                contract,
                args,
                terminal,
                allow_custom_instruction=allow_custom_instruction,
            )
        except OperatorStop:
            _finish_operator_stop(actuator)
            LOGGER.warning("Goal %r STOPPED; unexecuted policy output was discarded", task_name)
            outcome = "hold"
        except OperatorRelease:
            LOGGER.warning("Operator requested immediate orderly authority release")
            outcome = "release"
        except HandFeedbackOperatorHold as exc:
            LOGGER.warning(
                "Dex3 feedback crossed %.2f s; goal %r was abandoned in powered HOLD: %s",
                ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
                task_name,
                exc.detail,
                extra={"terminal_yellow": True},
            )
            outcome = "hold"
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                outcome = "release"
            else:
                _finish_operator_stop(actuator)
                outcome = "hold"
    checker = getattr(actuator, "immediate_control_requested", None)
    requested = checker() if callable(checker) else None
    if requested == "release":
        return "release"
    if requested == "hold":
        _finish_operator_stop(actuator)
        return "hold"
    return outcome


def _run_active_goal_controlled(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    actuator: SafeG1Dex3Actuator,
    task_name: str,
    instruction: str,
    contract: ModelContract,
    args: argparse.Namespace,
    terminal: _OperatorTerminal,
    *,
    allow_custom_instruction: bool,
) -> str:
    """Run one finite goal; return ``hold``, ``release``, or ``complete``."""

    completed_chunks = 0
    remaining_action_budget = args.execution_horizon * args.max_chunks
    while remaining_action_budget > 0:
        action = _active_command_action(_poll_active_command(terminal))
        if action == "hold":
            _finish_operator_stop(actuator)
            LOGGER.warning("Goal %r STOPPED in powered position hold", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release")
            return "release"

        request_horizon = min(args.execution_horizon, remaining_action_budget)
        chunk, inference_s = infer_chunk(
            policy,
            state_reader,
            camera,
            instruction,
            contract,
            request_horizon,
            actuator,
            command_conditioning=getattr(args, "command_conditioning", "xr"),
            show_camera=getattr(args, "show_camera", False),
            allow_custom_instruction=allow_custom_instruction,
        )

        # A command typed during synchronous inference is honored before the
        # newly returned chunk can be submitted. The request cannot be
        # cancelled, but its result is discarded and no further request is made.
        action = _active_command_action(_poll_active_command(terminal))
        if action == "hold":
            _finish_operator_stop(actuator)
            LOGGER.warning("Goal %r inferred chunk discarded; STOPPED", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release; inferred chunk discarded")
            return "release"

        try:
            sequence = actuator.submit(chunk)
        except HandFeedbackReplan as exc:
            LOGGER.warning(
                "Policy chunk was invalidated at installation by a Dex3 feedback pause; "
                "re-observing goal %r: %s",
                task_name,
                exc.detail,
                extra={"terminal_yellow": True},
            )
            policy.reset()
            continue
        completion = actuator.wait_completed(
            sequence,
            timeout_s=args.execution_horizon / CONTROL_HZ + 1.0,
        )
        if completion == "hold":
            LOGGER.warning("Goal %r STOPPED during action execution", task_name)
            return "hold"
        if completion == "release":
            LOGGER.warning("Operator requested immediate orderly authority release")
            return "release"
        if completion == "replan":
            detail = getattr(actuator, "last_replan_detail", None)
            if not isinstance(detail, dict):
                detail = {}
            if detail.get("reason") == "hand_state_recovered":
                LOGGER.warning(
                    "AUTOMATIC REPLAN after transient Dex3 feedback recovery during goal %r: "
                    "pause=%.3fs, safety readings=%s/3; discarded %s old actions and "
                    "refetching from a fresh observation",
                    task_name,
                    float(detail.get("pause_s", 0.0)),
                    detail.get("recovery_samples", "?"),
                    detail.get("discarded_actions", "?"),
                    extra={"terminal_yellow": True},
                )
            else:
                LOGGER.warning(
                    "AUTOMATIC REPLAN after scheduler discontinuity during goal %r: sequence=%s "
                    "action=%s lateness=%.3fs; discarded %s stale actions and refetching from a "
                    "fresh observation without entering operator HOLD; full timing ring retained "
                    "for post-release dump",
                    task_name,
                    detail.get("sequence", sequence),
                    detail.get("action_index", "?"),
                    float(detail.get("lateness_s", 0.0)),
                    detail.get("discarded_actions", "?"),
                    extra={"terminal_yellow": True},
                )
            executed_before_gap = max(0, int(detail.get("action_index", 0)))
            remaining_action_budget = max(0, remaining_action_budget - executed_before_gap)
            if remaining_action_budget == 0:
                return "complete"
            policy.reset()
            continue
        completed_chunks += 1
        remaining_action_budget = max(0, remaining_action_budget - chunk.length)
        LOGGER.info(
            "Completed live chunk %d/%d for %r (%d actions, inference %.3fs)",
            completed_chunks,
            args.max_chunks,
            task_name,
            chunk.length,
            inference_s,
        )
        action = _active_command_action(_poll_active_command(terminal))
        if action == "hold":
            _finish_operator_stop(actuator)
            LOGGER.warning("Goal %r STOPPED in powered position hold", task_name)
            return "hold"
        if action == "release":
            LOGGER.warning("Operator requested orderly authority release")
            return "release"
    return "complete"


def _rtc_previous_action(plan: ActionChunk, start_index: int) -> dict[str, np.ndarray]:
    return {
        "left_arm": np.ascontiguousarray(plan.arm[start_index:, :7], dtype=np.float32)[None],
        "right_arm": np.ascontiguousarray(plan.arm[start_index:, 7:], dtype=np.float32)[None],
        "left_hand": np.ascontiguousarray(plan.left_hand[start_index:], dtype=np.float32)[None],
        "right_hand": np.ascontiguousarray(plan.right_hand[start_index:], dtype=np.float32)[None],
    }


def _rtc_options(
    plan: ActionChunk,
    start_index: int,
    frozen_steps: int,
    ramp_rate: float | None,
) -> dict[str, object]:
    overlap = plan.length - start_index
    options: dict[str, object] = {
        "inference_mode": "rtc",
        "rtc_previous_action": _rtc_previous_action(plan, start_index),
        "rtc_overlap_steps": overlap,
        "rtc_frozen_steps": frozen_steps,
    }
    if ramp_rate is not None:
        options["rtc_ramp_rate"] = ramp_rate
    return options


def _drain_rtc_worker(
    worker: _RtcInferenceWorker,
    actuator: SafeG1Dex3Actuator,
) -> None:
    deadline = time.monotonic() + 6.0
    while worker.busy and time.monotonic() < deadline:
        actuator.heartbeat()
        actuator.assert_healthy()
        time.sleep(0.01)
    while worker.poll() is not None:
        pass
    # If the timeout-bounded request still has not returned, do not spend a
    # second blocking join interval without servicing the actuator heartbeat.
    # The daemon owns its ZMQ socket and will close it when the request exits.
    worker.close(wait=not worker.busy)


def _run_active_goal_rtc(
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
    with _OperatorTerminal(actuator) as terminal:
        try:
            remaining_action_budget = args.execution_horizon * args.max_chunks
            while True:
                outcome = _run_active_goal_rtc_controlled(
                    policy,
                    state_reader,
                    camera,
                    actuator,
                    task_name,
                    instruction,
                    contract,
                    args,
                    terminal,
                    allow_custom_instruction=allow_custom_instruction,
                    action_budget_override=remaining_action_budget,
                )
                if outcome != "replan":
                    break
                detail = getattr(actuator, "last_replan_detail", None)
                if not isinstance(detail, dict):
                    detail = {}
                consumed = int(detail.get("rtc_total_actions", detail.get("action_index", 0)))
                remaining_action_budget = max(0, remaining_action_budget - max(0, consumed))
                if detail.get("reason") == "hand_state_recovered":
                    LOGGER.warning(
                        "AUTOMATIC RTC REPLAN after transient Dex3 feedback recovery during goal %r: "
                        "pause=%.3fs, safety readings=%s/3; discarded %s old actions and "
                        "refetching a fresh plan (remaining action budget=%d)",
                        task_name,
                        float(detail.get("pause_s", 0.0)),
                        detail.get("recovery_samples", "?"),
                        detail.get("discarded_actions", "?"),
                        remaining_action_budget,
                        extra={"terminal_yellow": True},
                    )
                else:
                    LOGGER.warning(
                        "AUTOMATIC RTC REPLAN after scheduler discontinuity during goal %r: "
                        "sequence=%s action=%s lateness=%.3fs; discarded %s stale actions and "
                        "refetching a fresh plan without entering operator HOLD (remaining action "
                        "budget=%d); full timing ring retained for post-release dump",
                        task_name,
                        detail.get("sequence", "?"),
                        detail.get("action_index", "?"),
                        float(detail.get("lateness_s", 0.0)),
                        detail.get("discarded_actions", "?"),
                        remaining_action_budget,
                        extra={"terminal_yellow": True},
                    )
                if remaining_action_budget == 0:
                    outcome = "complete"
                    break
                policy.reset()
        except OperatorStop:
            _finish_operator_stop(actuator)
            LOGGER.warning("Goal %r STOPPED; pending RTC result will be discarded", task_name)
            outcome = "hold"
        except OperatorRelease:
            LOGGER.warning("Operator requested immediate orderly authority release during RTC")
            outcome = "release"
        except HandFeedbackOperatorHold as exc:
            LOGGER.warning(
                "Dex3 feedback crossed %.2f s; RTC goal %r was abandoned in powered HOLD: %s",
                ACTUATOR_HAND_STATE_OPERATOR_HOLD_AGE_S,
                task_name,
                exc.detail,
                extra={"terminal_yellow": True},
            )
            outcome = "hold"
        except ImmediateControlEvent as exc:
            if exc.action == "release":
                outcome = "release"
            else:
                _finish_operator_stop(actuator)
                outcome = "hold"
    checker = getattr(actuator, "immediate_control_requested", None)
    requested = checker() if callable(checker) else None
    if requested == "release":
        return "release"
    if requested == "hold":
        _finish_operator_stop(actuator)
        return "hold"
    return outcome


def _run_active_goal_rtc_controlled(
    policy: Gr00tClient,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    actuator: SafeG1Dex3Actuator,
    task_name: str,
    instruction: str,
    contract: ModelContract,
    args: argparse.Namespace,
    terminal: _OperatorTerminal,
    *,
    allow_custom_instruction: bool,
    action_budget_override: int | None = None,
) -> str:
    """Run child-timed asynchronous RTC while the main thread services safety."""

    plan, initial_inference_s = infer_plan(
        policy,
        state_reader,
        camera,
        instruction,
        contract,
        actuator,
        command_conditioning=getattr(args, "command_conditioning", "xr"),
        show_camera=getattr(args, "show_camera", False),
        allow_custom_instruction=allow_custom_instruction,
    )
    initial_command = _active_command_action(_poll_active_command(terminal))
    if initial_command == "hold":
        _finish_operator_stop(actuator)
        LOGGER.warning("Goal %r STOPPED; initial RTC plan was discarded", task_name)
        return "hold"
    if initial_command == "release":
        LOGGER.warning("Operator requested immediate orderly authority release during RTC")
        return "release"
    action_budget = (
        args.execution_horizon * args.max_chunks
        if action_budget_override is None
        else int(action_budget_override)
    )
    try:
        current_sequence = actuator.start_rtc(plan, action_budget=action_budget)
    except HandFeedbackReplan:
        return "replan"
    current_plan = plan
    # Store the post-capture part of recent delays. Each request adds its own
    # measured capture delay exactly once when selecting the frozen prefix.
    response_delay_history: deque[int] = deque(maxlen=RTC_DELAY_HISTORY)
    response_delay_history.append(max(1, int(np.ceil(initial_inference_s * CONTROL_HZ)) + 1))
    worker = _RtcInferenceWorker(args.policy_host, args.policy_port)
    pending: dict[str, object] | None = None
    request_count = 0
    handoff_count = 0
    next_snapshot_at = 0.0
    hand_pause_seen = False

    LOGGER.info(
        "RTC live plan started for %r: horizon=%d, minimum execution=%d, action budget=%d",
        task_name,
        plan.length,
        args.execution_horizon,
        action_budget,
    )
    try:
        while True:
            actuator.heartbeat()
            command_action = _active_command_action(_poll_active_command(terminal))
            if command_action == "hold":
                _finish_operator_stop(actuator)
                LOGGER.warning(
                    "Goal %r STOPPED; pending RTC result will be discarded",
                    task_name,
                )
                _drain_rtc_worker(worker, actuator)
                return "hold"
            if command_action == "release":
                LOGGER.warning("Operator requested immediate orderly authority release during RTC")
                worker.close(wait=False)
                return "release"

            event = actuator.poll_rtc_event()
            if event is not None:
                outcome, detail = event
                if outcome == "complete":
                    LOGGER.info(
                        "RTC goal %r completed %d actions with %d requests and %d handoffs",
                        task_name,
                        action_budget,
                        request_count,
                        handoff_count,
                    )
                    return "complete"
                if outcome == "replan":
                    _drain_rtc_worker(worker, actuator)
                    return "replan"
                LOGGER.error("RTC plan underrun/rejection; child entered powered HOLD: %s", detail)
                _drain_rtc_worker(worker, actuator)
                return "hold"

            if actuator.hand_state_paused:
                hand_pause_seen = True
            if hand_pause_seen:
                # The child discarded the time-indexed plan at pause entry.
                # Do not consume an in-flight response or request another one;
                # recovery will surface as a fresh-replan event, while the
                # 1.25 s boundary surfaces as terminal HOLD.
                time.sleep(0.002)
                continue

            response = worker.poll()
            if response is not None:
                if pending is None or response.generation != pending["sequence"]:
                    actuator.hold()
                    LOGGER.error("Stale RTC inference generation was discarded; entered powered HOLD")
                    _drain_rtc_worker(worker, actuator)
                    return "hold"
                if response.error is not None or response.action is None:
                    actuator.hold()
                    LOGGER.error("RTC inference failed; entered powered HOLD: %s", response.error)
                    _drain_rtc_worker(worker, actuator)
                    return "hold"
                reference_state = pending["state"]
                try:
                    replacement = parse_action_plan(
                        response.action,
                        model_horizon=contract.action_horizon,
                        current_arm=reference_state.arm,
                        current_left=reference_state.left_hand,
                        current_right=reference_state.right_hand,
                        # B[0] is aligned to the old commanded tail, while the child
                        # independently validates the actual old-target -> B[k]
                        # handoff after measuring how many actions elapsed.
                        validate_initial_step=False,
                        validate_target_steps=getattr(args, "command_conditioning", "xr") == "none",
                        end_effector=contract.end_effector,
                    )
                    current_sequence, actual_delay = actuator.replace_rtc(
                        replacement,
                        expected_sequence=int(pending["sequence"]),
                        request_index=int(pending["request_index"]),
                        expected_overlap=int(pending["overlap"]),
                    )
                except RtcTerminalEvent as exc:
                    if exc.outcome == "hold":
                        _drain_rtc_worker(worker, actuator)
                        return "hold"
                    if exc.outcome == "replan":
                        _drain_rtc_worker(worker, actuator)
                        return "replan"
                    return "complete"
                except ImmediateControlEvent:
                    raise
                except (DeploymentError, TimeoutError) as exc:
                    LOGGER.error("RTC response validation/handoff failed; entering powered HOLD: %s", exc)
                    actuator.hold()
                    _drain_rtc_worker(worker, actuator)
                    return "hold"
                current_plan = replacement
                capture_actions = int(pending["capture_actions"])
                response_delay_history.append(max(1, actual_delay - capture_actions))
                handoff_count += 1
                LOGGER.info(
                    "RTC handoff %d for %r: actual delay=%d actions (%.3fs inference), new index=%d",
                    handoff_count,
                    task_name,
                    actual_delay,
                    response.inference_s,
                    actual_delay,
                )
                pending = None
                continue

            now = time.monotonic()
            if pending is None and now >= next_snapshot_at:
                next_snapshot_at = now + 0.01
                try:
                    snapshot = actuator.rtc_snapshot()
                except RtcTerminalEvent as exc:
                    if exc.outcome == "hold":
                        _drain_rtc_worker(worker, actuator)
                        return "hold"
                    if exc.outcome == "replan":
                        _drain_rtc_worker(worker, actuator)
                        return "replan"
                    return "complete"
                if snapshot.sequence != current_sequence:
                    actuator.hold()
                    LOGGER.error("RTC child generation changed without an acknowledged handoff")
                    _drain_rtc_worker(worker, actuator)
                    return "hold"
                if snapshot.total_actions < snapshot.action_budget and snapshot.action_index >= args.execution_horizon:
                    request_snapshot = snapshot
                    # Camera and DDS sockets remain on the main thread. The
                    # pre-capture index is the RTC origin, so child-measured k
                    # includes observation capture as real end-to-end latency.
                    observation, reference_state = capture_policy_observation(
                        state_reader,
                        camera,
                        instruction,
                        contract,
                        actuator,
                        show_camera=getattr(args, "show_camera", False),
                        allow_custom_instruction=allow_custom_instruction,
                    )
                    try:
                        post_capture_snapshot = actuator.rtc_snapshot()
                    except RtcTerminalEvent as exc:
                        if exc.outcome == "hold":
                            _drain_rtc_worker(worker, actuator)
                            return "hold"
                        if exc.outcome == "replan":
                            _drain_rtc_worker(worker, actuator)
                            return "replan"
                        return "complete"
                    if (
                        post_capture_snapshot.sequence != current_sequence
                        or post_capture_snapshot.action_index >= current_plan.length
                    ):
                        continue
                    overlap = current_plan.length - request_snapshot.action_index
                    configured_frozen = getattr(args, "rtc_frozen_steps", None)
                    frozen_steps = (
                        int(configured_frozen)
                        if configured_frozen is not None
                        else max(response_delay_history)
                        + (post_capture_snapshot.action_index - request_snapshot.action_index)
                    )
                    if not 1 <= frozen_steps <= overlap:
                        actuator.hold()
                        LOGGER.error(
                            "RTC has only %d overlapping actions but needs %d frozen latency actions; entered HOLD",
                            overlap,
                            frozen_steps,
                        )
                        _drain_rtc_worker(worker, actuator)
                        return "hold"
                    request_count += 1
                    worker.submit(
                        _RtcRequest(
                            generation=request_snapshot.sequence,
                            observation=observation,
                            options=_rtc_options(
                                current_plan,
                                request_snapshot.action_index,
                                frozen_steps,
                                getattr(args, "rtc_ramp_rate", None),
                            ),
                        )
                    )
                    pending = {
                        "sequence": request_snapshot.sequence,
                        "request_index": request_snapshot.action_index,
                        "overlap": overlap,
                        "capture_actions": (post_capture_snapshot.action_index - request_snapshot.action_index),
                        "state": reference_state,
                    }
                    LOGGER.info(
                        "RTC request %d launched at plan index %d with overlap=%d, frozen=%d",
                        request_count,
                        request_snapshot.action_index,
                        overlap,
                        frozen_steps,
                    )
            time.sleep(0.002)
    finally:
        if worker.busy:
            worker.close(wait=False)
        else:
            worker.close(wait=True)


def _run_shadow_rtc(
    initial_plan: ActionChunk,
    initial_inference_s: float,
    state_reader: G1Dex3StateReader,
    camera: TeleimagerCamera,
    instruction: str,
    contract: ModelContract,
    args: argparse.Namespace,
    *,
    allow_custom_instruction: bool,
) -> None:
    """Exercise RTC transport/conditioning against a publisher-free virtual clock."""

    LOGGER.warning(
        "RTC SHADOW uses a virtual 30 Hz action clock and sends no commands. The robot state "
        "does not follow predictions, so this validates timing/protocol—not closed-loop RTC quality."
    )
    action_budget = args.execution_horizon * args.max_chunks
    plan = initial_plan
    plan_index = 0
    total_actions = 0
    generation = 1
    request_count = 0
    handoff_count = 0
    pending: dict[str, object] | None = None
    response_delay_history: deque[int] = deque(
        [max(1, int(np.ceil(initial_inference_s * CONTROL_HZ)) + 1)],
        maxlen=RTC_DELAY_HISTORY,
    )
    worker = _RtcInferenceWorker(args.policy_host, args.policy_port)
    next_action_at = time.monotonic() + 1.0 / CONTROL_HZ
    try:
        while total_actions < action_budget:
            now = time.monotonic()
            while now >= next_action_at and total_actions < action_budget:
                if plan_index >= plan.length:
                    raise DeploymentError("RTC shadow action buffer underrun")
                plan_index += 1
                total_actions += 1
                next_action_at += 1.0 / CONTROL_HZ

            response = worker.poll()
            if response is not None:
                if pending is None or response.generation != pending["generation"]:
                    raise DeploymentError("RTC shadow received a stale inference generation")
                if response.error is not None or response.action is None:
                    raise DeploymentError(f"RTC shadow inference failed: {response.error}")
                reference_state = pending["state"]
                replacement = parse_action_plan(
                    response.action,
                    model_horizon=contract.action_horizon,
                    current_arm=reference_state.arm,
                    current_left=reference_state.left_hand,
                    current_right=reference_state.right_hand,
                    validate_initial_step=False,
                    validate_target_steps=getattr(args, "command_conditioning", "xr") == "none",
                    end_effector=contract.end_effector,
                )
                actual_delay = plan_index - int(pending["request_index"])
                if actual_delay < 0 or actual_delay >= int(pending["overlap"]):
                    raise DeploymentError("RTC shadow response exhausted its previous-action overlap before handoff")
                suffix = ActionChunk(
                    arm=np.ascontiguousarray(replacement.arm[actual_delay:]),
                    left_hand=np.ascontiguousarray(replacement.left_hand[actual_delay:]),
                    right_hand=np.ascontiguousarray(replacement.right_hand[actual_delay:]),
                    end_effector=replacement.end_effector,
                )
                previous_index = max(0, plan_index - 1)
                if getattr(args, "command_conditioning", "xr") == "none":
                    validate_action_chunk(
                        suffix,
                        plan.arm[previous_index],
                        plan.left_hand[previous_index],
                        plan.right_hand[previous_index],
                    )
                plan = replacement
                plan_index = actual_delay
                generation += 1
                capture_actions = int(pending["capture_actions"])
                response_delay_history.append(max(1, actual_delay - capture_actions))
                handoff_count += 1
                LOGGER.info(
                    "RTC shadow handoff %d: delay=%d actions, inference=%.3fs",
                    handoff_count,
                    actual_delay,
                    response.inference_s,
                )
                pending = None

            if pending is None and total_actions < action_budget and plan_index >= args.execution_horizon:
                request_index = plan_index
                observation, reference_state = capture_policy_observation(
                    state_reader,
                    camera,
                    instruction,
                    contract,
                    show_camera=getattr(args, "show_camera", False),
                    allow_custom_instruction=allow_custom_instruction,
                )
                # Account for virtual actions that elapsed while camera capture
                # blocked. As in live RTC, the request remains conditioned on
                # the tail unconsumed at the pre-capture reference time.
                now = time.monotonic()
                while now >= next_action_at and total_actions < action_budget:
                    if plan_index >= plan.length:
                        raise DeploymentError("RTC shadow action buffer underrun during observation capture")
                    plan_index += 1
                    total_actions += 1
                    next_action_at += 1.0 / CONTROL_HZ
                if total_actions >= action_budget:
                    break
                capture_actions = plan_index - request_index
                overlap = plan.length - request_index
                configured_frozen = getattr(args, "rtc_frozen_steps", None)
                frozen_steps = (
                    int(configured_frozen)
                    if configured_frozen is not None
                    else max(response_delay_history) + capture_actions
                )
                if not 1 <= frozen_steps <= overlap:
                    raise DeploymentError(f"RTC shadow has overlap={overlap}, insufficient for frozen={frozen_steps}")
                request_count += 1
                worker.submit(
                    _RtcRequest(
                        generation=generation,
                        observation=observation,
                        options=_rtc_options(
                            plan,
                            request_index,
                            frozen_steps,
                            getattr(args, "rtc_ramp_rate", None),
                        ),
                    )
                )
                pending = {
                    "generation": generation,
                    "request_index": request_index,
                    "overlap": overlap,
                    "capture_actions": capture_actions,
                    "state": reference_state,
                }
                LOGGER.info(
                    "RTC shadow request %d: index=%d overlap=%d frozen=%d",
                    request_count,
                    request_index,
                    overlap,
                    frozen_steps,
                )
            time.sleep(0.001)
        LOGGER.info(
            "RTC shadow completed %d virtual actions with %d requests and %d handoffs",
            total_actions,
            request_count,
            handoff_count,
        )
    finally:
        worker.close(wait=True)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    repository_root = Path(__file__).resolve().parents[2]
    os.chdir(repository_root)
    voice_enabled = bool(getattr(args, "voice", False))
    initial_voice_server: VoiceCommandServer | None = None
    try:
        if voice_enabled and args.task is None and getattr(args, "custom_goal", None) is None:
            initial_voice_server = _start_voice_server(args)
        task_name, instruction = select_instruction(
            args.task,
            getattr(args, "custom_goal", None),
            voice_server=initial_voice_server,
            confirm_voice_text=bool(getattr(args, "confirm_text", False)),
        )
    finally:
        if initial_voice_server is not None:
            initial_voice_server.close()
    allow_custom_instruction = task_name == "custom-goal"
    if allow_custom_instruction and instruction not in TASKS.values():
        LOGGER.warning(
            "Using custom goal text that was not selected from the exact trained-task allowlist: %r",
            instruction,
        )
        confirm_custom_goal(instruction)
    end_effector = getattr(args, "end_effector", "dex3")
    configured_initialization = load_initialization_spec(
        getattr(args, "initialization", "measured"),
        task_name=task_name,
        pose_file=getattr(args, "initial_pose_file", None),
        end_effector=end_effector,
    )
    warmup1_enabled = bool(getattr(args, "warmup1", False))
    warmup1 = training_start_spec() if warmup1_enabled else None
    return_to_start_enabled = bool(getattr(args, "return_to_start", False))
    return_to_start_spec = warmup1 if warmup1 is not None else configured_initialization
    image_host = args.image_host or ("127.0.0.1" if args.sim else "192.168.123.164")

    policy: Gr00tClient | None = None
    camera: TeleimagerCamera | None = None
    state_reader: G1Dex3StateReader | None = None
    actuator: SafeG1Dex3Actuator | None = None
    voice_server: VoiceCommandServer | None = None
    cleanup_error: Exception | None = None
    try:
        policy = Gr00tClient(args.policy_host, args.policy_port)
        if not policy.ping():
            raise DeploymentError(f"GR00T server at {args.policy_host}:{args.policy_port} did not answer ping")
        contract = validate_model_contract(
            policy.get_modality_config(),
            end_effector=end_effector,
        )
        policy_metadata = policy.get_policy_metadata()
        requires_surface_normals = getattr(
            contract,
            "requires_surface_normals",
            contract.video_keys == ("ego_view", SURFACE_NORMAL_OUTPUT_KEY),
        )
        requires_depth_gray = getattr(
            contract,
            "requires_depth_gray",
            contract.video_keys == ("ego_view", DEPTH_OUTPUT_KEY),
        )
        visual_encoding = validate_policy_metadata(
            policy_metadata,
            end_effector=end_effector,
            requires_depth=requires_depth_gray,
            requires_surface_normals=requires_surface_normals,
            vision_input_contract=getattr(contract, "vision_input_contract", None),  # earlyfusion
        )
        if args.execution_horizon > contract.action_horizon:
            raise DeploymentError(
                f"Checkpoint action horizon is only {contract.action_horizon}, but "
                f"{args.execution_horizon} steps were requested"
            )
        inference_mode = getattr(args, "inference_mode", "synchronous")
        if inference_mode == "rtc" and contract.action_horizon < RTC_MIN_MODEL_HORIZON:
            raise DeploymentError(
                f"RTC requires a checkpoint trained for at least {RTC_MIN_MODEL_HORIZON} actions "
                f"(RTC_MIN_MODEL_HORIZON={RTC_MIN_MODEL_HORIZON}); "
                f"this checkpoint exposes {contract.action_horizon}. Use synchronous mode."
            )
        rtc_frozen_steps = getattr(args, "rtc_frozen_steps", None)
        first_overlap = contract.action_horizon - args.execution_horizon
        if inference_mode == "rtc" and rtc_frozen_steps is not None and rtc_frozen_steps > first_overlap:
            raise DeploymentError(
                f"--rtc-frozen-steps={rtc_frozen_steps} cannot fit the first RTC overlap of "
                f"{first_overlap} actions (model horizon {contract.action_horizon} minus "
                f"execution horizon {args.execution_horizon})"
            )
        if inference_mode == "rtc":
            rtc_capability = policy_metadata.get("rtc")
            expected_rtc = {
                "protocol_version": 1,
                "physical_action_tail": True,
                "backend": "pytorch",
            }
            if not isinstance(rtc_capability, dict) or any(
                rtc_capability.get(key) != value for key, value in expected_rtc.items()
            ):
                raise DeploymentError(
                    "GR00T server does not advertise the required RTC physical-tail protocol "
                    "and PyTorch backend; restart it with the RTC-capable server code"
                )
        LOGGER.info(
            "GR00T contract verified: video=%s + four G1/%s state/action keys, horizon %d",
            ",".join(contract.video_keys),
            end_effector,
            contract.action_horizon,
        )
        if end_effector == "inspire-dfx":
            LOGGER.info(
                "Inspire DFX shadow contract enabled: native normalized 6-DoF hand values; "
                "no command publisher or live conditioner exists"
            )
        elif getattr(args, "command_conditioning", "xr") == "xr":
            LOGGER.info(
                "XR command conditioning enabled: 100 Hz arm lead 0.08->0.12 rad/5s; "
                "no second Dex3 low-pass; nominal final slew arm=0.03 rad/write, "
                "hand=0.06857 thumb0/0.12 other joints rad/write"
            )
        else:
            LOGGER.warning("XR command conditioning disabled; raw policy target-step rejection is active")

        initialize_dds(args.sim, args.network_interface)
        state_reader = (
            G1Dex3StateReader(simulation=args.sim)
            if end_effector == "dex3"
            else G1InspireDfxStateReader(simulation=args.sim)
        )
        depth_encoding = visual_encoding if isinstance(visual_encoding, DepthEncodingContract) else None
        surface_normal_encoding = (
            visual_encoding if isinstance(visual_encoding, SurfaceNormalEncodingContract) else None
        )
        camera = TeleimagerCamera(
            image_host,
            depth_encoding=depth_encoding,
            surface_normal_encoding=surface_normal_encoding,
        )
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
        preflight_validation = args.actuate and not (
            configured_initialization.moves
            or warmup1 is not None
            or getattr(args, "policy_warm_start", True)
        )
        if inference_mode == "rtc":
            preflight, inference_s = infer_plan(
                policy,
                state_reader,
                camera,
                instruction,
                contract,
                camera_timeout_s=3.0,
                show_camera=getattr(args, "show_camera", False),
                allow_custom_instruction=allow_custom_instruction,
                validate_initial_step=preflight_validation,
                command_conditioning=getattr(args, "command_conditioning", "xr"),
            )
        else:
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
                # Shadow never executes or warm-starts this result, so its passive
                # measured pose is not a meaningful execution reference. Raw
                # structure/finiteness/ranges remain hard; with XR conditioning,
                # executable step checks occur later on final child commands.
                validate_initial_step=preflight_validation,
                command_conditioning=getattr(args, "command_conditioning", "xr"),
            )
        preflight_label = (
            "Publisher-free raw-contract preflight"
            if getattr(args, "command_conditioning", "xr") == "xr"
            else "Publisher-free preflight"
        )
        LOGGER.info(
            "%s passed in %.3fs: %s",
            preflight_label,
            inference_s,
            chunk_delta_summary(preflight, state_reader),
        )

        if not args.actuate:
            LOGGER.info("SHADOW MODE: no command publishers were created")
            if inference_mode == "rtc":
                _run_shadow_rtc(
                    preflight,
                    inference_s,
                    state_reader,
                    camera,
                    instruction,
                    contract,
                    args,
                    allow_custom_instruction=allow_custom_instruction,
                )
                return
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
                    # No shadow action is executed, so each request remains
                    # anchored to the unchanged passive robot pose.  Report
                    # that pose gap below. Raw structure/finiteness/ranges remain
                    # hard, but there is no actuator child to exercise conditioning.
                    validate_initial_step=False,
                    command_conditioning=getattr(args, "command_conditioning", "xr"),
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
        actuator_args = (
            args.sim,
            args.network_interface,
            getattr(args, "command_conditioning", "xr"),
        )
        # Parsed CLI namespaces always carry this value.  Direct library
        # callers with older hand-built Namespaces retain the actuator's
        # default-on behavior without changing their constructor call shape.
        actuator_kwargs = {}
        if hasattr(args, "gravity_feedforward"):
            actuator_kwargs["gravity_feedforward"] = args.gravity_feedforward
        run_log_dir = getattr(args, "_run_log_dir", None)
        if run_log_dir is None:
            # Keep direct library callers and existing test doubles compatible.
            actuator = SafeG1Dex3Actuator(*actuator_args, **actuator_kwargs)
        else:
            actuator = SafeG1Dex3Actuator(
                *actuator_args,
                run_log_dir=run_log_dir,
                **actuator_kwargs,
            )
        _run_blocking_motion_with_immediate_release(actuator, actuator.start)
        _run_blocking_motion_with_immediate_release(actuator, actuator.arm)
        LOGGER.warning("%s COMMAND MODE ARMED", "SIMULATION" if args.sim else "REAL ROBOT")

        confirm_initialization(actuator, configured_initialization)
        _run_blocking_motion_with_immediate_release(
            actuator,
            lambda: actuator.initialize(configured_initialization),
        )
        LOGGER.warning("Initialization completed: %s", configured_initialization.label)
        if warmup1_enabled:
            assert warmup1 is not None
            confirm_initialization(actuator, warmup1, stage="WARMUP1")
            _run_blocking_motion_with_immediate_release(
                actuator,
                lambda: actuator.warmup_pose(warmup1),
            )
            LOGGER.warning(
                "Warmup1 completed: %s (source episode=%d frame=%d)",
                warmup1.label,
                TRAINING_START_SOURCE["episode_index"],
                TRAINING_START_SOURCE["frame_index"],
            )
        confirm_policy_start(
            actuator,
            warmup1 if warmup1 is not None else configured_initialization,
        )
        if voice_enabled:
            voice_server = _start_voice_server(
                args,
                on_proposal=actuator.request_immediate_hold,
            )

        # The publisher-free result predates initialization and is deliberately
        # discarded. Every initial or replacement goal resets and re-observes.
        # Warmup2 is governed by --warmup2 for the first goal and for the next
        # goal after an explicit Return-to-Start reset. Direct replacement goals
        # use the independent --future-goal-warmup2 setting. None repeats
        # initialization or Warmup1.
        if sys.stdin.isatty():
            print(
                "\nACTIVE GOAL CONTROLS (NO ENTER): press s to STOP in a powered position hold; "
                "press q to release authority and exit. Ctrl-C also releases from any state."
            )

        first_goal = True
        warmup2_after_return_to_start = False
        custom_goal_mode = allow_custom_instruction
        while True:
            warmup2_enabled = (
                bool(getattr(args, "policy_warm_start", True))
                if first_goal or warmup2_after_return_to_start
                else bool(getattr(args, "future_goal_warmup2", True))
            )
            preparation = _prepare_policy_goal(
                policy,
                state_reader,
                camera,
                actuator,
                instruction,
                contract,
                args,
                allow_custom_instruction=allow_custom_instruction,
                warmup2_enabled=warmup2_enabled,
            )
            if preparation == "release":
                LOGGER.warning("Operator requested orderly authority release during goal transition")
                break
            if preparation == "hold":
                outcome = "hold"
            else:
                warmup2_after_return_to_start = False
                runner = _run_active_goal_rtc if inference_mode == "rtc" else _run_active_goal
                outcome = runner(
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
            if outcome == "release":
                break
            if outcome == "complete":
                if not return_to_start_enabled:
                    break
                _run_blocking_motion_with_immediate_release(actuator, actuator.hold)
                LOGGER.warning(
                    "Goal %r completed; entered powered HOLD with Return-to-Start available",
                    task_name,
                )
            elif outcome != "hold":
                raise DeploymentError(f"Unexpected active-goal outcome {outcome!r}")

            mode_state = {"custom_goal_mode": custom_goal_mode}
            while True:
                selector_kwargs = {
                    # A run launched with --custom-goal returns to custom mode;
                    # otherwise the operator's last Tab-selected mode persists.
                    "custom_goal_mode": custom_goal_mode,
                    "mode_state": mode_state,
                }
                # Preserve compatibility for direct callers/test doubles that
                # implement the pre-feature selector signature.
                if return_to_start_enabled:
                    selector_kwargs["return_to_start"] = True
                if voice_server is not None:
                    selector_kwargs["voice_server"] = voice_server
                    selector_kwargs["confirm_voice_text"] = bool(getattr(args, "confirm_text", False))
                next_goal = _select_next_goal_while_holding(actuator, **selector_kwargs)
                custom_goal_mode = mode_state["custom_goal_mode"]
                if next_goal is not None:
                    acknowledge = getattr(actuator, "acknowledge_hand_operator_hold", None)
                    if callable(acknowledge):
                        acknowledge()
                if next_goal != RETURN_TO_START:
                    break
                decision = _run_return_to_start_from_hold(actuator, return_to_start_spec)
                if decision == "release":
                    next_goal = None
                    break
                if decision == "hold":
                    continue
                warmup2_after_return_to_start = True
            if next_goal is None:
                LOGGER.warning("Operator requested orderly authority release from HOLD")
                break
            task_name, instruction = next_goal
            allow_custom_instruction = task_name == "custom-goal"
            first_goal = False
            LOGGER.warning("Next goal accepted while holding: %r — %s", task_name, instruction)
    finally:
        active_exception = sys.exc_info()[1]
        active_error = active_exception is not None
        if voice_server is not None:
            voice_server.close()
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
    parser = argparse.ArgumentParser(description="Colour/RGBD GR00T runner for Unitree G1-29")
    parser.add_argument(
        "--end-effector",
        choices=("dex3", "inspire-dfx"),
        default="dex3",
        help="Hand data/transport contract (default: dex3); Inspire DFX is currently shadow-only",
    )
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument("--task", choices=tuple(TASKS), help="Trained task ID; omit for a menu")
    goal_group.add_argument(
        "--custom-goal",
        metavar="TEXT",
        help="Send custom language goal text instead of one of the exact trained task strings",
    )
    parser.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accept reviewed goals from GrootVoiceCommander over authenticated local TCP while "
            "retaining all terminal controls"
        ),
    )
    parser.add_argument(
        "--confirm-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --voice, require an additional local terminal confirmation of phone text",
    )
    parser.add_argument(
        "--voice-listen-host",
        default="0.0.0.0",
        help="Local address for the voice TCP listener (default: all local interfaces)",
    )
    parser.add_argument(
        "--voice-port",
        type=int,
        default=DEFAULT_VOICE_PORT,
        help=f"Voice TCP listener port (default: {DEFAULT_VOICE_PORT})",
    )
    parser.add_argument(
        "--voice-session-token-file",
        type=Path,
        metavar="PATH",
        help=(
            "File containing the voice session token; otherwise read "
            "GROOT_VOICE_SESSION_TOKEN from the environment"
        ),
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
        "--inference-mode",
        choices=("synchronous", "rtc"),
        default="synchronous",
        help="Blocking chunk execution (default) or experimental asynchronous Real-Time Chunking",
    )
    parser.add_argument(
        "--command-conditioning",
        choices=("xr", "none"),
        default="xr",
        help=(
            "Condition final policy commands in the 100 Hz actuator (default: xr): apply XR's "
            "measured-relative arm command-lead limiter and final arm/hand slew limits. Dex3 policy "
            "targets are not low-pass filtered a second time. Raw shape/finite/joint-limit "
            "validation remains mandatory; use none only for comparison"
        ),
    )
    parser.add_argument(
        "--gravity-feedforward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Publish XR-compatible pose-dependent arm gravity feed-forward torque (default: "
            "enabled); --no-gravity-feedforward restores zero outgoing arm tau"
        ),
    )
    parser.add_argument(
        "--rtc-frozen-steps",
        type=int,
        help=(
            "Conservative total RTC delay budget in 30 Hz actions, including observation capture, "
            "serialization/network, inference, parsing, and handoff; default estimates recent timings"
        ),
    )
    parser.add_argument(
        "--rtc-ramp-rate",
        type=float,
        help="Optional RTC denoising ramp override (default: checkpoint model configuration)",
    )
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_MODES,
        default="measured",
        help=(
            "Pose before policy execution: preserve measured q (default), target XR's arm+hand "
            "joint-zero home with guarded motion, or load an experimental reviewed task-bound JSON pose. "
            "This stage runs before Warmup1"
        ),
    )
    parser.add_argument(
        "--initial-pose-file",
        help="Reviewed initialization JSON; required only with --initialization pose-file",
    )
    parser.add_argument(
        "--warmup1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After initialization, slowly move once to the frozen measured pose from training "
            "episode 0 frame 0 before policy inference (default: enabled)"
        ),
    )
    parser.add_argument(
        "--warmup2",
        "--policy-warm-start",
        dest="policy_warm_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For actuated runs, smoothly reach the first target of one fresh inferred chunk "
            "at startup and after an explicit Return-to-Start reset, discard that chunk, "
            "reset/re-observe, then begin live execution (default: enabled; "
            "--[no-]policy-warm-start remains a compatibility alias)"
        ),
    )
    parser.add_argument(
        "--future-goal-warmup2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For each direct replacement goal that does not follow Return-to-Start, perform "
            "the guarded Warmup2 first-target transition before live execution (default: "
            "enabled). Explicit Return-to-Start resets follow --warmup2 instead; initialization "
            "and Warmup1 are never repeated"
        ),
    )
    parser.add_argument(
        "--return-to-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Offer Shift+Tab in powered HOLD to repeat Warmup1 when enabled, otherwise the "
            "explicit xr-home/pose-file initialization target (default: disabled); it is "
            "invalid with --no-warmup1 --initialization measured"
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
        help="Create command publishers after preflight and the single-key r confirmation",
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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        metavar="DIR",
        help=(
            "Root directory for a new per-run log folder (default: ./logs). "
            "Each run records parent.log, run.json, actuator.log when actuation starts, "
            "and bounded active-loop timing JSON diagnostics."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_log_dir = create_run_directory(args.log_dir)
        configure_process_logging(run_log_dir / "parent.log")
        args._run_log_dir = str(run_log_dir)
        manifest_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        }
        write_json(
            run_log_dir / "run.json",
            {
                "schema_version": 1,
                "argv": list(sys.argv),
                "arguments": manifest_args,
                "launch_cwd": str(Path.cwd()),
                "pid": os.getpid(),
                "python": sys.version,
                "run_log_dir": str(run_log_dir),
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Could not create the run log directory: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    LOGGER.info("RUN_LOG_DIR=%s", run_log_dir)
    LOGGER.info("RUN_START cwd=%s argv=%r arguments=%r", Path.cwd(), sys.argv, manifest_args)

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
        LOGGER.warning("Stop requested")
    except (DeploymentError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
