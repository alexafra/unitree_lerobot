"""Guarded G1-29 + Dex3 state, camera, and command runtime.

Shadow use creates only subscribers.  Live command publishers exist in a spawned
child process with an independent heartbeat deadline, so blocked inference cannot
leave an action chunk advancing indefinitely.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
import multiprocessing as mp
from multiprocessing.queues import Queue as MpQueue
import queue
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
import zmq

from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.image_server.rgbd_protocol import RGBD_PROTOCOL
from unitree_lerobot.eval_robot.groot_contract import (
    ActionChunk,
    CONTROL_HZ,
    DepthEncodingContract,
    EXPECTED_DEPTH_VIEW_SHAPE,
    EXPECTED_EGO_VIEW_SHAPE,
    validate_action_chunk,
    validate_measured_state,
)
from unitree_lerobot.utils.depth_encoding import encode_depth_gray_rgb


LOGGER = logging.getLogger(__name__)

STATE_MAX_AGE_S = 0.25
ACTUATOR_STATE_MAX_AGE_S = 0.075
HEARTBEAT_TIMEOUT_S = 1.0
CHUNK_MAX_AGE_S = 0.25
ARM_AUTHORITY_RAMP_S = 1.5
ARM_RELEASE_RAMP_S = 1.0
PUBLISH_HZ = 100.0
DDS_WRITE_TIMEOUT_S = 0.5
MAX_ARM_DQ_RAD_S = 6.0
MAX_ARM_TRACKING_ERROR_RAD = 0.35
MAX_HAND_TRACKING_ERROR_RAD = 0.50
TRACKING_GRACE_S = 0.50
MAX_ACTION_LATENESS_S = 0.02
QUALIFIED_REAL_MODE_MACHINE = 5
PREARM_STATIONARY_DWELL_S = 0.5
PREARM_STATE_MAX_AGE_S = 0.05
PREARM_MAX_ARM_DQ_RAD_S = 0.10
PREARM_MAX_POSITION_DRIFT_RAD = 0.02
PREARM_MIN_DISTINCT_SAMPLES = 5

# IsaacLab publishes/consumes its right hand as thumb, middle, index.  The real
# Dex3 and recorded dataset use thumb, index, middle.  This permutation is its own
# inverse and must be applied to simulation states and commands only.
SIM_RIGHT_HAND_PERMUTATION = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)
TELEIMAGER_CONFIG_PORT = 60000
TELEIMAGER_CONFIG_TIMEOUT_S = 1.0
RGBD_MAX_RECEIVE_AGE_S = 0.15


@dataclass(frozen=True)
class RobotState:
    captured_at: float
    mode_machine: int
    arm: np.ndarray
    arm_dq: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


@dataclass(frozen=True)
class CameraImages:
    rgb: np.ndarray
    depth_gray: np.ndarray | None = None
    sequence: int | None = None


def initialize_dds(simulation: bool, network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    domain = 1 if simulation else 0
    if network_interface:
        ChannelFactoryInitialize(domain, networkInterface=network_interface)
    else:
        ChannelFactoryInitialize(domain)


class G1Dex3StateReader:
    """Read arm and hand state without constructing command publishers."""

    def __init__(self, simulation: bool = False, max_age_s: float = STATE_MAX_AGE_S):
        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex
        from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
            Dex3_1_Left_JointIndex,
            Dex3_1_Right_JointIndex,
        )
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        self._simulation = simulation
        self._max_age_s = max_age_s
        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        self._left_indices = tuple(int(index) for index in Dex3_1_Left_JointIndex)
        self._right_indices = tuple(int(index) for index in Dex3_1_Right_JointIndex)
        self._lock = threading.Lock()
        self._messages: dict[str, Any] = {"arm": None, "left": None, "right": None}
        self._updated_at = {key: 0.0 for key in self._messages}
        self._subscribers = {
            "arm": ChannelSubscriber("rt/lowstate", LowState_),
            "left": ChannelSubscriber("rt/dex3/left/state", HandState_),
            "right": ChannelSubscriber("rt/dex3/right/state", HandState_),
        }
        for key, subscriber in self._subscribers.items():
            subscriber.Init(handler=self._make_handler(key))

    def _make_handler(self, key: str):
        def update(message: Any) -> None:
            if message is None:
                return
            with self._lock:
                self._messages[key] = message
                self._updated_at[key] = time.monotonic()

        return update

    def latest(self) -> RobotState:
        now = time.monotonic()
        with self._lock:
            messages = dict(self._messages)
            updated_at = dict(self._updated_at)
        missing = [
            key for key, message in messages.items() if message is None or now - updated_at[key] > self._max_age_s
        ]
        if missing:
            raise TimeoutError(f"Stale Unitree state: {', '.join(missing)}")

        arm_message = messages["arm"]
        left_message = messages["left"]
        right_message = messages["right"]
        arm = np.array([arm_message.motor_state[index].q for index in self._arm_indices], dtype=np.float64)
        arm_dq = np.array([arm_message.motor_state[index].dq for index in self._arm_indices], dtype=np.float64)
        left = np.array([left_message.motor_state[index].q for index in self._left_indices], dtype=np.float64)
        right = np.array([right_message.motor_state[index].q for index in self._right_indices], dtype=np.float64)
        if self._simulation:
            right = right[SIM_RIGHT_HAND_PERMUTATION]
        validate_measured_state(arm, arm_dq, left, right)
        return RobotState(
            captured_at=min(updated_at.values()),
            mode_machine=int(getattr(arm_message, "mode_machine", 0)),
            arm=arm,
            arm_dq=arm_dq,
            left_hand=left,
            right_hand=right,
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
        raise TimeoutError(f"Timed out waiting for fresh Unitree state ({last_error})")

    def close(self) -> None:
        for subscriber in self._subscribers.values():
            with contextlib.suppress(Exception):
                subscriber.Close()


def _decode_color_jpeg_rgb(jpg: bytes | None, camera_config: dict[str, Any]) -> np.ndarray:
    if not jpg:
        raise TimeoutError("Head-camera transport has no fresh JPEG")
    encoded = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise DeploymentError("TeleImager returned an undecodable head-camera JPEG")

    try:
        head = camera_config["head_camera"]
        configured_height, configured_width = (int(value) for value in head["image_shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("TeleImager head-camera configuration is incomplete") from exc
    if bgr.shape != (configured_height, configured_width, 3):
        raise DeploymentError(
            f"Head frame {bgr.shape} does not match TeleImager config {(configured_height, configured_width, 3)}"
        )
    if head.get("binocular", False):
        if configured_width % 2:
            raise DeploymentError(f"Binocular frame width {configured_width} is not even")
        bgr = bgr[:, : configured_width // 2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    if list(rgb.shape) != EXPECTED_EGO_VIEW_SHAPE:
        raise DeploymentError(
            f"Decoded color_0 shape {rgb.shape} does not match the training contract {tuple(EXPECTED_EGO_VIEW_SHAPE)}"
        )
    return rgb


def decode_color_0_rgb(frame: Any, camera_config: dict[str, Any]) -> np.ndarray:
    """Decode a fresh TeleImage JPEG and reproduce recorded camera ``color_0``."""

    return _decode_color_jpeg_rgb(getattr(frame, "jpg", None), camera_config)


def _decode_depth_png_u16(encoded_png: bytes) -> np.ndarray:
    depth = cv2.imdecode(np.frombuffer(encoded_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise DeploymentError("RGBD packet contains an undecodable aligned-depth PNG")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise DeploymentError(
            f"RGBD aligned depth must decode to an HxW uint16 image; got shape={depth.shape}, dtype={depth.dtype}"
        )
    return depth


def request_live_camera_config(
    host: str,
    port: int = TELEIMAGER_CONFIG_PORT,
    timeout_s: float = TELEIMAGER_CONFIG_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch TeleImager config without its requester's local-YAML fallback."""

    timeout_ms = max(1, round(timeout_s * 1000.0))
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(f"tcp://{host}:{port}")
        socket.send(b"GET_DATA")
        if not socket.poll(timeout_ms, zmq.POLLIN):
            raise DeploymentError(
                f"Live TeleImager config server {host}:{port} did not respond; "
                "refusing the client's local YAML fallback"
            )
        config = socket.recv_json()
        if not isinstance(config, dict):
            raise DeploymentError("Live TeleImager config response is not a JSON object")
        return config
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(f"Could not fetch live TeleImager config from {host}:{port}") from exc
    finally:
        socket.close(linger=0)
        context.term()


def _validate_live_head_config(
    config: dict[str, Any],
    *,
    requires_depth: bool,
) -> tuple[int, float | None]:
    head = config.get("head_camera")
    if not isinstance(head, dict):
        raise DeploymentError("TeleImager configuration has no head_camera object")
    if not head.get("enable_zmq", False):
        raise DeploymentError("TeleImager head-camera ZMQ stream is disabled")
    try:
        fps = float(head["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError("TeleImager head camera has no numeric FPS") from exc
    if not np.isfinite(fps) or fps != CONTROL_HZ:
        raise DeploymentError(
            f"TeleImager head camera is configured for {head.get('fps')!r} FPS; "
            f"the training contract requires {CONTROL_HZ:g} FPS"
        )

    port_key = "zmq_port"
    depth_scale: float | None = None
    if requires_depth:
        if str(head.get("type", "")).lower() != "realsense":
            raise DeploymentError("RGBD checkpoint requires a RealSense TeleImager head camera")
        if not head.get("enable_depth", False):
            raise DeploymentError("RGBD checkpoint requires TeleImager aligned depth")
        if head.get("binocular", False):
            raise DeploymentError("RGBD checkpoint does not support a binocular head-camera layout")
        if head.get("rgbd_protocol") != RGBD_PROTOCOL:
            raise DeploymentError(
                f"RGBD checkpoint requires rgbd_protocol={RGBD_PROTOCOL!r}; got {head.get('rgbd_protocol')!r}"
            )
        port_key = "rgbd_zmq_port"
        try:
            depth_scale = float(head["depth_scale_m_per_unit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentError("TeleImager has no valid live RealSense depth scale") from exc
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise DeploymentError(f"TeleImager reported invalid depth scale {depth_scale!r}")
    try:
        port = int(head[port_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError(f"TeleImager has no valid {port_key}") from exc
    if not 1 <= port <= 65535:
        raise DeploymentError(f"TeleImager reported invalid {port_key} {port!r}")
    return port, depth_scale


class TeleimagerCamera:
    """TeleImager client selected automatically for colour or atomic RGBD."""

    def __init__(self, host: str, depth_encoding: DepthEncodingContract | None = None):
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

        self._client = None
        self._depth_encoding = depth_encoding
        self._requires_depth = depth_encoding is not None
        self._last_rgbd_sequence: int | None = None
        live_config = request_live_camera_config(host)
        stream_port, depth_scale = _validate_live_head_config(
            live_config,
            requires_depth=self._requires_depth,
        )
        if depth_scale is not None:
            self._depth_scale_m_per_unit = depth_scale
        try:
            self._client = ImageClient(
                host=host,
                request_bgr=False,
                request_rgbd=self._requires_depth,
            )
        except Exception:
            self.close()
            raise
        self.config = self._client.get_cam_config()
        if self.config != live_config:
            self.close()
            raise DeploymentError("TeleImager ImageClient config differs from the config returned by the live server")
        try:
            stream_key = (host, stream_port)
            self._head_subscriber = self._client._subscriber_manager._subscriber_threads[stream_key]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.close()
            raise DeploymentError("Could not verify the live TeleImager head subscriber") from exc
        if not self._head_subscriber.is_alive():
            self.close()
            raise DeploymentError("TeleImager head subscriber stopped during startup")
        self._reported_stream_fps = False

    def _assert_subscriber_alive(self) -> None:
        if not self._head_subscriber.is_alive():
            raise DeploymentError("TeleImager head subscriber stopped")

    def read_rgb(self, timeout_s: float = 3.0) -> np.ndarray:
        if getattr(self, "_requires_depth", False):
            raise DeploymentError("RGBD camera must be read atomically with read()")
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._assert_subscriber_alive()
                frame = self._client.get_head_frame()
                try:
                    measured_fps = float(getattr(frame, "fps", 0.0))
                except (TypeError, ValueError) as exc:
                    raise TimeoutError("TeleImager stream FPS is not numeric") from exc
                if not np.isfinite(measured_fps) or measured_fps <= 0.0:
                    raise TimeoutError("TeleImager stream has not established a live rolling FPS")
                rgb = decode_color_0_rgb(frame, self.config)
                if not self._reported_stream_fps:
                    LOGGER.info("TeleImager head stream is live at %.1f measured FPS", measured_fps)
                    self._reported_stream_fps = True
                return rgb
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for a fresh TeleImager frame ({last_error})")

    def _read_rgbd(self, timeout_s: float) -> CameraImages:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._assert_subscriber_alive()
                frame = self._client.get_head_rgbd_frame()
                if frame is None:
                    raise TimeoutError("TeleImager RGBD transport has no fresh packet")
                try:
                    measured_fps = float(self._client.get_head_rgbd_fps())
                except (TypeError, ValueError) as exc:
                    raise TimeoutError("TeleImager RGBD stream FPS is not numeric") from exc
                if not np.isfinite(measured_fps) or measured_fps <= 0.0:
                    raise TimeoutError("TeleImager RGBD stream has not established a live rolling FPS")
                if frame.received_monotonic_ns is None:
                    raise DeploymentError("TeleImager RGBD packet has no local receive timestamp")
                age_s = (time.monotonic_ns() - frame.received_monotonic_ns) / 1_000_000_000.0
                if age_s < 0.0 or age_s > RGBD_MAX_RECEIVE_AGE_S:
                    raise TimeoutError(f"TeleImager RGBD packet is stale ({age_s:.3f}s old)")
                if self._last_rgbd_sequence is not None:
                    if frame.sequence < self._last_rgbd_sequence:
                        raise DeploymentError(
                            "TeleImager RGBD sequence regressed from "
                            f"{self._last_rgbd_sequence} to {frame.sequence}; restart the policy runner"
                        )
                    if frame.sequence == self._last_rgbd_sequence:
                        raise TimeoutError(f"TeleImager RGBD sequence {frame.sequence} is not new")

                rgb = _decode_color_jpeg_rgb(frame.color_jpeg, self.config)
                depth_u16 = _decode_depth_png_u16(frame.aligned_depth_png)
                if list(depth_u16.shape) != EXPECTED_DEPTH_VIEW_SHAPE[:2]:
                    raise DeploymentError(
                        f"Aligned depth shape {depth_u16.shape} does not match the training contract "
                        f"{tuple(EXPECTED_DEPTH_VIEW_SHAPE[:2])}"
                    )
                assert self._depth_encoding is not None
                try:
                    depth_gray = encode_depth_gray_rgb(
                        depth_u16,
                        scale_m_per_unit=self._depth_scale_m_per_unit,
                        near_m=self._depth_encoding.near_m,
                        far_m=self._depth_encoding.far_m,
                    )
                except ValueError as exc:
                    raise DeploymentError(f"Could not encode aligned depth: {exc}") from exc
                self._last_rgbd_sequence = frame.sequence
                if not self._reported_stream_fps:
                    LOGGER.info("TeleImager atomic RGBD stream is live at %.1f measured FPS", measured_fps)
                    self._reported_stream_fps = True
                return CameraImages(rgb=rgb, depth_gray=depth_gray, sequence=frame.sequence)
            except TimeoutError as exc:
                last_error = exc
                time.sleep(0.005)
            except (TypeError, ValueError) as exc:
                raise DeploymentError(f"Invalid TeleImager RGBD packet: {exc}") from exc
        raise TimeoutError(f"Timed out waiting for a fresh TeleImager RGBD frame ({last_error})")

    def read(self, timeout_s: float = 3.0) -> CameraImages:
        if self._requires_depth:
            return self._read_rgbd(timeout_s)
        return CameraImages(rgb=self.read_rgb(timeout_s=timeout_s))

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            self._client = None
            client.close()


# Existing colour-only imports remain valid.  The default constructor still
# selects the legacy head-colour stream.
TeleimagerColourCamera = TeleimagerCamera


class _G1Dex3CommandBackend:
    """DDS publishers used only inside the actuator child process."""

    def __init__(self, simulation: bool, network_interface: str | None):
        initialize_dds(simulation, network_interface)
        self.simulation = simulation
        self.reader = G1Dex3StateReader(
            simulation=simulation,
            max_age_s=ACTUATOR_STATE_MAX_AGE_S,
        )
        initial = self.reader.read(timeout_s=5.0)
        if not simulation and initial.mode_machine != QUALIFIED_REAL_MODE_MACHINE:
            raise DeploymentError(
                f"Real G1 mode_machine is {initial.mode_machine}; this adapter is qualified only "
                f"for mode {QUALIFIED_REAL_MODE_MACHINE} (g1_29dof_with_hand_rev_1_0)"
            )

        from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_JointArmIndex
        from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
            Dex3_1_Left_JointIndex,
            Dex3_1_Right_JointIndex,
        )
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self._arm_indices = tuple(int(index) for index in G1_29_JointArmIndex)
        self._left_indices = tuple(int(index) for index in Dex3_1_Left_JointIndex)
        self._right_indices = tuple(int(index) for index in Dex3_1_Right_JointIndex)
        arm_topic = "rt/lowcmd" if simulation else "rt/arm_sdk"

        # Construct every resource before the first Write.  A partial constructor
        # failure therefore cannot acquire arm authority or move a hand.
        self._arm_publisher = ChannelPublisher(arm_topic, LowCmd_)
        self._left_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self._right_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._arm_publisher.Init()
        self._left_publisher.Init()
        self._right_publisher.Init()
        self._arm_message = unitree_hg_msg_dds__LowCmd_()
        self._left_message = unitree_hg_msg_dds__HandCmd_()
        self._right_message = unitree_hg_msg_dds__HandCmd_()
        self._crc = CRC()
        self._weight = 0.0
        self._released = False
        self._has_published = False
        self._arm_target = initial.arm.copy()
        self._left_target = initial.left_hand.copy()
        self._right_target = initial.right_hand.copy()
        self._configure_messages(initial)

    def _configure_messages(self, initial: RobotState) -> None:
        self._arm_message.mode_pr = 0
        self._arm_message.mode_machine = initial.mode_machine
        wrist_offsets = {4, 5, 6, 11, 12, 13}
        for offset, index in enumerate(self._arm_indices):
            command = self._arm_message.motor_cmd[index]
            command.mode = 1
            command.q = float(initial.arm[offset])
            command.dq = 0.0
            command.tau = 0.0
            command.kp = 40.0 if offset in wrist_offsets else 80.0
            command.kd = 1.5 if offset in wrist_offsets else 3.0

        for message, indices in (
            (self._left_message, self._left_indices),
            (self._right_message, self._right_indices),
        ):
            for index in indices:
                command = message.motor_cmd[index]
                command.mode = (index & 0x0F) | (0x01 << 4)
                command.q = 0.0
                command.dq = 0.0
                command.tau = 0.0
                command.kp = 1.5
                command.kd = 0.2

    def _validate_runtime_state(self, state: RobotState) -> RobotState:
        if not self.simulation and state.mode_machine != QUALIFIED_REAL_MODE_MACHINE:
            raise DeploymentError(
                f"Robot mode_machine changed from qualified mode {QUALIFIED_REAL_MODE_MACHINE} to {state.mode_machine}"
            )
        if np.max(np.abs(state.arm_dq)) > MAX_ARM_DQ_RAD_S:
            raise DeploymentError(f"Arm velocity exceeded {MAX_ARM_DQ_RAD_S:.1f} rad/s")
        return state

    def state(self) -> RobotState:
        return self._validate_runtime_state(self.reader.latest())

    def prepare_measured_hold(self) -> RobotState:
        state = self._validate_runtime_state(self.reader.read(timeout_s=0.5))
        if not self.simulation:
            reference = state
            last_captured_at = float("-inf")
            distinct_samples = 0
            deadline = time.monotonic() + PREARM_STATIONARY_DWELL_S
            while time.monotonic() < deadline:
                state = self._validate_runtime_state(self.reader.read(timeout_s=0.1))
                if time.monotonic() - state.captured_at > PREARM_STATE_MAX_AGE_S:
                    raise DeploymentError("Robot state is not fresh enough to arm")
                if np.max(np.abs(state.arm_dq)) > PREARM_MAX_ARM_DQ_RAD_S:
                    raise DeploymentError(
                        f"Arm must be stationary before arming ({PREARM_MAX_ARM_DQ_RAD_S:.2f} rad/s limit)"
                    )
                if (
                    max(
                        np.max(np.abs(state.arm - reference.arm)),
                        np.max(np.abs(state.left_hand - reference.left_hand)),
                        np.max(np.abs(state.right_hand - reference.right_hand)),
                    )
                    > PREARM_MAX_POSITION_DRIFT_RAD
                ):
                    raise DeploymentError("Robot position changed during the pre-arm stationary dwell")
                if state.captured_at > last_captured_at:
                    distinct_samples += 1
                    last_captured_at = state.captured_at
                time.sleep(0.005)
            if distinct_samples < PREARM_MIN_DISTINCT_SAMPLES:
                raise DeploymentError(
                    "Too few distinct robot-state samples arrived during the pre-arm stationary dwell"
                )
        self._arm_message.mode_machine = state.mode_machine if self.simulation else QUALIFIED_REAL_MODE_MACHINE
        self.set_target(state.arm, state.left_hand, state.right_hand)
        return state

    def set_target(self, arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self._arm_target = np.asarray(arm, dtype=np.float64).copy()
        self._left_target = np.asarray(left, dtype=np.float64).copy()
        self._right_target = np.asarray(right, dtype=np.float64).copy()

    def _publish_arm(self, require_qualified_state: bool = True) -> None:
        for offset, index in enumerate(self._arm_indices):
            self._arm_message.motor_cmd[index].q = float(self._arm_target[offset])
        if not self.simulation:
            # Never copy an unqualified/transient mode into a real arm command.
            # Operational writes also re-check the latest state immediately
            # before publishing.  Cleanup can skip the freshness dependency so
            # an attempted authority release is not defeated by the same state
            # fault that triggered it.
            self._arm_message.mode_machine = QUALIFIED_REAL_MODE_MACHINE
            if require_qualified_state:
                self._validate_runtime_state(self.reader.latest())
            self._arm_message.motor_cmd[29].q = float(self._weight)
        self._arm_message.crc = self._crc.Crc(self._arm_message)
        if self._arm_publisher.Write(self._arm_message, timeout=DDS_WRITE_TIMEOUT_S) is not True:
            raise DeploymentError("Arm DDS Write failed")
        self._has_published = True

    def _publish_hands(self) -> None:
        left_command = self._left_target
        right_command = self._right_target
        if self.simulation:
            right_command = right_command[SIM_RIGHT_HAND_PERMUTATION]
        for offset, index in enumerate(self._left_indices):
            self._left_message.motor_cmd[index].q = float(left_command[offset])
        for offset, index in enumerate(self._right_indices):
            self._right_message.motor_cmd[index].q = float(right_command[offset])

        if self._left_publisher.Write(self._left_message, timeout=DDS_WRITE_TIMEOUT_S) is not True:
            raise DeploymentError("Left Dex3 DDS Write failed")
        if self._right_publisher.Write(self._right_message, timeout=DDS_WRITE_TIMEOUT_S) is not True:
            raise DeploymentError("Right Dex3 DDS Write failed")

    def _stop_hands(self) -> None:
        """Send Unitree's documented Dex3 ``stopMotors`` command once per hand."""

        for message, indices in (
            (self._left_message, self._left_indices),
            (self._right_message, self._right_indices),
        ):
            for index in indices:
                command = message.motor_cmd[index]
                command.mode = (index & 0x0F) | (0x01 << 4) | (0x01 << 7)
                command.q = 0.0
                command.dq = 0.0
                command.tau = 0.0
                command.kp = 0.0
                command.kd = 0.0
        failures = []
        for name, publisher, message in (
            ("Left", self._left_publisher, self._left_message),
            ("Right", self._right_publisher, self._right_message),
        ):
            try:
                if publisher.Write(message, timeout=DDS_WRITE_TIMEOUT_S) is not True:
                    failures.append(f"{name} Dex3 stop Write failed")
            except Exception as exc:
                failures.append(f"{name} Dex3 stop Write raised {exc!r}")
        if failures:
            raise DeploymentError("; ".join(failures))

    def publish(self) -> None:
        self._publish_arm()
        self._publish_hands()

    def set_weight(self, weight: float) -> None:
        self._weight = float(np.clip(weight, 0.0, 1.0))

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if not self._has_published:
            return
        if self.simulation:
            try:
                measured = self.reader.latest()
                self.set_target(measured.arm, measured.left_hand, measured.right_hand)
                self.publish()
            except Exception as exc:
                raise DeploymentError(f"Failed to leave simulator holding measured position: {exc}") from exc
            return
        failures = []
        start_weight = self._weight
        steps = max(1, round(ARM_RELEASE_RAMP_S * PUBLISH_HZ))
        period = 1.0 / PUBLISH_HZ
        # Release does not depend on policy, camera, fresh state, or IK.
        for step in range(steps):
            self.set_weight(start_weight * (1.0 - (step + 1) / steps))
            try:
                self._publish_arm(require_qualified_state=False)
            except Exception as exc:
                failures.append(f"arm_sdk authority release failed: {exc}")
                break
            time.sleep(period)
        # Do this after the arm release so a blocked hand DDS Write cannot
        # prevent the higher-priority arm_sdk weight ramp from being attempted.
        try:
            self._stop_hands()
        except Exception as exc:
            failures.append(f"Dex3 stopMotors failed: {exc}")
        if failures:
            raise DeploymentError("; ".join(failures))

    def close(self) -> None:
        self.reader.close()
        for publisher in (
            self._arm_publisher,
            self._left_publisher,
            self._right_publisher,
        ):
            with contextlib.suppress(Exception):
                publisher.Close()


def _heartbeat_age(heartbeat: Any) -> float:
    with heartbeat.get_lock():
        last = float(heartbeat.value)
    return time.monotonic() - last


def _status(status_queue: MpQueue, kind: str, payload: Any = None) -> None:
    try:
        status_queue.put((kind, payload), timeout=0.2)
    except queue.Full:
        pass


def _actuator_main(
    simulation: bool,
    network_interface: str | None,
    command_queue: MpQueue,
    status_queue: MpQueue,
    stop_event: Any,
    heartbeat: Any,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if threading.current_thread() is threading.main_thread():

        def stop_on_signal(_signum: int, _frame: object) -> None:
            stop_event.set()

        for signal_name in ("SIGTERM", "SIGINT", "SIGHUP"):
            signum = getattr(signal, signal_name, None)
            if signum is not None:
                signal.signal(signum, stop_on_signal)
    backend: _G1Dex3CommandBackend | None = None
    last_sequence = 0
    tracking_checks_after = float("inf")
    try:
        backend = _G1Dex3CommandBackend(simulation, network_interface)
        _status(status_queue, "ready")

        # Publishers now exist but no message has been written.  Wait for the
        # parent's explicit arm request.
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.05)
            except queue.Empty:
                if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                    raise DeploymentError("Parent heartbeat expired before arm request")
                continue
            if command[0] == "arm":
                break
        else:
            return

        backend.prepare_measured_hold()

        if not simulation:
            steps = max(1, round(ARM_AUTHORITY_RAMP_S * PUBLISH_HZ))
            for step in range(steps):
                if stop_event.is_set() or _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                    raise DeploymentError("Heartbeat expired while ramping arm authority")
                measured = backend.state()
                backend.set_target(measured.arm, measured.left_hand, measured.right_hand)
                backend.set_weight((step + 1) / steps)
                backend.publish()
                stop_event.wait(1.0 / PUBLISH_HZ)
        tracking_checks_after = time.monotonic() + TRACKING_GRACE_S
        _status(status_queue, "armed")

        chunk: ActionChunk | None = None
        chunk_sequence = 0
        chunk_index = 0
        next_action_at = 0.0
        period = 1.0 / PUBLISH_HZ
        action_period = 1.0 / CONTROL_HZ

        while not stop_event.is_set():
            loop_started = time.monotonic()
            if _heartbeat_age(heartbeat) > HEARTBEAT_TIMEOUT_S:
                raise DeploymentError("Parent heartbeat expired")

            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if command is not None:
                kind = command[0]
                if kind != "chunk":
                    raise DeploymentError(f"Unexpected actuator command {kind!r}")
                if chunk is not None:
                    raise DeploymentError("Received a new chunk before the prior chunk completed")
                _, sequence, created_at, arm, left, right = command
                if sequence <= last_sequence:
                    raise DeploymentError(f"Stale/out-of-order action chunk {sequence}")
                if time.monotonic() - created_at > CHUNK_MAX_AGE_S:
                    raise DeploymentError(f"Action chunk {sequence} expired before execution")
                state = backend.state()
                proposed = ActionChunk(arm=arm, left_hand=left, right_hand=right)
                validate_action_chunk(proposed, state.arm, state.left_hand, state.right_hand)
                chunk = proposed
                chunk_sequence = int(sequence)
                chunk_index = 0
                next_action_at = time.monotonic()
                last_sequence = int(sequence)

            now = time.monotonic()
            if chunk is not None and now >= next_action_at:
                if chunk_index < chunk.length:
                    if now - next_action_at > MAX_ACTION_LATENESS_S:
                        raise DeploymentError(f"Action scheduler is {now - next_action_at:.3f}s late")
                    backend.set_target(
                        chunk.arm[chunk_index],
                        chunk.left_hand[chunk_index],
                        chunk.right_hand[chunk_index],
                    )
                    chunk_index += 1
                    # Keep the 30 Hz phase on the 100 Hz publication loop.  A
                    # lateness ceiling above prevents burst catch-up after a stall.
                    next_action_at += action_period
                else:
                    _status(status_queue, "completed", chunk_sequence)
                    chunk = None

            state = backend.state()
            if now >= tracking_checks_after:
                arm_error = float(np.max(np.abs(state.arm - backend._arm_target)))
                hand_error = float(
                    max(
                        np.max(np.abs(state.left_hand - backend._left_target)),
                        np.max(np.abs(state.right_hand - backend._right_target)),
                    )
                )
                if arm_error > MAX_ARM_TRACKING_ERROR_RAD:
                    raise DeploymentError(f"Arm tracking error is {arm_error:.3f} rad")
                if hand_error > MAX_HAND_TRACKING_ERROR_RAD:
                    raise DeploymentError(f"Hand tracking error is {hand_error:.3f} rad")

            backend.publish()
            elapsed = time.monotonic() - loop_started
            stop_event.wait(max(0.0, period - elapsed))
    except BaseException as exc:
        _status(status_queue, "fault", f"{type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            try:
                backend.release()
            except BaseException as exc:
                _status(status_queue, "release_failed", f"{type(exc).__name__}: {exc}")
            try:
                backend.close()
            except BaseException as exc:
                _status(status_queue, "release_failed", f"close {type(exc).__name__}: {exc}")
        _status(status_queue, "stopped")


class SafeG1Dex3Actuator:
    """Parent-side handle for the watchdog-owning actuator process."""

    def __init__(self, simulation: bool, network_interface: str | None):
        context = mp.get_context("spawn")
        self._command_queue = context.Queue(maxsize=1)
        self._status_queue = context.Queue(maxsize=32)
        self._stop_event = context.Event()
        self._heartbeat = context.Value("d", time.monotonic())
        self._process = context.Process(
            target=_actuator_main,
            args=(
                simulation,
                network_interface,
                self._command_queue,
                self._status_queue,
                self._stop_event,
                self._heartbeat,
            ),
            name="groot-g1-dex3-actuator",
        )
        self._sequence = 0
        self._closed = False

    def heartbeat(self) -> None:
        with self._heartbeat.get_lock():
            self._heartbeat.value = time.monotonic()

    def _wait_status(self, expected: str, timeout_s: float, payload: Any = None) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.heartbeat()
            try:
                kind, value = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                if not self._process.is_alive():
                    raise DeploymentError("Actuator process exited unexpectedly")
                continue
            if kind == "fault":
                raise DeploymentError(f"Actuator fault: {value}")
            if kind == expected and (payload is None or value == payload):
                return
        raise TimeoutError(f"Timed out waiting for actuator status '{expected}'")

    def start(self) -> None:
        self._process.start()
        self._wait_status("ready", timeout_s=8.0)

    def arm(self) -> None:
        self.heartbeat()
        self._command_queue.put(("arm",), timeout=0.2)
        self._wait_status("armed", timeout_s=ARM_AUTHORITY_RAMP_S + 3.0)

    def assert_healthy(self) -> None:
        if not self._process.is_alive():
            fault = None
            try:
                while True:
                    kind, value = self._status_queue.get_nowait()
                    if kind == "fault":
                        fault = value
            except queue.Empty:
                pass
            raise DeploymentError(f"Actuator process stopped{f': {fault}' if fault else ''}")

    def submit(self, chunk: ActionChunk) -> int:
        self.assert_healthy()
        self._sequence += 1
        self.heartbeat()
        command = (
            "chunk",
            self._sequence,
            time.monotonic(),
            chunk.arm,
            chunk.left_hand,
            chunk.right_hand,
        )
        try:
            self._command_queue.put(command, timeout=0.2)
        except queue.Full as exc:
            raise DeploymentError("Actuator command queue is full; refusing to buffer") from exc
        return self._sequence

    def wait_completed(self, sequence: int, timeout_s: float) -> None:
        self._wait_status("completed", timeout_s=timeout_s, payload=sequence)

    def close(self) -> None:
        if self._closed:
            return
        self.heartbeat()
        self._stop_event.set()
        if self._process.pid is None:
            self._closed = True
            return
        forced_kill = False
        self._process.join(timeout=ARM_RELEASE_RAMP_S + 2.0)
        if self._process.is_alive():
            LOGGER.critical("Actuator did not acknowledge release; requesting SIGTERM shutdown")
            self._process.terminate()
            self._process.join(timeout=ARM_RELEASE_RAMP_S + 2.0)
        if self._process.is_alive():
            LOGGER.critical("Actuator is unresponsive; forcing child process exit")
            forced_kill = True
            self._process.kill()
            self._process.join(timeout=1.0)
        self._closed = True
        issues = []
        if self._process.is_alive():
            issues.append("actuator process remains alive after forced shutdown")
        stopped_acknowledged = False
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                kind, value = self._status_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "stopped":
                stopped_acknowledged = True
                break
            elif kind == "release_failed":
                issues.append(f"release failed: {value}")
            elif kind == "fault":
                issues.append(f"actuator fault: {value}")
        if not stopped_acknowledged:
            issues.append("no orderly stopped acknowledgment was received")
        if forced_kill:
            issues.append("actuator required SIGKILL")
        if self._process.exitcode not in (None, 0):
            issues.append(f"actuator exited with code {self._process.exitcode}")
        if issues:
            raise DeploymentError("DDS release is unconfirmed: " + "; ".join(issues))
