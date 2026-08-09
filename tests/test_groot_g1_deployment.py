from __future__ import annotations

import argparse
import queue
import sys
from types import SimpleNamespace
from types import ModuleType
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import validate_args
from unitree_lerobot.eval_robot.groot_client import DeploymentError, MsgSerializer
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_EGO_VIEW_SHAPE,
    EXPECTED_JOINT_NAMES,
    EXPECTED_ROBOT_TYPE,
    TASKS,
    make_observation,
    parse_action_chunk,
    validate_model_contract,
    validate_policy_metadata,
    validate_measured_state,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    G1Dex3StateReader,
    RobotState,
    SafeG1Dex3Actuator,
    SIM_RIGHT_HAND_PERMUTATION,
    TeleimagerColourCamera,
    _G1Dex3CommandBackend,
    _actuator_main,
    decode_color_0_rgb,
    request_live_camera_config,
)


def modality_config(action_horizon: int = 16):
    return {
        "video": {"delta_indices": [0], "modality_keys": ["ego_view"]},
        "state": {"delta_indices": [0], "modality_keys": list(ACTION_KEYS)},
        "action": {
            "delta_indices": list(range(action_horizon)),
            "modality_keys": list(ACTION_KEYS),
            "action_configs": [
                {
                    "rep": "RELATIVE" if "arm" in key else "ABSOLUTE",
                    "type": "NON_EEF",
                    "format": "DEFAULT",
                    "state_key": None,
                }
                for key in ACTION_KEYS
            ],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }


def valid_action(horizon: int = 16):
    return {key: np.zeros((1, horizon, 7), dtype=np.float32) for key in ACTION_KEYS}


class FakeHeartbeat:
    def __init__(self, value: float):
        self.value = value
        self._lock = threading.Lock()

    def get_lock(self):
        return self._lock


class FakeBackend:
    instance = None

    def __init__(self, _simulation, _network_interface):
        type(self).instance = self
        self.publishes = 0
        self.released = False
        self.closed = False
        self._arm_target = np.zeros(14)
        self._left_target = np.zeros(7)
        self._right_target = np.zeros(7)

    def state(self):
        return RobotState(
            captured_at=time.monotonic(),
            mode_machine=0,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )

    def set_weight(self, _weight):
        pass

    def prepare_measured_hold(self):
        return self.state()

    def set_target(self, arm, left, right):
        self._arm_target = arm
        self._left_target = left
        self._right_target = right

    def publish(self):
        self.publishes += 1

    def release(self):
        self.released = True

    def close(self):
        self.closed = True


class GrootG1DeploymentTests(unittest.TestCase):
    def test_serializer_is_compatible_with_gr00t_server(self):
        try:
            from gr00t.policy.server_client import MsgSerializer as ServerSerializer
        except ImportError as exc:
            self.skipTest(f"Isaac-GR00T is not installed in this environment: {exc}")
        value = {
            "video": {"ego_view": np.zeros((1, 1, 4, 5, 3), dtype=np.uint8)},
            "state": {"left_arm": np.arange(7, dtype=np.float32)[None, None]},
        }

        decoded_by_server = ServerSerializer.from_bytes(MsgSerializer.to_bytes(value))
        np.testing.assert_array_equal(decoded_by_server["state"]["left_arm"], value["state"]["left_arm"])
        decoded_by_client = MsgSerializer.from_bytes(ServerSerializer.to_bytes(value))
        np.testing.assert_array_equal(decoded_by_client["video"]["ego_view"], value["video"]["ego_view"])

    def test_server_modality_config_round_trips_without_client_gr00t_types(self):
        try:
            from gr00t.data.types import ModalityConfig
            from gr00t.policy.server_client import MsgSerializer as ServerSerializer
        except ImportError as exc:
            self.skipTest(f"Isaac-GR00T is not installed in this environment: {exc}")
        config = {name: ModalityConfig(**value) for name, value in modality_config().items()}

        decoded = MsgSerializer.from_bytes(ServerSerializer.to_bytes(config))

        self.assertEqual(validate_model_contract(decoded).action_horizon, 16)

    def test_model_contract_rejects_a_different_embodiment_layout(self):
        config = modality_config()
        config["state"]["modality_keys"].append("waist")

        with self.assertRaisesRegex(DeploymentError, "Unsupported state keys"):
            validate_model_contract(config)

    def test_policy_metadata_requires_the_explicit_g1_training_tag(self):
        metadata = {
            "protocol_version": 1,
            "embodiment_tag": "new_embodiment",
            "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
            "dataset_contract": {
                "robot_type": EXPECTED_ROBOT_TYPE,
                "fps": 30.0,
                "observation_state_names": EXPECTED_JOINT_NAMES,
                "action_names": EXPECTED_JOINT_NAMES,
                "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
            },
        }
        validate_policy_metadata(metadata)
        with self.assertRaisesRegex(DeploymentError, "deployment protocol"):
            validate_policy_metadata({**metadata, "protocol_version": 2})
        with self.assertRaisesRegex(DeploymentError, "no deployment dataset contract"):
            validate_policy_metadata(
                {
                    "protocol_version": 1,
                    "embodiment_tag": "new_embodiment",
                    "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
                }
            )
        with self.assertRaisesRegex(DeploymentError, "requires GR00T training tag"):
            validate_policy_metadata({**metadata, "embodiment_tag": "unitree_g1_sonic"})
        with self.assertRaisesRegex(DeploymentError, "contract mismatch for fps"):
            validate_policy_metadata(
                {
                    **metadata,
                    "dataset_contract": {**metadata["dataset_contract"], "fps": 50.0},
                }
            )
        with self.assertRaisesRegex(DeploymentError, "contract mismatch for observation_state_names"):
            validate_policy_metadata(
                {
                    **metadata,
                    "dataset_contract": {
                        **metadata["dataset_contract"],
                        "observation_state_names": list(reversed(EXPECTED_JOINT_NAMES)),
                    },
                }
            )
        with self.assertRaisesRegex(DeploymentError, "no checkpoint action output contract"):
            validate_policy_metadata({key: value for key, value in metadata.items() if key != "action_output_contract"})
        with self.assertRaisesRegex(DeploymentError, "Unsupported checkpoint action output contract"):
            validate_policy_metadata(
                {
                    **metadata,
                    "action_output_contract": {
                        **EXPECTED_ACTION_OUTPUT_CONTRACT,
                        "use_relative_action": False,
                    },
                }
            )

    def test_model_contract_rejects_delta_joint_actions(self):
        config = modality_config()
        config["action"]["action_configs"][0]["rep"] = "DELTA"

        with self.assertRaisesRegex(DeploymentError, "Unsupported action representation"):
            validate_model_contract(config)

    def test_measured_state_must_be_finite_and_inside_hardware_limits(self):
        safe_arm = np.zeros(14)
        with self.assertRaisesRegex(DeploymentError, "outside its physical range"):
            unsafe_arm = safe_arm.copy()
            unsafe_arm[0] = 100.0
            validate_measured_state(unsafe_arm, np.zeros(14), np.zeros(7), np.zeros(7))
        with self.assertRaisesRegex(DeploymentError, "NaN or infinity"):
            unsafe_hand = np.zeros(7)
            unsafe_hand[3] = np.nan
            validate_measured_state(safe_arm, np.zeros(14), unsafe_hand, np.zeros(7))

    def test_state_reader_callbacks_are_fresh_bounded_and_closed(self):
        class FakeSubscriber:
            instances = {}

            def __init__(self, topic, _message_type):
                self.topic = topic
                self.handler = None
                self.closed = False
                self.instances[topic] = self

            def Init(self, handler=None, queueLen=0):
                self.handler = handler
                self.queue_len = queueLen

            def Close(self):
                self.closed = True

        def fake_module(name, **attributes):
            module = ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        fake_modules = {
            "unitree_lerobot.eval_robot.robot_control.robot_arm": fake_module(
                "robot_arm", G1_29_JointArmIndex=tuple(range(14))
            ),
            "unitree_lerobot.eval_robot.robot_control.robot_hand_unitree": fake_module(
                "robot_hand_unitree",
                Dex3_1_Left_JointIndex=tuple(range(7)),
                Dex3_1_Right_JointIndex=tuple(range(7)),
            ),
            "unitree_sdk2py": fake_module("unitree_sdk2py"),
            "unitree_sdk2py.core": fake_module("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": fake_module("unitree_sdk2py.core.channel", ChannelSubscriber=FakeSubscriber),
            "unitree_sdk2py.idl": fake_module("unitree_sdk2py.idl"),
            "unitree_sdk2py.idl.unitree_hg": fake_module("unitree_sdk2py.idl.unitree_hg"),
            "unitree_sdk2py.idl.unitree_hg.msg": fake_module("unitree_sdk2py.idl.unitree_hg.msg"),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": fake_module(
                "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                HandState_=object,
                LowState_=object,
            ),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            reader = G1Dex3StateReader(max_age_s=0.1)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                reader.read(timeout_s=0.01)
            self.assertLess(time.monotonic() - started, 0.2)

            arm_message = SimpleNamespace(
                mode_machine=5,
                motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(14)],
            )
            hand_message = SimpleNamespace(motor_state=[SimpleNamespace(q=0.0) for _ in range(7)])
            FakeSubscriber.instances["rt/lowstate"].handler(arm_message)
            FakeSubscriber.instances["rt/dex3/left/state"].handler(hand_message)
            FakeSubscriber.instances["rt/dex3/right/state"].handler(hand_message)

            state = reader.latest()
            self.assertEqual(state.mode_machine, 5)
            np.testing.assert_array_equal(state.arm, np.zeros(14))
            reader.close()

        self.assertTrue(all(item.closed for item in FakeSubscriber.instances.values()))

    def test_observation_matches_saved_g1_dex3_modality_shapes(self):
        observation = make_observation(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.arange(14, dtype=np.float64),
            np.arange(7, dtype=np.float64),
            np.arange(7, dtype=np.float64),
            TASKS["pick-red-cup"],
        )

        self.assertEqual(observation["video"]["ego_view"].shape, (1, 1, 480, 640, 3))
        self.assertEqual(observation["video"]["ego_view"].dtype, np.uint8)
        np.testing.assert_array_equal(observation["state"]["left_arm"][0, 0], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(observation["state"]["right_arm"][0, 0], np.arange(7, 14, dtype=np.float32))
        self.assertEqual(
            observation["language"]["annotation.human.task_description"],
            [[TASKS["pick-red-cup"]]],
        )

    def test_action_parser_combines_arm_order_and_keeps_execution_prefix(self):
        action = valid_action()
        action["left_arm"][:] = 0.01
        action["right_arm"][:] = 0.02

        chunk = parse_action_chunk(
            action,
            model_horizon=16,
            execution_horizon=8,
            current_arm=np.zeros(14),
            current_left=np.zeros(7),
            current_right=np.zeros(7),
        )

        self.assertEqual(chunk.arm.shape, (8, 14))
        np.testing.assert_allclose(chunk.arm[0], [0.01] * 7 + [0.02] * 7)

    def test_action_parser_rejects_non_finite_values(self):
        for unsafe in (np.nan, np.inf):
            with self.subTest(unsafe=unsafe):
                action = valid_action()
                action["right_arm"][0, 0, 0] = unsafe
                with self.assertRaisesRegex(DeploymentError, "NaN or infinity"):
                    parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_action_parser_rejects_large_first_step_instead_of_clipping(self):
        action = valid_action()
        action["right_arm"][0, :, 0] = 0.20

        with self.assertRaisesRegex(DeploymentError, "jump is too large"):
            parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_action_parser_rejects_invalid_unexecuted_tail(self):
        action = valid_action()
        action["left_arm"][0, 15, 0] = 100.0

        with self.assertRaisesRegex(DeploymentError, "outside"):
            parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

    def test_action_parser_accepts_calibrated_one_milliradian_hand_endpoint(self):
        action = valid_action()
        action["right_hand"][0, :, 3] = -0.001

        chunk = parse_action_chunk(action, 16, 8, np.zeros(14), np.zeros(7), np.zeros(7))

        np.testing.assert_allclose(chunk.right_hand[:, 3], -0.001)

    def test_camera_requires_fresh_jpeg_even_if_cached_bgr_exists(self):
        stale = SimpleNamespace(jpg=None, bgr=np.zeros((2, 4, 3), dtype=np.uint8))
        config = {"head_camera": {"image_shape": [2, 4], "binocular": False}}

        with self.assertRaisesRegex(TimeoutError, "fresh JPEG"):
            decode_color_0_rgb(stale, config)

    def test_camera_config_request_refuses_local_fallback_on_live_timeout(self):
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.closed = False

            def setsockopt(self, *_args):
                pass

            def connect(self, _endpoint):
                pass

            def send(self, message):
                self.sent = message

            def poll(self, _timeout_ms, _event):
                return 0

            def close(self, linger=None):
                self.closed = True
                self.linger = linger

        class FakeContext:
            def __init__(self, socket):
                self.socket_instance = socket
                self.terminated = False

            def socket(self, _kind):
                return self.socket_instance

            def term(self):
                self.terminated = True

        socket = FakeSocket()
        context = FakeContext(socket)
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.zmq.Context",
            return_value=context,
        ):
            with self.assertRaisesRegex(DeploymentError, "refusing.*local YAML fallback"):
                request_live_camera_config("camera-host", timeout_s=0.001)

        self.assertEqual(socket.sent, b"GET_DATA")
        self.assertTrue(socket.closed)
        self.assertTrue(context.terminated)

    def test_camera_rejects_image_client_config_that_was_not_live(self):
        live_config = {
            "head_camera": {
                "enable_zmq": True,
                "fps": 30,
                "image_shape": [480, 640],
            }
        }
        stale_config = {
            "head_camera": {
                "enable_zmq": True,
                "fps": 30,
                "image_shape": [720, 1280],
            }
        }

        class FakeImageClient:
            instance = None

            def __init__(self, **_kwargs):
                type(self).instance = self
                self.closed = False
                self._requester = None

            def get_cam_config(self):
                return stale_config

            def close(self):
                self.closed = True

        module_name = "unitree_lerobot.eval_robot.image_server.image_client"
        fake_module = ModuleType(module_name)
        fake_module.ImageClient = FakeImageClient
        with (
            mock.patch.dict(sys.modules, {module_name: fake_module}),
            mock.patch(
                "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.request_live_camera_config",
                return_value=live_config,
            ),
        ):
            with self.assertRaisesRegex(DeploymentError, "differs.*live server"):
                TeleimagerColourCamera("camera-host")

        self.assertTrue(FakeImageClient.instance.closed)

    def test_camera_waits_for_a_live_rolling_stream_rate(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        frames = [
            SimpleNamespace(jpg=encoded.tobytes(), fps=0.0),
            SimpleNamespace(jpg=encoded.tobytes(), fps=30.0),
        ]

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def get_head_frame(self):
                frame = frames[min(self.calls, len(frames) - 1)]
                self.calls += 1
                return frame

        camera = object.__new__(TeleimagerColourCamera)
        camera._client = FakeClient()
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: True)
        camera._reported_stream_fps = False
        camera.config = {
            "head_camera": {
                "image_shape": [480, 640],
                "binocular": False,
            }
        }

        rgb = camera.read_rgb(timeout_s=0.1)

        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(camera._client.calls, 2)
        self.assertTrue(camera._reported_stream_fps)

    def test_camera_rejects_a_stopped_subscriber_before_using_cached_data(self):
        camera = object.__new__(TeleimagerColourCamera)
        camera._client = SimpleNamespace(get_head_frame=mock.Mock())
        camera._head_subscriber = SimpleNamespace(is_alive=lambda: False)
        camera._reported_stream_fps = False
        camera.config = {}

        with self.assertRaisesRegex(DeploymentError, "subscriber stopped"):
            camera.read_rgb(timeout_s=0.1)

        camera._client.get_head_frame.assert_not_called()

    def test_camera_uses_configured_binocular_crop_and_converts_to_rgb(self):
        bgr = np.zeros((480, 1280, 3), dtype=np.uint8)
        bgr[:, :640] = [1, 2, 3]
        bgr[:, 640:] = [10, 20, 30]
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self.assertTrue(ok)
        frame = SimpleNamespace(jpg=encoded.tobytes())
        config = {"head_camera": {"image_shape": [480, 1280], "binocular": True}}

        rgb = decode_color_0_rgb(frame, config)

        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertTrue(rgb.flags.c_contiguous)
        np.testing.assert_allclose(rgb[0, 0], [3, 2, 1], atol=3)

    def test_camera_rejects_a_self_consistent_wrong_training_shape(self):
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)

        with self.assertRaisesRegex(DeploymentError, "training contract"):
            decode_color_0_rgb(
                SimpleNamespace(jpg=encoded.tobytes()),
                {"head_camera": {"image_shape": [720, 1280], "binocular": False}},
            )

    def test_sim_right_hand_permutation_round_trips_dataset_order(self):
        dataset_order = np.array(["thumb0", "thumb1", "thumb2", "index0", "index1", "middle0", "middle1"])
        expected_simulator_order = np.array(["thumb0", "thumb1", "thumb2", "middle0", "middle1", "index0", "index1"])
        simulator_order = dataset_order[SIM_RIGHT_HAND_PERMUTATION]

        np.testing.assert_array_equal(simulator_order, expected_simulator_order)
        np.testing.assert_array_equal(simulator_order[SIM_RIGHT_HAND_PERMUTATION], dataset_order)

    def test_real_actuation_requires_interface_and_local_policy(self):
        base = dict(
            execution_horizon=8,
            max_chunks=1,
            actuate=True,
            sim=False,
            network_interface=None,
            policy_host="127.0.0.1",
            allow_unqualified_real=False,
        )
        with self.assertRaisesRegex(DeploymentError, "network-interface"):
            validate_args(argparse.Namespace(**base))

        base["network_interface"] = "eth0"
        with self.assertRaisesRegex(DeploymentError, "fail-closed"):
            validate_args(argparse.Namespace(**base))

        base["allow_unqualified_real"] = True
        base["policy_host"] = "192.168.1.5"
        with self.assertRaisesRegex(DeploymentError, "loopback"):
            validate_args(argparse.Namespace(**base))

    def test_sim_actuation_requires_an_isolated_auto_dds_network(self):
        base = dict(
            execution_horizon=8,
            max_chunks=1,
            actuate=True,
            sim=True,
            network_interface=None,
            policy_host="127.0.0.1",
            allow_unqualified_real=False,
            confirm_sim_network_isolated=False,
        )
        with self.assertRaisesRegex(DeploymentError, "not a physical safety boundary"):
            validate_args(argparse.Namespace(**base))

        base["confirm_sim_network_isolated"] = True
        validate_args(argparse.Namespace(**base))

        base["network_interface"] = "lo"
        with self.assertRaisesRegex(DeploymentError, "may not match"):
            validate_args(argparse.Namespace(**base))

    def test_actuator_backend_does_not_publish_before_arm_command(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            self.assertEqual(FakeBackend.instance.publishes, 0)
            stop.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(FakeBackend.instance.publishes, 0)
        self.assertTrue(FakeBackend.instance.closed)

    def test_dex3_cleanup_uses_unitree_stop_motors_command(self):
        class Publisher:
            def __init__(self):
                self.writes = []

            def Write(self, message, timeout=None):
                self.writes.append((message, timeout))
                return True

        backend = object.__new__(_G1Dex3CommandBackend)
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._left_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._right_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._left_publisher = Publisher()
        backend._right_publisher = Publisher()

        backend._stop_hands()

        for message in (backend._left_message, backend._right_message):
            for index, command in enumerate(message.motor_cmd):
                self.assertEqual(command.mode, 0x90 | index)
                self.assertEqual(
                    (command.q, command.dq, command.tau, command.kp, command.kd),
                    (0.0, 0.0, 0.0, 0.0, 0.0),
                )
        self.assertEqual(len(backend._left_publisher.writes), 1)
        self.assertEqual(len(backend._right_publisher.writes), 1)

    def test_dex3_cleanup_attempts_right_stop_when_left_stop_fails(self):
        class Publisher:
            def __init__(self, result):
                self.result = result
                self.writes = 0

            def Write(self, _message, timeout=None):
                self.writes += 1
                return self.result

        backend = object.__new__(_G1Dex3CommandBackend)
        backend._left_indices = tuple(range(7))
        backend._right_indices = tuple(range(7))
        backend._left_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._right_message = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        backend._left_publisher = Publisher(False)
        backend._right_publisher = Publisher(True)

        with self.assertRaisesRegex(DeploymentError, "Left Dex3 stop Write failed"):
            backend._stop_hands()

        self.assertEqual(backend._left_publisher.writes, 1)
        self.assertEqual(backend._right_publisher.writes, 1)

    def test_prepare_measured_hold_rejects_unqualified_transient_mode(self):
        unsafe_state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=2,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        backend = object.__new__(_G1Dex3CommandBackend)
        backend.simulation = False
        backend.reader = SimpleNamespace(read=lambda timeout_s: unsafe_state)
        backend._arm_message = SimpleNamespace(mode_machine=None)

        with self.assertRaisesRegex(DeploymentError, "qualified mode 5"):
            backend.prepare_measured_hold()

        self.assertIsNone(backend._arm_message.mode_machine)

    def test_prepare_measured_hold_requires_fresh_stationary_real_state(self):
        def backend_with_state(state):
            backend = object.__new__(_G1Dex3CommandBackend)
            backend.simulation = False
            backend.reader = SimpleNamespace(read=lambda timeout_s: state)
            backend._arm_message = SimpleNamespace(mode_machine=None)
            return backend

        stale = RobotState(
            captured_at=time.monotonic() - 1.0,
            mode_machine=5,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        with self.assertRaisesRegex(DeploymentError, "not fresh enough"):
            backend_with_state(stale).prepare_measured_hold()

        moving = RobotState(
            captured_at=time.monotonic(),
            mode_machine=5,
            arm=np.zeros(14),
            arm_dq=np.full(14, 0.2),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        with self.assertRaisesRegex(DeploymentError, "stationary before arming"):
            backend_with_state(moving).prepare_measured_hold()

    def test_actuator_heartbeat_expiry_faults_and_releases(self):
        commands = queue.Queue(maxsize=1)
        statuses = queue.Queue(maxsize=32)
        stop = threading.Event()
        heartbeat = FakeHeartbeat(time.monotonic())
        with mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3._G1Dex3CommandBackend",
            FakeBackend,
        ):
            thread = threading.Thread(
                target=_actuator_main,
                args=(True, None, commands, statuses, stop, heartbeat),
                daemon=True,
            )
            thread.start()
            self.assertEqual(statuses.get(timeout=1.0)[0], "ready")
            commands.put(("arm",))
            self.assertEqual(statuses.get(timeout=1.0)[0], "armed")
            with heartbeat.get_lock():
                heartbeat.value = time.monotonic() - 10.0
            kind, message = statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("heartbeat expired", message.lower())
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(FakeBackend.instance.released)
        self.assertTrue(FakeBackend.instance.closed)

    def test_parent_surfaces_child_release_failure(self):
        class StoppedProcess:
            pid = 123
            exitcode = 0

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._closed = False
        actuator._heartbeat = FakeHeartbeat(time.monotonic())
        actuator._stop_event = threading.Event()
        actuator._process = StoppedProcess()
        actuator._status_queue = queue.Queue()
        actuator._status_queue.put(("release_failed", "right hand stop failed"))
        actuator._status_queue.put(("stopped", None))

        with self.assertRaisesRegex(DeploymentError, "release is unconfirmed.*right hand"):
            actuator.close()

        self.assertTrue(actuator._closed)


if __name__ == "__main__":
    unittest.main()
