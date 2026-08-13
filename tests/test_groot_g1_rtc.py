from __future__ import annotations

from contextlib import ExitStack
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _RtcInferenceWorker,
    _RtcRequest,
    _run_active_goal_rtc,
    _run_active_goal_rtc_controlled,
    _rtc_options,
    _rtc_previous_action,
    build_parser,
    run,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError, Gr00tClient
from unitree_lerobot.eval_robot.groot_contract import (
    ActionChunk,
    ARM_UPPER,
    MAX_ARM_STEP_RAD,
    load_initialization_spec,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    MAX_CONDITIONED_ARM_STEP_RAD,
    MAX_CONDITIONED_HAND_STEP_RAD,
    RobotState,
    RtcTerminalEvent,
    SafeG1Dex3Actuator,
    XrPolicyOutputConditioner,
    _actuator_main,
)


_SAFE_MODULE = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"


class _LiveHeartbeat:
    """Thread-test heartbeat that is always fresh."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        return time.monotonic()

    @value.setter
    def value(self, _value: float) -> None:
        pass

    def get_lock(self) -> threading.Lock:
        return self._lock


class _RecordingBackend:
    instance: _RecordingBackend | None = None
    follow_commanded_target = True

    def __init__(self, simulation: bool, network_interface: str | None):
        type(self).instance = self
        self.simulation = simulation
        self.network_interface = network_interface
        self._arm_target = np.zeros(14, dtype=np.float64)
        self._left_target = np.zeros(7, dtype=np.float64)
        self._right_target = np.zeros(7, dtype=np.float64)
        self._target_lock = threading.Lock()
        self.targets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.publishes = 0
        self.released = False
        self.closed = False

    def state(self) -> RobotState:
        if self.follow_commanded_target:
            arm = self._arm_target
            left = self._left_target
            right = self._right_target
        else:
            # Emulate ordinary position-control lag. RTC boundary slew must use
            # the last commanded target; measured-target tracking is checked by
            # the actuator independently.
            arm = np.zeros(14, dtype=np.float64)
            left = np.zeros(7, dtype=np.float64)
            right = np.zeros(7, dtype=np.float64)
        return RobotState(
            captured_at=time.monotonic(),
            mode_machine=0,
            arm=np.asarray(arm).copy(),
            arm_dq=np.zeros(14, dtype=np.float64),
            left_hand=np.asarray(left).copy(),
            right_hand=np.asarray(right).copy(),
        )

    def set_weight(self, _weight: float) -> None:
        pass

    def prepare_measured_hold(self) -> RobotState:
        return self.state()

    def set_target(
        self,
        arm: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> None:
        with self._target_lock:
            self._arm_target = np.asarray(arm, dtype=np.float64).copy()
            self._left_target = np.asarray(left_hand, dtype=np.float64).copy()
            self._right_target = np.asarray(right_hand, dtype=np.float64).copy()
            self.targets.append(
                (
                    self._arm_target.copy(),
                    self._left_target.copy(),
                    self._right_target.copy(),
                )
            )

    def target_snapshot(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self._target_lock:
            return [(arm.copy(), left.copy(), right.copy()) for arm, left, right in self.targets]

    def clear_targets(self) -> None:
        with self._target_lock:
            self.targets.clear()

    def publish(self) -> None:
        self.publishes += 1

    def release(self) -> None:
        self.released = True

    def close(self) -> None:
        self.closed = True


class _LaggingRecordingBackend(_RecordingBackend):
    instance: _LaggingRecordingBackend | None = None
    follow_commanded_target = False


class _GatedRecordingBackend(_RecordingBackend):
    """Pause one publisher iteration so queue/event ordering is deterministic."""

    instance: _GatedRecordingBackend | None = None

    def __init__(self, simulation: bool, network_interface: str | None):
        super().__init__(simulation, network_interface)
        type(self).instance = self
        self.pause_next_publish = threading.Event()
        self.publish_paused = threading.Event()
        self.continue_publish = threading.Event()

    def publish(self) -> None:
        super().publish()
        if self.pause_next_publish.is_set():
            self.pause_next_publish.clear()
            self.publish_paused.set()
            self.continue_publish.wait(timeout=1.0)


class _RecordingConditioner(XrPolicyOutputConditioner):
    instance: _RecordingConditioner | None = None

    def __init__(self) -> None:
        super().__init__()
        type(self).instance = self
        self.reset_count = 0
        self.desired: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def reset(self, arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self.reset_count += 1
        super().reset(arm, left, right)

    def set_desired(
        self,
        arm: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        *,
        now: float,
    ) -> None:
        self.desired.append(
            (
                np.asarray(arm).copy(),
                np.asarray(left).copy(),
                np.asarray(right).copy(),
            )
        )
        super().set_desired(arm, left, right, now=now)


class _ChildHarness:
    def __init__(
        self,
        backend_type: type[_RecordingBackend] = _RecordingBackend,
        *,
        command_conditioning: str = "none",
        conditioner_type: type[XrPolicyOutputConditioner] | None = None,
    ):
        self.backend_type = backend_type
        self.command_conditioning = command_conditioning
        self.conditioner_type = conditioner_type
        self.commands: queue.Queue = queue.Queue(maxsize=1)
        self.statuses: queue.Queue = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.urgent_hold = threading.Event()
        self.heartbeat = _LiveHeartbeat()
        self.thread: threading.Thread | None = None
        self._stack = ExitStack()

    @property
    def backend(self) -> _RecordingBackend:
        backend = self.backend_type.instance
        assert backend is not None
        return backend

    def __enter__(self) -> _ChildHarness:
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}._G1Dex3CommandBackend", self.backend_type))
        if self.conditioner_type is not None:
            self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.XrPolicyOutputConditioner", self.conditioner_type))
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.INITIALIZATION_START_DWELL_S", 0.0))
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.INITIALIZATION_CONVERGENCE_DWELL_S", 0.0))
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.INITIALIZATION_MIN_DISTINCT_SAMPLES", 1))
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.INITIALIZATION_MIN_MOVE_S", 0.0))
        self._stack.enter_context(mock.patch(f"{_SAFE_MODULE}.MAX_ACTION_LATENESS_S", 0.10))
        self.thread = threading.Thread(
            target=_actuator_main,
            args=(
                True,
                None,
                self.commands,
                self.statuses,
                self.stop,
                self.heartbeat,
                self.urgent_hold,
                self.command_conditioning,
            ),
            daemon=True,
        )
        self.thread.start()
        self.assert_status("ready")
        self.commands.put(("arm",), timeout=0.2)
        self.assert_status("armed")
        self.commands.put(
            (
                "initialize",
                time.monotonic(),
                load_initialization_spec("measured", task_name="pick-red-cup"),
            ),
            timeout=0.2,
        )
        self.assert_status("initializing")
        self.assert_status("initialized")
        self.backend.clear_targets()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self._stack.close()

    def assert_status(self, expected: str, timeout: float = 1.0):
        kind, value = self.statuses.get(timeout=timeout)
        if kind != expected:
            raise AssertionError(f"expected status {expected!r}, got {(kind, value)!r}")
        return value

    def wait_for_target_count(self, count: int, timeout: float = 0.5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.backend.target_snapshot()) >= count:
                return
            time.sleep(0.002)
        raise AssertionError(f"expected at least {count} targets, got {len(self.backend.target_snapshot())}")


def _plan(horizon: int, *, arm_step: float = 0.01) -> ActionChunk:
    arm = np.zeros((horizon, 14), dtype=np.float64)
    arm[:, 0] = np.arange(horizon, dtype=np.float64) * arm_step
    return ActionChunk(
        arm=arm,
        left_hand=np.zeros((horizon, 7), dtype=np.float64),
        right_hand=np.zeros((horizon, 7), dtype=np.float64),
    )


def _gripping_plan(horizon: int) -> ActionChunk:
    """Return a plan whose hand targets deliberately differ from feedback."""

    plan = _plan(horizon)
    plan.arm[:, 0] += 0.02
    plan.left_hand[:] = np.array([0.20, 0.15, 0.30, -0.20, -0.25, -0.20, -0.25])
    plan.right_hand[:] = np.array([-0.20, -0.15, -0.30, 0.20, 0.25, 0.20, 0.25])
    return plan


def _rtc_start_command(sequence: int, plan: ActionChunk, action_budget: int) -> tuple:
    return (
        "rtc_start",
        sequence,
        time.monotonic(),
        action_budget,
        plan.arm,
        plan.left_hand,
        plan.right_hand,
        plan.length,
    )


def _parent_handle_for_child(child: _ChildHarness) -> SafeG1Dex3Actuator:
    class _ThreadProcess:
        def is_alive(self) -> bool:
            return child.thread is not None and child.thread.is_alive()

    actuator = object.__new__(SafeG1Dex3Actuator)
    actuator._command_queue = child.commands
    actuator._status_queue = child.statuses
    actuator._stop_event = child.stop
    actuator._urgent_hold_event = child.urgent_hold
    actuator._heartbeat = child.heartbeat
    actuator._process = _ThreadProcess()
    actuator._sequence = 0
    actuator._started = True
    actuator._armed = True
    actuator._initialized = True
    actuator._warm_started = True
    actuator._holding = False
    actuator._chunk_in_flight = False
    actuator._pending_sequence = None
    actuator._rtc_active = False
    actuator._rtc_terminal = None
    actuator._command_conditioning = child.command_conditioning
    actuator._control_lock = threading.Lock()
    actuator._immediate_hold_requested = threading.Event()
    actuator._immediate_release_requested = threading.Event()
    actuator._stopped_acknowledged = False
    actuator._closed = False
    return actuator


class GrootG1RtcTests(unittest.TestCase):
    @staticmethod
    def _conditioner_jump_plan(horizon: int = 1) -> ActionChunk:
        arm = np.zeros((horizon, 14), dtype=np.float64)
        left = np.zeros((horizon, 7), dtype=np.float64)
        right = np.zeros((horizon, 7), dtype=np.float64)
        arm[:, 0] = 0.5
        left[:, 0] = 1.0
        right[:, 0] = 1.0
        return ActionChunk(arm=arm, left_hand=left, right_hand=right)

    def test_conditioned_child_sync_publishes_final_outputs_not_raw_policy_targets(self):
        with _ChildHarness(command_conditioning="xr") as child:
            plan = self._conditioner_jump_plan()
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            child.wait_for_target_count(1)

            first_arm, first_left, first_right = child.backend.target_snapshot()[0]
            self.assertAlmostEqual(first_arm[0], MAX_CONDITIONED_ARM_STEP_RAD)
            self.assertAlmostEqual(first_left[0], MAX_CONDITIONED_HAND_STEP_RAD[0])
            self.assertAlmostEqual(first_right[0], MAX_CONDITIONED_HAND_STEP_RAD[0])
            self.assertFalse(np.array_equal(first_arm, plan.arm[0]))
            self.assertFalse(np.array_equal(first_left, plan.left_hand[0]))
            self.assertTrue(child.thread.is_alive())

    def test_conditioned_child_rtc_publishes_final_outputs_not_raw_policy_targets(self):
        with _ChildHarness(command_conditioning="xr") as child:
            plan = self._conditioner_jump_plan(horizon=4)
            child.commands.put(_rtc_start_command(1, plan, action_budget=1), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            child.wait_for_target_count(1)

            first_arm, first_left, first_right = child.backend.target_snapshot()[0]
            self.assertAlmostEqual(first_arm[0], MAX_CONDITIONED_ARM_STEP_RAD)
            self.assertAlmostEqual(first_left[0], MAX_CONDITIONED_HAND_STEP_RAD[0])
            self.assertAlmostEqual(first_right[0], MAX_CONDITIONED_HAND_STEP_RAD[0])
            self.assertFalse(np.array_equal(first_arm, plan.arm[0]))
            self.assertFalse(np.array_equal(first_left, plan.left_hand[0]))
            self.assertTrue(child.thread.is_alive())

    def test_conditioner_persists_across_sync_chunks_and_powered_hold_resets_it(self):
        with _ChildHarness(
            command_conditioning="xr",
            conditioner_type=_RecordingConditioner,
        ) as child:
            conditioner = _RecordingConditioner.instance
            assert conditioner is not None
            self.assertEqual(conditioner.reset_count, 1)  # initialization endpoint

            first = self._conditioner_jump_plan()
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    first.arm,
                    first.left_hand,
                    first.right_hand,
                ),
                timeout=0.2,
            )
            self.assertEqual(child.assert_status("completed"), 1)

            second = self._conditioner_jump_plan()
            second.arm[:, 0] = 0.7
            second.left_hand[:, 0] = -0.5
            second.right_hand[:, 0] = -0.5
            child.commands.put(
                (
                    "chunk",
                    2,
                    time.monotonic(),
                    second.arm,
                    second.left_hand,
                    second.right_hand,
                ),
                timeout=0.2,
            )
            self.assertEqual(child.assert_status("completed"), 2)
            self.assertEqual(conditioner.reset_count, 1)
            self.assertEqual(len(conditioner.desired), 2)
            np.testing.assert_array_equal(conditioner.desired[-1][0], second.arm[0])

            child.commands.put(("hold", time.monotonic()), timeout=0.2)
            self.assertEqual(child.assert_status("holding"), 2)
            self.assertEqual(conditioner.reset_count, 2)
            self.assertIsNone(conditioner._started_at)
            np.testing.assert_array_equal(conditioner._desired_arm, child.backend._arm_target)
            np.testing.assert_array_equal(conditioner._desired_left, child.backend._left_target)
            np.testing.assert_array_equal(conditioner._desired_right, child.backend._right_target)

            child.urgent_hold.set()
            self.assertEqual(child.assert_status("holding"), 2)
            self.assertEqual(conditioner.reset_count, 3)
            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 2)
            self.assertEqual(conditioner.reset_count, 4)

    def test_conditioner_persists_across_rtc_replacement(self):
        with _ChildHarness(
            command_conditioning="xr",
            conditioner_type=_RecordingConditioner,
        ) as child:
            conditioner = _RecordingConditioner.instance
            assert conditioner is not None
            original = self._conditioner_jump_plan(horizon=32)
            child.commands.put(_rtc_start_command(1, original, action_budget=100), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            child.wait_for_target_count(1)

            child.commands.put(("rtc_snapshot", 1), timeout=0.2)
            snapshot = child.assert_status("rtc_snapshot")
            request_index = int(snapshot["action_index"])
            overlap = original.length - request_index
            replacement = self._conditioner_jump_plan(horizon=32)
            replacement.arm[:, 0] = -0.5
            replacement.left_hand[:, 0] = -0.5
            replacement.right_hand[:, 0] = -0.5
            child.commands.put(
                (
                    "rtc_replace",
                    2,
                    time.monotonic(),
                    1,
                    request_index,
                    overlap,
                    replacement.arm,
                    replacement.left_hand,
                    replacement.right_hand,
                    replacement.length,
                ),
                timeout=0.2,
            )
            acknowledgement = child.assert_status("rtc_replaced")
            self.assertGreaterEqual(int(acknowledgement["action_index"]), 0)

            deadline = time.monotonic() + 0.2
            while (
                not any(desired_arm[0] < 0.0 for desired_arm, _, _ in conditioner.desired)
                and time.monotonic() < deadline
            ):
                time.sleep(0.002)
            replacement_desired = [desired for desired in conditioner.desired if desired[0][0] < 0.0]
            self.assertGreaterEqual(len(replacement_desired), 1)
            self.assertEqual(conditioner.reset_count, 1)
            np.testing.assert_array_equal(replacement_desired[0][0], replacement.arm[0])
            self.assertTrue(child.thread.is_alive())

    def test_conditioned_child_still_rejects_raw_nan_and_out_of_range_targets(self):
        cases = []
        nan_plan = self._conditioner_jump_plan()
        nan_plan.arm[0, 0] = np.nan
        cases.append(("NaN", nan_plan, "NaN or infinity"))
        out_of_range = self._conditioner_jump_plan()
        out_of_range.arm[0, 0] = ARM_UPPER[0] + 1.0
        cases.append(("out of range", out_of_range, "outside its safety-margined joint range"))

        for name, plan, expected in cases:
            with self.subTest(name=name), _ChildHarness(command_conditioning="xr") as child:
                child.commands.put(
                    (
                        "chunk",
                        1,
                        time.monotonic(),
                        plan.arm,
                        plan.left_hand,
                        plan.right_hand,
                    ),
                    timeout=0.2,
                )
                kind, detail = child.statuses.get(timeout=1.0)
                self.assertEqual(kind, "fault")
                self.assertIn(expected, detail)
                child.thread.join(timeout=1.0)
                self.assertFalse(child.thread.is_alive())
                self.assertEqual(child.backend.target_snapshot(), [])

    def test_conditioned_child_rtc_still_rejects_raw_out_of_range_target(self):
        with _ChildHarness(command_conditioning="xr") as child:
            plan = self._conditioner_jump_plan(horizon=4)
            plan.arm[2, 0] = ARM_UPPER[0] + 1.0
            child.commands.put(_rtc_start_command(1, plan, action_budget=4), timeout=0.2)
            kind, detail = child.statuses.get(timeout=1.0)
            self.assertEqual(kind, "fault")
            self.assertIn("outside its safety-margined joint range", detail)
            child.thread.join(timeout=1.0)
            self.assertFalse(child.thread.is_alive())
            self.assertEqual(child.backend.target_snapshot(), [])

    def test_initial_rtc_inference_result_is_discarded_after_immediate_operator_key(self):
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        args = SimpleNamespace(
            execution_horizon=8,
            max_chunks=20,
            policy_host="127.0.0.1",
            policy_port=5555,
            show_camera=False,
            rtc_frozen_steps=None,
            rtc_ramp_rate=None,
        )
        for command, expected in (("hold", "hold"), ("release", "release")):
            with self.subTest(command=command):
                actuator = mock.Mock()
                actuator.immediate_control_requested.return_value = command
                terminal = SimpleNamespace(poll_control=mock.Mock(return_value=command))
                with mock.patch(f"{module}.infer_plan", return_value=(_plan(32), 0.1)):
                    outcome = _run_active_goal_rtc_controlled(
                        mock.Mock(),
                        mock.Mock(),
                        mock.Mock(),
                        actuator,
                        "pick-red-cup",
                        "pick up the red cup.",
                        SimpleNamespace(action_horizon=32),
                        args,
                        terminal,
                        allow_custom_instruction=False,
                    )

                self.assertEqual(outcome, expected)
                actuator.start_rtc.assert_not_called()
                if command == "hold":
                    actuator.finish_immediate_hold.assert_called_once_with()
                else:
                    actuator.finish_immediate_hold.assert_not_called()

    def test_inference_worker_owns_its_client_and_surfaces_timeout(self):
        main_thread = threading.get_ident()
        thread_ids: list[int] = []

        class _TimeoutClient:
            def __init__(self, _host: str, _port: int):
                thread_ids.append(threading.get_ident())

            def get_action(self, _observation, _options):
                thread_ids.append(threading.get_ident())
                raise TimeoutError("injected policy timeout")

            def close(self):
                thread_ids.append(threading.get_ident())

        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with mock.patch(f"{module}.Gr00tClient", _TimeoutClient):
            worker = _RtcInferenceWorker("127.0.0.1", 5555)
            worker.submit(_RtcRequest(7, {}, {"inference_mode": "rtc"}))
            deadline = time.monotonic() + 1.0
            response = None
            while response is None and time.monotonic() < deadline:
                response = worker.poll()
                time.sleep(0.001)
            worker.close(wait=True)

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.generation, 7)
        self.assertIsInstance(response.error, TimeoutError)
        self.assertFalse(worker.busy)
        self.assertEqual(len(thread_ids), 3)
        self.assertEqual(len(set(thread_ids)), 1)
        self.assertNotEqual(thread_ids[0], main_thread)

    def test_active_rtc_hold_and_release_commands_remain_distinct(self):
        class _IdleWorker:
            instances: list[_IdleWorker] = []

            def __init__(self, _host: str, _port: int):
                self.busy = False
                self.closed: list[bool] = []
                type(self).instances.append(self)

            def poll(self):
                return None

            def close(self, *, wait: bool):
                self.closed.append(wait)

        plan = _plan(32)
        state = RobotState(
            captured_at=time.monotonic(),
            mode_machine=0,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(7),
            right_hand=np.zeros(7),
        )
        args = SimpleNamespace(
            execution_horizon=8,
            max_chunks=20,
            policy_host="127.0.0.1",
            policy_port=5555,
            show_camera=False,
            rtc_frozen_steps=None,
            rtc_ramp_rate=None,
        )
        contract = SimpleNamespace(action_horizon=32)
        module = "unitree_lerobot.eval_robot.eval_groot_g1"

        for terminal_input, expected_outcome in (("s", "hold"), ("q", "release")):
            with self.subTest(terminal_input=terminal_input):
                actuator = mock.Mock()
                actuator.start_rtc.return_value = 1
                actuator.poll_rtc_event.return_value = None
                _IdleWorker.instances.clear()
                with (
                    mock.patch(f"{module}.infer_plan", return_value=(plan, 0.1)),
                    mock.patch(f"{module}._RtcInferenceWorker", _IdleWorker),
                    mock.patch(
                        f"{module}._poll_active_command",
                        side_effect=(None, terminal_input),
                    ),
                ):
                    outcome = _run_active_goal_rtc(
                        mock.Mock(),
                        mock.Mock(read=mock.Mock(return_value=state)),
                        mock.Mock(),
                        actuator,
                        "pick-red-cup",
                        "pick up the red cup.",
                        contract,
                        args,
                        allow_custom_instruction=False,
                    )

                self.assertEqual(outcome, expected_outcome)
                if expected_outcome == "hold":
                    actuator.hold.assert_called_once_with()
                else:
                    actuator.hold.assert_not_called()
                self.assertTrue(_IdleWorker.instances[0].closed)

    def test_client_requires_per_response_rtc_acknowledgement(self):
        plan = _plan(32)
        options = _rtc_options(plan, 8, frozen_steps=6, ramp_rate=None)
        action = {"left_arm": np.zeros((1, 32, 7), dtype=np.float32)}
        client = object.__new__(Gr00tClient)

        client.call = mock.Mock(return_value=(action, {}))
        with self.assertRaisesRegex(DeploymentError, "RTC ramp rate"):
            client.get_action({}, options)

        client.call = mock.Mock(
            return_value=(
                action,
                {
                    "rtc_applied": True,
                    "rtc_previous_action_horizon": 24,
                    "rtc_overlap_steps": 24,
                    "rtc_frozen_steps": 6,
                    "rtc_ramp_rate": 6.0,
                },
            )
        )
        self.assertIs(client.get_action({}, options), action)

    def test_impossible_explicit_frozen_prefix_fails_before_dds_initialization(self):
        args = build_parser().parse_args(
            [
                "--task",
                "pick-red-cup",
                "--inference-mode",
                "rtc",
                "--execution-horizon",
                "8",
                "--rtc-frozen-steps",
                "25",
            ]
        )
        policy = mock.Mock()
        policy.ping.return_value = True
        policy.get_modality_config.return_value = {}
        policy.get_policy_metadata.return_value = {
            "rtc": {
                "protocol_version": 1,
                "physical_action_tail": True,
                "backend": "pytorch",
            }
        }
        module = "unitree_lerobot.eval_robot.eval_groot_g1"
        with (
            mock.patch(f"{module}.Gr00tClient", return_value=policy),
            mock.patch(f"{module}.validate_model_contract", return_value=mock.Mock(action_horizon=32)),
            mock.patch(f"{module}.validate_policy_metadata", return_value=None),
            mock.patch(f"{module}.initialize_dds") as initialize_dds,
            self.assertRaisesRegex(DeploymentError, "cannot fit the first RTC overlap"),
        ):
            run(args)

        initialize_dds.assert_not_called()
        policy.close.assert_called_once_with()

    def test_physical_tail_options_preserve_group_order_shape_and_origin(self):
        horizon = 6
        arm = np.arange(horizon * 14, dtype=np.float32).reshape(horizon, 14) / 100.0
        left = np.arange(horizon * 7, dtype=np.float32).reshape(horizon, 7) / 200.0
        right = -left
        plan = ActionChunk(arm=arm, left_hand=left, right_hand=right)

        previous = _rtc_previous_action(plan, 2)
        options = _rtc_options(plan, 2, frozen_steps=3, ramp_rate=6.0)

        self.assertEqual(set(previous), {"left_arm", "right_arm", "left_hand", "right_hand"})
        for value in previous.values():
            self.assertEqual(value.shape, (1, 4, 7))
            self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(previous["left_arm"][0], arm[2:, :7])
        np.testing.assert_array_equal(previous["right_arm"][0], arm[2:, 7:])
        np.testing.assert_array_equal(previous["left_hand"][0], left[2:])
        np.testing.assert_array_equal(previous["right_hand"][0], right[2:])
        self.assertEqual(options["inference_mode"], "rtc")
        self.assertEqual(options["rtc_overlap_steps"], 4)
        self.assertEqual(options["rtc_frozen_steps"], 3)
        self.assertEqual(options["rtc_ramp_rate"], 6.0)
        self.assertNotIn("action_horizon", options)

    def test_child_handoff_skips_exact_elapsed_prefix_and_uses_commanded_boundary(self):
        # Measured q deliberately stays at zero while the target moves. A handoff
        # accepted after the command exceeds the configured per-step ceiling
        # proves the boundary is checked from the last commanded target rather
        # than from lagging measured q.
        with _ChildHarness(_LaggingRecordingBackend) as child:
            arm_step = MAX_ARM_STEP_RAD * 0.4
            original = _plan(12, arm_step=arm_step)
            child.commands.put(_rtc_start_command(1, original, action_budget=50), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            child.wait_for_target_count(2)

            child.commands.put(("rtc_snapshot", 1), timeout=0.2)
            snapshot = child.assert_status("rtc_snapshot")
            request_index = int(snapshot["action_index"])
            overlap = original.length - request_index
            self.assertGreaterEqual(request_index, 1)
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline:
                targets = child.backend.target_snapshot()
                if targets and targets[-1][0][0] > MAX_ARM_STEP_RAD:
                    break
                time.sleep(0.002)
            else:
                self.fail("old commanded target did not move beyond measured-q step tolerance")

            replacement = _plan(12, arm_step=arm_step)
            replacement.arm[:, 0] = (request_index + np.arange(replacement.length, dtype=np.float64)) * arm_step
            replacement.arm[:, 1] = 0.01 + np.arange(replacement.length, dtype=np.float64) * 0.002
            before_replace = len(child.backend.target_snapshot())
            child.commands.put(
                (
                    "rtc_replace",
                    2,
                    time.monotonic(),
                    1,
                    request_index,
                    overlap,
                    replacement.arm,
                    replacement.left_hand,
                    replacement.right_hand,
                    replacement.length,
                ),
                timeout=0.2,
            )
            acknowledgement = child.assert_status("rtc_replaced")
            elapsed = int(acknowledgement["action_index"])
            self.assertGreater(elapsed, 0)
            self.assertLess(elapsed, overlap)

            deadline = time.monotonic() + 0.3
            first_replacement = None
            while time.monotonic() < deadline:
                for arm_target, _, _ in child.backend.target_snapshot()[before_replace:]:
                    if arm_target[1] > 0.0:
                        first_replacement = arm_target
                        break
                if first_replacement is not None:
                    break
                time.sleep(0.002)
            self.assertIsNotNone(first_replacement)
            np.testing.assert_allclose(first_replacement, replacement.arm[elapsed], atol=0.0)
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

    def test_child_executes_exact_action_budget_then_enters_powered_hold(self):
        with _ChildHarness() as child:
            plan = _plan(8)
            child.commands.put(_rtc_start_command(1, plan, action_budget=3), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            self.assertEqual(child.assert_status("rtc_completed"), 3)

            targets = child.backend.target_snapshot()
            self.assertGreaterEqual(len(targets), 4)  # three actions plus measured HOLD
            for index in range(3):
                np.testing.assert_array_equal(targets[index][0], plan.arm[index])
            self.assertFalse(any(np.array_equal(target[0], plan.arm[3]) for target in targets))
            np.testing.assert_array_equal(targets[-1][0], plan.arm[2])
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

    def test_rtc_completion_holds_measured_arm_without_relaxing_commanded_grip(self):
        with _ChildHarness(_LaggingRecordingBackend) as child:
            plan = _gripping_plan(8)
            child.commands.put(_rtc_start_command(1, plan, action_budget=1), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            self.assertEqual(child.assert_status("rtc_completed"), 1)

            targets = child.backend.target_snapshot()
            self.assertGreaterEqual(len(targets), 2)
            np.testing.assert_array_equal(targets[-2][0], plan.arm[0])
            np.testing.assert_array_equal(targets[-2][1], plan.left_hand[0])
            np.testing.assert_array_equal(targets[-2][2], plan.right_hand[0])
            np.testing.assert_array_equal(targets[-1][0], np.zeros(14))
            np.testing.assert_array_equal(targets[-1][1], targets[-2][1])
            np.testing.assert_array_equal(targets[-1][2], targets[-2][2])
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

    def test_urgent_stop_preempts_synchronous_chunk_before_its_next_action(self):
        with _ChildHarness() as child:
            plan = _plan(16)
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            child.wait_for_target_count(2)
            requested_at = time.monotonic()
            child.urgent_hold.set()

            self.assertEqual(child.assert_status("holding"), 1)
            self.assertLess(time.monotonic() - requested_at, 0.2)
            targets = child.backend.target_snapshot()
            self.assertLess(len(targets), plan.length + 1)
            np.testing.assert_array_equal(targets[-1][0], targets[-2][0])
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 1)
            self.assertFalse(child.urgent_hold.is_set())

            # STOP keeps authority and permits a later plan generation.
            next_plan = _plan(1)
            child.commands.put(
                (
                    "chunk",
                    2,
                    time.monotonic(),
                    next_plan.arm,
                    next_plan.left_hand,
                    next_plan.right_hand,
                ),
                timeout=0.2,
            )
            self.assertEqual(child.assert_status("completed"), 2)

    def test_urgent_stop_and_barrier_hold_measured_arm_without_relaxing_commanded_grip(self):
        with _ChildHarness(_LaggingRecordingBackend) as child:
            plan = _gripping_plan(16)
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            child.wait_for_target_count(2)
            commanded_left = child.backend._left_target.copy()
            commanded_right = child.backend._right_target.copy()

            child.urgent_hold.set()
            self.assertEqual(child.assert_status("holding"), 1)
            immediate_hold = child.backend.target_snapshot()[-1]
            np.testing.assert_array_equal(immediate_hold[0], np.zeros(14))
            np.testing.assert_array_equal(immediate_hold[1], commanded_left)
            np.testing.assert_array_equal(immediate_hold[2], commanded_right)

            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 1)
            barrier_hold = child.backend.target_snapshot()[-1]
            np.testing.assert_array_equal(barrier_hold[0], np.zeros(14))
            np.testing.assert_array_equal(barrier_hold[1], commanded_left)
            np.testing.assert_array_equal(barrier_hold[2], commanded_right)
            self.assertFalse(child.urgent_hold.is_set())
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

    def test_urgent_stop_latch_keeps_publishing_without_advancing_policy_targets(self):
        with _ChildHarness() as child:
            plan = _plan(16)
            child.commands.put(
                (
                    "chunk",
                    1,
                    time.monotonic(),
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                ),
                timeout=0.2,
            )
            child.wait_for_target_count(2)
            child.urgent_hold.set()
            self.assertEqual(child.assert_status("holding"), 1)

            held_targets = child.backend.target_snapshot()
            held_publish_count = child.backend.publishes
            deadline = time.monotonic() + 0.3
            while child.backend.publishes < held_publish_count + 5 and time.monotonic() < deadline:
                time.sleep(0.002)

            self.assertGreaterEqual(child.backend.publishes, held_publish_count + 5)
            self.assertEqual(len(child.backend.target_snapshot()), len(held_targets))
            np.testing.assert_array_equal(child.backend.target_snapshot()[-1][0], held_targets[-1][0])
            self.assertTrue(child.urgent_hold.is_set())
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 1)
            self.assertFalse(child.urgent_hold.is_set())

    def test_urgent_stop_wins_when_sync_chunk_is_already_queued_but_unconsumed(self):
        with _ChildHarness(_GatedRecordingBackend) as child:
            backend = child.backend
            backend.pause_next_publish.set()
            self.assertTrue(backend.publish_paused.wait(timeout=0.5))
            actuator = _parent_handle_for_child(child)
            plan = _plan(8)
            plan.arm[:, 0] += 0.02

            try:
                sequence = actuator.submit(plan)
                actuator.request_immediate_hold()
            finally:
                backend.continue_publish.set()

            actuator.finish_immediate_hold()
            self.assertEqual(sequence, 1)
            self.assertEqual(actuator._sequence, 1)
            self.assertTrue(actuator._holding)
            self.assertFalse(actuator._chunk_in_flight)
            self.assertIsNone(actuator._pending_sequence)
            targets = backend.target_snapshot()
            self.assertGreaterEqual(len(targets), 1)
            for arm_target, _, _ in targets:
                np.testing.assert_array_equal(arm_target, np.zeros(14))
            self.assertTrue(child.thread.is_alive())

    def test_urgent_stop_wins_when_rtc_start_is_already_queued_but_unconsumed(self):
        with _ChildHarness(_GatedRecordingBackend) as child:
            backend = child.backend
            backend.pause_next_publish.set()
            self.assertTrue(backend.publish_paused.wait(timeout=0.5))
            actuator = _parent_handle_for_child(child)
            plan = _plan(8)
            plan.arm[:, 0] += 0.02
            result: list[int] = []
            errors: list[BaseException] = []

            def start_rtc() -> None:
                try:
                    result.append(actuator.start_rtc(plan, action_budget=20))
                except BaseException as exc:
                    errors.append(exc)

            starter = threading.Thread(target=start_rtc, daemon=True)
            starter.start()
            deadline = time.monotonic() + 0.5
            while child.commands.empty() and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertFalse(child.commands.empty())
            try:
                actuator.request_immediate_hold()
            finally:
                backend.continue_publish.set()
            starter.join(timeout=1.0)

            self.assertFalse(starter.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(result, [1])
            actuator.finish_immediate_hold()
            self.assertEqual(actuator._sequence, 1)
            self.assertTrue(actuator._holding)
            self.assertFalse(actuator._rtc_active)
            self.assertFalse(actuator._chunk_in_flight)
            targets = backend.target_snapshot()
            self.assertGreaterEqual(len(targets), 1)
            for arm_target, _, _ in targets:
                np.testing.assert_array_equal(arm_target, np.zeros(14))
            self.assertTrue(child.thread.is_alive())

    def test_urgent_stop_preempts_rtc_plan_and_keeps_powered_authority(self):
        with _ChildHarness() as child:
            plan = _plan(16)
            child.commands.put(_rtc_start_command(1, plan, action_budget=100), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            child.wait_for_target_count(2)
            requested_at = time.monotonic()
            child.urgent_hold.set()

            self.assertEqual(child.assert_status("holding"), 1)
            self.assertLess(time.monotonic() - requested_at, 0.2)
            targets = child.backend.target_snapshot()
            self.assertLess(len(targets), plan.length + 1)
            np.testing.assert_array_equal(targets[-1][0], targets[-2][0])
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

            child.commands.put(("urgent_hold_barrier",), timeout=0.2)
            self.assertEqual(child.assert_status("urgent_holding"), 1)
            self.assertFalse(child.urgent_hold.is_set())

    def test_child_underrun_holds_and_accepts_the_next_generation(self):
        with _ChildHarness() as child:
            short_plan = _plan(2)
            child.commands.put(_rtc_start_command(1, short_plan, action_budget=10), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            detail = child.assert_status("rtc_underrun")
            self.assertEqual(detail["sequence"], 1)
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

            # An underrun does not consume a replacement generation. The next
            # plan must therefore be sequence 2, not 3.
            next_plan = _plan(2)
            child.commands.put(_rtc_start_command(2, next_plan, action_budget=1), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 2)
            self.assertEqual(child.assert_status("rtc_completed"), 1)

    def test_child_stale_replacement_holds_without_consuming_sequence(self):
        with _ChildHarness() as child:
            plan = _plan(8)
            child.commands.put(_rtc_start_command(1, plan, action_budget=20), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 1)
            child.commands.put(
                (
                    "rtc_replace",
                    2,
                    time.monotonic(),
                    999,
                    0,
                    plan.length,
                    plan.arm,
                    plan.left_hand,
                    plan.right_hand,
                    plan.length,
                ),
                timeout=0.2,
            )
            message = child.assert_status("rtc_rejected")
            self.assertIn("stale", message)
            self.assertTrue(child.thread.is_alive())
            self.assertFalse(child.backend.released)

            child.commands.put(_rtc_start_command(2, plan, action_budget=1), timeout=0.2)
            self.assertEqual(child.assert_status("rtc_started"), 2)
            self.assertEqual(child.assert_status("rtc_completed"), 1)

    def test_parent_terminal_race_does_not_create_a_generation_gap(self):
        class _AliveProcess:
            @staticmethod
            def is_alive() -> bool:
                return True

        actuator = object.__new__(SafeG1Dex3Actuator)
        actuator._initialized = True
        actuator._holding = False
        actuator._chunk_in_flight = True
        actuator._pending_sequence = 7
        actuator._sequence = 7
        actuator._rtc_active = True
        actuator._rtc_terminal = None
        actuator._control_lock = threading.Lock()
        actuator._immediate_hold_requested = threading.Event()
        actuator._immediate_release_requested = threading.Event()
        actuator._stopped_acknowledged = False
        actuator._command_queue = queue.Queue(maxsize=1)
        actuator._status_queue = queue.Queue(maxsize=32)
        actuator._status_queue.put(("rtc_underrun", {"sequence": 7}))
        actuator._process = _AliveProcess()
        actuator._heartbeat = _LiveHeartbeat()

        with self.assertRaises(RtcTerminalEvent) as raised:
            actuator.replace_rtc(
                _plan(4),
                expected_sequence=7,
                request_index=1,
                expected_overlap=3,
            )
        self.assertEqual(raised.exception.outcome, "hold")
        self.assertEqual(actuator._sequence, 7)
        self.assertTrue(actuator._holding)

        with mock.patch.object(actuator, "_wait_status", return_value=8):
            next_sequence = actuator.start_rtc(_plan(4), action_budget=1)
        self.assertEqual(next_sequence, 8)
        command = actuator._command_queue.get_nowait()
        self.assertEqual(command[0], "rtc_start")
        self.assertEqual(command[1], 8)


if __name__ == "__main__":
    unittest.main()
