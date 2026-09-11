from __future__ import annotations

import contextlib
from types import ModuleType, SimpleNamespace
from unittest import mock
import sys
import unittest

import numpy as np

from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser, run, validate_args
from unitree_lerobot.eval_robot.g1_end_effectors import (
    DEX3_PROFILE,
    INSPIRE_DFX_PROFILE,
    get_end_effector_profile,
    inspire_dfx_dataset_contract,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    ARM_JOINT_NAMES,
    ActionChunk,
    COLOUR_VIDEO_KEYS,
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_EGO_VIEW_SHAPE,
    InitializationSpec,
    TASKS,
    make_observation,
    parse_action_plan,
    validate_model_contract,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.robot_control.g1_inspire_dfx import (
    INSPIRE_DFX_COMMAND_MAX_STEP,
    G1InspireDfxStateReader,
    InspireDfxCommandWriter,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    HandStateFreshnessGate,
    RobotState,
    SafeG1Dex3Actuator,
    XrPolicyOutputConditioner,
    _G1Dex3CommandBackend,
    _ramp_real_arm_authority,
    build_initialization_chunk,
)
from unitree_lerobot.utils.constants import ROBOT_CONFIGS


_CASE = unittest.TestCase()


def _fake_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _modality_config(action_horizon: int = 16) -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": list(COLOUR_VIDEO_KEYS)},
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


def _layout(hand_dof: int) -> dict:
    return {
        "left_arm": {"start": 0, "end": 7, "dim": 7},
        "right_arm": {"start": 7, "end": 14, "dim": 7},
        "left_hand": {"start": 14, "end": 14 + hand_dof, "dim": hand_dof},
        "right_hand": {
            "start": 14 + hand_dof,
            "end": 14 + 2 * hand_dof,
            "dim": hand_dof,
        },
    }


def _inspire_metadata() -> dict:
    profile = INSPIRE_DFX_PROFILE
    joint_names = list(ARM_JOINT_NAMES + profile.joint_names)
    vector_dim = 14 + 2 * profile.hand_dof
    layout = _layout(profile.hand_dof)
    return {
        "protocol_version": 1,
        "embodiment_tag": "new_embodiment",
        "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
        "dataset_contract": {
            "robot_type": profile.robot_type,
            "fps": 30.0,
            "observation_state_shape": [vector_dim],
            "action_shape": [vector_dim],
            "observation_state_names": joint_names,
            "action_names": joint_names,
            "state_layout": layout,
            "action_layout": layout,
            "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
            "video_shapes": {"ego_view": EXPECTED_EGO_VIEW_SHAPE},
            "end_effector": inspire_dfx_dataset_contract(),
        },
    }


def _profiles_are_exact_immutable_and_dex3_remains_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", "pick-red-cup"])
    assert args.end_effector == "dex3"
    assert get_end_effector_profile(args.end_effector) is DEX3_PROFILE
    assert INSPIRE_DFX_PROFILE.robot_type == "Unitree_G1_Inspire_HeadOnly"
    assert INSPIRE_DFX_PROFILE.hand_dof == 6
    assert len(INSPIRE_DFX_PROFILE.joint_names) == 12
    assert INSPIRE_DFX_PROFILE.max_step is None
    np.testing.assert_array_equal(INSPIRE_DFX_PROFILE.conditioned_step, np.full(6, 0.2))
    assert not INSPIRE_DFX_PROFILE.left_lower.flags.writeable
    with _CASE.assertRaises(ValueError):
        INSPIRE_DFX_PROFILE.left_lower[0] = -1.0
    assert "Unitree_G1_Inspire_HeadOnly" in ROBOT_CONFIGS
    inspire_dataset = ROBOT_CONFIGS["Unitree_G1_Inspire_HeadOnly"]
    assert len(inspire_dataset.motors) == 26
    assert inspire_dataset.cameras == ["ego_view"]


def _inspire_live_cli_requires_the_exact_authorized_gates() -> None:
    parser = build_parser()
    shadow = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
        ]
    )
    validate_args(shadow)

    live = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--actuate",
            "--network-interface",
            "eth-test",
            "--allow-unqualified-real",
        ]
    )
    validate_args(live)

    missing_override = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--actuate",
            "--network-interface",
            "eth-test",
        ]
    )
    with _CASE.assertRaisesRegex(DeploymentError, "allow-unqualified-real"):
        validate_args(missing_override)

    unconditioned = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--actuate",
            "--network-interface",
            "eth-test",
            "--allow-unqualified-real",
            "--command-conditioning",
            "none",
        ]
    )
    with _CASE.assertRaisesRegex(DeploymentError, "requires --command-conditioning xr"):
        validate_args(unconditioned)

    simulation = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--sim",
        ]
    )
    with _CASE.assertRaisesRegex(DeploymentError, "simulation is not qualified"):
        validate_args(simulation)

    fixed_home = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--initialization",
            "xr-home",
            "--actuate",
            "--network-interface",
            "eth-test",
            "--allow-unqualified-real",
        ]
    )
    with _CASE.assertRaisesRegex(DeploymentError, "freshly measured state"):
        validate_args(fixed_home)


def _inspire_metadata_requires_native_26d_shapes_and_layouts() -> None:
    contract = validate_model_contract(_modality_config(), end_effector="inspire-dfx")
    assert contract.end_effector == "inspire-dfx"
    validate_policy_metadata(_inspire_metadata(), end_effector="inspire-dfx")

    missing_layout = _inspire_metadata()
    del missing_layout["dataset_contract"]["action_layout"]
    with _CASE.assertRaisesRegex(DeploymentError, "action_layout"):
        validate_policy_metadata(missing_layout, end_effector="inspire-dfx")

    wrong_shape = _inspire_metadata()
    wrong_shape["dataset_contract"]["observation_state_shape"] = [28]
    with _CASE.assertRaisesRegex(DeploymentError, "observation_state_shape"):
        validate_policy_metadata(wrong_shape, end_effector="inspire-dfx")

    missing_provenance = _inspire_metadata()
    del missing_provenance["dataset_contract"]["end_effector"]
    with _CASE.assertRaisesRegex(DeploymentError, "end_effector"):
        validate_policy_metadata(missing_provenance, end_effector="inspire-dfx")

    dex3_checkpoint = _inspire_metadata()
    dex3_checkpoint["dataset_contract"]["robot_type"] = DEX3_PROFILE.robot_type
    with _CASE.assertRaisesRegex(DeploymentError, "robot_type"):
        validate_policy_metadata(dex3_checkpoint, end_effector="inspire-dfx")

    ftp_checkpoint = _inspire_metadata()
    ftp_checkpoint["dataset_contract"]["end_effector"]["protocol"] = "ftp"
    with _CASE.assertRaisesRegex(DeploymentError, "end_effector"):
        validate_policy_metadata(ftp_checkpoint, end_effector="inspire-dfx")


def _inspire_native_six_dof_observation_and_action_contract() -> None:
    observation = make_observation(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros(14),
        np.zeros(6),
        np.ones(6),
        TASKS["pick-red-cup"],
        end_effector="inspire-dfx",
    )
    assert observation["state"]["left_hand"].shape == (1, 1, 6)
    assert observation["state"]["right_hand"].shape == (1, 1, 6)

    horizon = 16
    action = {
        "left_arm": np.zeros((1, horizon, 7), dtype=np.float32),
        "right_arm": np.zeros((1, horizon, 7), dtype=np.float32),
        "left_hand": np.zeros((1, horizon, 6), dtype=np.float32),
        "right_hand": np.ones((1, horizon, 6), dtype=np.float32),
    }
    plan = parse_action_plan(
        action,
        horizon,
        np.zeros(14),
        np.zeros(6),
        np.ones(6),
        end_effector="inspire-dfx",
    )
    assert plan.end_effector == "inspire-dfx"
    assert plan.left_hand.shape == (horizon, 6)

    action["left_hand"] = np.zeros((1, horizon, 7), dtype=np.float32)
    with _CASE.assertRaisesRegex(DeploymentError, r"shape \(1, 16, 6\)"):
        parse_action_plan(
            action,
            horizon,
            np.zeros(14),
            np.zeros(6),
            np.ones(6),
            end_effector="inspire-dfx",
        )


def _inspire_checkpoint_mismatch_stops_before_dds_initialization() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--no-gravity-feedforward",
        ]
    )
    metadata = _inspire_metadata()
    metadata["dataset_contract"]["action_shape"] = [28]

    class FakePolicy:
        def __init__(self, *_args, **_kwargs):
            pass

        def ping(self) -> bool:
            return True

        def get_modality_config(self) -> dict:
            return _modality_config()

        def get_policy_metadata(self) -> dict:
            return metadata

        def close(self) -> None:
            pass

    with (
        mock.patch("unitree_lerobot.eval_robot.eval_groot_g1.Gr00tClient", FakePolicy),
        mock.patch("unitree_lerobot.eval_robot.eval_groot_g1.initialize_dds") as initialize_dds,
        _CASE.assertRaisesRegex(DeploymentError, "action_shape"),
    ):
        run(args)
    initialize_dds.assert_not_called()


def _dfx_combined_state_uses_official_states_field_and_freezes_only_lost_side() -> None:
    class FakeSubscriber:
        instances: dict[str, "FakeSubscriber"] = {}

        def __init__(self, topic: str, _message_type: object):
            self.topic = topic
            self.handler = None
            self.closed = False
            self.instances[topic] = self

        def Init(self, handler=None, queueLen=0):
            self.handler = handler
            self.queue_len = queueLen

        def Close(self):
            self.closed = True

    fake_modules = {
        "unitree_lerobot.eval_robot.robot_control.robot_arm": _fake_module(
            "robot_arm", G1_29_JointArmIndex=tuple(range(14))
        ),
        "unitree_sdk2py": _fake_module("unitree_sdk2py"),
        "unitree_sdk2py.core": _fake_module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": _fake_module("unitree_sdk2py.core.channel", ChannelSubscriber=FakeSubscriber),
        "unitree_sdk2py.idl": _fake_module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_go": _fake_module("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": _fake_module("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg.dds_", MotorStates_=object
        ),
        "unitree_sdk2py.idl.unitree_hg": _fake_module("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": _fake_module("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_", LowState_=object
        ),
    }

    def hands(*, right_q: float, left_q: float, right_lost: int, left_lost: int):
        # Official MotorStates_ field/order: states=[right six, left six].
        return SimpleNamespace(
            states=[
                *[SimpleNamespace(q=right_q, lost=right_lost) for _ in range(6)],
                *[SimpleNamespace(q=left_q, lost=left_lost) for _ in range(6)],
            ]
        )

    FakeSubscriber.instances.clear()
    with mock.patch.dict(sys.modules, fake_modules):
        reader = G1InspireDfxStateReader(max_age_s=1.0)
        arm_message = SimpleNamespace(
            mode_machine=6,
            motor_state=[SimpleNamespace(q=0.0, dq=0.0) for _ in range(14)],
        )
        FakeSubscriber.instances["rt/lowstate"].handler(arm_message)

        # First callback is baseline-only.  This prevents an uninitialized
        # cached q from being accepted when DFX starts during a failed read.
        FakeSubscriber.instances["rt/inspire/state"].handler(hands(right_q=0.4, left_q=0.0, right_lost=0, left_lost=0))
        with _CASE.assertRaisesRegex(TimeoutError, "two consecutive clean"):
            reader.latest()

        FakeSubscriber.instances["rt/inspire/state"].handler(hands(right_q=0.4, left_q=0.0, right_lost=0, left_lost=0))
        initial = reader.latest()
        np.testing.assert_array_equal(initial.left_hand, np.zeros(6))
        np.testing.assert_array_equal(initial.right_hand, np.full(6, 0.4))

        # Combined callbacks continue, but a right-side lost increment accepts
        # the new left sample only and retains the last known-good right q/time.
        FakeSubscriber.instances["rt/inspire/state"].handler(hands(right_q=0.9, left_q=0.2, right_lost=1, left_lost=0))
        one_side_drop = reader.latest()
        np.testing.assert_array_equal(one_side_drop.left_hand, np.full(6, 0.2))
        np.testing.assert_array_equal(one_side_drop.right_hand, np.full(6, 0.4))
        assert one_side_drop.left_hand_received_at > initial.left_hand_received_at
        assert one_side_drop.right_hand_received_at == initial.right_hand_received_at
        assert one_side_drop.right_hand_lost == (1,) * 6

        # The next unchanged lost counters prove a clean right-hand read.
        FakeSubscriber.instances["rt/inspire/state"].handler(hands(right_q=0.7, left_q=0.3, right_lost=1, left_lost=0))
        recovered = reader.latest()
        np.testing.assert_array_equal(recovered.right_hand, np.full(6, 0.7))
        diagnostics = reader.diagnostics()
        assert diagnostics["right"]["drop_events"] == 1
        assert diagnostics["right"]["lost_increments"] == 1
        assert diagnostics["left"]["drop_events"] == 0

        # A service restart/counter regression immediately invalidates only
        # that side and requires a new baseline plus a later clean sample.
        FakeSubscriber.instances["rt/inspire/state"].handler(
            hands(right_q=0.8, left_q=0.4, right_lost=0, left_lost=0)
        )
        with _CASE.assertRaisesRegex(TimeoutError, "right"):
            reader.latest()
        assert reader.diagnostics()["right"]["resets"] == 1

        # A syntactically wrong combined state updates neither hand.
        FakeSubscriber.instances["rt/inspire/state"].handler(
            SimpleNamespace(states=[SimpleNamespace(q=0.5, lost=1) for _ in range(11)])
        )
        assert reader.diagnostics()["malformed_messages"] == 1
        reader.close()

    assert set(FakeSubscriber.instances) == {"rt/lowstate", "rt/inspire/state"}
    assert all(subscriber.closed for subscriber in FakeSubscriber.instances.values())


def _dfx_state_reader_closes_both_subscribers_when_second_init_fails() -> None:
    class FakeSubscriber:
        instances: list["FakeSubscriber"] = []

        def __init__(self, topic: str, _message_type: object):
            self.topic = topic
            self.closed = False
            self.instances.append(self)

        def Init(self, handler=None, queueLen=0):
            del handler, queueLen
            if self.topic == "rt/inspire/state":
                raise RuntimeError("synthetic second Init failure")

        def Close(self):
            self.closed = True

    fake_modules = {
        "unitree_lerobot.eval_robot.robot_control.robot_arm": _fake_module(
            "robot_arm", G1_29_JointArmIndex=tuple(range(14))
        ),
        "unitree_sdk2py": _fake_module("unitree_sdk2py"),
        "unitree_sdk2py.core": _fake_module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": _fake_module(
            "unitree_sdk2py.core.channel", ChannelSubscriber=FakeSubscriber
        ),
        "unitree_sdk2py.idl": _fake_module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_go": _fake_module("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg"
        ),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg.dds_", MotorStates_=object
        ),
        "unitree_sdk2py.idl.unitree_hg": _fake_module("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": _fake_module(
            "unitree_sdk2py.idl.unitree_hg.msg"
        ),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_", LowState_=object
        ),
    }

    FakeSubscriber.instances.clear()
    with (
        mock.patch.dict(sys.modules, fake_modules),
        _CASE.assertRaisesRegex(RuntimeError, "second Init failure"),
    ):
        G1InspireDfxStateReader(max_age_s=1.0)
    assert len(FakeSubscriber.instances) == 2
    assert all(subscriber.closed for subscriber in FakeSubscriber.instances)

    class ConstructorFailingSubscriber(FakeSubscriber):
        instances: list["ConstructorFailingSubscriber"] = []

        def __init__(self, topic: str, message_type: object):
            if topic == "rt/inspire/state":
                raise RuntimeError("synthetic second constructor failure")
            super().__init__(topic, message_type)

        def Init(self, handler=None, queueLen=0):
            del handler, queueLen

    constructor_modules = dict(fake_modules)
    constructor_modules["unitree_sdk2py.core.channel"] = _fake_module(
        "unitree_sdk2py.core.channel",
        ChannelSubscriber=ConstructorFailingSubscriber,
    )
    ConstructorFailingSubscriber.instances.clear()
    with (
        mock.patch.dict(sys.modules, constructor_modules),
        _CASE.assertRaisesRegex(RuntimeError, "second constructor failure"),
    ):
        G1InspireDfxStateReader(max_age_s=1.0)
    assert len(ConstructorFailingSubscriber.instances) == 1
    assert ConstructorFailingSubscriber.instances[0].closed


def _dfx_combined_writer_is_atomic_right_first_and_never_writes_during_init() -> None:
    class FakePublisher:
        instances: list["FakePublisher"] = []

        def __init__(self, topic: str, message_type: object):
            self.topic = topic
            self.message_type = message_type
            self.initialized = False
            self.write_result = True
            self.writes: list[tuple[list[float], float]] = []
            self.closed = False
            self.instances.append(self)

        def Init(self):
            self.initialized = True

        def Write(self, message, timeout=None):
            self.writes.append(([float(command.q) for command in message.cmds], float(timeout)))
            return self.write_result

        def Close(self):
            self.closed = True

    class FakeMotorCmds:
        def __init__(self):
            self.cmds = []

    fake_modules = {
        "unitree_sdk2py": _fake_module("unitree_sdk2py"),
        "unitree_sdk2py.core": _fake_module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": _fake_module(
            "unitree_sdk2py.core.channel",
            ChannelPublisher=FakePublisher,
        ),
        "unitree_sdk2py.idl": _fake_module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": _fake_module(
            "unitree_sdk2py.idl.default",
            unitree_go_msg_dds__MotorCmd_=lambda: SimpleNamespace(q=0.0),
        ),
        "unitree_sdk2py.idl.unitree_go": _fake_module("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": _fake_module("unitree_sdk2py.idl.unitree_go.msg"),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg.dds_",
            MotorCmds_=FakeMotorCmds,
        ),
    }
    FakePublisher.instances.clear()
    with mock.patch.dict(sys.modules, fake_modules):
        writer = InspireDfxCommandWriter(np.zeros(6), np.zeros(6))
        publisher = FakePublisher.instances[0]
        assert publisher.topic == "rt/inspire/cmd"
        assert publisher.initialized
        assert publisher.writes == []

        completed_at, left, right = writer.write(np.full(6, 0.2), np.full(6, 0.1))
        assert completed_at > 0.0
        np.testing.assert_array_equal(left, np.full(6, 0.2))
        np.testing.assert_array_equal(right, np.full(6, 0.1))
        assert publisher.writes == [([0.1] * 6 + [0.2] * 6, 0.5)]

        previous_wire = list(publisher.writes[-1][0])
        with _CASE.assertRaisesRegex(DeploymentError, "COMMAND_MAX_STEP"):
            writer.write(np.full(6, 0.5), np.full(6, 0.1))
        assert publisher.writes[-1][0] == previous_wire
        assert len(publisher.writes) == 1

        publisher.write_result = False
        with _CASE.assertRaisesRegex(DeploymentError, "combined DDS Write failed"):
            writer.write(np.full(6, 0.3), np.full(6, 0.2))
        np.testing.assert_array_equal(writer._last_left, np.full(6, 0.2))
        np.testing.assert_array_equal(writer._last_right, np.full(6, 0.1))
        publisher.write_result = True
        writer.refresh_last_successful()
        assert publisher.writes[-1] == ([0.1] * 6 + [0.2] * 6, 0.5)
        with _CASE.assertRaisesRegex(DeploymentError, "cannot be reseeded after a write"):
            writer.reseed_before_first_write(np.full(6, 0.8), np.full(6, 0.8))
        writer.close()
        assert publisher.closed

        # Construction-time state can legitimately be far from the final
        # measured pre-arm hold. Reseeding is allowed only before the first
        # lease packet and avoids an artificial first-write discontinuity.
        second = InspireDfxCommandWriter(np.zeros(6), np.zeros(6))
        second.reseed_before_first_write(np.full(6, 0.8), np.full(6, 0.7))
        second.write(np.full(6, 0.8), np.full(6, 0.7))
        assert FakePublisher.instances[-1].writes == [([0.7] * 6 + [0.8] * 6, 0.5)]
        second.close()


def _dfx_writer_closes_initialized_publisher_when_message_build_fails() -> None:
    class FakePublisher:
        instance: "FakePublisher | None" = None

        def __init__(self, _topic: str, _message_type: object):
            self.closed = False
            self.initialized = False
            self.__class__.instance = self

        def Init(self):
            self.initialized = True

        def Close(self):
            self.closed = True

    class BrokenMotorCmds:
        def __init__(self):
            raise RuntimeError("synthetic message construction failure")

    fake_modules = {
        "unitree_sdk2py": _fake_module("unitree_sdk2py"),
        "unitree_sdk2py.core": _fake_module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": _fake_module(
            "unitree_sdk2py.core.channel", ChannelPublisher=FakePublisher
        ),
        "unitree_sdk2py.idl": _fake_module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": _fake_module(
            "unitree_sdk2py.idl.default",
            unitree_go_msg_dds__MotorCmd_=lambda: SimpleNamespace(q=0.0),
        ),
        "unitree_sdk2py.idl.unitree_go": _fake_module("unitree_sdk2py.idl.unitree_go"),
        "unitree_sdk2py.idl.unitree_go.msg": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg"
        ),
        "unitree_sdk2py.idl.unitree_go.msg.dds_": _fake_module(
            "unitree_sdk2py.idl.unitree_go.msg.dds_", MotorCmds_=BrokenMotorCmds
        ),
    }

    with (
        mock.patch.dict(sys.modules, fake_modules),
        _CASE.assertRaisesRegex(RuntimeError, "message construction failure"),
    ):
        InspireDfxCommandWriter(np.zeros(6), np.zeros(6))
    assert FakePublisher.instance is not None
    assert FakePublisher.instance.initialized
    assert FakePublisher.instance.closed


def _dfx_backend_records_one_success_timestamp_for_both_hands() -> None:
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_DFX_PROFILE
    backend._left_target = np.full(6, 0.2)
    backend._right_target = np.full(6, 0.4)
    backend._left_hand_publish_history = []
    backend._right_hand_publish_history = []
    backend._authority_ramp_timing_enabled = False
    backend._hand_writer = SimpleNamespace(
        write=mock.Mock(return_value=(12.5, np.full(6, 0.2), np.full(6, 0.4)))
    )

    backend._publish_hands()

    assert backend._left_hand_publish_history[0].completed_at == 12.5
    assert backend._right_hand_publish_history[0].completed_at == 12.5
    np.testing.assert_array_equal(backend._left_hand_publish_history[0].target, np.full(6, 0.2))
    np.testing.assert_array_equal(backend._right_hand_publish_history[0].target, np.full(6, 0.4))

    backend._left_hand_publish_history.clear()
    backend._right_hand_publish_history.clear()
    backend._hand_writer.write = mock.Mock(
        side_effect=DeploymentError("synthetic combined write failure")
    )
    with _CASE.assertRaisesRegex(DeploymentError, "synthetic combined write failure"):
        backend._publish_hands()
    assert backend._left_hand_publish_history == []
    assert backend._right_hand_publish_history == []


def _dfx_backend_constructor_unwinds_every_partial_resource() -> None:
    resources: dict[str, SimpleNamespace] = {}

    def resource(name: str) -> SimpleNamespace:
        item = SimpleNamespace(closed=False)
        item.close = lambda: setattr(item, "closed", True)
        item.Close = item.close
        resources[name] = item
        return item

    initial = RobotState(
        captured_at=1.0,
        mode_machine=6,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.zeros(6),
        right_hand=np.zeros(6),
    )

    class FakeReader:
        def __init__(self, **_kwargs):
            self.closed = False
            resources["reader"] = self

        @staticmethod
        def read(timeout_s: float):
            assert timeout_s == 5.0
            return initial

        def close(self):
            self.closed = True

    def fail_after_partial_construction(backend, _initial):
        backend._arm_publisher = resource("arm")
        backend._hand_writer = resource("hand")
        raise RuntimeError("synthetic backend construction failure")

    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    inspire_module = "unitree_lerobot.eval_robot.robot_control.g1_inspire_dfx"
    with (
        mock.patch(f"{module}.initialize_dds"),
        mock.patch(f"{inspire_module}.G1InspireDfxStateReader", FakeReader),
        mock.patch.object(
            _G1Dex3CommandBackend,
            "_initialize_command_resources",
            fail_after_partial_construction,
        ),
        _CASE.assertRaisesRegex(RuntimeError, "synthetic backend construction failure"),
    ):
        _G1Dex3CommandBackend(
            False,
            "eth-test",
            gravity_feedforward=False,
            end_effector="inspire-dfx",
        )

    assert resources["reader"].closed
    assert resources["arm"].closed
    assert resources["hand"].closed


def _dfx_backend_cleanup_attempts_every_partial_resource() -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail

        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

        def Close(self):
            self.close()

    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend._hand_writer = Resource("hand", fail=True)
    backend._right_publisher = Resource("right")
    backend._left_publisher = Resource("left")
    backend._arm_publisher = Resource("arm")
    backend.reader = Resource("reader")

    with _CASE.assertRaisesRegex(DeploymentError, "hand failed"):
        backend.close()
    assert calls == ["hand", "right", "left", "arm", "reader"]
    assert backend._hand_writer is not None
    assert backend._right_publisher is None
    assert backend._left_publisher is None
    assert backend._arm_publisher is None
    assert backend.reader is None


def _dfx_authority_ramp_refreshes_the_lease_after_every_arm_write() -> None:
    now = [10.0]
    events: list[tuple[str, float]] = []

    class ClockedStop:
        @staticmethod
        def is_set() -> bool:
            return False

        @staticmethod
        def wait(timeout: float) -> bool:
            now[0] += max(0.0, float(timeout))
            return False

    class Heartbeat:
        value = 10.0

        @staticmethod
        def get_lock():
            return contextlib.nullcontext()

    class Backend:
        simulation = False
        profile = INSPIRE_DFX_PROFILE

        def __init__(self):
            self._arm_target = np.zeros(14)
            self._left_target = np.zeros(6)
            self._right_target = np.zeros(6)

        @staticmethod
        def _validate_prearm_takeover_state(_state):
            return None

        def _reseed_inspire_hands_before_first_write(self, state):
            self._left_target = state.left_hand.copy()
            self._right_target = state.right_hand.copy()
            events.append(("reseed", now[0]))

        @staticmethod
        def set_weight(_weight):
            return None

        def state(self):
            return RobotState(
                captured_at=now[0],
                mode_machine=6,
                arm=np.zeros(14),
                arm_dq=np.zeros(14),
                left_hand=np.zeros(6),
                right_hand=np.zeros(6),
                arm_received_at=now[0],
            )

        @staticmethod
        def _publish_arm(**_kwargs):
            events.append(("arm", now[0]))

        @staticmethod
        def _publish_hands():
            events.append(("hands", now[0]))

    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    with (
        mock.patch(f"{module}.time.monotonic", side_effect=lambda: now[0]),
        mock.patch(f"{module}.PUBLISH_HZ", 10.0),
        mock.patch(f"{module}.HEARTBEAT_TIMEOUT_S", 10.0),
        mock.patch(f"{module}.ARM_GRAVITY_RAMP_S", 1.2),
        mock.patch(f"{module}.ARM_TAKEOVER_SETTLE_DWELL_S", 0.2),
        mock.patch(f"{module}.ARM_TAKEOVER_SETTLE_TIMEOUT_S", 0.5),
    ):
        assert _ramp_real_arm_authority(Backend(), ClockedStop(), Heartbeat())

    assert events[:3] == [("reseed", 10.0), ("arm", 10.0), ("hands", 10.0)]
    hand_times = [timestamp for kind, timestamp in events if kind == "hands"]
    assert hand_times[-1] - hand_times[0] > 1.0
    assert max(np.diff(hand_times)) <= 0.1 + 1e-12
    for index, (kind, _timestamp) in enumerate(events):
        if kind == "hands":
            assert index > 0 and events[index - 1][0] == "arm"


def _dfx_release_holds_lease_until_arm_authority_is_zero_then_closes() -> None:
    now = [20.0]
    events: list[tuple[str, float, float | None]] = []
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_DFX_PROFILE
    backend._released = False
    backend._has_published = True
    backend.simulation = False
    backend._weight = 1.0
    backend._hand_writer = SimpleNamespace(has_written=True)
    backend._publish_last_arm_for_release = lambda: events.append(
        ("arm", now[0], backend._weight)
    )
    backend._refresh_last_successful_hands_for_release = lambda: events.append(
        ("hands", now[0], None)
    )
    backend._stop_hands = lambda _callback=None: events.append(("close", now[0], None))

    def sleep(duration: float) -> None:
        now[0] += max(0.0, float(duration))

    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    with (
        mock.patch(f"{module}.time.monotonic", side_effect=lambda: now[0]),
        mock.patch(f"{module}.time.sleep", side_effect=sleep),
        mock.patch(f"{module}.PUBLISH_HZ", 10.0),
        mock.patch(f"{module}.ARM_RELEASE_RAMP_S", 1.2),
    ):
        backend.release()

    assert events[-1][0] == "close"
    assert events[-2][0] == "hands"
    assert events[-3][0] == "arm" and events[-3][2] == 0.0
    hand_times = [timestamp for kind, timestamp, _weight in events if kind == "hands"]
    assert hand_times[-1] - hand_times[0] > 1.0
    assert max(np.diff(hand_times)) <= 0.1 + 1e-12
    for index, (kind, _timestamp, _weight) in enumerate(events[:-1]):
        if kind == "hands":
            assert index > 0 and events[index - 1][0] == "arm"


def _dfx_release_never_acquires_a_hand_lease_that_was_not_written() -> None:
    events: list[str] = []
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_DFX_PROFILE
    backend._released = False
    backend._has_published = True  # The first arm takeover Write succeeded.
    backend.simulation = False
    backend._weight = 0.0
    backend._hand_writer = SimpleNamespace(has_written=False)
    backend._publish_last_arm_for_release = lambda: events.append("arm")
    backend._refresh_last_successful_hands_for_release = lambda: events.append("hands")
    backend._stop_hands = lambda _callback=None: events.append("close")

    backend.release()

    assert events == ["arm", "close"]


def _inspire_conditioner_and_internal_warmup_use_native_six_dof_and_point_two_steps() -> None:
    conditioner = XrPolicyOutputConditioner(INSPIRE_DFX_PROFILE)
    arm = np.zeros(14)
    left = np.zeros(6)
    right = np.zeros(6)
    conditioner.reset(arm, left, right)
    conditioner.set_desired(arm, np.ones(6), np.ones(6), now=1.0)
    first = conditioner.next_command(arm, arm, left, right, now=1.0)
    assert first.end_effector == "inspire-dfx"
    np.testing.assert_allclose(first.left_hand[0], INSPIRE_DFX_COMMAND_MAX_STEP)
    np.testing.assert_allclose(first.right_hand[0], INSPIRE_DFX_COMMAND_MAX_STEP)

    state = RobotState(
        captured_at=1.0,
        mode_machine=6,
        arm=arm,
        arm_dq=np.zeros(14),
        left_hand=left,
        right_hand=right,
    )
    measured = InitializationSpec(
        mode="measured",
        label="measured",
        arm=None,
        left_hand=None,
        right_hand=None,
        end_effector="inspire-dfx",
    )
    hold = build_initialization_chunk(state, measured)
    assert hold.end_effector == "inspire-dfx"
    assert hold.left_hand.shape == (1, 6)

    warmup = InitializationSpec(
        mode="pose-file",
        label="first policy target",
        arm=np.zeros(14),
        left_hand=np.ones(6),
        right_hand=np.ones(6),
        end_effector="inspire-dfx",
    )
    with _CASE.assertRaisesRegex(DeploymentError, "internal validated policy Warmup2"):
        build_initialization_chunk(state, warmup)
    path = build_initialization_chunk(state, warmup, allow_policy_warm_start=True)
    assert path.end_effector == "inspire-dfx"
    assert np.max(np.abs(np.diff(np.vstack((left, path.left_hand)), axis=0))) <= 0.2 + 1e-12


def _dfx_lost_increment_enters_pause_immediately_and_needs_three_clean_pairs() -> None:
    gate = HandStateFreshnessGate()

    def state(at: float, lost: int) -> RobotState:
        return RobotState(
            captured_at=at,
            mode_machine=6,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(6),
            right_hand=np.zeros(6),
            left_hand_received_at=at,
            right_hand_received_at=at,
            left_hand_lost=(lost,) * 6,
            right_hand_lost=(0,) * 6,
        )

    assert gate.check(state(1.000, 0), now=1.001).ready
    entered = gate.check(state(1.010, 1), now=1.011)
    assert entered.entered and not entered.ready and entered.stale_hands == ("left",)
    for index in range(2):
        progress = gate.check(state(1.020 + index * 0.010, 1), now=1.021 + index * 0.010)
        assert progress.recovery_progressed and not progress.ready
    recovered = gate.check(state(1.040, 1), now=1.041)
    assert recovered.ready and recovered.recovered and recovered.fresh_samples == 3


def _dfx_pause_age_enters_operator_hold_even_while_samples_are_fresh() -> None:
    gate = HandStateFreshnessGate()

    def state(at: float, lost: int) -> RobotState:
        return RobotState(
            captured_at=at,
            mode_machine=6,
            arm=np.zeros(14),
            arm_dq=np.zeros(14),
            left_hand=np.zeros(6),
            right_hand=np.zeros(6),
            left_hand_received_at=at,
            right_hand_received_at=at,
            left_hand_lost=(lost,) * 6,
            right_hand_lost=(0,) * 6,
        )

    assert gate.check(state(1.0, 0), now=1.0).ready
    assert gate.check(state(1.1, 1), now=1.1).entered
    # One clean pair is deliberately insufficient for recovery. Even though
    # that pair is fresh, total pause duration must still revoke auto-resume.
    result = gate.check(state(2.35, 1), now=2.35)
    assert not result.ready
    assert result.operator_hold_entered
    assert result.pause_s >= 1.25


def _actuator_profile_identity_is_passed_to_child_and_mismatch_fails_before_queue() -> None:
    context = mock.Mock()
    process = mock.Mock()
    context.Process.return_value = process
    context.Queue.side_effect = [mock.Mock(), mock.Mock()]
    context.Event.side_effect = [mock.Mock(), mock.Mock()]
    context.Value.return_value = mock.Mock()
    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    with mock.patch(f"{module}.mp.get_context", return_value=context):
        actuator = SafeG1Dex3Actuator(
            False,
            "eth-test",
            "xr",
            end_effector="inspire-dfx",
        )
    assert actuator.end_effector == "inspire-dfx"
    assert context.Process.call_args.kwargs["kwargs"]["end_effector"] == "inspire-dfx"

    wrong = ActionChunk(
        arm=np.zeros((1, 14)),
        left_hand=np.zeros((1, 7)),
        right_hand=np.zeros((1, 7)),
    )
    with _CASE.assertRaisesRegex(DeploymentError, "payload is tagged 'dex3'"):
        actuator.submit(wrong)


class InspireDfxShadowTests(unittest.TestCase):
    test_profiles = staticmethod(_profiles_are_exact_immutable_and_dex3_remains_default)
    test_cli_gates = staticmethod(_inspire_live_cli_requires_the_exact_authorized_gates)
    test_metadata = staticmethod(_inspire_metadata_requires_native_26d_shapes_and_layouts)
    test_native_shapes = staticmethod(_inspire_native_six_dof_observation_and_action_contract)
    test_pre_dds_mismatch = staticmethod(_inspire_checkpoint_mismatch_stops_before_dds_initialization)
    test_dfx_lost = staticmethod(_dfx_combined_state_uses_official_states_field_and_freezes_only_lost_side)
    test_dfx_state_init_cleanup = staticmethod(
        _dfx_state_reader_closes_both_subscribers_when_second_init_fails
    )
    test_dfx_writer = staticmethod(_dfx_combined_writer_is_atomic_right_first_and_never_writes_during_init)
    test_dfx_writer_init_cleanup = staticmethod(
        _dfx_writer_closes_initialized_publisher_when_message_build_fails
    )
    test_dfx_backend_history = staticmethod(_dfx_backend_records_one_success_timestamp_for_both_hands)
    test_dfx_backend_constructor_cleanup = staticmethod(
        _dfx_backend_constructor_unwinds_every_partial_resource
    )
    test_dfx_backend_cleanup = staticmethod(
        _dfx_backend_cleanup_attempts_every_partial_resource
    )
    test_dfx_authority_lease = staticmethod(
        _dfx_authority_ramp_refreshes_the_lease_after_every_arm_write
    )
    test_dfx_release_lease = staticmethod(
        _dfx_release_holds_lease_until_arm_authority_is_zero_then_closes
    )
    test_dfx_release_without_lease = staticmethod(
        _dfx_release_never_acquires_a_hand_lease_that_was_not_written
    )
    test_conditioner_and_warmup = staticmethod(
        _inspire_conditioner_and_internal_warmup_use_native_six_dof_and_point_two_steps
    )
    test_lost_immediate_pause = staticmethod(
        _dfx_lost_increment_enters_pause_immediately_and_needs_three_clean_pairs
    )
    test_pause_age_operator_hold = staticmethod(
        _dfx_pause_age_enters_operator_hold_even_while_samples_are_fresh
    )
    test_actuator_identity = staticmethod(
        _actuator_profile_identity_is_passed_to_child_and_mismatch_fails_before_queue
    )
