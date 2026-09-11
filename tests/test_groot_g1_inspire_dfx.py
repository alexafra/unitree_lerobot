from __future__ import annotations

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
    COLOUR_VIDEO_KEYS,
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_EGO_VIEW_SHAPE,
    TASKS,
    make_observation,
    parse_action_plan,
    validate_model_contract,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.robot_control.g1_inspire_dfx import G1InspireDfxStateReader
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
    assert not INSPIRE_DFX_PROFILE.left_lower.flags.writeable
    with _CASE.assertRaises(ValueError):
        INSPIRE_DFX_PROFILE.left_lower[0] = -1.0
    assert "Unitree_G1_Inspire_HeadOnly" in ROBOT_CONFIGS
    inspire_dataset = ROBOT_CONFIGS["Unitree_G1_Inspire_HeadOnly"]
    assert len(inspire_dataset.motors) == 26
    assert inspire_dataset.cameras == ["ego_view"]


def _inspire_shadow_cli_is_explicit_and_live_paths_fail_before_dds() -> None:
    parser = build_parser()
    shadow = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-dfx",
            "--no-warmup1",
            "--no-gravity-feedforward",
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
            "--no-gravity-feedforward",
            "--actuate",
            "--network-interface",
            "eth-test",
            "--allow-unqualified-real",
        ]
    )
    with _CASE.assertRaisesRegex(DeploymentError, "shadow/read-only only"):
        validate_args(live)


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


class InspireDfxShadowTests(unittest.TestCase):
    test_profiles = staticmethod(_profiles_are_exact_immutable_and_dex3_remains_default)
    test_cli_gates = staticmethod(_inspire_shadow_cli_is_explicit_and_live_paths_fail_before_dds)
    test_metadata = staticmethod(_inspire_metadata_requires_native_26d_shapes_and_layouts)
    test_native_shapes = staticmethod(_inspire_native_six_dof_observation_and_action_contract)
    test_pre_dds_mismatch = staticmethod(_inspire_checkpoint_mismatch_stops_before_dds_initialization)
    test_dfx_lost = staticmethod(_dfx_combined_state_uses_official_states_field_and_freezes_only_lost_side)
