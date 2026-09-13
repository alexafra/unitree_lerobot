from __future__ import annotations

import contextlib
from types import ModuleType, SimpleNamespace
from unittest import mock
import sys

import numpy as np
import pytest

from unitree_lerobot.eval_robot.eval_groot_g1 import (
    _run_return_to_start_from_hold,
    build_parser,
    confirm_policy_continue,
    confirm_policy_start,
    resolve_runtime_end_effector,
    run,
    validate_args,
)
from unitree_lerobot.eval_robot.g1_end_effectors import (
    DEX3_PROFILE,
    INSPIRE_DFX_PROFILE,
    INSPIRE_FTP_PROFILE,
    get_end_effector_profile,
    inspire_ftp_dataset_contract,
)
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    ARM_JOINT_NAMES,
    ActionChunk,
    COLOUR_VIDEO_KEYS,
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_EGO_VIEW_SHAPE,
    INSPIRE_XR_HOME_ARM,
    INSPIRE_XR_HOME_ELBOW_RAD,
    load_initialization_spec,
    validate_policy_metadata,
)
from unitree_lerobot.eval_robot.robot_control.g1_inspire_ftp import (
    G1InspireFtpStateReader,
    INSPIRE_FTP_COMMAND_MAX_STEP,
    InspireFtpCommandWriter,
    InspireFtpPartialWriteError,
)
from unitree_lerobot.eval_robot.robot_control.safe_g1_dex3 import (
    INSPIRE_FTP_REAL_MODE_MACHINE,
    INSPIRE_FTP_WAIST_INDICES,
    INSPIRE_FTP_WAIST_KD,
    INSPIRE_FTP_WAIST_KP,
    MAX_WAIST_DQ_RAD_S,
    MAX_WAIST_HOLD_ERROR_RAD,
    QUALIFIED_REAL_MODE_MACHINE,
    RobotState,
    _G1Dex3CommandBackend,
    _ramp_real_arm_authority,
)


def _module(name: str, **attributes: object) -> ModuleType:
    result = ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


def _modality_config(action_horizon: int = 32) -> dict:
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


def _layout() -> dict:
    return {
        "left_arm": {"start": 0, "end": 7, "dim": 7},
        "right_arm": {"start": 7, "end": 14, "dim": 7},
        "left_hand": {"start": 14, "end": 20, "dim": 6},
        "right_hand": {"start": 20, "end": 26, "dim": 6},
    }


def _metadata(*, protocol: str = "ftp") -> dict:
    profile = INSPIRE_FTP_PROFILE
    joint_names = list(ARM_JOINT_NAMES + profile.joint_names)
    end_effector = inspire_ftp_dataset_contract()
    end_effector["protocol"] = protocol
    return {
        "protocol_version": 1,
        "embodiment_tag": "new_embodiment",
        "task_contract": {
            "schema_version": 1,
            "instructions": ["pick up the red cup.", "put down the red cup."],
            "sha256": "1dc8e65b8210cd017ad48081d860208805ad4638785496059eef743fd1d633e7",
        },
        "vision_input_contract": {
            "version": 1,
            "mode": "separate_views",
            "input_channels": 3,
            "channel_layout": ["ego_view:0", "ego_view:1", "ego_view:2"],
            "patch_embed_init": "original_rgb",
            "wire_video_keys": ["ego_view"],
        },
        "action_output_contract": EXPECTED_ACTION_OUTPUT_CONTRACT,
        "dataset_contract": {
            "robot_type": profile.robot_type,
            "fps": 30.0,
            "observation_state_shape": [26],
            "action_shape": [26],
            "observation_state_names": joint_names,
            "action_names": joint_names,
            "state_layout": _layout(),
            "action_layout": _layout(),
            "ego_view_shape": EXPECTED_EGO_VIEW_SHAPE,
            "video_shapes": {"ego_view": EXPECTED_EGO_VIEW_SHAPE},
            "end_effector": end_effector,
        },
    }


class FakePublisher:
    instances: list["FakePublisher"] = []

    def __init__(self, topic: str, message_type: object):
        self.topic = topic
        self.message_type = message_type
        self.initialized = False
        self.closed = False
        self.write_result = True
        self.writes: list[tuple[list[int], int, float | None]] = []
        self.instances.append(self)

    def Init(self):
        self.initialized = True

    def Write(self, message, timeout=None):
        self.writes.append((list(message.angle_set), int(message.mode), timeout))
        return self.write_result

    def Close(self):
        self.closed = True


class FakeSubscriber:
    instances: dict[str, "FakeSubscriber"] = {}

    def __init__(self, topic: str, message_type: object):
        self.topic = topic
        self.message_type = message_type
        self.handler = None
        self.closed = False
        self.instances[topic] = self

    def Init(self, handler=None):
        self.handler = handler

    def Close(self):
        self.closed = True


class FakeControl:
    def __init__(self):
        self.angle_set = []
        self.mode = 0


def _sdk_modules() -> dict[str, ModuleType]:
    inspire_dds = _module(
        "inspire_sdkpy.inspire_dds",
        inspire_hand_ctrl=type("inspire_hand_ctrl", (), {}),
        inspire_hand_state=type("inspire_hand_state", (), {}),
    )
    inspire_package = _module("inspire_sdkpy", inspire_dds=inspire_dds)
    return {
        "inspire_sdkpy": inspire_package,
        "inspire_sdkpy.inspire_dds": inspire_dds,
        "inspire_sdkpy.inspire_hand_defaut": _module(
            "inspire_sdkpy.inspire_hand_defaut",
            get_inspire_hand_ctrl=FakeControl,
        ),
        "unitree_sdk2py": _module("unitree_sdk2py"),
        "unitree_sdk2py.core": _module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": _module(
            "unitree_sdk2py.core.channel",
            ChannelPublisher=FakePublisher,
            ChannelSubscriber=FakeSubscriber,
        ),
        "unitree_sdk2py.idl": _module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.unitree_hg": _module("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": _module("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": _module(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            LowState_=type("LowState_", (), {}),
        ),
        "unitree_lerobot.eval_robot.robot_control.robot_arm": _module(
            "unitree_lerobot.eval_robot.robot_control.robot_arm",
            G1_29_JointArmIndex=tuple(range(15, 29)),
            G1_29_JointIndex=SimpleNamespace(
                kWaistYaw=12,
                kWaistRoll=13,
                kWaistPitch=14,
            ),
        ),
    }


def test_ftp_profile_contract_and_explicit_live_gate():
    assert get_end_effector_profile("dex3") is DEX3_PROFILE
    assert get_end_effector_profile("inspire-ftp") is INSPIRE_FTP_PROFILE
    assert INSPIRE_FTP_PROFILE.hand_dof == 6
    assert INSPIRE_FTP_PROFILE.left_state_topic == "rt/inspire_hand/state/l"
    assert INSPIRE_FTP_PROFILE.right_state_topic == "rt/inspire_hand/state/r"
    assert INSPIRE_FTP_PROFILE.left_command_topic == "rt/inspire_hand/ctrl/l"
    assert INSPIRE_FTP_PROFILE.right_command_topic == "rt/inspire_hand/ctrl/r"
    np.testing.assert_array_equal(INSPIRE_FTP_PROFILE.conditioned_step, np.full(6, 0.2))
    np.testing.assert_array_equal(INSPIRE_FTP_PROFILE.home, np.ones(6))
    assert inspire_ftp_dataset_contract()["protocol"] == "ftp"
    assert resolve_runtime_end_effector("inspire-ftp", True) == "inspire-dfx"
    assert resolve_runtime_end_effector("inspire-ftp", False) == "inspire-ftp"
    assert resolve_runtime_end_effector("dex3", True) == "dex3"
    assert INSPIRE_DFX_PROFILE.supports_simulation
    assert not INSPIRE_FTP_PROFILE.supports_simulation
    assert not INSPIRE_FTP_PROFILE.uses_lost_counters
    validate_policy_metadata(_metadata(), end_effector="inspire-ftp")
    with pytest.raises(DeploymentError, match="end_effector"):
        validate_policy_metadata(_metadata(protocol="dfx"), end_effector="inspire-ftp")

    parser = build_parser()
    shadow = parser.parse_args(
        ["--task", "pick-red-cup", "--end-effector", "inspire-ftp", "--no-warmup1"]
    )
    validate_args(shadow)
    live = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-ftp",
            "--no-warmup1",
            "--actuate",
            "--network-interface",
            "eth-test",
            "--allow-unqualified-real",
        ]
    )
    with pytest.raises(DeploymentError, match="allow-inspire-ftp-unverified-stop"):
        validate_args(live)
    live.allow_inspire_ftp_unverified_stop = True
    validate_args(live)
    simulation = parser.parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-ftp",
            "--no-warmup1",
            "--sim",
            "--actuate",
            "--confirm-sim-network-isolated",
        ]
    )
    validate_args(simulation)
    wrong_profile = parser.parse_args(
        ["--task", "pick-red-cup", "--allow-inspire-ftp-unverified-stop"]
    )
    with pytest.raises(DeploymentError, match="valid only"):
        validate_args(wrong_profile)


def test_ftp_sim_validates_ftp_metadata_then_uses_combined_dfx_runtime():
    args = build_parser().parse_args(
        [
            "--task",
            "pick-red-cup",
            "--end-effector",
            "inspire-ftp",
            "--no-warmup1",
            "--sim",
        ]
    )
    policy = mock.Mock()
    policy.ping.return_value = True
    policy.get_modality_config.return_value = _modality_config()
    policy.get_policy_metadata.return_value = _metadata()
    reader = SimpleNamespace(close=mock.Mock())
    camera = SimpleNamespace(
        config={
            "head_camera": {
                "type": "Head",
                "image_shape": [480, 640],
                "binocular": False,
                "fps": 30,
            }
        },
        close=mock.Mock(),
    )
    chunk = ActionChunk(
        arm=np.zeros((1, 14)),
        left_hand=np.zeros((1, 6)),
        right_hand=np.zeros((1, 6)),
        end_effector="inspire-dfx",
    )
    module = "unitree_lerobot.eval_robot.eval_groot_g1"
    with (
        mock.patch(f"{module}.Gr00tClient", return_value=policy),
        mock.patch(f"{module}.require_inspire_ftp_sdk") as require_ftp_sdk,
        mock.patch(f"{module}.initialize_dds") as initialize_dds,
        mock.patch(f"{module}.G1InspireDfxStateReader", return_value=reader) as dfx_reader,
        mock.patch(f"{module}.G1InspireFtpStateReader") as ftp_reader,
        mock.patch(f"{module}.TeleimagerCamera", return_value=camera) as camera_factory,
        mock.patch(f"{module}.infer_chunk", return_value=(chunk, 0.01)) as infer_chunk,
        mock.patch(f"{module}.chunk_delta_summary", return_value="synthetic preflight"),
    ):
        run(args)

    # FTP remains the checkpoint provenance gate: the real validator accepted
    # _metadata() only because it advertises protocol=ftp. The rewritten tag is
    # introduced afterwards and is limited to the simulator's DFX wire path.
    assert infer_chunk.call_args.args[4].end_effector == "inspire-dfx"
    require_ftp_sdk.assert_not_called()
    initialize_dds.assert_called_once_with(True, None)
    dfx_reader.assert_called_once_with(simulation=True)
    ftp_reader.assert_not_called()
    assert camera_factory.call_args.kwargs["prefer_atomic_rgbd"] is False
    policy.reset.assert_called_once_with()
    policy.close.assert_called_once_with()
    reader.close.assert_called_once_with()
    camera.close.assert_called_once_with()


def test_ftp_xr_home_and_return_to_start_are_profile_aware():
    spec = load_initialization_spec(
        "xr-home",
        task_name="pick-red-cup",
        end_effector="inspire-ftp",
    )
    assert spec.end_effector == "inspire-ftp"
    expected_arm = np.zeros(14)
    expected_arm[[3, 10]] = -0.15
    np.testing.assert_array_equal(spec.arm, expected_arm)
    np.testing.assert_array_equal(spec.arm, INSPIRE_XR_HOME_ARM)
    assert spec.arm is not INSPIRE_XR_HOME_ARM
    assert INSPIRE_XR_HOME_ELBOW_RAD == -0.15
    assert "both elbows -0.15 rad" in spec.label
    np.testing.assert_array_equal(spec.left_hand, np.ones(6))
    np.testing.assert_array_equal(spec.right_hand, np.ones(6))

    parser = build_parser()
    common = [
        "--task",
        "pick-red-cup",
        "--end-effector",
        "inspire-ftp",
        "--no-warmup1",
        "--actuate",
        "--network-interface",
        "eth-test",
        "--allow-unqualified-real",
        "--allow-inspire-ftp-unverified-stop",
        "--return-to-start",
    ]
    with pytest.raises(DeploymentError, match="requires.*xr-home"):
        validate_args(parser.parse_args(common))
    validate_args(parser.parse_args([*common, "--initialization", "xr-home"]))


def test_missing_ftp_sdk_fails_before_parent_dds_initialization():
    args = build_parser().parse_args(
        ["--task", "pick-red-cup", "--end-effector", "inspire-ftp", "--no-warmup1"]
    )

    class FakePolicy:
        def __init__(self, *_args):
            pass

        @staticmethod
        def ping():
            return True

        @staticmethod
        def get_modality_config():
            return _modality_config()

        @staticmethod
        def get_policy_metadata():
            return _metadata()

        @staticmethod
        def close():
            pass

    module = "unitree_lerobot.eval_robot.eval_groot_g1"
    with (
        mock.patch(f"{module}.Gr00tClient", FakePolicy),
        mock.patch(
            f"{module}.require_inspire_ftp_sdk",
            side_effect=DeploymentError("synthetic missing inspire_sdkpy"),
        ),
        mock.patch(f"{module}.initialize_dds") as initialize_dds,
        pytest.raises(DeploymentError, match="missing inspire_sdkpy"),
    ):
        run(args)
    initialize_dds.assert_not_called()


def test_incompatible_ftp_sdk_fails_before_parent_dds_initialization():
    args = build_parser().parse_args(
        ["--task", "pick-red-cup", "--end-effector", "inspire-ftp", "--no-warmup1"]
    )

    class FakePolicy:
        def __init__(self, *_args):
            pass

        ping = staticmethod(lambda: True)
        get_modality_config = staticmethod(_modality_config)
        get_policy_metadata = staticmethod(_metadata)
        close = staticmethod(lambda: None)

    modules = _sdk_modules()
    modules["inspire_sdkpy"].inspire_dds = _module(
        "inspire_sdkpy.inspire_dds",
        inspire_hand_ctrl=type("inspire_hand_ctrl", (), {}),
        # Deliberately missing inspire_hand_state.
    )
    modules["inspire_sdkpy.inspire_dds"] = modules["inspire_sdkpy"].inspire_dds
    module = "unitree_lerobot.eval_robot.eval_groot_g1"
    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch(f"{module}.Gr00tClient", FakePolicy),
        mock.patch(f"{module}.initialize_dds") as initialize_dds,
        pytest.raises(DeploymentError, match="IDL types are missing"),
    ):
        run(args)
    initialize_dds.assert_not_called()


def test_ftp_state_uses_independent_topics_and_normalizes_angle_codes():
    FakeSubscriber.instances.clear()
    with mock.patch.dict(sys.modules, _sdk_modules()):
        reader = G1InspireFtpStateReader(max_age_s=1.0)
        motor_state = [SimpleNamespace(q=0.0, dq=0.0) for _ in range(35)]
        for index, q in zip((12, 13, 14), (0.25, -0.04, 0.06), strict=True):
            motor_state[index] = SimpleNamespace(q=q, dq=0.01)
        arm = SimpleNamespace(
            mode_machine=5,
            motor_state=motor_state,
        )
        FakeSubscriber.instances["rt/lowstate"].handler(arm)
        FakeSubscriber.instances["rt/inspire_hand/state/l"].handler(
            SimpleNamespace(angle_act=[0, 200, 400, 600, 800, 1000])
        )
        with pytest.raises(TimeoutError, match="right"):
            reader.latest()
        FakeSubscriber.instances["rt/inspire_hand/state/r"].handler(
            SimpleNamespace(angle_act=[1000, 800, 600, 400, 200, 0])
        )
        state = reader.latest()
        np.testing.assert_array_equal(state.arm, np.zeros(14))
        np.testing.assert_allclose(state.waist, [0.25, -0.04, 0.06])
        np.testing.assert_allclose(state.waist_dq, [0.01, 0.01, 0.01])
        np.testing.assert_allclose(state.left_hand, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(state.right_hand, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        right_received_at = state.right_hand_received_at

        # A malformed left callback cannot overwrite either accepted side.
        FakeSubscriber.instances["rt/inspire_hand/state/l"].handler(
            SimpleNamespace(angle_act=[0, 0, 0, 0, 0, 1001])
        )
        retained = reader.latest()
        np.testing.assert_array_equal(retained.left_hand, state.left_hand)
        assert retained.right_hand_received_at == right_received_at
        assert reader.diagnostics()["left"]["malformed_messages"] == 1

        FakeSubscriber.instances["rt/inspire_hand/state/l"].handler(
            SimpleNamespace(angle_act=[0] * 6)
        )
        FakeSubscriber.instances["rt/inspire_hand/state/r"].handler(
            SimpleNamespace(angle_act=[0] * 6)
        )
        all_zero = reader.latest()
        np.testing.assert_array_equal(all_zero.left_hand, np.zeros(6))
        np.testing.assert_array_equal(all_zero.right_hand, np.zeros(6))
        reader.close()
    assert set(FakeSubscriber.instances) == {
        "rt/lowstate",
        "rt/inspire_hand/state/l",
        "rt/inspire_hand/state/r",
    }
    assert all(subscriber.closed for subscriber in FakeSubscriber.instances.values())


def test_ftp_completion_prompts_require_visual_hand_confirmation():
    module = "unitree_lerobot.eval_robot.eval_groot_g1"
    spec = load_initialization_spec(
        "xr-home",
        task_name="pick-red-cup",
        end_effector="inspire-ftp",
    )
    actuator = SimpleNamespace(end_effector="inspire-ftp")

    with mock.patch(f"{module}._confirm_while_armed") as confirm:
        confirm_policy_start(actuator, spec)
    startup_message = confirm.call_args.args[1]
    assert "Startup arm pose" in startup_message
    assert "hand convergence is not software-verified" in startup_message
    assert "Visually confirm both hands" in startup_message

    with mock.patch(f"{module}._confirm_goal_transition", return_value="continue") as confirm:
        assert confirm_policy_continue(actuator, "inspire-ftp") == "continue"
    warmup_message = confirm.call_args.args[1]
    assert "Warmup2 arm target converged" in warmup_message
    assert "hand convergence is not software-verified" in warmup_message
    assert "Visually confirm both hands" in warmup_message


def test_ftp_return_to_start_requires_a_post_motion_visual_gate():
    module = "unitree_lerobot.eval_robot.eval_groot_g1"
    spec = load_initialization_spec(
        "xr-home",
        task_name="pick-red-cup",
        end_effector="inspire-ftp",
    )
    events: list[str] = []
    actuator = SimpleNamespace(
        wait_for_hand_feedback=lambda: events.append("feedback"),
        warmup_pose=lambda _spec: events.append("motion"),
    )

    def run_motion(_actuator, operation, **_kwargs):
        operation()

    def visual_gate(_actuator, message, required):
        events.append("visual")
        assert "arm target" in message
        assert "hand convergence is not software-verified" in message
        assert "Visually confirm both hands" in message
        assert required == "CONFIRM RETURN-TO-START VISUAL CHECK"
        return "continue"

    with (
        mock.patch(f"{module}.confirm_return_to_start", return_value="continue"),
        mock.patch(
            f"{module}._run_blocking_motion_with_immediate_release",
            side_effect=run_motion,
        ),
        mock.patch(f"{module}._confirm_goal_transition", side_effect=visual_gate),
    ):
        assert _run_return_to_start_from_hold(actuator, spec) == "continue"
    assert events == ["feedback", "motion", "visual"]


def test_ftp_state_reader_unwinds_partial_subscriber_initialization():
    class FailingSubscriber(FakeSubscriber):
        instances: dict[str, "FailingSubscriber"] = {}

        def Init(self, handler=None):
            super().Init(handler=handler)
            if self.topic == "rt/inspire_hand/state/r":
                raise RuntimeError("synthetic right subscriber Init failure")

    modules = _sdk_modules()
    modules["unitree_sdk2py.core.channel"] = _module(
        "unitree_sdk2py.core.channel",
        ChannelPublisher=FakePublisher,
        ChannelSubscriber=FailingSubscriber,
    )
    FailingSubscriber.instances.clear()
    with (
        mock.patch.dict(sys.modules, modules),
        pytest.raises(RuntimeError, match="right subscriber Init failure"),
    ):
        G1InspireFtpStateReader(max_age_s=1.0)
    assert len(FailingSubscriber.instances) == 3
    assert all(subscriber.closed for subscriber in FailingSubscriber.instances.values())


def test_ftp_writer_matches_teleop_scaling_and_tracks_partial_success():
    FakePublisher.instances.clear()
    initial = np.array([0.0009, 0.001, 0.1239, 0.5, 0.9999, 1.0])
    with mock.patch.dict(sys.modules, _sdk_modules()):
        writer = InspireFtpCommandWriter(initial, initial)
        left, right = FakePublisher.instances
        assert left.writes == [] and right.writes == []
        result = writer.write(initial, initial)
        expected = [0, 1, 123, 500, 999, 1000]
        assert left.writes == [(expected, 1, 0.5)]
        assert right.writes == [(expected, 1, 0.5)]
        np.testing.assert_array_equal(result.left, np.asarray(expected) / 1000.0)
        np.testing.assert_array_equal(result.right, np.asarray(expected) / 1000.0)
        assert result.left_completed_at <= result.right_completed_at
        assert writer.left_has_written and writer.right_has_written

        with pytest.raises(DeploymentError, match="COMMAND_MAX_STEP"):
            writer.write(np.zeros(6), initial)
        assert len(left.writes) == len(right.writes) == 1
        writer.close()
        assert left.closed and right.closed

        FakePublisher.instances.clear()
        partial = InspireFtpCommandWriter(np.zeros(6), np.zeros(6))
        partial_left, partial_right = FakePublisher.instances
        partial_right.write_result = False
        with pytest.raises(InspireFtpPartialWriteError) as captured:
            partial.write(np.full(6, 0.1), np.full(6, 0.2))
        np.testing.assert_array_equal(captured.value.left, np.full(6, 0.1))
        assert partial.left_has_written and not partial.right_has_written
        assert len(partial_left.writes) == 1
        assert len(partial_right.writes) == 1
        partial.close()

        FakePublisher.instances.clear()
        raised = InspireFtpCommandWriter(np.zeros(6), np.zeros(6))
        raised_left, raised_right = FakePublisher.instances
        raised_right.Write = mock.Mock(side_effect=RuntimeError("synthetic transport error"))
        with pytest.raises(InspireFtpPartialWriteError, match="right DDS Write raised"):
            raised.write(np.full(6, 0.1), np.full(6, 0.2))
        assert raised.left_has_written and not raised.right_has_written
        assert len(raised_left.writes) == 1
        raised.close()


def test_ftp_backend_records_partial_history_and_cleanup_only_closes():
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_FTP_PROFILE
    backend._left_target = np.full(6, 0.1)
    backend._right_target = np.full(6, 0.2)
    backend._left_hand_publish_history = []
    backend._right_hand_publish_history = []
    backend._authority_ramp_timing_enabled = False
    backend._hand_writer = SimpleNamespace(
        write=mock.Mock(
            side_effect=InspireFtpPartialWriteError(
                "synthetic right failure",
                left_completed_at=12.5,
                left=np.full(6, 0.1),
            )
        ),
        close=mock.Mock(),
        left_has_written=True,
        right_has_written=False,
    )
    with pytest.raises(InspireFtpPartialWriteError, match="right failure"):
        backend._publish_hands()
    assert len(backend._left_hand_publish_history) == 1
    assert backend._left_hand_publish_history[0].completed_at == 12.5
    assert backend._right_hand_publish_history == []

    phases = []
    backend._stop_hands(lambda name, detail: phases.append((name, detail)))
    backend._hand_writer.close.assert_called_once_with()
    assert [name for name, _detail in phases] == [
        "inspire_ftp_publishers_close_begin",
        "inspire_ftp_publishers_close_end",
    ]
    assert phases[-1][1]["motor_stop_acknowledged"] is False
    assert phases[-1][1]["left_dds_write_completed"] is True
    assert phases[-1][1]["right_dds_write_completed"] is False
    assert phases[-1][1]["assume_written_setpoint_may_persist"] is True


def test_ftp_backend_missing_sdk_fails_before_child_dds_initialization():
    safe_module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    ftp_module = "unitree_lerobot.eval_robot.robot_control.g1_inspire_ftp"
    with (
        mock.patch(
            f"{ftp_module}.require_inspire_ftp_sdk",
            side_effect=DeploymentError("synthetic child SDK failure"),
        ),
        mock.patch(f"{safe_module}.initialize_dds") as initialize_dds,
        pytest.raises(DeploymentError, match="child SDK failure"),
    ):
        _G1Dex3CommandBackend(
            False,
            "eth-test",
            gravity_feedforward=False,
            end_effector="inspire-ftp",
        )
    initialize_dds.assert_not_called()


def _mode5_state(
    *,
    waist: np.ndarray | None = None,
    waist_dq: np.ndarray | None = None,
    mode_machine: int = INSPIRE_FTP_REAL_MODE_MACHINE,
) -> RobotState:
    return RobotState(
        captured_at=1.0,
        mode_machine=mode_machine,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.full(6, 0.8),
        right_hand=np.full(6, 0.7),
        waist=np.array([0.25, -0.04, 0.06]) if waist is None else waist,
        waist_dq=np.zeros(3) if waist_dq is None else waist_dq,
    )


def _low_command() -> SimpleNamespace:
    return SimpleNamespace(
        mode_pr=-1,
        mode_machine=-1,
        crc=0,
        motor_cmd=[
            SimpleNamespace(mode=-1, q=-99.0, dq=-99.0, tau=-99.0, kp=-99.0, kd=-99.0)
            for _ in range(35)
        ],
    )


@pytest.mark.parametrize(
    ("waist", "waist_dq", "message"),
    (
        (None, None, "requires measured waist q/dq"),
        (np.array([0.0, np.nan, 0.0]), np.zeros(3), "NaN or infinity"),
        (np.zeros(3), np.array([0.0, np.inf, 0.0]), "NaN or infinity"),
        (np.array([2.599, 0.0, 0.0]), np.zeros(3), "waist_yaw.*outside the guarded"),
        (np.array([0.0, 0.501, 0.0]), np.zeros(3), "waist_roll.*outside the guarded"),
        (np.array([0.0, 0.0, -0.501]), np.zeros(3), "waist_pitch.*outside the guarded"),
    ),
)
def test_ftp_mode5_invalid_waist_fails_before_command_resources_exist(
    waist,
    waist_dq,
    message,
):
    base = _mode5_state()
    initial = RobotState(
        captured_at=base.captured_at,
        mode_machine=base.mode_machine,
        arm=base.arm,
        arm_dq=base.arm_dq,
        left_hand=base.left_hand,
        right_hand=base.right_hand,
        waist=waist,
        waist_dq=waist_dq,
    )
    reader = SimpleNamespace(read=mock.Mock(return_value=initial), close=mock.Mock())
    safe_module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    ftp_module = "unitree_lerobot.eval_robot.robot_control.g1_inspire_ftp"
    with (
        mock.patch(f"{ftp_module}.require_inspire_ftp_sdk"),
        mock.patch(f"{safe_module}.initialize_dds"),
        mock.patch(f"{ftp_module}.G1InspireFtpStateReader", return_value=reader),
        mock.patch.object(_G1Dex3CommandBackend, "_initialize_command_resources") as resources,
        pytest.raises(DeploymentError, match=message),
    ):
        _G1Dex3CommandBackend(
            False,
            "eth-test",
            gravity_feedforward=False,
            end_effector="inspire-ftp",
        )
    resources.assert_not_called()
    reader.close.assert_called_once_with()


def test_ftp_mode5_and_dex3_mode6_gates_are_profile_specific():
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.simulation = False
    backend.profile = INSPIRE_FTP_PROFILE
    backend._waist_target = np.array([0.25, -0.04, 0.06])
    assert backend._qualified_real_mode_machine() == INSPIRE_FTP_REAL_MODE_MACHINE == 5
    state = _mode5_state()
    assert backend._validate_runtime_state(state) is state
    with pytest.raises(DeploymentError, match="required mode 5"):
        backend._validate_runtime_state(_mode5_state(mode_machine=QUALIFIED_REAL_MODE_MACHINE))

    backend.profile = DEX3_PROFILE
    assert backend._qualified_real_mode_machine() == QUALIFIED_REAL_MODE_MACHINE == 6
    dex3_state = RobotState(
        captured_at=1.0,
        mode_machine=QUALIFIED_REAL_MODE_MACHINE,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.zeros(7),
        right_hand=np.zeros(7),
    )
    assert backend._validate_runtime_state(dex3_state) is dex3_state
    with pytest.raises(DeploymentError, match="required mode 6"):
        backend._validate_runtime_state(
            RobotState(
                captured_at=1.0,
                mode_machine=INSPIRE_FTP_REAL_MODE_MACHINE,
                arm=np.zeros(14),
                arm_dq=np.zeros(14),
                left_hand=np.zeros(7),
                right_hand=np.zeros(7),
            )
        )


def test_ftp_mode5_waist_is_measured_held_and_unchanged_by_policy_or_release():
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.simulation = False
    backend.profile = INSPIRE_FTP_PROFILE
    backend._arm_indices = tuple(range(15, 29))
    backend._waist_indices = INSPIRE_FTP_WAIST_INDICES
    backend._arm_message = _low_command()
    lower_body_before = [
        vars(backend._arm_message.motor_cmd[index]).copy() for index in range(12)
    ]
    backend._waist_target = None
    initial = _mode5_state()
    backend._configure_messages(initial)

    assert [
        vars(backend._arm_message.motor_cmd[index]).copy() for index in range(12)
    ] == lower_body_before

    expected_waist = initial.waist.copy()
    for offset, index in enumerate(INSPIRE_FTP_WAIST_INDICES):
        command = backend._arm_message.motor_cmd[index]
        assert command.mode == 1
        assert command.q == expected_waist[offset]
        assert command.dq == 0.0
        assert command.tau == 0.0
        assert command.kp == INSPIRE_FTP_WAIST_KP == 300.0
        assert command.kd == INSPIRE_FTP_WAIST_KD == 3.0

    writes: list[tuple[int, list[dict[str, float]], np.ndarray, list[dict[str, float]]]] = []

    def write(message, timeout=None):
        del timeout
        writes.append(
            (
                int(message.mode_machine),
                [vars(message.motor_cmd[index]).copy() for index in INSPIRE_FTP_WAIST_INDICES],
                np.array([message.motor_cmd[index].q for index in backend._arm_indices]),
                [vars(message.motor_cmd[index]).copy() for index in range(12)],
            )
        )
        return True

    backend._arm_gravity = None
    backend._arm_publisher = SimpleNamespace(Write=write)
    backend._crc = SimpleNamespace(Crc=lambda _message: 123)
    backend._weight = 1.0
    backend._authority_ramp_timing_enabled = False
    backend._last_publish_timing_ms = {}
    backend._has_published = False
    backend._last_published_arm_q = None
    backend._last_published_arm_tau = None

    first_arm = np.linspace(-0.2, 0.2, 14)
    second_arm = np.linspace(0.3, -0.3, 14)
    backend.set_target(first_arm, np.zeros(6), np.ones(6))
    np.testing.assert_array_equal(backend._waist_target, expected_waist)
    backend._publish_arm(require_qualified_state=False)
    backend.set_target(second_arm, np.ones(6), np.zeros(6))
    np.testing.assert_array_equal(backend._waist_target, expected_waist)
    backend._publish_arm(require_qualified_state=False)
    backend._publish_last_arm_for_release()

    assert [mode for mode, _waist, _arm, _lower in writes] == [5, 5, 5]
    for _mode, waist_commands, _arm, lower_body in writes:
        assert lower_body == lower_body_before
        for offset, command in enumerate(waist_commands):
            assert command == {
                "mode": 1,
                "q": expected_waist[offset],
                "dq": 0.0,
                "tau": 0.0,
                "kp": 300.0,
                "kd": 3.0,
            }
    np.testing.assert_array_equal(writes[0][2], first_arm)
    np.testing.assert_array_equal(writes[1][2], second_arm)
    np.testing.assert_array_equal(writes[2][2], second_arm)


def test_ftp_mode5_runtime_allows_teleop_compatible_waist_motion():
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.simulation = False
    backend.profile = INSPIRE_FTP_PROFILE
    backend._arm_target = np.zeros(14)
    backend._left_target = np.full(6, 0.8)
    backend._right_target = np.full(6, 0.7)
    backend._waist_target = np.array([0.25, -0.04, 0.06])

    deviated = backend._waist_target.copy()
    deviated[1] += MAX_WAIST_HOLD_ERROR_RAD + 1e-6
    runtime_moving = np.zeros(3)
    runtime_moving[1] = MAX_WAIST_DQ_RAD_S + 1.0
    state = _mode5_state(waist=deviated, waist_dq=runtime_moving)
    assert backend._validate_runtime_state(state) is state

    moving = np.zeros(3)
    moving[2] = 0.101
    with (
        mock.patch(
            "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3.time.monotonic",
            return_value=1.0,
        ),
        pytest.raises(DeploymentError, match="Waist is not stationary.*waist_pitch"),
    ):
        backend._validate_prearm_takeover_state(_mode5_state(waist_dq=moving))


def test_ftp_release_reaches_zero_arm_weight_without_hand_refresh():
    now = [20.0]
    events: list[tuple[str, float | None]] = []
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_FTP_PROFILE
    backend._released = False
    backend._has_published = True
    backend.simulation = False
    backend._weight = 1.0
    # Model the safety-significant partial case: only the left write was ever
    # written through DDS. FTP cleanup must not synthesize or refresh either side.
    backend._hand_writer = SimpleNamespace(
        has_written=True,
        left_has_written=True,
        right_has_written=False,
    )
    backend._publish_last_arm_for_release = lambda: events.append(("arm", backend._weight))
    backend._refresh_last_successful_hands_for_release = lambda: events.append(("hands", None))
    backend._stop_hands = lambda _callback=None: events.append(("close", None))

    def sleep(duration: float) -> None:
        now[0] += max(0.0, float(duration))

    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    with (
        mock.patch(f"{module}.time.monotonic", side_effect=lambda: now[0]),
        mock.patch(f"{module}.time.sleep", side_effect=sleep),
        mock.patch(f"{module}.PUBLISH_HZ", 10.0),
        mock.patch(f"{module}.ARM_RELEASE_RAMP_S", 0.2),
    ):
        backend.release()
    assert events[-1] == ("close", None)
    assert events[-2] == ("arm", 0.0)
    assert not any(kind == "hands" for kind, _value in events)


def test_ftp_final_measured_reseed_is_used_before_first_write():
    backend = _G1Dex3CommandBackend.__new__(_G1Dex3CommandBackend)
    backend.profile = INSPIRE_FTP_PROFILE
    backend._left_target = np.zeros(6)
    backend._right_target = np.zeros(6)
    backend._hand_writer = SimpleNamespace(reseed_before_first_write=mock.Mock())
    measured = RobotState(
        captured_at=1.0,
        mode_machine=6,
        arm=np.zeros(14),
        arm_dq=np.zeros(14),
        left_hand=np.full(6, 0.8),
        right_hand=np.full(6, 0.7),
    )
    backend._reseed_inspire_hands_before_first_write(measured)
    np.testing.assert_array_equal(backend._left_target, measured.left_hand)
    np.testing.assert_array_equal(backend._right_target, measured.right_hand)
    backend._hand_writer.reseed_before_first_write.assert_called_once()
    assert INSPIRE_FTP_COMMAND_MAX_STEP == 0.2


@pytest.mark.parametrize("settle_waist_dq", (0.0, 0.2))
def test_ftp_authority_ramp_requires_stationary_waist(settle_waist_dq):
    now = [10.0]
    events: list[tuple[str, float]] = []

    class ClockedStop:
        @staticmethod
        def is_set():
            return False

        @staticmethod
        def wait(timeout):
            now[0] += max(0.0, float(timeout))
            return False

    class Heartbeat:
        value = 10.0

        @staticmethod
        def get_lock():
            return contextlib.nullcontext()

    class FreshnessGate:
        def __init__(self):
            self.checks = 0

        def check(self, _state):
            self.checks += 1
            events.append(("fresh", now[0]))
            return SimpleNamespace(ready=True, stale_hands=(), max_age_s=0.0)

    class Backend:
        simulation = False
        profile = INSPIRE_FTP_PROFILE

        def __init__(self):
            self._arm_target = np.zeros(14)
            self._left_target = np.zeros(6)
            self._right_target = np.zeros(6)
            self._waist_target = np.array([0.25, -0.04, 0.06])

        @staticmethod
        def _uses_mode5_waist_hold():
            return True

        @staticmethod
        def _require_mode5_waist_state(state):
            return state.waist, state.waist_dq

        @staticmethod
        def _validate_prearm_takeover_state(_state):
            pass

        def _reseed_inspire_hands_before_first_write(self, state):
            self._left_target = state.left_hand.copy()
            self._right_target = state.right_hand.copy()
            events.append(("reseed", now[0]))

        @staticmethod
        def set_weight(_weight):
            pass

        def state(self):
            return RobotState(
                captured_at=now[0],
                mode_machine=5,
                arm=np.zeros(14),
                arm_dq=np.zeros(14),
                left_hand=np.full(6, 0.8),
                right_hand=np.full(6, 0.7),
                waist=self._waist_target.copy(),
                waist_dq=np.array([0.0, settle_waist_dq, 0.0]),
                left_hand_received_at=now[0],
                right_hand_received_at=now[0],
                arm_received_at=now[0],
            )

        @staticmethod
        def _publish_arm(**_kwargs):
            events.append(("arm", now[0]))

        @staticmethod
        def _publish_hands():
            events.append(("hands", now[0]))

    gate = FreshnessGate()
    module = "unitree_lerobot.eval_robot.robot_control.safe_g1_dex3"
    with (
        mock.patch(f"{module}.time.monotonic", side_effect=lambda: now[0]),
        mock.patch(f"{module}.PUBLISH_HZ", 10.0),
        mock.patch(f"{module}.HEARTBEAT_TIMEOUT_S", 10.0),
        mock.patch(f"{module}.ARM_GRAVITY_RAMP_S", 0.2),
        mock.patch(f"{module}.ARM_TAKEOVER_SETTLE_DWELL_S", 0.2),
        mock.patch(f"{module}.ARM_TAKEOVER_SETTLE_TIMEOUT_S", 0.5),
    ):
        if settle_waist_dq == 0.0:
            assert _ramp_real_arm_authority(Backend(), ClockedStop(), Heartbeat(), gate)
        else:
            with pytest.raises(DeploymentError, match="max waist dq=0.200"):
                _ramp_real_arm_authority(Backend(), ClockedStop(), Heartbeat(), gate)
    assert gate.checks > 1
    assert [kind for kind, _at in events[:4]] == ["fresh", "reseed", "arm", "hands"]
    assert sum(kind == "hands" for kind, _at in events) == 1
