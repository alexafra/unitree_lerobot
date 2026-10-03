"""Offline custom-goal regressions: opt out of membership, never contracts."""

from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest

from unitree_lerobot.eval_robot import eval_groot_g1 as runner
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    EXPECTED_ACTION_OUTPUT_CONTRACT,
    EXPECTED_DEPTH_VIEW_SHAPE,
    EXPECTED_EGO_VIEW_SHAPE,
    EXPECTED_JOINT_NAMES,
    EXPECTED_ROBOT_TYPE,
    TASKS,
    DepthEncodingContract,
    ModelContract,
    _task_contract_sha256,
    validate_policy_metadata,
)


CUSTOM_GOAL = "stack the three cups."
TRAINED_INSTRUCTIONS = [TASKS["pick-red-cup"], TASKS["down-red-cup"]]


def _metadata():
    return {
        "protocol_version": 1,
        "embodiment_tag": "new_embodiment",
        "task_contract": {
            "schema_version": 1,
            "instructions": list(TRAINED_INSTRUCTIONS),
            "sha256": _task_contract_sha256(TRAINED_INSTRUCTIONS),
        },
        "action_output_contract": deepcopy(EXPECTED_ACTION_OUTPUT_CONTRACT),
        "vision_input_contract": {"input_channels": 3},
        "dataset_contract": {
            "robot_type": EXPECTED_ROBOT_TYPE,
            "fps": 30.0,
            "observation_state_names": list(EXPECTED_JOINT_NAMES),
            "action_names": list(EXPECTED_JOINT_NAMES),
            "ego_view_shape": list(EXPECTED_EGO_VIEW_SHAPE),
            "video_shapes": {
                "ego_view": list(EXPECTED_EGO_VIEW_SHAPE),
                "depth_gray_view": list(EXPECTED_DEPTH_VIEW_SHAPE),
            },
            "depth_encoding": {
                "source_key": "depth_0",
                "feature_key": "observation.images.depth_gray_view",
                "encoding": "linear_grayscale_replicated_rgb",
                "near_m": 0.25,
                "far_m": 1.0,
                "invalid_value": 0,
                "valid_value_range": [1, 255],
            },
        },
        "rtc": {
            "protocol_version": 1,
            "physical_action_tail": True,
            "backend": "pytorch",
        },
    }


def test_custom_metadata_opt_in_keeps_full_contract_validation():
    metadata = _metadata()
    kwargs = {
        "instruction": CUSTOM_GOAL,
        "requires_depth": True,
        "vision_input_contract": {"input_channels": 3},
    }
    with pytest.raises(DeploymentError, match="not advertised by this checkpoint"):
        validate_policy_metadata(metadata, **kwargs)
    assert validate_policy_metadata(metadata, allow_custom_instruction=True, **kwargs) == DepthEncodingContract(
        near_m=0.25, far_m=1.0
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("task_contract",), None, "no checkpoint task contract"),
        (("task_contract", "schema_version"), 2, "Unsupported GR00T task contract schema"),
        (("task_contract", "sha256"), "0" * 64, "SHA-256 mismatch"),
        (("task_contract", "instructions"), [], "unique"),
        (("task_contract", "instructions"), ["same", "same"], "unique"),
        (("protocol_version",), 2, "deployment protocol"),
        (("embodiment_tag",), "wrong", "requires GR00T training tag"),
        (("dataset_contract",), None, "no deployment dataset contract"),
        (("dataset_contract", "robot_type"), "wrong", "contract mismatch for robot_type"),
        (("dataset_contract", "fps"), 50.0, "contract mismatch for fps"),
        (("dataset_contract", "observation_state_names"), [], "observation_state_names"),
        (("dataset_contract", "action_names"), [], "contract mismatch for action_names"),
        (("dataset_contract", "ego_view_shape"), [1, 1, 3], "ego_view_shape"),
        (("dataset_contract", "depth_encoding", "encoding"), "per_frame", "Unsupported depth encoding"),
        (("action_output_contract",), None, "no checkpoint action output contract"),
        (("action_output_contract", "use_relative_action"), False, "Unsupported checkpoint action output contract"),
        (("vision_input_contract", "input_channels"), 4, "vision input contract mismatch"),
    ],
)
def test_custom_goal_does_not_bypass_malformed_metadata(path, value, message):
    metadata = _metadata()
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DeploymentError, match=message):
        validate_policy_metadata(
            metadata,
            instruction=CUSTOM_GOAL,
            allow_custom_instruction=True,
            requires_depth=True,
            vision_input_contract={"input_channels": 3},
        )


@pytest.mark.parametrize("from_menu", [False, True], ids=["cli", "menu"])
def test_runner_accepts_explicit_custom_goal_before_dds(from_menu):
    argv = ["--end-effector", "dex3", "--sim", "--no-warmup1", "--no-return-to-start"]
    if not from_menu:
        argv += ["--custom-goal", CUSTOM_GOAL]
    args = runner.build_parser().parse_args(argv)
    args.actuate = False
    policy = mock.Mock()
    policy.ping.return_value = True
    policy.get_policy_metadata.return_value = _metadata()

    class ReachedMockDDS(Exception):
        pass

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(runner, "Gr00tClient", return_value=policy))
        stack.enter_context(mock.patch.object(runner, "validate_model_contract", return_value=ModelContract(32)))
        validate = stack.enter_context(
            mock.patch.object(runner, "validate_policy_metadata", wraps=validate_policy_metadata)
        )
        dds = stack.enter_context(mock.patch.object(runner, "initialize_dds", side_effect=ReachedMockDDS))
        reader = stack.enter_context(mock.patch.object(runner, "G1Dex3StateReader"))
        if from_menu:
            stack.enter_context(
                mock.patch.object(
                    runner, "_readline_before_authority", side_effect=[runner.GOAL_MODE_TOGGLE, CUSTOM_GOAL]
                )
            )
        with pytest.raises(ReachedMockDDS):
            runner.run(args)

    assert validate.call_args.kwargs["instruction"] == CUSTOM_GOAL
    assert validate.call_args.kwargs["allow_custom_instruction"] is True
    dds.assert_called_once_with(True, None)
    reader.assert_not_called()
    policy.get_action.assert_not_called()
    policy.close.assert_called_once_with()


@pytest.mark.parametrize("inference_mode", ["synchronous", "rtc"])
def test_held_trained_custom_trained_goals_keep_real_metadata_checks(inference_mode):
    args = runner.build_parser().parse_args(
        [
            "--end-effector",
            "dex3",
            "--task",
            "pick-red-cup",
            "--actuate",
            "--sim",
            "--confirm-sim-network-isolated",
            "--no-warmup1",
            "--no-warmup2",
            "--no-return-to-start",
            "--inference-mode",
            inference_mode,
        ]
    )
    policy = mock.Mock()
    policy.ping.return_value = True
    policy.get_policy_metadata.return_value = _metadata()
    reader = SimpleNamespace(close=mock.Mock())
    camera = SimpleNamespace(config={"head_camera": {}}, close=mock.Mock())
    actuator = SimpleNamespace(start=mock.Mock(), arm=mock.Mock(), initialize=mock.Mock(), close=mock.Mock())
    selected_goals = iter(
        [
            ("custom-goal", CUSTOM_GOAL),
            ("down-red-cup", TASKS["down-red-cup"]),
            None,
        ]
    )

    def select_next(_actuator, *, custom_goal_mode, mode_state, return_to_start=False):
        goal = next(selected_goals)
        if goal is not None:
            mode_state["custom_goal_mode"] = goal[0] == "custom-goal"
        return goal

    def fake_motion(_actuator, action, *, stage):
        action()

    with ExitStack() as stack:
        replacements = {
            "Gr00tClient": {"return_value": policy},
            "validate_model_contract": {"return_value": ModelContract(32)},
            "initialize_dds": {},
            "G1Dex3StateReader": {"return_value": reader},
            "TeleimagerCamera": {"return_value": camera},
            "SafeG1Dex3Actuator": {"return_value": actuator},
            "chunk_delta_summary": {"return_value": "bounded"},
            "confirm_actuation": {},
            "confirm_initialization": {},
            "confirm_policy_start": {},
            "_run_blocking_motion_with_immediate_release": {"side_effect": fake_motion},
            "_select_next_goal_while_holding": {"side_effect": select_next},
        }
        for name, options in replacements.items():
            stack.enter_context(mock.patch.object(runner, name, **options))
        validate = stack.enter_context(
            mock.patch.object(runner, "validate_policy_metadata", wraps=validate_policy_metadata)
        )
        infer_name = "infer_plan" if inference_mode == "rtc" else "infer_chunk"
        infer = stack.enter_context(
            mock.patch.object(runner, infer_name, return_value=(SimpleNamespace(length=1), 0.01))
        )
        prepare = stack.enter_context(mock.patch.object(runner, "_prepare_policy_goal", return_value="ready"))
        active_name = "_run_active_goal_rtc" if inference_mode == "rtc" else "_run_active_goal"
        active = stack.enter_context(mock.patch.object(runner, active_name, return_value="hold"))
        runner.run(args)

    expected = [
        (TASKS["pick-red-cup"], False),
        (CUSTOM_GOAL, True),
        (TASKS["down-red-cup"], False),
    ]
    assert [
        (call.kwargs["instruction"], call.kwargs["allow_custom_instruction"]) for call in validate.call_args_list
    ] == expected
    assert [(call.args[4], call.kwargs["allow_custom_instruction"]) for call in prepare.call_args_list] == expected
    assert [(call.args[5], call.kwargs["allow_custom_instruction"]) for call in active.call_args_list] == expected
    assert infer.call_args.args[3] == TASKS["pick-red-cup"]
    assert infer.call_args.kwargs["allow_custom_instruction"] is False
    policy.get_action.assert_not_called()
    actuator.close.assert_called_once_with()
    policy.close.assert_called_once_with()
