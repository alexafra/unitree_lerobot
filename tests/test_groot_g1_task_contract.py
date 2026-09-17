from __future__ import annotations

from unittest import mock

import pytest

from unitree_lerobot.eval_robot.eval_groot_g1 import build_parser, run
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ModelContract,
    TASKS,
    _task_contract_sha256,
    validate_policy_instruction,
)


RED_CUP_TASKS = ["pick up the red cup.", "put down the red cup."]


def test_left_to_right_cup_pyramid_task_is_exactly_registered():
    assert TASKS["build-cup-pyramid-left-to-right"] == (
        "build a cup pyramid left-to-right."
    )


def _metadata(instructions=RED_CUP_TASKS):
    instructions = list(instructions)
    return {
        "protocol_version": 1,
        "embodiment_tag": "new_embodiment",
        "task_contract": {
            "schema_version": 1,
            "instructions": instructions,
            "sha256": _task_contract_sha256(instructions),
        },
    }


def test_exact_red_cup_task_allowlist_matches_server_hash_contract():
    metadata = _metadata()

    assert metadata["task_contract"]["sha256"] == (
        "1dc8e65b8210cd017ad48081d860208805ad4638785496059eef743fd1d633e7"
    )
    assert validate_policy_instruction(metadata, "pick up the red cup.") == tuple(RED_CUP_TASKS)
    assert validate_policy_instruction(metadata, "put down the red cup.") == tuple(RED_CUP_TASKS)


@pytest.mark.parametrize(
    "instruction",
    [
        "pick up the wooden block.",
        "Pick up the red cup.",
        "pick up the red cup",
        "pick up the red cup. ",
    ],
)
def test_unadvertised_or_inexact_instruction_is_rejected(instruction):
    with pytest.raises(DeploymentError, match="not advertised by this checkpoint"):
        validate_policy_instruction(_metadata(), instruction)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata.pop("task_contract"), "no checkpoint task contract"),
        (
            lambda metadata: metadata["task_contract"].update(sha256="0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda metadata: metadata["task_contract"].update(schema_version=2),
            "Unsupported GR00T task contract schema",
        ),
        (
            lambda metadata: metadata["task_contract"].update(
                instructions=["pick up the red cup.", "pick up the red cup."]
            ),
            "unique",
        ),
    ],
)
def test_malformed_task_contract_fails_closed(mutate, message):
    metadata = _metadata()
    mutate(metadata)

    with pytest.raises(DeploymentError, match=message):
        validate_policy_instruction(metadata, "pick up the red cup.")


@pytest.mark.parametrize(
    "goal_args",
    [
        ["--task", "pick-wooden-block"],
        ["--custom-goal", "pick up the blue cup."],
    ],
)
def test_runner_rejects_unadvertised_instruction_before_dds(goal_args):
    args = build_parser().parse_args(goal_args)
    policy = mock.Mock()
    policy.ping.return_value = True
    policy.get_modality_config.return_value = {}
    policy.get_policy_metadata.return_value = _metadata()
    module = "unitree_lerobot.eval_robot.eval_groot_g1"

    with (
        mock.patch(f"{module}.Gr00tClient", return_value=policy),
        mock.patch(
            f"{module}.validate_model_contract",
            return_value=ModelContract(action_horizon=32),
        ),
        mock.patch(f"{module}.initialize_dds") as initialize_dds,
        pytest.raises(DeploymentError, match="not advertised by this checkpoint"),
    ):
        run(args)

    initialize_dds.assert_not_called()
    policy.close.assert_called_once_with()
