"""Named measured Warmup1 poses: exact frozen values, not filesystem fixtures."""

from unittest import mock

import numpy as np
import pytest

import unitree_lerobot.eval_robot.eval_groot_g1 as runner
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import TRAINING_START_MODE
from unitree_lerobot.eval_robot.training_start_pose import (
    load_warmup1_pose,
    training_start_source,
    training_start_spec,
)


# Frozen from the reviewed candidate JSONs, not loaded from a user's directory.
# Each list is arm14 + left hand6 + right hand6, from observation.state frame0.
PYRAMID1 = [
    -0.3238137662410736, 0.11266370117664337, 0.260237455368042,
    1.071677327156067, -0.34684744477272034, -1.010689616203308,
    -0.844001054763794, -0.3882769048213959, -0.19183149933815002,
    -0.057380471378564835, 1.0802819728851318, -0.0228659026324749,
    -0.9447644352912903, 0.28523653745651245,
    0.6859999895095825, 0.777999997138977, 0.8489999771118164,
    0.8619999885559082, 0.9990000128746033, 0.8289999961853027,
    0.6710000038146973, 0.7749999761581421, 0.8259999752044678,
    0.871999979019165, 0.8809999823570251, 0.6759999990463257,
]
PYRAMID2 = [
    -0.33673277497291565, 0.17343570291996002, 0.3664536476135254,
    1.0695921182632446, -0.3935379981994629, -0.870522141456604,
    -0.8195532560348511, -0.4010041654109955, -0.16189490258693695,
    -0.16660469770431519, 0.9246788620948792, 0.20451080799102783,
    -0.859544575214386, 0.5700176954269409,
    0.6880000233650208, 0.7870000004768372, 0.8550000190734863,
    0.8730000257492065, 0.9959999918937683, 0.9039999842643738,
    0.5289999842643738, 0.625, 0.6909999847412109,
    0.7670000195503235, 0.8610000014305115, 0.7839999794960022,
]
SOURCE_DATASET = (
    "/home/alex/Development/Datasets/lerobot2/inspire/"
    "pyramid_sideways_09_19_normals_range_mask_v2/train"
)


@pytest.mark.parametrize("profile", ["dex3", "inspire-ftp", "inspire-dfx"])
@pytest.mark.parametrize("name", [None, "default"])
def test_default_name_and_omission_preserve_existing_pose_bitwise(profile, name):
    expected = training_start_spec(profile)
    actual, source = load_warmup1_pose(profile, pose_name=name)
    assert (actual.mode, actual.label, actual.end_effector) == (
        expected.mode, expected.label, expected.end_effector,
    )
    for field in ("arm", "left_hand", "right_hand"):
        actual_values = getattr(actual, field)
        expected_values = getattr(expected, field)
        assert actual_values.dtype == expected_values.dtype
        assert actual_values.tobytes() == expected_values.tobytes()
    assert source == {**training_start_source(profile), "pose_name": "default"}


@pytest.mark.parametrize("profile", ["inspire-ftp", "inspire-dfx"])
@pytest.mark.parametrize(
    "name,expected,episode,source_episode",
    [("pyramid1", PYRAMID1, 11, "episode_0018"), ("pyramid2", PYRAMID2, 46, "episode_0068")],
)
def test_named_pyramids_match_reviewed_complete_measured_poses(profile, name, expected, episode, source_episode):
    spec, source = load_warmup1_pose(profile, pose_name=name)
    assert spec.mode == TRAINING_START_MODE
    assert spec.label.startswith("Warmup1:")
    assert spec.end_effector == profile
    actual = np.concatenate((spec.arm, spec.left_hand, spec.right_hand))
    assert actual.dtype == np.float64
    assert actual.tobytes() == np.asarray(expected, dtype=np.float64).tobytes()
    assert spec.arm.shape == (14,)
    assert spec.left_hand.shape == spec.right_hand.shape == (6,)
    for array in (spec.arm, spec.left_hand, spec.right_hand):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0.0
    assert source["pose_name"] == name
    assert source["dataset_path"] == SOURCE_DATASET
    assert source["episode_index"] == episode
    assert source["frame_index"] == 0
    assert source["quantity"] == "observation.state"
    assert source["source_episode"] == source_episode
    assert source["task"] == "build a cup pyramid left-to-right."


@pytest.mark.parametrize("name", ["pyramid1", "pyramid2"])
def test_named_pose_source_copy_cannot_mutate_later_loads(name):
    first, first_source = load_warmup1_pose("inspire-ftp", pose_name=name)
    first_source["pose_name"] = "tampered"
    first_source["episode_index"] = -1
    second, second_source = load_warmup1_pose("inspire-ftp", pose_name=name)
    assert second_source["pose_name"] == name
    assert second_source["episode_index"] in (11, 46)
    np.testing.assert_array_equal(first.arm, second.arm)


@pytest.mark.parametrize("name", ["pyramid1", "pyramid2"])
def test_pyramid_names_reject_wrong_hand_profile(name):
    with pytest.raises(DeploymentError):
        load_warmup1_pose("dex3", pose_name=name)


@pytest.mark.parametrize("name", ["pyramid3", "", "PYRAMID1", "unknown"])
def test_unknown_name_is_not_a_silent_default(name):
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", pose_name=name)


@pytest.mark.parametrize("name", ["default", "pyramid1", "pyramid2"])
def test_name_and_file_are_mutually_exclusive_in_loader(tmp_path, name):
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", tmp_path / "not-read.json", pose_name=name)


def test_parser_defaults_remain_unchanged_and_named_option_is_validated():
    parser = runner.build_parser()
    default = parser.parse_args([])
    assert default.warmup1_pose is None
    assert default.warmup1_pose_file is None
    for name in ("default", "pyramid1", "pyramid2"):
        selected = parser.parse_args(["--warmup1-pose", name])
        assert selected.warmup1_pose == name
        assert selected.warmup1_pose_file is None
        for field in ("warmup1", "policy_warm_start", "return_to_start", "actuate", "inference_mode", "debug_actions"):
            assert getattr(selected, field) == getattr(default, field)
    with pytest.raises(SystemExit):
        parser.parse_args(["--warmup1-pose", "pyramid3"])


def test_parser_rejects_both_named_and_path_pose_options():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([
            "--warmup1-pose", "pyramid1", "--warmup1-pose-file", "/offline/pose.json",
        ])


@pytest.mark.parametrize("name", ["default", "pyramid1", "pyramid2"])
def test_explicit_name_rejected_with_disabled_warmup1(name):
    args = runner.build_parser().parse_args([
        "--no-actuate", "--task", "pick-red-cup", "--no-warmup1", "--warmup1-pose", name,
    ])
    with pytest.raises(DeploymentError, match="warmup1"):
        runner.validate_args(args)


def test_runner_passes_named_choice_to_loader_before_external_boundaries(monkeypatch):
    class PoseLoadReached(Exception):
        pass

    args = runner.build_parser().parse_args([
        "--no-actuate", "--task", "pick-red-cup", "--warmup1-pose", "pyramid2",
    ])
    loader = mock.Mock(side_effect=PoseLoadReached)
    policy = mock.Mock(side_effect=AssertionError("no network allowed"))
    dds = mock.Mock(side_effect=AssertionError("no DDS allowed"))
    monkeypatch.setattr(runner, "load_warmup1_pose", loader)
    monkeypatch.setattr(runner, "Gr00tClient", policy)
    monkeypatch.setattr(runner, "initialize_dds", dds)
    monkeypatch.setattr(runner.os, "chdir", mock.Mock())
    with pytest.raises(PoseLoadReached):
        runner.run(args)
    loader.assert_called_once_with("inspire-ftp", None, pose_name="pyramid2")
    policy.assert_not_called()
    dds.assert_not_called()
