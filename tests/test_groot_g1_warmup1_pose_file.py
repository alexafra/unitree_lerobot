"""CPU-only validation of whole-pose Warmup1 files; no publishers are created."""

import hashlib
import json

import numpy as np
import pytest

from unitree_lerobot.eval_robot.g1_end_effectors import get_end_effector_profile
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import ARM_JOINT_NAMES, TRAINING_START_MODE
from unitree_lerobot.eval_robot.training_start_pose import (
    WARMUP1_POSE_MAX_BYTES,
    load_warmup1_pose,
    training_start_source,
    training_start_spec,
)


def _payload(end_effector="inspire-ftp"):
    spec = training_start_spec(end_effector)
    profile = get_end_effector_profile(end_effector)
    return {
        "schema_version": 1,
        "name": "stack example",
        "robot_type": "g1",
        "end_effector": end_effector,
        "arm_unit": "rad",
        "hand_unit": profile.value_unit,
        "joint_names": {
            "arm": list(ARM_JOINT_NAMES),
            "left_hand": list(profile.left_joint_names),
            "right_hand": list(profile.right_joint_names),
        },
        "arm": spec.arm.tolist(),
        "left_hand": spec.left_hand.tolist(),
        "right_hand": spec.right_hand.tolist(),
        "source": {
            "dataset_path": "/historical/dataset/no-longer-present/train",
            "episode_index": 428,
            "frame_index": 0,
            "quantity": "observation.state",
            "source_episode": "episode_0084",
            "task": "stack the three red cups.",
            "timestamp_s": 0.0,
        },
    }


def _write(tmp_path, payload):
    path = tmp_path / "warmup1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("profile", ["dex3", "inspire-dfx", "inspire-ftp"])
def test_no_file_preserves_exact_default_pose_and_source(profile):
    expected = training_start_spec(profile)
    spec, source = load_warmup1_pose(profile)
    assert (spec.mode, spec.label, spec.end_effector) == (
        expected.mode, expected.label, expected.end_effector,
    )
    for field in ("arm", "left_hand", "right_hand"):
        actual = getattr(spec, field)
        reference = getattr(expected, field)
        assert actual.dtype == reference.dtype
        assert actual.tobytes() == reference.tobytes()
        assert not np.shares_memory(actual, reference)
    assert source == {**training_start_source(profile), "pose_name": "default"}


@pytest.mark.parametrize("profile", ["dex3", "inspire-dfx", "inspire-ftp"])
def test_valid_file_preserves_values_provenance_and_freezes_arrays(tmp_path, profile):
    payload = _payload(profile)
    path = _write(tmp_path, payload)
    spec, source = load_warmup1_pose(profile, path)
    assert spec.mode == ("pose-file" if profile == "dex3" else TRAINING_START_MODE)
    assert spec.label == "Warmup1: stack example"
    assert spec.end_effector == profile
    for field in ("arm", "left_hand", "right_hand"):
        actual = getattr(spec, field)
        np.testing.assert_array_equal(actual, payload[field])
        assert actual.dtype == np.float64
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual[0] = 0.0
    assert source == {
        **payload["source"],
        "pose_name": payload["name"],
        "pose_file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_optional_provenance_fields_are_optional_and_dataset_need_not_exist(tmp_path):
    payload = _payload()
    for key in ("source_episode", "task", "timestamp_s"):
        del payload["source"][key]
    spec, source = load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))
    assert spec.moves
    assert source["dataset_path"] == payload["source"]["dataset_path"]
    assert "timestamp_s" not in source


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True), ("schema_version", 1.0), ("schema_version", 2),
        ("name", ""), ("name", " " * 3), ("name", "x" * 101), ("name", "bad\nname"),
        ("robot_type", "g1_other"), ("end_effector", "inspire-dfx"),
        ("arm_unit", "deg"), ("hand_unit", "rad"),
        ("arm", None), ("left_hand", "measured"), ("right_hand", None),
        ("source", None), ("joint_names", []),
    ],
)
def test_invalid_top_level_value_is_rejected(tmp_path, field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))


@pytest.mark.parametrize("field", ["arm", "left_hand", "right_hand"])
@pytest.mark.parametrize("bad_value", [True, "0.5", None, float("nan"), float("inf"), 1e100])
def test_nonfinite_boolean_nonnumeric_or_out_of_range_target_is_rejected(tmp_path, field, bad_value):
    payload = _payload()
    payload[field][0] = bad_value
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))


@pytest.mark.parametrize("field", ["arm", "left_hand", "right_hand"])
def test_wrong_vector_length_and_joint_order_are_rejected(tmp_path, field):
    payload = _payload()
    payload[field].pop()
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))
    payload = _payload()
    payload["joint_names"][field] = list(reversed(payload["joint_names"][field]))
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_path", ""), ("dataset_path", 1),
        ("episode_index", -1), ("episode_index", True), ("episode_index", 1.0),
        ("frame_index", -1), ("frame_index", True), ("frame_index", 0.0),
        ("quantity", "action"), ("quantity", "state"),
        ("task", None), ("source_episode", ""),
        ("timestamp_s", -1.0), ("timestamp_s", True),
        ("timestamp_s", float("nan")), ("timestamp_s", float("inf")),
        ("timestamp_s", "0"), ("timestamp_s", 10**400),
    ],
)
def test_invalid_source_provenance_is_rejected(tmp_path, field, value):
    payload = _payload()
    payload["source"][field] = value
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))


@pytest.mark.parametrize("location", [None, "source", "joint_names"])
def test_extra_and_missing_fields_are_rejected(tmp_path, location):
    payload = _payload()
    target = payload if location is None else payload[location]
    target["unexpected"] = "bad"
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))
    payload = _payload()
    target = payload if location is None else payload[location]
    del target["arm" if location in (None, "joint_names") else "quantity"]
    with pytest.raises(DeploymentError):
        load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))


@pytest.mark.parametrize("field", ["schema_version", "dataset_path", "arm"])
def test_duplicate_json_fields_rejected_at_every_object_depth(tmp_path, field):
    payload = _payload()
    raw = json.dumps(payload)
    needle = json.dumps(field) + ":"
    raw = raw.replace(needle, needle + " null, " + needle, 1)
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(DeploymentError, match="duplicate"):
        load_warmup1_pose("inspire-ftp", path)


@pytest.mark.parametrize("profile", ["inspire-dfx", "dex3"])
def test_file_must_match_selected_profile(tmp_path, profile):
    with pytest.raises(DeploymentError, match="end_effector"):
        load_warmup1_pose(profile, _write(tmp_path, _payload("inspire-ftp")))


def test_existing_profile_limits_are_not_relaxed(tmp_path):
    for field, value in [("left_hand", -0.000001), ("right_hand", 1.000001), ("arm", 100.0)]:
        payload = _payload()
        payload[field][0] = value
        with pytest.raises(DeploymentError):
            load_warmup1_pose("inspire-ftp", _write(tmp_path, payload))
    payload = _payload("dex3")
    payload["left_hand"][0] = 100.0
    with pytest.raises(DeploymentError):
        load_warmup1_pose("dex3", _write(tmp_path, payload))


def test_size_limit_bad_encoding_malformed_json_and_nonobject_are_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    for raw in [b" " * (WARMUP1_POSE_MAX_BYTES + 1), b"\xff", b"{", b"[]", b"null"]:
        path.write_bytes(raw)
        with pytest.raises(DeploymentError):
            load_warmup1_pose("inspire-ftp", path)


def test_missing_file_and_directory_are_rejected(tmp_path):
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(DeploymentError):
            load_warmup1_pose("inspire-ftp", path)


def test_name_limit_and_exact_64_kib_file_are_accepted(tmp_path):
    payload = _payload()
    payload["name"] = "x" * 100
    raw = json.dumps(payload).encode("utf-8")
    path = tmp_path / "limit.json"
    path.write_bytes(raw + b" " * (WARMUP1_POSE_MAX_BYTES - len(raw)))
    spec, source = load_warmup1_pose("inspire-ftp", path)
    assert spec.label == "Warmup1: " + "x" * 100
    assert source["pose_file"] == str(path.resolve())
