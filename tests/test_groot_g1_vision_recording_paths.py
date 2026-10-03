from datetime import datetime, timezone
from pathlib import Path

import pytest

from unitree_lerobot.eval_robot.vision_recording_paths import recording_model_label, recording_output_path


NOW = datetime(2026, 9, 21, 0, 43, 47, 523522, tzinfo=timezone.utc)
IDENTITY = {"model_name": "inspire_rgbd_turbo_late_pre", "checkpoint": "checkpoint-30000"}


def test_name_has_model_checkpoint_and_minutes_only(tmp_path: Path):
    result = recording_output_path(tmp_path, IDENTITY, now=NOW)
    assert result == tmp_path / IDENTITY["model_name"] / "checkpoint-30000" / "2026-09-21_00-43"
    assert result.parent.is_dir()
    assert not result.exists()  # Recorder retains its atomic no-overwrite creation.


def test_repeat_same_minute_gets_counter_without_overwrite(tmp_path: Path):
    first = recording_output_path(tmp_path, IDENTITY, now=NOW)
    first.mkdir()
    marker = first / "keep.txt"
    marker.write_text("original")
    second = recording_output_path(tmp_path, IDENTITY, now=NOW)
    assert second.name == "run02_2026-09-21_00-43"
    second.mkdir()
    assert recording_output_path(tmp_path, IDENTITY, now=NOW).name == "run03_2026-09-21_00-43"
    assert marker.read_text() == "original"


@pytest.mark.parametrize("identity", [None, {}, "malformed"])
def test_old_server_is_honestly_labelled_unknown(tmp_path: Path, identity):
    result = recording_output_path(tmp_path, identity, now=NOW)
    assert result.parent == tmp_path / "unknown-model" / "unknown-checkpoint"


def test_model_names_cannot_escape_recording_root(tmp_path: Path):
    result = recording_output_path(tmp_path, {"model_name": "../../outside", "checkpoint": "../30000"}, now=NOW)
    assert result.is_relative_to(tmp_path)
    assert len(result.relative_to(tmp_path).parts) == 3


def test_existing_symlink_parent_is_rejected(tmp_path: Path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / IDENTITY["model_name"]).symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        recording_output_path(tmp_path, IDENTITY, now=NOW)
    assert not list((tmp_path / "elsewhere").iterdir())


def test_long_model_names_stay_distinct_and_fit_filesystem(tmp_path: Path):
    paths = [recording_output_path(tmp_path, {**IDENTITY, "model_name": "m" * 300 + suffix}, now=NOW)
             for suffix in ("a", "b")]
    assert paths[0] != paths[1]
    assert all(len(path.parent.parent.name.encode()) <= 240 for path in paths)


@pytest.mark.parametrize(("architecture", "label"), [
    ("rgb", "rgb"),
    ("rgbd_turbo_late_fusion_pre_adapter_4x_linear_rgb50_geo50", "turbo_latepre"),
    ("rgbd_turbo_separate_views", "turbo_separate"),
    ("rgb_surface_normals_late_fusion_pre_adapter_4x_linear_rgb50_geo50", "normals_latepre"),
    ("rgb_surface_normals_late_fusion_post_adapter_4x_linear_rgb50_geo50", "normals_latepost"),
    ("rgb_surface_normals_separate_views", "normals_separate"),
    ("normals_6ch_early_fusion", "normals_early"),
    ("rgbd_late_fusion_pre_adapter_4x_linear_rgb50_geo50", "depth_latepre"),
    ("d1_4ch_early_fusion", "depth_early"),
])
def test_short_labels_use_architecture_and_training_count_not_dataset_total(architecture, label):
    name = (f"inspire_c_{architecture}_patch_frozen_bf16_batch_32_acc_1_30k_"
            "final_1090eps_train870eps_20260920_normals_range_mask_v2_moddrop05each_independent")
    assert recording_model_label(name) == f"{label}_final_870ep"


def test_real_rgb_name_has_no_normals_suffix_confusion(tmp_path: Path):
    identity = {"model_name": "inspire_c_rgb_patch_tuned_bf16_batch_32_acc_1_35k_final_1090eps_train870eps_20260920_normals_range_mask_v2", "checkpoint": "checkpoint-30000"}
    result = recording_output_path(tmp_path, identity, now=NOW)
    assert result.relative_to(tmp_path).as_posix() == "rgb_final_870ep/checkpoint-30000/2026-09-21_00-43"
    assert identity["model_name"].endswith("normals_range_mask_v2")  # Original identity untouched.
