from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

import unitree_lerobot.eval_robot.eval_groot_g1 as eval_groot_g1
from unitree_lerobot.eval_robot.groot_client import DeploymentError
from unitree_lerobot.eval_robot.groot_contract import (
    ACTION_KEYS,
    RGBD_VIDEO_KEYS,
    TASKS,
    ModelContract,
    validate_model_contract,
)
from unitree_lerobot.eval_robot.vision_recorder import NonBlockingVisionRecorder
from unitree_lerobot.utils.depth_colormap import (
    FIXED_TURBO_DEPTH_COLORMAP,
    FIXED_TURBO_DEPTH_LUT,
    FIXED_TURBO_DEPTH_LUT_SHA256,
    apply_fixed_turbo_depth_colormap_array,
    fixed_turbo_depth_colormap_contract,
)


def _turbo_modality_config(*, late_fusion: bool = False) -> dict[str, object]:
    video: dict[str, object] = {
        "delta_indices": [0],
        "modality_keys": list(RGBD_VIDEO_KEYS),
        "depth_colormap": FIXED_TURBO_DEPTH_COLORMAP,
    }
    if late_fusion:
        video.update(
            post_vision_fusion=True,
            post_vision_fusion_stage="pre_vision_language_adapter",
        )
    return {
        "video": video,
        "state": {"delta_indices": [0], "modality_keys": list(ACTION_KEYS)},
        "action": {
            "delta_indices": list(range(8)),
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


def _gray_frame() -> np.ndarray:
    gray = np.tile(np.arange(640, dtype=np.uint16) % 256, (480, 1)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _wait_for_path(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def test_fixed_turbo_lut_is_exact_and_reserves_invalid_black() -> None:
    digest = "sha256:" + hashlib.sha256(FIXED_TURBO_DEPTH_LUT.tobytes()).hexdigest()

    assert digest == FIXED_TURBO_DEPTH_LUT_SHA256
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[0], [0, 0, 0])
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[1], [48, 18, 59])
    np.testing.assert_array_equal(FIXED_TURBO_DEPTH_LUT[255], [122, 4, 2])


@pytest.mark.parametrize("late_fusion", [False, True])
def test_model_contract_accepts_exact_checkpoint_bound_turbo(late_fusion: bool) -> None:
    contract = validate_model_contract(_turbo_modality_config(late_fusion=late_fusion))

    assert contract.depth_colormap == FIXED_TURBO_DEPTH_COLORMAP
    assert contract.vision_input_contract["depth_colormap"] == (fixed_turbo_depth_colormap_contract())


def test_model_contract_rejects_unknown_or_early_fused_turbo() -> None:
    unknown = _turbo_modality_config()
    unknown["video"]["depth_colormap"] = "turbo_from_runtime_library"
    with pytest.raises(DeploymentError, match="depth_colormap"):
        validate_model_contract(unknown)

    early = _turbo_modality_config()
    early["video"]["channel_fusion"] = [
        {"key": "ego_view", "channels": [0, 1, 2]},
        {"key": "depth_gray_view", "channels": [0]},
    ]
    with pytest.raises(DeploymentError, match="cannot use early channel fusion"):
        validate_model_contract(early)


def test_preview_renders_exact_turbo_without_mutating_wire_depth() -> None:
    rgb = np.zeros((1, 6, 3), dtype=np.uint8)
    gray_codes = np.array([[0, 1, 64, 128, 192, 255]], dtype=np.uint8)
    depth = np.repeat(gray_codes[..., None], 3, axis=-1)
    original = depth.copy()

    with (
        mock.patch.object(eval_groot_g1.cv2, "imshow") as imshow,
        mock.patch.object(eval_groot_g1, "_pump_camera_preview_events"),
    ):
        eval_groot_g1.show_camera_preview(
            rgb,
            depth,
            depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
        )

    rendered_bgr = imshow.call_args_list[1].args[1]
    expected_rgb = apply_fixed_turbo_depth_colormap_array(depth)
    np.testing.assert_array_equal(rendered_bgr, cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(depth, original)
    eval_groot_g1.close_camera_preview()


def test_capture_keeps_policy_and_recorder_wire_depth_gray() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = _gray_frame()
    state = SimpleNamespace(
        arm=np.zeros(14, dtype=np.float64),
        left_hand=np.zeros(7, dtype=np.float64),
        right_hand=np.zeros(7, dtype=np.float64),
    )
    reader = SimpleNamespace(read=mock.Mock(return_value=state))
    camera = SimpleNamespace(
        read=mock.Mock(
            return_value=SimpleNamespace(
                rgb=rgb,
                depth_gray=depth,
                surface_normals=None,
            )
        )
    )
    recorder = mock.Mock()
    contract = ModelContract(
        action_horizon=8,
        video_keys=RGBD_VIDEO_KEYS,
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )

    with mock.patch.object(eval_groot_g1, "show_camera_preview") as preview:
        observation, _ = eval_groot_g1.capture_policy_observation(
            reader,
            camera,
            TASKS["pick-red-cup"],
            contract,
            show_camera=True,
            vision_recorder=recorder,
        )

    preview.assert_called_once_with(
        rgb,
        depth,
        "depth_gray_view",
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )
    np.testing.assert_array_equal(observation["video"]["depth_gray_view"][0, 0], depth)
    assert recorder.submit_observation.call_args.args[0] is observation
    np.testing.assert_array_equal(
        recorder.submit_observation.call_args.args[0]["video"]["depth_gray_view"][0, 0],
        depth,
    )


def test_recorder_saves_model_visible_turbo_and_records_contract(tmp_path: Path) -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = _gray_frame()
    original = depth.copy()
    observation = {
        "video": {
            "ego_view": rgb[None, None],
            "depth_gray_view": depth[None, None],
        }
    }
    output_dir = tmp_path / "recording"
    recorder = NonBlockingVisionRecorder(
        output_dir,
        RGBD_VIDEO_KEYS,
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )
    depth_path = output_dir / "depth_gray_view" / "frame-000000.png"
    try:
        assert recorder.submit_observation(observation) is True
        _wait_for_path(depth_path)
    finally:
        recorder.close()

    saved_bgr = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
    assert saved_bgr is not None
    saved_rgb = cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(saved_rgb, apply_fixed_turbo_depth_colormap_array(depth))
    np.testing.assert_array_equal(depth, original)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["view_transforms"] == {"depth_gray_view": fixed_turbo_depth_colormap_contract()}


def test_recording_metadata_includes_exact_turbo_contract() -> None:
    metadata = eval_groot_g1._vision_recording_metadata(
        end_effector="inspire-ftp",
        execution_horizon=8,
        image_host="192.168.123.164",
        inference_mode="synchronous",
        surface_normal_encoding=None,
        depth_colormap=FIXED_TURBO_DEPTH_COLORMAP,
    )

    assert metadata["depth_colormap"] == fixed_turbo_depth_colormap_contract()
