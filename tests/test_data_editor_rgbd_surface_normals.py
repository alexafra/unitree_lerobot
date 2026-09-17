from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np

from unitree_lerobot.utils.camera_calibration import D435I_254322071415_CALIBRATION
from unitree_lerobot.utils.depth_encoding import encode_depth_gray_rgb
from unitree_lerobot.utils.surface_normal_encoding import (
    REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
    encode_surface_normals_rgb,
)

EDITOR_PATH = Path(__file__).parents[1] / "data_editor" / "data_editor_EN_rgbd.py"


class _QtObject:
    pass


class _QImage(_QtObject):
    Format_RGB888 = object()


def _pyqt_signal(*args):
    return object()


def _load_editor_with_pyqt_stubs():
    pyqt = types.ModuleType("PyQt5")
    qt_core = types.ModuleType("PyQt5.QtCore")
    qt_core.Qt = types.SimpleNamespace()
    qt_core.QTimer = _QtObject
    qt_core.QRect = _QtObject
    qt_core.pyqtSignal = _pyqt_signal

    qt_gui = types.ModuleType("PyQt5.QtGui")
    qt_gui.QPixmap = _QtObject
    qt_gui.QImage = _QImage
    qt_gui.QPainter = _QtObject
    qt_gui.QColor = _QtObject
    qt_gui.QPen = _QtObject
    qt_gui.QBrush = _QtObject

    qt_widgets = types.ModuleType("PyQt5.QtWidgets")
    for name in (
        "QApplication",
        "QWidget",
        "QLabel",
        "QPushButton",
        "QGridLayout",
        "QHBoxLayout",
        "QVBoxLayout",
        "QMessageBox",
        "QFrame",
        "QFileDialog",
        "QComboBox",
        "QLineEdit",
    ):
        setattr(qt_widgets, name, type(name, (_QtObject,), {}))

    stubs = {
        "PyQt5": pyqt,
        "PyQt5.QtCore": qt_core,
        "PyQt5.QtGui": qt_gui,
        "PyQt5.QtWidgets": qt_widgets,
    }
    module_name = "_data_editor_rgbd_under_test"
    spec = importlib.util.spec_from_file_location(module_name, EDITOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load editor from {EDITOR_PATH}")
    module = importlib.util.module_from_spec(spec)

    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class _Label:
    def __init__(self):
        self.placeholder = None
        self.pixmap = None

    def set_placeholder(self, text):
        self.placeholder = text

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap


class _Pixmap:
    def isNull(self):
        return False


class DataEditorRgbdSurfaceNormalsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.editor = _load_editor_with_pyqt_stubs()

    def test_bottom_row_places_derived_normals_left_and_raw_depth_right(self):
        self.assertEqual(
            self.editor.DatasetPlayer.DISPLAY_GRID_POSITIONS,
            {
                "color_0": (0, 0),
                "depth_0": (0, 1),
                "surface_normals_view": (1, 0),
                "raw_depth_0": (1, 1),
            },
        )
        self.assertEqual(
            self.editor.DatasetPlayer.DISPLAY_STREAMS["surface_normals_view"][1],
            "surface_normals",
        )
        self.assertEqual(
            self.editor.DatasetPlayer.DISPLAY_STREAMS["raw_depth_0"][1],
            "depth",
        )

    def test_color_only_cli_is_explicit_and_preserves_qt_arguments(self):
        args, qt_args = self.editor.parse_cli_args([])
        self.assertFalse(args.color_only)
        self.assertEqual(qt_args, [])

        args, qt_args = self.editor.parse_cli_args(
            ["--color-only", "-platform", "offscreen"]
        )
        self.assertTrue(args.color_only)
        self.assertEqual(qt_args, ["-platform", "offscreen"])

        args, qt_args = self.editor.parse_cli_args(["--color"])
        self.assertFalse(args.color_only)
        self.assertEqual(qt_args, ["--color"])

    def test_color_only_display_configuration_contains_only_rgb(self):
        streams, positions = self.editor.DatasetPlayer.active_display_configuration(False)
        self.assertIs(streams, self.editor.DatasetPlayer.DISPLAY_STREAMS)
        self.assertIs(positions, self.editor.DatasetPlayer.DISPLAY_GRID_POSITIONS)

        streams, positions = self.editor.DatasetPlayer.active_display_configuration(True)
        self.assertEqual(streams, {"color_0": ("RGB Camera 0", "color")})
        self.assertEqual(positions, {"color_0": (0, 0)})

    def test_color_only_ignores_invalid_unused_depth_scale(self):
        payload = {"info": {"depth": {"scale_m_per_unit": "invalid"}}}
        with mock.patch.object(
            self.editor,
            "resolve_depth_scale_m_per_unit",
            side_effect=AssertionError("depth scale must not be inspected"),
        ) as resolver:
            self.assertIsNone(
                self.editor.resolve_display_depth_scale(payload, color_only=True)
            )
        resolver.assert_not_called()

        with self.assertRaises(ValueError):
            self.editor.resolve_display_depth_scale(payload, color_only=False)

    def test_editor_canonicalizes_realsense_reported_scale_without_byte_change(self):
        payload = {
            "info": {
                "depth": {
                    "scale_m_per_unit": 0.001,
                    "scale_reported_m_per_unit": 0.0010000000474974513,
                }
            }
        }
        depth = np.array([[0, 250, 625, 1000]], dtype=np.uint16)

        scale = self.editor.resolve_depth_scale_m_per_unit(payload)

        self.assertEqual(scale, 0.001)
        np.testing.assert_array_equal(
            self.editor.depth_to_gray_rgb(depth, scale),
            self.editor.depth_to_gray_rgb(depth, 0.0010000000474974513),
        )

    def test_randomized_depth_preview_is_byte_exact_with_production_encoder(self):
        generator = np.random.default_rng(4012)
        depth = generator.integers(0, 5001, size=(73, 101), dtype=np.uint16)
        depth[::7, ::11] = 0

        preview = self.editor.depth_to_gray_rgb(
            depth,
            0.0010000000474974513,
        )
        production = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=0.001,
            near_m=0.25,
            far_m=1.0,
        )

        np.testing.assert_array_equal(preview, production)
        self.assertGreater(np.unique(preview[..., 0]).size, 100)

    def test_randomized_recorded_k_normals_preview_is_byte_exact_with_production(self):
        generator = np.random.default_rng(254322071415)
        y, x = np.indices((480, 640), dtype=np.uint16)
        depth = (500 + x // 8 + y // 12).astype(np.uint16)
        depth += generator.integers(0, 3, size=depth.shape, dtype=np.uint16)
        depth[::53, ::47] = 0
        intrinsics = self.editor.resolve_surface_normal_intrinsics(
            {"info": {"depth": {"calibration": D435I_254322071415_CALIBRATION}}}
        )

        preview = self.editor.aligned_depth_to_surface_normals_rgb(
            depth,
            0.0010000000474974513,
            intrinsics,
        )
        production = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
        )

        np.testing.assert_array_equal(preview, production)
        self.assertGreater(np.count_nonzero(preview), 100_000)

    def test_normals_preview_masks_samples_outside_depth_preview_range(self):
        intrinsics = self.editor.PinholeIntrinsics(
            width=7,
            height=5,
            fx=100.0,
            fy=100.0,
            cx=3.0,
            cy=2.0,
        )
        depth = np.full((5, 7), 600, dtype=np.uint16)
        depth[2, 2] = 1_001

        normals = self.editor.aligned_depth_to_surface_normals_rgb(
            depth,
            0.001,
            intrinsics,
        )

        np.testing.assert_array_equal(normals[2, 3], [0, 0, 0])

    def test_editor_uses_recorded_color_intrinsics_for_normals(self):
        payload = {
            "info": {
                "depth": {"calibration": D435I_254322071415_CALIBRATION}
            }
        }

        self.assertEqual(
            self.editor.resolve_surface_normal_intrinsics(payload),
            REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
        )
        self.assertEqual(
            self.editor.resolve_surface_normal_intrinsics({}),
            self.editor.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        )

    def test_legacy_inspire_uses_replacement_camera_k_while_dex3_stays_old(self):
        inspire = {"info": {"end_effector": {"type": "inspire"}, "depth": {}}}
        dex3 = {"info": {"end_effector": {"type": "dex3"}, "depth": {}}}

        self.assertEqual(
            self.editor.resolve_surface_normal_intrinsics(inspire),
            REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
        )
        self.assertEqual(
            self.editor.resolve_surface_normal_intrinsics(dex3),
            self.editor.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        )

    def test_derived_view_uses_the_aligned_depth_path_not_raw_depth(self):
        paths = self.editor.resolve_frame_display_paths(
            {
                "colors": {"color_0": "colors/000007_color_0.jpg"},
                "depths": {
                    "depth_0": "depths/000007_depth_0.png",
                    "raw_depth_0": "raw_depths/000007_raw_depth_0.png",
                },
            }
        )

        self.assertEqual(paths["surface_normals_view"], paths["depth_0"])
        self.assertNotEqual(paths["surface_normals_view"], paths["raw_depth_0"])

    def test_derived_view_calls_the_shared_encoder_with_known_color_intrinsics(self):
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        expected = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch.object(
            self.editor,
            "encode_surface_normals_rgb",
            return_value=expected,
        ) as encoder:
            result = self.editor.aligned_depth_to_surface_normals_rgb(depth, 0.001)

        self.assertIs(result, expected)
        encoder.assert_called_once_with(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=self.editor.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
            depth_near_m=0.25,
            depth_far_m=1.0,
        )

    def test_show_frame_dispatches_aligned_depth_to_normals_renderer(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.frame_keys = [7]
        player.frames_map = {
            7: {
                "color_0": "/episode/colors/000007_color_0.jpg",
                "depth_0": "/episode/depths/000007_depth_0.png",
                "surface_normals_view": "/episode/depths/000007_depth_0.png",
                "raw_depth_0": "/episode/raw_depths/000007_raw_depth_0.png",
            }
        }
        player.current_episode_name = "episode_0001"
        player.play_selection_only = False
        player.is_playing = False
        player.color_only = False
        player.depth_scale_m_per_unit = 0.001
        player.surface_normal_intrinsics = (
            self.editor.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480
        )
        player.info_label = mock.Mock()
        player.active_display_streams = player.DISPLAY_STREAMS
        player.image_labels = {key: _Label() for key in player.active_display_streams}
        player.range_slider = mock.Mock()

        with (
            mock.patch.object(self.editor.os.path, "isfile", return_value=True),
            mock.patch.object(self.editor, "QPixmap", return_value=_Pixmap()),
            mock.patch.object(self.editor, "load_depth_pixmap", return_value=_Pixmap()) as load_depth,
            mock.patch.object(
                self.editor,
                "load_surface_normals_pixmap",
                return_value=_Pixmap(),
            ) as load_normals,
        ):
            player.show_frame(0)

        load_normals.assert_called_once_with(
            "/episode/depths/000007_depth_0.png",
            0.001,
            self.editor.DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        )
        self.assertEqual(
            load_depth.call_args_list,
            [
                mock.call("/episode/depths/000007_depth_0.png", 0.001),
                mock.call("/episode/raw_depths/000007_raw_depth_0.png", 0.001),
            ],
        )

    def test_color_only_show_frame_never_dispatches_depth_renderers(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.frame_keys = [7]
        player.frames_map = {
            7: {
                "color_0": "/episode/colors/000007_color_0.jpg",
                "depth_0": "/episode/depths/000007_depth_0.png",
                "surface_normals_view": "/episode/depths/000007_depth_0.png",
                "raw_depth_0": "/episode/raw_depths/000007_raw_depth_0.png",
            }
        }
        player.current_episode_name = "episode_0001"
        player.play_selection_only = False
        player.is_playing = False
        player.color_only = True
        player.depth_scale_m_per_unit = None
        player.info_label = mock.Mock()
        player.active_display_streams = {"color_0": player.DISPLAY_STREAMS["color_0"]}
        player.image_labels = {"color_0": _Label()}
        player.range_slider = mock.Mock()

        with (
            mock.patch.object(self.editor.os.path, "isfile", return_value=True),
            mock.patch.object(
                self.editor,
                "QPixmap",
                return_value=_Pixmap(),
            ) as load_color,
            mock.patch.object(self.editor, "load_depth_pixmap") as load_depth,
            mock.patch.object(
                self.editor,
                "load_surface_normals_pixmap",
            ) as load_normals,
        ):
            player.show_frame(0)

        load_color.assert_called_once_with("/episode/colors/000007_color_0.jpg")
        load_depth.assert_not_called()
        load_normals.assert_not_called()
        self.assertIn("Visuals: Color only", player.info_label.setText.call_args.args[0])

    def test_health_findings_show_header_and_enable_full_details(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.health_warning_label = mock.Mock()
        player.health_details_btn = mock.Mock()
        player._health_report_text = ""
        scan = types.SimpleNamespace(findings=(object(),), warnings=(object(),), serious=())

        with (
            mock.patch.object(
                self.editor,
                "health_header_text",
                return_value="yellow warning",
            ),
            mock.patch.object(self.editor, "render_report", return_value="full reasons"),
        ):
            player._apply_episode_health_scan(scan)

        player.health_warning_label.setText.assert_called_once_with("yellow warning")
        self.assertEqual(player._health_report_text, "full reasons")
        player.health_warning_label.setStyleSheet.assert_called_once_with(
            self.editor.EPISODE_HEALTH_WARNING_STYLE
        )
        player.health_warning_label.show.assert_called_once_with()
        player.health_details_btn.show.assert_called_once_with()

    def test_serious_health_findings_show_red_header(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.health_warning_label = mock.Mock()
        player.health_details_btn = mock.Mock()
        player._health_report_text = ""
        scan = types.SimpleNamespace(findings=(object(),), warnings=(), serious=(object(),))

        with (
            mock.patch.object(self.editor, "health_header_text", return_value="red issue"),
            mock.patch.object(self.editor, "render_report", return_value="full reasons"),
        ):
            player._apply_episode_health_scan(scan)

        player.health_warning_label.setText.assert_called_once_with("red issue")
        player.health_warning_label.setStyleSheet.assert_called_once_with(
            self.editor.EPISODE_HEALTH_SERIOUS_STYLE
        )

    def test_clean_health_scan_shows_explicit_green_status(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.health_warning_label = mock.Mock()
        player.health_details_btn = mock.Mock()
        player._health_report_text = ""
        scan = types.SimpleNamespace(findings=())

        with (
            mock.patch.object(self.editor, "health_header_text", return_value="clean status"),
            mock.patch.object(self.editor, "render_report", return_value="clean report"),
        ):
            player._apply_episode_health_scan(scan)

        player.health_warning_label.setText.assert_called_once_with("clean status")
        self.assertEqual(player._health_report_text, "clean report")
        player.health_warning_label.setStyleSheet.assert_called_once_with(
            self.editor.EPISODE_HEALTH_CLEAN_STYLE
        )
        player.health_warning_label.show.assert_called_once_with()
        player.health_details_btn.show.assert_called_once_with()

    def test_stale_async_health_result_is_discarded(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.root_dir = "/new/root"
        player._health_scan_closed = False
        player._health_scan_generation = 2
        player._apply_episode_health_scan = mock.Mock()
        player._schedule_episode_health_poll = mock.Mock()
        future = Future()
        future.set_result(object())

        player._poll_episode_health_scan(future, 1, "/old/root")

        player._apply_episode_health_scan.assert_not_called()
        player._schedule_episode_health_poll.assert_not_called()

    def test_marking_health_stale_cancels_prior_scan_and_stays_yellow(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player._health_scan_generation = 4
        player._health_scan_future = mock.Mock()
        player._show_episode_health_banner = mock.Mock()

        player.mark_episode_health_stale("disk change started")

        self.assertEqual(player._health_scan_generation, 5)
        player._show_episode_health_banner.assert_called_once()
        text, style, details = player._show_episode_health_banner.call_args.args
        self.assertIn("stale", text)
        self.assertEqual(style, self.editor.EPISODE_HEALTH_WARNING_STYLE)
        self.assertIn("previous health result no longer applies", details)

    def test_failed_trim_marks_health_stale_before_and_after_mutation(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.frame_keys = [0, 1]
        player.range_slider = mock.Mock()
        player.range_slider.get_selected_range.return_value = (0, 0)
        player.current_episode_name = "episode_0001"
        player.is_playing = True
        player.update_play_button_text = mock.Mock()
        calls = mock.Mock()
        player.mark_episode_health_stale = calls.mark_stale
        player.delete_and_renumber_frames = calls.trim
        calls.trim.side_effect = RuntimeError("partial failure")

        with (
            mock.patch.object(self.editor.QMessageBox, "Yes", 1, create=True),
            mock.patch.object(self.editor.QMessageBox, "No", 0, create=True),
            mock.patch.object(
                self.editor.QMessageBox,
                "question",
                return_value=1,
                create=True,
            ),
            mock.patch.object(self.editor.QMessageBox, "critical", create=True),
        ):
            player.trim_selected_frames()

        self.assertEqual(calls.mock_calls[0][0], "mark_stale")
        self.assertEqual(calls.mock_calls[1], mock.call.trim([0]))
        self.assertEqual(calls.mock_calls[2][0], "mark_stale")

    def test_successful_trim_stays_paused_without_health_rescan(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.frame_keys = [0, 1, 2]
        player.range_slider = mock.Mock()
        player.range_slider.get_selected_range.return_value = (0, 0)
        player.current_episode_index = 0
        player.current_episode_name = "episode_0001"
        player.is_playing = True
        player.update_play_button_text = mock.Mock()
        player.mark_episode_health_stale = mock.Mock()
        player.delete_and_renumber_frames = mock.Mock()
        player.load_episode = mock.Mock()
        player.refresh_episode_health = mock.Mock()

        with (
            mock.patch.object(self.editor.QMessageBox, "Yes", 1, create=True),
            mock.patch.object(self.editor.QMessageBox, "No", 0, create=True),
            mock.patch.object(
                self.editor.QMessageBox,
                "question",
                return_value=1,
                create=True,
            ),
            mock.patch.object(self.editor.QMessageBox, "information", create=True),
        ):
            player.trim_selected_frames()

        player.delete_and_renumber_frames.assert_called_once_with([0])
        player.load_episode.assert_called_once_with(0)
        player.refresh_episode_health.assert_not_called()
        player.mark_episode_health_stale.assert_called_once()
        self.assertFalse(player.is_playing)
        player.update_play_button_text.assert_called_once_with()

    def test_failed_episode_delete_marks_health_stale_before_and_after_mutation(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.episodes = ["episode_0001"]
        player.current_episode_name = "episode_0001"
        player.current_episode_index = 0
        player.root_dir = "/selected/task"
        player.is_playing = True
        player.update_play_button_text = mock.Mock()
        calls = mock.Mock()
        player.mark_episode_health_stale = calls.mark_stale

        def fail_delete(path):
            calls.delete(path)
            raise OSError("partial failure")

        with (
            mock.patch.object(self.editor.QMessageBox, "Yes", 1, create=True),
            mock.patch.object(self.editor.QMessageBox, "Cancel", 0, create=True),
            mock.patch.object(
                self.editor.QMessageBox,
                "question",
                return_value=1,
                create=True,
            ),
            mock.patch.object(self.editor.QMessageBox, "critical", create=True),
            mock.patch.object(self.editor.os.path, "isdir", return_value=True),
            mock.patch.object(self.editor.shutil, "rmtree", side_effect=fail_delete),
        ):
            player.delete_current_episode()

        self.assertEqual(calls.mock_calls[0][0], "mark_stale")
        self.assertEqual(
            calls.mock_calls[1],
            mock.call.delete("/selected/task/episode_0001"),
        )
        self.assertEqual(calls.mock_calls[2][0], "mark_stale")

    def test_successful_episode_delete_stays_paused_without_health_rescan(self):
        player = object.__new__(self.editor.DatasetPlayer)
        player.episodes = ["episode_0001", "episode_0002"]
        player.current_episode_name = "episode_0001"
        player.current_episode_index = 0
        player.root_dir = "/selected/task"
        player.is_playing = True
        player.mark_episode_health_stale = mock.Mock()
        player.update_play_button_text = mock.Mock()
        player.update_range_info = mock.Mock()
        player.find_episodes = mock.Mock(return_value=["episode_0002"])
        player.load_episode = mock.Mock()
        player.refresh_episode_health = mock.Mock()

        with (
            mock.patch.object(self.editor.QMessageBox, "Yes", 1, create=True),
            mock.patch.object(self.editor.QMessageBox, "Cancel", 0, create=True),
            mock.patch.object(
                self.editor.QMessageBox,
                "question",
                return_value=1,
                create=True,
            ),
            mock.patch.object(self.editor.QMessageBox, "information", create=True),
            mock.patch.object(self.editor.os.path, "isdir", return_value=True),
            mock.patch.object(self.editor.shutil, "rmtree") as rmtree,
        ):
            player.delete_current_episode()

        rmtree.assert_called_once_with("/selected/task/episode_0001")
        player.load_episode.assert_called_once_with(0)
        player.refresh_episode_health.assert_not_called()
        self.assertFalse(player.is_playing)
        player.update_play_button_text.assert_called()
        player.update_range_info.assert_called()

    def test_trim_color_only_episode_without_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = Path(temp_dir) / "episode_0001"
            colors = episode / "colors"
            colors.mkdir(parents=True)
            frames = []
            for index, contents in enumerate((b"zero", b"one", b"two")):
                filename = f"{index:06d}_color_0.jpg"
                (colors / filename).write_bytes(contents)
                frames.append(
                    {
                        "idx": index,
                        "timestamp_s": index * 0.0165,
                        "colors": {"color_0": f"colors/{filename}"},
                        "depths": {},
                    }
                )
            (episode / "data.json").write_text(
                json.dumps(
                    {
                        "info": {},
                        "timing": {
                            "frame_count": 3,
                            "sample_duration_s": 0.033,
                            "measured_fps": 2 / 0.033,
                            "max_frame_gap_s": 0.0165,
                            "recording_duration_s": 1.25,
                            "capture_start_utc": "2026-09-15T00:00:00+00:00",
                            "capture_stop_utc": "2026-09-15T00:00:01.250000+00:00",
                        },
                        "data": frames,
                    }
                ),
                encoding="utf-8",
            )

            player = object.__new__(self.editor.DatasetPlayer)
            player.root_dir = temp_dir
            player.current_episode_name = "episode_0001"
            player.delete_and_renumber_frames([1])

            payload = json.loads((episode / "data.json").read_text(encoding="utf-8"))
            self.assertEqual([frame["idx"] for frame in payload["data"]], [0, 1])
            self.assertEqual(
                [frame["colors"]["color_0"] for frame in payload["data"]],
                ["colors/000000_color_0.jpg", "colors/000001_color_0.jpg"],
            )
            self.assertEqual((colors / "000000_color_0.jpg").read_bytes(), b"zero")
            self.assertEqual((colors / "000001_color_0.jpg").read_bytes(), b"two")
            self.assertFalse((colors / "000002_color_0.jpg").exists())
            self.assertEqual(payload["timing"]["frame_count"], 2)
            self.assertAlmostEqual(payload["timing"]["sample_duration_s"], 0.033)
            self.assertAlmostEqual(payload["timing"]["measured_fps"], 1 / 0.033)
            self.assertAlmostEqual(payload["timing"]["max_frame_gap_s"], 0.033)
            self.assertEqual(payload["timing"]["recording_duration_s"], 1.25)
            self.assertEqual(payload["timing"]["capture_start_utc"], "2026-09-15T00:00:00+00:00")
            self.assertEqual(
                payload["timing"]["capture_stop_utc"],
                "2026-09-15T00:00:01.250000+00:00",
            )


if __name__ == "__main__":
    unittest.main()
