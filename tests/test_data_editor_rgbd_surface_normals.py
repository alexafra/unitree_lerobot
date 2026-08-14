from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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
        player.depth_scale_m_per_unit = 0.001
        player.info_label = mock.Mock()
        player.image_labels = {key: _Label() for key in player.DISPLAY_STREAMS}
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
        )
        self.assertEqual(
            load_depth.call_args_list,
            [
                mock.call("/episode/depths/000007_depth_0.png", 0.001),
                mock.call("/episode/raw_depths/000007_raw_depth_0.png", 0.001),
            ],
        )


if __name__ == "__main__":
    unittest.main()
