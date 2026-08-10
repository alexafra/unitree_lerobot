import os
import re
import sys
import json
import shutil
import tempfile
from collections import defaultdict

import numpy as np


from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QFrame,
    QFileDialog,
    QComboBox,
    QLineEdit
)


class ImageLabel(QLabel):
    """QLabel with automatic image scaling support."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: black;
                color: white;
                border: 1px solid #444;
            }
        """)
        self.setMinimumSize(320, 240)
        self._original_pixmap = None
        self._title = title
        self.set_placeholder(f"{title}\nWaiting for playback")

    def set_placeholder(self, text):
        self._original_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        elif self.text() == "":
            self.setText(f"{self._title}\nNo image")


DEFAULT_DEPTH_SCALE_M_PER_UNIT = 0.001
DEPTH_NEAR_M = 0.25
DEPTH_FAR_M = 1.0


def calculate_measured_fps(data_items):
    """Calculate the average recorded framerate from frame timestamps."""

    timestamps = [
        item.get("timestamp_s")
        for item in data_items
        if isinstance(item, dict)
        and isinstance(item.get("timestamp_s"), (int, float))
        and not isinstance(item.get("timestamp_s"), bool)
        and np.isfinite(item.get("timestamp_s"))
    ]

    if len(timestamps) < 2:
        return None

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return None

    return (len(timestamps) - 1) / elapsed


def resolve_depth_scale_m_per_unit(json_obj):
    """Read the episode depth scale, falling back only when it is absent."""

    stored_scale = (
        json_obj.get("info", {})
        .get("depth", {})
        .get("scale_m_per_unit")
    )
    if stored_scale is None:
        return DEFAULT_DEPTH_SCALE_M_PER_UNIT

    try:
        depth_scale_m_per_unit = float(stored_scale)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"depth scale must be numeric, got {stored_scale!r}"
        ) from error

    if not np.isfinite(depth_scale_m_per_unit) or depth_scale_m_per_unit <= 0:
        raise ValueError(
            f"depth scale must be positive and finite, got {stored_scale!r}"
        )
    return depth_scale_m_per_unit


def depth_to_gray_rgb(
    depth,
    depth_scale_m_per_unit=DEFAULT_DEPTH_SCALE_M_PER_UNIT,
):
    """Apply the training converter fixed metric-depth grayscale encoding."""

    if depth.ndim == 3:
        import cv2

        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    valid_mask = np.isfinite(depth) & (depth > 0)
    depth_m = depth.astype(np.float32) * depth_scale_m_per_unit
    normalized = np.clip(
        (depth_m - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M),
        0.0,
        1.0,
    )

    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid_mask] = 1 + np.round(254 * normalized[valid_mask]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def load_depth_pixmap(
    image_path,
    depth_scale_m_per_unit=DEFAULT_DEPTH_SCALE_M_PER_UNIT,
):
    """Load depth and display the same fixed-scale grayscale used for training."""

    import cv2

    depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        return QPixmap()

    gray_rgb = depth_to_gray_rgb(depth, depth_scale_m_per_unit)

    height, width, channels = gray_rgb.shape
    bytes_per_line = channels * width

    image = QImage(
        gray_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(image)

class RangeSlider(QFrame):
    """
    Progress bar + range selection bar:
    - Left click: jump to the frame
    - Left drag: preview frames in real time
    - Shift + left drag: select a frame range
    """
    rangeChanged = pyqtSignal(int, int)
    frameChanged = pyqtSignal(int)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setStyleSheet("background-color: #f2f2f2; border: 1px solid #bbb;")

        self.total_frames = 0
        self.current_frame = 0

        self.sel_start = 0
        self.sel_end = 0

        self.drag_mode = None   # None / "scrub" / "select"
        self.drag_anchor = 0

    def set_total_frames(self, total_frames):
        self.total_frames = max(0, total_frames)
        if self.total_frames <= 1:
            self.current_frame = 0
            self.sel_start = 0
            self.sel_end = 0
        else:
            max_index = self.total_frames - 1
            self.current_frame = min(self.current_frame, max_index)
            self.sel_start = min(self.sel_start, max_index)
            self.sel_end = min(self.sel_end, max_index)
        self.update()

    def set_current_frame(self, frame_index):
        self.current_frame = max(0, min(frame_index, max(0, self.total_frames - 1)))
        self.update()

    def reset_selection(self):
        if self.total_frames > 0:
            self.sel_start = 0
            self.sel_end = 0
        else:
            self.sel_start = 0
            self.sel_end = 0
        self.rangeChanged.emit(self.sel_start, self.sel_end)
        self.update()

    def get_selected_range(self):
        return min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)

    def frame_from_x(self, x):
        if self.total_frames <= 1:
            return 0
        left = 15
        right = self.width() - 15
        width = max(1, right - left)
        x = max(left, min(x, right))
        ratio = (x - left) / width
        frame = int(round(ratio * (self.total_frames - 1)))
        return max(0, min(frame, self.total_frames - 1))

    def x_from_frame(self, frame):
        if self.total_frames <= 1:
            return 15
        left = 15
        right = self.width() - 15
        width = max(1, right - left)
        ratio = frame / (self.total_frames - 1)
        return int(left + ratio * width)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.total_frames <= 0:
            return

        frame = self.frame_from_x(event.x())
        self.sliderPressed.emit()

        if event.modifiers() & Qt.ShiftModifier:
            self.drag_mode = "select"
            self.drag_anchor = frame
            self.sel_start = frame
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        else:
            self.drag_mode = "scrub"
            self.current_frame = frame
            self.frameChanged.emit(frame)

        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_mode is None or self.total_frames <= 0:
            return

        frame = self.frame_from_x(event.x())

        if self.drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        elif self.drag_mode == "scrub":
            self.current_frame = frame
            self.frameChanged.emit(frame)

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self.drag_mode is None:
            return

        frame = self.frame_from_x(event.x())

        if self.drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.get_selected_range())
        elif self.drag_mode == "scrub":
            self.current_frame = frame
            self.frameChanged.emit(frame)
            self.sliderReleased.emit(frame)

        self.drag_mode = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        left = 15
        right = self.width() - 15
        bar_y = 36
        bar_h = 12
        bar_w = max(1, right - left)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#cccccc"))
        painter.drawRoundedRect(left, bar_y, bar_w, bar_h, 6, 6)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            x1 = self.x_from_frame(start)
            x2 = self.x_from_frame(end)
            sel_x = min(x1, x2)
            sel_w = max(6, abs(x2 - x1))
            painter.setBrush(QColor("#6fa8dc"))
            painter.drawRoundedRect(sel_x, bar_y, sel_w, bar_h, 6, 6)

        if self.total_frames > 0:
            cur_x = self.x_from_frame(self.current_frame)
            painter.setPen(QPen(QColor("red"), 2))
            painter.drawLine(cur_x, 16, cur_x, 58)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            for frame in [start, end]:
                x = self.x_from_frame(frame)
                painter.setPen(QPen(QColor("#1c4587"), 2))
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(x - 5, bar_y - 4, 10, 20)

        painter.setPen(QColor("#222222"))
        top_text = "Drag directly: seek playback    Shift+drag: select trim/playback range"
        painter.drawText(QRect(10, 6, self.width() - 20, 20), Qt.AlignCenter, top_text)

        if self.total_frames > 0:
            start, end = self.get_selected_range()
            bottom_text = (
                f"Selected range: {start} - {end}    "
                f"Current frame: {self.current_frame} / {self.total_frames - 1}"
            )
        else:
            bottom_text = "No available frames"
        painter.drawText(QRect(10, 62, self.width() - 20, 20), Qt.AlignCenter, bottom_text)


class DatasetPlayer(QWidget):
    """
    Dataset playback interface:
    - Selectable root_dir
    - Left and right arrows to switch episodes
    - 2x2 multi-camera display in the center
    - Bottom progress bar + range selection + range playback + trimming
    - Delete current dataset
    """

    DISPLAY_STREAMS = {
        "color_0": ("RGB Camera 0", False),
        "depth_0": ("Aligned Depth", True),
        "raw_depth_0": ("Raw Depth", True),
    }

    FRAME_FILE_PATTERN = re.compile(
        r"^(\d+)(_.+)$",
        re.IGNORECASE,
    )

    def __init__(self, root_dir, interval_ms=100):
        super().__init__()
        self.root_dir = root_dir
        self.interval_ms = interval_ms
        self.playback_speed = 1
        self.playback_frame_step = 1

        self.episodes = []
        self.current_episode_index = 0
        self.current_episode_name = ""
        self.frame_keys = []
        self.frames_map = defaultdict(dict)
        self.current_frame_index = 0
        self.depth_scale_m_per_unit = DEFAULT_DEPTH_SCALE_M_PER_UNIT

        self.is_playing = True
        self.play_selection_only = False
        self.was_playing_before_drag = False
        self.measured_fps = None
        self.current_json_path = ""
        self.current_goal = ""

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.play_next_frame)
        self.timer.start(self.interval_ms)

        self.reload_dataset_root(self.root_dir, show_message=False)

    def init_ui(self):
        self.setWindowTitle("Unitree Dataset Editor V1.0")
        self.resize(1450, 1030)

        self.select_root_btn = QPushButton("Select Dataset Path")
        self.select_root_btn.clicked.connect(self.select_root_dir)
        self.select_root_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #2e86de;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #1f6fc1;
            }
        """)

        self.root_dir_label = QLabel("")
        self.root_dir_label.setWordWrap(True)
        self.root_dir_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.root_dir_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #222;
                padding: 8px 12px;
                border: 1px solid #cccccc;
                background: #fafafa;
                border-radius: 6px;
            }
        """)

        root_layout = QHBoxLayout()
        root_layout.addWidget(self.select_root_btn)
        root_layout.addWidget(self.root_dir_label, 1)

        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")

        for btn in (self.prev_btn, self.next_btn):
            btn.setFixedWidth(90)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 36px;
                    font-weight: bold;
                    background-color: #2d2d2d;
                    color: white;
                    border-radius: 10px;
                }
                QPushButton:disabled {
                    background-color: #777;
                    color: #bbb;
                }
                QPushButton:hover:!disabled {
                    background-color: #444;
                }
            """)

        self.prev_btn.clicked.connect(self.prev_episode)
        self.next_btn.clicked.connect(self.next_episode)

        self.episode_label = QLabel("Current Episode:")
        self.episode_label.setAlignment(Qt.AlignCenter)
        self.episode_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                padding: 8px;
            }
        """)

        self.fps_label = QLabel("Measured framerate: N/A")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
        """)

        self.goal_label = QLabel("Goal:")
        self.goal_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText("No goal specified")
        self.goal_edit.setMinimumWidth(400)
        self.goal_edit.setEnabled(False)
        self.goal_edit.setToolTip("Press Enter or click away to save the goal")
        self.goal_edit.editingFinished.connect(self.save_goal)
        self.goal_edit.setStyleSheet("""
            QLineEdit {
                font-size: 16px;
                padding: 5px 8px;
                border: 1px solid #aaa;
                border-radius: 4px;
                background: white;
            }
        """)

        episode_details_layout = QHBoxLayout()
        episode_details_layout.addStretch()
        episode_details_layout.addWidget(self.fps_label)
        episode_details_layout.addSpacing(30)
        episode_details_layout.addWidget(self.goal_label)
        episode_details_layout.addWidget(self.goal_edit, 1)
        episode_details_layout.addStretch()

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #333;
                padding-bottom: 8px;
            }
        """)

        self.image_labels = {
            stream_key: ImageLabel(title)
            for stream_key, (title, _) in self.DISPLAY_STREAMS.items()
        }

        grid = QGridLayout()
        grid.setSpacing(8)

        grid.addWidget(
            self.image_labels["color_0"],
            0,
            0,
        )

        grid.addWidget(
            self.image_labels["depth_0"],
            0,
            1,
        )

        grid.addWidget(
            self.image_labels["raw_depth_0"],
            1,
            0,
            1,
            2,
        )

        self.range_slider = RangeSlider()
        self.range_slider.rangeChanged.connect(self.on_range_changed)
        self.range_slider.frameChanged.connect(self.on_slider_frame_changed)
        self.range_slider.sliderPressed.connect(self.on_slider_pressed)
        self.range_slider.sliderReleased.connect(self.on_slider_released)

        self.range_info_label = QLabel("No range selected")
        self.range_info_label.setAlignment(Qt.AlignCenter)
        self.range_info_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #333;
                padding-top: 4px;
                padding-bottom: 4px;
            }
        """)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "4x", "8x", "16x", "32x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setMinimumWidth(75)
        self.speed_combo.currentTextChanged.connect(
            self.set_playback_speed
        )

        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)

        self.play_all_btn = QPushButton("Loop Full Episode")
        self.play_all_btn.clicked.connect(self.set_play_all_mode)

        self.play_selection_btn = QPushButton("Play Selected Range Only")
        self.play_selection_btn.clicked.connect(self.set_play_selection_mode)

        self.reset_range_btn = QPushButton("Reset Range")
        self.reset_range_btn.clicked.connect(self.reset_trim_range)

        self.trim_btn = QPushButton("Trim Selected Range")
        self.trim_btn.clicked.connect(self.trim_selected_frames)

        self.delete_episode_btn = QPushButton("Delete Current Episode")
        self.delete_episode_btn.clicked.connect(self.delete_current_episode)

        for btn in [
            self.play_pause_btn,
            self.play_all_btn,
            self.play_selection_btn,
            self.reset_range_btn,
            self.trim_btn,
            self.delete_episode_btn
        ]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 15px;
                    padding: 10px 18px;
                    border-radius: 8px;
                    background-color: #555555;
                    color: white;
                }
                QPushButton:hover:!disabled {
                    background-color: #444444;
                }
                QPushButton:disabled {
                    background-color: #999999;
                }
            """)

        self.trim_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #c0392b;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #999999;
            }
        """)

        self.delete_episode_btn.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 10px 18px;
                border-radius: 8px;
                background-color: #8e44ad;
                color: white;
            }
            QPushButton:hover:!disabled {
                background-color: #7d3c98;
            }
            QPushButton:disabled {
                background-color: #999999;
            }
        """)

        control_layout = QHBoxLayout()
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Speed:"))
        control_layout.addWidget(self.speed_combo)
        control_layout.addWidget(self.play_pause_btn)
        control_layout.addWidget(self.play_all_btn)
        control_layout.addWidget(self.play_selection_btn)
        control_layout.addWidget(self.reset_range_btn)
        control_layout.addWidget(self.trim_btn)
        control_layout.addWidget(self.delete_episode_btn)
        control_layout.addStretch()

        center_layout = QVBoxLayout()
        center_layout.addLayout(root_layout)
        center_layout.addWidget(self.episode_label)
        center_layout.addLayout(episode_details_layout)
        center_layout.addWidget(self.info_label)
        center_layout.addLayout(grid)
        center_layout.addSpacing(12)
        center_layout.addWidget(self.range_slider)
        center_layout.addWidget(self.range_info_label)
        center_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.prev_btn)
        main_layout.addLayout(center_layout, 1)
        main_layout.addWidget(self.next_btn)

        self.setLayout(main_layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.next_episode()
            event.accept()
        else:
            super().keyPressEvent(event)

    def update_root_dir_label(self):
        self.root_dir_label.setText(f"Current Dataset Path: {self.root_dir}")

    def set_playback_speed(self, speed_text):
        self.playback_speed = int(speed_text.removesuffix("x"))

        timer_speed = min(self.playback_speed, 8)

        self.playback_frame_step = max(1, self.playback_speed // timer_speed)

        playback_interval = max(1, round(self.interval_ms / timer_speed))

        if hasattr(self, "timer"):
            self.timer.setInterval(playback_interval)

        self.update_range_info()

    def update_framerate_label(self):
        if self.measured_fps is None:
            self.fps_label.setText("Measured framerate: N/A")
            self.fps_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #333;
                }
            """)
        else:
            self.fps_label.setText(
                f"Measured framerate: {self.measured_fps:.2f} FPS"
            )
            if self.measured_fps < 27:
                self.fps_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        font-weight: bold;
                        color: #c0392b;
                    }
                """)
            else:
                self.fps_label.setStyleSheet("""
                    QLabel {
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;
                    }
                """)

    def set_goal(self, goal, enabled):
        self.current_goal = goal
        self.goal_edit.setText(goal)
        self.goal_edit.setEnabled(enabled)

    def save_goal(self):
        if not self.current_json_path or not self.goal_edit.isEnabled():
            return

        # Persist the semantic goal, not accidental padding from typing or paste.
        # ``strip`` removes only leading/trailing whitespace; whitespace inside
        # the instruction is preserved.
        new_goal = self.goal_edit.text().strip()
        self.goal_edit.setText(new_goal)
        if new_goal == self.current_goal:
            return

        temporary_path = None

        try:
            with open(
                self.current_json_path,
                "r",
                encoding="utf-8",
            ) as file:
                json_obj = json.load(file)

            text_info = json_obj.get("text")
            if not isinstance(text_info, dict):
                text_info = {}
                json_obj["text"] = text_info

            text_info["goal"] = new_goal

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(self.current_json_path),
                prefix=".data_goal_",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = file.name
                json.dump(
                    json_obj,
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
                file.write("\n")

            os.replace(temporary_path, self.current_json_path)
            self.current_goal = new_goal

        except (OSError, json.JSONDecodeError) as error:
            if temporary_path and os.path.isfile(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
            self.goal_edit.setText(self.current_goal)
            QMessageBox.critical(
                self,
                "Error",
                f"Could not save the goal:\n{error}",
            )

    def clear_player_state(self, message="No available data"):
        self.measured_fps = None
        self.update_framerate_label()
        self.current_json_path = ""
        self.set_goal("", enabled=False)
        self.episodes = []
        self.current_episode_index = 0
        self.current_episode_name = ""
        self.frame_keys = []
        self.frames_map = defaultdict(dict)
        self.current_frame_index = 0
        self.depth_scale_m_per_unit = DEFAULT_DEPTH_SCALE_M_PER_UNIT
        self.play_selection_only = False

        self.episode_label.setText("Current Episode: None")
        self.info_label.setText(message)

        for stream_key, label in self.image_labels.items():
            title = self.DISPLAY_STREAMS[stream_key][0]
            label.set_placeholder(
                f"{title}\nNo image"
            )

        self.range_slider.set_total_frames(0)
        self.range_info_label.setText(message)
        self.trim_btn.setEnabled(False)
        self.delete_episode_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def find_episodes(self):
        episode_dirs = []
        if not os.path.isdir(self.root_dir):
            return episode_dirs

        for name in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, name)
            if os.path.isdir(full_path) and re.match(r"^episode_(\d+)$", name):
                episode_dirs.append(name)

        episode_dirs.sort(key=lambda x: int(re.match(r"^episode_(\d+)$", x).group(1)))
        return episode_dirs

    def reload_dataset_root(self, new_root_dir, show_message=True):
        if not new_root_dir or not os.path.isdir(new_root_dir):
            return

        self.is_playing = False
        self.update_play_button_text()

        self.root_dir = new_root_dir
        self.update_root_dir_label()

        self.episodes = self.find_episodes()

        if not self.episodes:
            self.clear_player_state("No episode_0001 / episode_0002 ... found in the current path")
            if show_message:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"No episode_XXXX dataset directories found in:\n{self.root_dir}"
                )
            return

        self.current_episode_index = 0
        self.load_episode(self.current_episode_index)
        self.is_playing = True
        self.update_play_button_text()
        self.update_range_info()

        if show_message:
            QMessageBox.information(
                self,
                "Done",
                f"Dataset path switched to:\n{self.root_dir}"
            )

    def select_root_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root Directory",
            self.root_dir if os.path.isdir(self.root_dir) else os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if selected_dir:
            self.reload_dataset_root(selected_dir, show_message=False)

    def update_nav_buttons(self):
        if not self.episodes:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        self.prev_btn.setEnabled(self.current_episode_index > 0)
        self.next_btn.setEnabled(self.current_episode_index < len(self.episodes) - 1)

    def update_play_button_text(self):
        self.play_pause_btn.setText("Pause" if self.is_playing else "Start")

    def update_range_info(self):
        if not self.frame_keys:
            self.range_info_label.setText("No available frames")
            return


        start, end = self.range_slider.get_selected_range()
        mode_text = "Play selected range only" if self.play_selection_only else "Loop full episode"
        state_text = "Playing" if self.is_playing else "Paused"
        speed_text = f"{self.playback_speed}x"

        if self.playback_frame_step > 1:
            speed_text += f" (frame step {self.playback_frame_step})"

        self.range_info_label.setText(
            f"Mode: {mode_text}    State: {state_text}    Speed: {speed_text}    "
            f"Selected range: frame {start} ~ frame {end}"
        )

    def on_range_changed(self, start, end):
        self.trim_btn.setEnabled(len(self.frame_keys) > 0 and end >= start)
        self.update_range_info()

    def on_slider_pressed(self):
        self.was_playing_before_drag = self.is_playing
        self.is_playing = False
        self.update_play_button_text()
        self.update_range_info()

    def on_slider_released(self, frame_index):
        self.current_frame_index = frame_index
        if self.was_playing_before_drag:
            self.is_playing = True
        self.update_play_button_text()
        self.update_range_info()

    def on_slider_frame_changed(self, frame_index):
        if not self.frame_keys:
            return
        self.current_frame_index = max(0, min(frame_index, len(self.frame_keys) - 1))
        self.show_frame(self.current_frame_index)

    def toggle_play_pause(self):
        if not self.frame_keys:
            return
        self.is_playing = not self.is_playing
        self.update_play_button_text()
        self.update_range_info()

    def set_play_all_mode(self):
        self.play_selection_only = False
        self.update_range_info()

    def set_play_selection_mode(self):
        if not self.frame_keys:
            return

        start, end = self.range_slider.get_selected_range()
        if start == end:
            QMessageBox.information(
                self,
                "Info",
                "The selected range currently contains only 1 frame. It can still be played. To select a longer range, hold Shift and drag on the progress bar."
            )

        self.play_selection_only = True
        self.current_frame_index = start
        self.show_frame(self.current_frame_index)
        self.update_range_info()

    def reset_trim_range(self):
        self.range_slider.reset_selection()
        self.update_range_info()

    def load_episode(self, episode_index):
        self.measured_fps = None
        self.update_framerate_label()
        self.current_json_path = ""
        self.set_goal("", enabled=False)
        if not self.episodes:
            self.clear_player_state("No available datasets")
            return

        self.current_episode_index = episode_index
        self.current_episode_name = self.episodes[episode_index]

        episode_dir = os.path.join(
            self.root_dir,
            self.current_episode_name,
        )

        json_path = os.path.join(
            episode_dir,
            "data.json",
        )

        if not os.path.isfile(json_path):
            self.frames_map = defaultdict(dict)
            self.frame_keys = []
            self.current_frame_index = 0

            self.episode_label.setText(
                f"Current Episode: {self.current_episode_name}"
            )

            self.info_label.setText(
                f"File not found: {json_path}"
            )

            for stream_key, label in self.image_labels.items():
                title = self.DISPLAY_STREAMS[stream_key][0]
                label.set_placeholder(
                    f"{title}\ndata.json not found"
                )

            self.range_slider.set_total_frames(0)
            self.range_info_label.setText(
                "data.json not found"
            )

            self.trim_btn.setEnabled(False)
            self.delete_episode_btn.setEnabled(True)
            self.update_nav_buttons()
            return

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as file:
                json_obj = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            self.clear_player_state(
                f"Could not read data.json: {error}"
            )
            return

        try:
            self.depth_scale_m_per_unit = resolve_depth_scale_m_per_unit(json_obj)
        except ValueError as error:
            self.clear_player_state(
                f"Invalid depth scale in {json_path}: {error}"
            )
            return

        data_items = json_obj.get("data")

        if not isinstance(data_items, list):
            self.clear_player_state(
                "Invalid data.json: missing data array"
            )
            return

        self.measured_fps = calculate_measured_fps(data_items)
        self.update_framerate_label()

        text_info = json_obj.get("text")
        goal = text_info.get("goal", "") if isinstance(text_info, dict) else ""
        if not isinstance(goal, str):
            goal = str(goal)
        self.current_json_path = json_path
        self.set_goal(goal, enabled=True)

        frames_map = defaultdict(dict)

        for item in data_items:
            if not isinstance(item, dict):
                continue

            frame_id = item.get("idx")

            if not isinstance(frame_id, int):
                continue

            colors = item.get("colors") or {}
            depths = item.get("depths") or {}

            relative_paths = {
                "color_0": colors.get("color_0"),
                "depth_0": depths.get("depth_0"),
                "raw_depth_0": depths.get("raw_depth_0"),
            }

            for stream_key, relative_path in relative_paths.items():
                if not isinstance(relative_path, str):
                    continue

                if not relative_path:
                    continue

                frames_map[frame_id][stream_key] = os.path.join(
                    episode_dir,
                    relative_path,
                )

            # Ensure the frame remains on the timeline even when
            # one or more images are missing.
            frames_map[frame_id]

        self.frames_map = frames_map
        self.frame_keys = sorted(frames_map.keys())
        self.current_frame_index = 0
        self.play_selection_only = False

        self.episode_label.setText(
            f"Current Episode: {self.current_episode_name}"
        )

        if not self.frame_keys:
            self.info_label.setText(
                "No playable frames found in data.json"
            )

            for stream_key, label in self.image_labels.items():
                title = self.DISPLAY_STREAMS[stream_key][0]
                label.set_placeholder(
                    f"{title}\nNo image"
                )

            self.range_slider.set_total_frames(0)
            self.range_info_label.setText(
                "No available frames"
            )
            self.trim_btn.setEnabled(False)

        else:
            self.range_slider.set_total_frames(
                len(self.frame_keys)
            )

            self.range_slider.reset_selection()
            self.show_frame(self.current_frame_index)
            self.trim_btn.setEnabled(True)

        self.delete_episode_btn.setEnabled(True)
        self.update_nav_buttons()
        self.update_play_button_text()
        self.update_range_info()


    def show_frame(self, frame_index):
        if not self.frame_keys:
            return

        frame_index = max(
            0,
            min(
                frame_index,
                len(self.frame_keys) - 1,
            ),
        )

        frame_id = self.frame_keys[frame_index]
        frame_file_map = self.frames_map[frame_id]

        mode_text = (
            "Selected range loop"
            if self.play_selection_only
            else "Full episode loop"
        )

        state_text = (
            "Playing"
            if self.is_playing
            else "Paused"
        )

        self.info_label.setText(
            f"Dataset: {self.current_episode_name}    "
            f"Current frame: "
            f"{frame_index}/{len(self.frame_keys) - 1}    "
            f"Frame ID: {frame_id:06d}    "
            f"Depth scale: {self.depth_scale_m_per_unit:g} m/unit    "
            f"Mode: {mode_text}    "
            f"State: {state_text}"
        )

        for stream_key, stream_info in self.DISPLAY_STREAMS.items():
            title, is_depth = stream_info

            label = self.image_labels[stream_key]
            image_path = frame_file_map.get(stream_key)

            if not image_path:
                label.set_placeholder(
                    f"{title}\nMissing frame"
                )
                continue

            if not os.path.isfile(image_path):
                label.set_placeholder(
                    f"{title}\nMissing frame"
                )
                continue

            if is_depth:
                pixmap = load_depth_pixmap(
                    image_path,
                    self.depth_scale_m_per_unit,
                )
            else:
                pixmap = QPixmap(image_path)

            if pixmap.isNull():
                label.set_placeholder(
                    f"{title}\nFailed to read image"
                )
            else:
                label.set_pixmap(pixmap)

        self.range_slider.set_current_frame(frame_index)

    def play_next_frame(self):
        if not self.frame_keys or not self.is_playing:
            return

        if self.play_selection_only:
            start, end = self.range_slider.get_selected_range()
            start = max(0, min(start, len(self.frame_keys) - 1))
            end = max(0, min(end, len(self.frame_keys) - 1))

            if self.current_frame_index < start or self.current_frame_index > end:
                self.current_frame_index = start

            self.show_frame(self.current_frame_index)

            selection_length = end - start + 1

            self.current_frame_index = start + (self.current_frame_index - start 
                + self.playback_frame_step
            ) % selection_length

        else:
            self.show_frame(self.current_frame_index)
            self.current_frame_index = (
                self.current_frame_index
                + self.playback_frame_step
            ) % len(self.frame_keys)


    def prev_episode(self):
        if self.current_episode_index > 0:
            self.load_episode(self.current_episode_index - 1)

    def next_episode(self):
        if self.current_episode_index < len(self.episodes) - 1:
            self.load_episode(self.current_episode_index + 1)

    def trim_selected_frames(self):
        if not self.frame_keys:
            QMessageBox.warning(self, "Warning", "There are no frames that can be trimmed in the current episode.")
            return

        start_idx, end_idx = self.range_slider.get_selected_range()

        if start_idx < 0 or end_idx >= len(self.frame_keys):
            QMessageBox.warning(self, "Warning", "Invalid selected range.")
            return

        delete_frame_ids = self.frame_keys[start_idx:end_idx + 1]
        delete_count = len(delete_frame_ids)
        remain_count = len(self.frame_keys) - delete_count

        if remain_count <= 0:
            QMessageBox.warning(
                self,
                "Warning",
                "You cannot trim all frames from the current episode. At least 1 frame must remain."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Trim",
            (
                f"Current dataset: {self.current_episode_name}\n"
                f"Frames to delete: {start_idx} ~ {end_idx} (total {delete_count} frames)\n"
                f"{remain_count} frames will remain after deletion and be renumbered.\n\n"
                f"This operation will directly modify files on disk. Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.is_playing = False
        self.update_play_button_text()

        try:
            self.delete_and_renumber_frames(delete_frame_ids)
            self.load_episode(self.current_episode_index)
            QMessageBox.information(
                self,
                "Done",
                f"Trim completed.\n{delete_count} frames were deleted and the remaining frames were renumbered."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Trim failed:\n{str(e)}")

    def delete_current_episode(self):
        if not self.episodes or not self.current_episode_name:
            QMessageBox.warning(self, "Warning", "There is no dataset to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            "This operation will delete the current dataset!",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )

        if reply != QMessageBox.Yes:
            return

        self.is_playing = False
        self.update_play_button_text()

        episode_path = os.path.join(self.root_dir, self.current_episode_name)
        current_index = self.current_episode_index

        try:
            if not os.path.isdir(episode_path):
                raise RuntimeError(f"Dataset directory does not exist: {episode_path}")

            shutil.rmtree(episode_path)

            self.episodes = self.find_episodes()

            if not self.episodes:
                self.clear_player_state("No datasets remain in the current path")
                QMessageBox.information(self, "Done", "The current dataset has been deleted.")
                return

            next_index = min(current_index, len(self.episodes) - 1)
            self.load_episode(next_index)
            self.is_playing = True
            self.update_play_button_text()
            self.update_range_info()

            QMessageBox.information(self, "Done", "The current dataset has been deleted.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete the current dataset:\n{str(e)}")

    def delete_and_renumber_frames(
        self,
        delete_frame_ids,
    ):
        episode_dir = os.path.join(
            self.root_dir,
            self.current_episode_name,
        )

        json_path = os.path.join(
            episode_dir,
            "data.json",
        )

        if not os.path.isfile(json_path):
            raise RuntimeError(
                f"data.json does not exist: {json_path}"
            )

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:
            json_obj = json.load(file)

        old_data_list = json_obj.get("data")

        if not isinstance(old_data_list, list):
            raise RuntimeError(
                "Invalid data.json format: missing data array."
            )

        delete_frame_set = set(delete_frame_ids)

        valid_items = [
            item
            for item in old_data_list
            if (
                isinstance(item, dict)
                and isinstance(item.get("idx"), int)
            )
        ]

        valid_items.sort(
            key=lambda item: item["idx"]
        )

        remaining_items = [
            item
            for item in valid_items
            if item["idx"] not in delete_frame_set
        ]

        if not remaining_items:
            raise RuntimeError(
                "No frame data remains after trimming."
            )

        new_id_map = {
            item["idx"]: new_idx
            for new_idx, item in enumerate(remaining_items)
        }

        media_directories = [
            "colors",
            "depths",
            "raw_depths",
        ]

        for directory_name in media_directories:
            directory_path = os.path.join(
                episode_dir,
                directory_name,
            )

            if not os.path.isdir(directory_path):
                continue

            file_records = []

            for filename in os.listdir(directory_path):
                match = self.FRAME_FILE_PATTERN.match(
                    filename
                )

                if not match:
                    continue

                old_frame_id = int(match.group(1))

                old_path = os.path.join(
                    directory_path,
                    filename,
                )

                file_records.append(
                    (
                        old_frame_id,
                        filename,
                        old_path,
                    )
                )

            # Delete every selected RGB/depth/raw-depth file.
            for old_frame_id, filename, old_path in file_records:
                if old_frame_id not in delete_frame_set:
                    continue

                if os.path.isfile(old_path):
                    os.remove(old_path)

            # First rename retained files to temporary names.
            # This prevents collisions such as frame 2 becoming frame 1
            # while the old frame 1 file still exists.
            temporary_records = []
            temp_counter = 0

            for old_frame_id, filename, old_path in file_records:
                if old_frame_id not in new_id_map:
                    continue

                if not os.path.isfile(old_path):
                    continue

                filename_match = self.FRAME_FILE_PATTERN.match(
                    filename
                )

                if filename_match is None:
                    continue

                filename_suffix = filename_match.group(2)

                temporary_name = (
                    f"__trim_tmp__"
                    f"{temp_counter:08d}_"
                    f"{filename}"
                )

                temporary_path = os.path.join(
                    directory_path,
                    temporary_name,
                )

                os.rename(
                    old_path,
                    temporary_path,
                )

                temporary_records.append(
                    (
                        old_frame_id,
                        filename_suffix,
                        temporary_path,
                    )
                )

                temp_counter += 1

            # Rename temporary files to their final sequential IDs.
            for (
                old_frame_id,
                filename_suffix,
                temporary_path,
            ) in temporary_records:
                new_frame_id = new_id_map[old_frame_id]

                new_filename = (
                    f"{new_frame_id:06d}"
                    f"{filename_suffix}"
                )

                new_path = os.path.join(
                    directory_path,
                    new_filename,
                )

                os.rename(
                    temporary_path,
                    new_path,
                )

        new_data_list = []

        for item in remaining_items:
            old_idx = item["idx"]
            new_idx = new_id_map[old_idx]

            new_item = json.loads(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
            )

            new_item["idx"] = new_idx

            if isinstance(
                new_item.get("colors"),
                dict,
            ):
                new_item["colors"] = (
                    self.renumber_media_paths(
                        new_item["colors"],
                        new_idx,
                    )
                )

            if isinstance(
                new_item.get("depths"),
                dict,
            ):
                # Handles both:
                #
                # depth_0:
                # depths/XXXXXX_depth_0.png
                #
                # raw_depth_0:
                # raw_depths/XXXXXX_raw_depth_0.png
                new_item["depths"] = (
                    self.renumber_media_paths(
                        new_item["depths"],
                        new_idx,
                    )
                )

            if isinstance(
                new_item.get("audios"),
                dict,
            ):
                new_audios = {}

                for (
                    audio_key,
                    relative_path,
                ) in new_item["audios"].items():
                    if not isinstance(relative_path, str):
                        new_audios[audio_key] = relative_path
                        continue

                    path = os.path.normpath(
                        relative_path
                    )

                    directory_name = os.path.dirname(path)
                    extension = os.path.splitext(path)[1]

                    filename = (
                        f"audio_"
                        f"{new_idx:06d}_"
                        f"{audio_key}"
                        f"{extension}"
                    )

                    new_audios[audio_key] = os.path.join(
                        directory_name,
                        filename,
                    ).replace(
                        os.sep,
                        "/",
                    )

                new_item["audios"] = new_audios

            new_data_list.append(new_item)

        json_obj["data"] = new_data_list

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                json_obj,
                file,
                ensure_ascii=False,
                indent=4,
            )


    @staticmethod
    def renumber_media_paths(
        media_paths,
        new_idx,
    ):
        """
        Replace the leading frame number while preserving
        the directory, stream name and extension.
        """
        renumbered = {}

        for media_key, relative_path in media_paths.items():
            if not isinstance(relative_path, str):
                renumbered[media_key] = relative_path
                continue

            normalized_path = os.path.normpath(
                relative_path
            )

            directory_name = os.path.dirname(
                normalized_path
            )

            old_filename = os.path.basename(
                normalized_path
            )

            match = re.match(
                r"^\d+(_.+)$",
                old_filename,
            )

            if match is None:
                renumbered[media_key] = relative_path
                continue

            new_filename = (
                f"{new_idx:06d}"
                f"{match.group(1)}"
            )

            renumbered[media_key] = os.path.join(
                directory_name,
                new_filename,
            ).replace(
                os.sep,
                "/",
            )

        return renumbered

def main():
    app = QApplication(sys.argv)

    root_dir = ""

    player = DatasetPlayer(root_dir=root_dir, interval_ms=100)
    player.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
