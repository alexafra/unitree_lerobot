"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot-type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --push-to-hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --include-depth Include aligned depth_0 as a normalized three-channel visual feature
# --depth-near-m Fixed near bound used for depth normalization
# --depth-far-m  Fixed far bound used for depth normalization
# --include-surface-normals Include camera-frame normals derived from aligned depth_0

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot-type Unitree_G1_Dex3_HeadOnly \
    --include-depth \
    --depth-near-m 0.25 \
    --depth-far-m 1.0 \
    --push-to-hub
"""

import os
import av
import cv2
import tqdm
import tyro
import json
import dataclasses
import shutil
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info

from unitree_lerobot.utils.constants import ROBOT_CONFIGS
from unitree_lerobot.utils.depth_encoding import (
    DEFAULT_DEPTH_FAR_M,
    DEFAULT_DEPTH_NEAR_M,
    DEFAULT_DEPTH_SCALE_M_PER_UNIT,
    DEPTH_COLOR_SOURCE_KEY,
    DEPTH_ENCODING,
    DEPTH_OUTPUT_KEY,
    DEPTH_SOURCE_KEY,
    encode_depth_gray_rgb,
)
from unitree_lerobot.utils.surface_normal_encoding import (
    DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
    SURFACE_NORMAL_OUTPUT_KEY,
    PinholeIntrinsics,
    encode_surface_normals_rgb,
    surface_normals_encoding_metadata,
)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def encode_lossless_geometry_video(
    image_dir: Path,
    video_path: Path,
    fps: int,
) -> None:
    """Encode geometry bytes without a lossy RGB-to-YUV round trip."""

    input_paths = sorted(image_dir.glob("frame-[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not input_paths:
        raise FileNotFoundError(f"No geometry frames found in {image_dir}")

    with Image.open(input_paths[0]) as first_image:
        width, height = first_image.size

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(
            "libx264rgb",
            fps,
            options={"g": "2", "crf": "0"},
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgb24"

        for input_path in input_paths:
            with Image.open(input_path) as input_image:
                frame = av.VideoFrame.from_image(input_image.convert("RGB"))
            for packet in stream.encode(frame):
                output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)

    if not video_path.is_file():
        raise OSError(f"Geometry video encoding did not create {video_path}")


class GeometryVideoLeRobotDataset(LeRobotDataset):
    """Use byte-exact RGB H.264 for model geometry views only."""

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        geometry_video_keys = {
            f"observation.images.{DEPTH_OUTPUT_KEY}",
            f"observation.images.{SURFACE_NORMAL_OUTPUT_KEY}",
        }
        if video_key not in geometry_video_keys:
            return super()._encode_temporary_episode_video(video_key, episode_index)

        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_lossless_geometry_video(
            img_dir,
            temp_path,
            self.fps,
        )
        shutil.rmtree(img_dir)
        return temp_path


class JsonDataset:
    def __init__(
        self,
        data_dirs: Path,
        robot_type: str,
        *,
        include_depth: bool = False,
        depth_near_m: float = DEFAULT_DEPTH_NEAR_M,
        depth_far_m: float = DEFAULT_DEPTH_FAR_M,
        include_surface_normals: bool = False,
        surface_normal_intrinsics: PinholeIntrinsics = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        surface_normal_max_neighbor_depth_delta_m: float = (DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M),
    ) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = Path(data_dirs)
        self.json_file = "data.json"

        if depth_far_m <= depth_near_m:
            raise ValueError(f"depth_far_m ({depth_far_m}) must be greater than depth_near_m ({depth_near_m})")

        self.include_depth = include_depth
        self.depth_near_m = depth_near_m
        self.depth_far_m = depth_far_m
        self.include_surface_normals = include_surface_normals
        self.surface_normal_intrinsics = surface_normal_intrinsics
        self.surface_normal_max_neighbor_depth_delta_m = surface_normal_max_neighbor_depth_delta_m
        # Validate the complete geometry contract before reading any frames.
        surface_normals_encoding_metadata(
            intrinsics=surface_normal_intrinsics,
            max_neighbor_depth_delta_m=surface_normal_max_neighbor_depth_delta_m,
        )

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        if not self.data_dirs.is_dir():
            raise NotADirectoryError(f"Raw dataset directory does not exist: {self.data_dirs}")

        direct_episode_paths = sorted(
            path for path in self.data_dirs.iterdir() if path.is_dir() and (path / self.json_file).is_file()
        )

        if direct_episode_paths:
            self.task_paths = [self.data_dirs]
            self.episode_paths = direct_episode_paths
        else:
            self.task_paths = []
            self.episode_paths = []
            for task_path in sorted(path for path in self.data_dirs.iterdir() if path.is_dir()):
                episode_paths = sorted(
                    path for path in task_path.iterdir() if path.is_dir() and (path / self.json_file).is_file()
                )
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)

        if not self.episode_paths:
            raise FileNotFoundError(
                f"No episode directories containing {self.json_file} were found under {self.data_dirs}"
            )

        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> list:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def _parse_depth_derived_images(
        self,
        episode_path: str,
        episode_data: dict,
        *,
        include_depth: bool,
        include_surface_normals: bool,
    ) -> dict[str, list[np.ndarray]]:
        """Load each aligned PNG16 once and generate requested model views."""

        images = defaultdict(list)

        depth_info = episode_data.get("info", {}).get("depth", {})

        stored_scale = depth_info.get("scale_m_per_unit")

        if stored_scale is None:
            depth_scale = DEFAULT_DEPTH_SCALE_M_PER_UNIT
            print(f"Warning: depth scale missing for {episode_path}; assuming {depth_scale} m/unit")
        else:
            depth_scale = float(stored_scale)

        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError(f"Invalid depth scale in {episode_path}: {depth_scale}")

        # depth_0 is aligned with color_0.
        rgb_camera_key = self.camera_to_image_key.get(DEPTH_COLOR_SOURCE_KEY)

        if rgb_camera_key is None:
            raise ValueError(f"No image mapping exists for {DEPTH_COLOR_SOURCE_KEY}")

        for sample_data in episode_data["data"]:
            relative_path = sample_data.get("depths", {}).get(DEPTH_SOURCE_KEY)

            if not relative_path:
                raise ValueError(f"Missing {DEPTH_SOURCE_KEY} in frame {sample_data.get('idx')}")

            depth_path = os.path.join(episode_path, relative_path)

            depth_u16 = cv2.imread(
                depth_path,
                cv2.IMREAD_UNCHANGED,
            )

            if depth_u16 is None:
                raise RuntimeError(f"Failed to read depth image: {depth_path}")

            if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
                raise ValueError(
                    f"Expected HxW uint16 depth at {depth_path}; got shape={depth_u16.shape}, dtype={depth_u16.dtype}"
                )

            if include_depth:
                depth_rgb = encode_depth_gray_rgb(
                    depth_u16,
                    scale_m_per_unit=depth_scale,
                    near_m=self.depth_near_m,
                    far_m=self.depth_far_m,
                )
                images[DEPTH_OUTPUT_KEY].append(depth_rgb)

            if include_surface_normals:
                surface_normals_rgb = encode_surface_normals_rgb(
                    depth_u16,
                    scale_m_per_unit=depth_scale,
                    intrinsics=self.surface_normal_intrinsics,
                    max_neighbor_depth_delta_m=self.surface_normal_max_neighbor_depth_delta_m,
                )
                images[SURFACE_NORMAL_OUTPUT_KEY].append(surface_normals_rgb)

        return images

    def _parse_depth_images(
        self,
        episode_path: str,
        episode_data: dict,
    ) -> dict[str, list[np.ndarray]]:
        """Preserve the existing depth-only helper for downstream callers."""

        return self._parse_depth_derived_images(
            episode_path,
            episode_data,
            include_depth=True,
            include_surface_normals=False,
        )

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        if index is None:
            index = int(np.random.randint(len(self.episode_paths)))

        file_path = self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")
        if not isinstance(task, str):
            raise TypeError(f"Episode {file_path} text.goal must be a string")
        task = task.strip()
        if not task:
            raise ValueError(f"Episode {file_path} has an empty text.goal after trimming whitespace")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)
        if self.include_depth or self.include_surface_normals:
            cameras.update(
                self._parse_depth_derived_images(
                    file_path,
                    episode_data,
                    include_depth=self.include_depth,
                    include_surface_normals=self.include_surface_normals,
                )
            )

        if not cameras:
            raise ValueError(f"No camera frames found for episode {file_path}")

        reference_shape = next(iter(cameras.values()))[0].shape
        for camera_key, frames in cameras.items():
            if len(frames) != episode_length:
                raise ValueError(
                    f"Frame count mismatch for {camera_key} in {file_path}: "
                    f"expected {episode_length}, got {len(frames)}"
                )
            for frame_index, frame in enumerate(frames):
                if frame.dtype != np.uint8:
                    raise ValueError(
                        f"Expected uint8 image for {camera_key} frame {frame_index} in {file_path}; got {frame.dtype}"
                    )
                if frame.shape != reference_shape:
                    raise ValueError(
                        f"Image shape mismatch for {camera_key} frame {frame_index} in "
                        f"{file_path}: expected {reference_shape}, got {frame.shape}"
                    )

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    include_depth: bool = False,
    include_surface_normals: bool = False,
    surface_normal_intrinsics: PinholeIntrinsics = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    robot_config = ROBOT_CONFIGS[robot_type]
    motors = robot_config.motors
    cameras = robot_config.cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if include_depth:
        rgb_camera_key = robot_config.camera_to_image_key.get(DEPTH_COLOR_SOURCE_KEY)
        if rgb_camera_key is None:
            raise ValueError(f"No image mapping exists for {DEPTH_COLOR_SOURCE_KEY}")

        features[f"observation.images.{DEPTH_OUTPUT_KEY}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if include_surface_normals:
        rgb_camera_key = robot_config.camera_to_image_key.get(DEPTH_COLOR_SOURCE_KEY)
        if rgb_camera_key is None:
            raise ValueError(f"No image mapping exists for {DEPTH_COLOR_SOURCE_KEY}")

        features[f"observation.images.{SURFACE_NORMAL_OUTPUT_KEY}"] = {
            "dtype": mode,
            "shape": (surface_normal_intrinsics.height, surface_normal_intrinsics.width, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset_type = GeometryVideoLeRobotDataset if include_depth or include_surface_normals else LeRobotDataset
    return dataset_type.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    *,
    include_depth: bool = False,
    depth_near_m: float = DEFAULT_DEPTH_NEAR_M,
    depth_far_m: float = DEFAULT_DEPTH_FAR_M,
    include_surface_normals: bool = False,
    surface_normal_intrinsics: PinholeIntrinsics = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    surface_normal_max_neighbor_depth_delta_m: float = DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
) -> LeRobotDataset:
    json_dataset = JsonDataset(
        raw_dir,
        robot_type,
        include_depth=include_depth,
        depth_near_m=depth_near_m,
        depth_far_m=depth_far_m,
        include_surface_normals=include_surface_normals,
        surface_normal_intrinsics=surface_normal_intrinsics,
        surface_normal_max_neighbor_depth_delta_m=surface_normal_max_neighbor_depth_delta_m,
    )
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    include_depth: bool = False,
    depth_near_m: float = DEFAULT_DEPTH_NEAR_M,
    depth_far_m: float = DEFAULT_DEPTH_FAR_M,
    include_surface_normals: bool = False,
    surface_normals_width: int = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.width,
    surface_normals_height: int = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.height,
    surface_normals_fx: float = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.fx,
    surface_normals_fy: float = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.fy,
    surface_normals_cx: float = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.cx,
    surface_normals_cy: float = DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.cy,
    surface_normals_max_neighbor_depth_delta_m: float = DEFAULT_SURFACE_NORMAL_MAX_NEIGHBOR_DEPTH_DELTA_M,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    surface_normal_intrinsics = PinholeIntrinsics(
        width=surface_normals_width,
        height=surface_normals_height,
        fx=surface_normals_fx,
        fy=surface_normals_fy,
        cx=surface_normals_cx,
        cy=surface_normals_cy,
    )
    # Validate even when the feature is disabled so invalid explicit CLI values
    # fail deterministically rather than being silently ignored.
    surface_normal_metadata = surface_normals_encoding_metadata(
        intrinsics=surface_normal_intrinsics,
        max_neighbor_depth_delta_m=surface_normals_max_neighbor_depth_delta_m,
    )

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        include_depth=include_depth,
        include_surface_normals=include_surface_normals,
        surface_normal_intrinsics=surface_normal_intrinsics,
        dataset_config=dataset_config,
    )
    if include_depth:
        dataset.meta.info["depth_encoding"] = {
            "source_key": DEPTH_SOURCE_KEY,
            "feature_key": f"observation.images.{DEPTH_OUTPUT_KEY}",
            "encoding": DEPTH_ENCODING,
            "default_scale_m_per_unit": DEFAULT_DEPTH_SCALE_M_PER_UNIT,
            "near_m": depth_near_m,
            "far_m": depth_far_m,
            "invalid_value": 0,
            "valid_value_range": [1, 255],
        }
    if include_surface_normals:
        dataset.meta.info["surface_normals_encoding"] = surface_normal_metadata
    # ``LeRobotDataset.finalize`` closes writers but does not write info.json.
    # Persist custom geometry contracts before frame conversion so they survive
    # image-mode datasets and interrupted conversions as well as video mode.
    if include_depth or include_surface_normals:
        write_info(dataset.meta.info, dataset.meta.root)
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        include_depth=include_depth,
        depth_near_m=depth_near_m,
        depth_far_m=depth_far_m,
        include_surface_normals=include_surface_normals,
        surface_normal_intrinsics=surface_normal_intrinsics,
        surface_normal_max_neighbor_depth_delta_m=surface_normals_max_neighbor_depth_delta_m,
    )
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
