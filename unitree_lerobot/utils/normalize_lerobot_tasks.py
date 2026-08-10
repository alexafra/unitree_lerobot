#!/usr/bin/env python3
"""Normalize task text in LeRobot v2.1 and processed-raw datasets safely.

The command is a dry run unless ``--apply`` is supplied.  For a root containing
``train/``, ``validation/`` (or ``validate/``), and ``test/``, the normalized
task order is taken from ``train/meta/tasks.jsonl`` and shared by every split.

Examples:

    python -m unitree_lerobot.utils.normalize_lerobot_tasks /data/my_dataset
    python -m unitree_lerobot.utils.normalize_lerobot_tasks /data/my_dataset --apply

LeRobot writes use a sibling staging directory and an atomic directory swap.
Processed-raw writes touch only ``data.json``, ``split_manifest.json``, and
``flatten_manifest.json`` files, with a JSON-only backup and rollback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_META_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json")
SPLIT_ORDER = ("train", "validation", "test")
VALIDATION_NAMES = ("validation", "validate")
RAW_JSON_FILENAMES = ("data.json", "flatten_manifest.json", "split_manifest.json")
MISSPELLED_TOKEN = re.compile(r"(?<![A-Za-z0-9_])toothepaste(?![A-Za-z0-9_])")


class NormalizationError(RuntimeError):
    """Raised when normalizing would be ambiguous or unsafe."""


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    relative_parquet_path: Path
    old_task_values: np.ndarray
    new_task_values: np.ndarray

    @property
    def task_ids_changed(self) -> bool:
        return not np.array_equal(self.old_task_values, self.new_task_values)


@dataclass(frozen=True)
class SplitPlan:
    name: str
    relative_root: Path
    old_to_new: dict[int, int]
    collisions: dict[str, tuple[int, ...]]
    episodes: tuple[EpisodePlan, ...]
    new_info: dict[str, Any]
    new_tasks: tuple[dict[str, Any], ...]
    new_episode_records: tuple[dict[str, Any], ...]
    new_episode_stats: tuple[dict[str, Any], ...]
    new_stats: dict[str, Any]
    metadata_changed: bool

    @property
    def changed_parquet_files(self) -> int:
        return sum(episode.task_ids_changed for episode in self.episodes)

    @property
    def changed_frames(self) -> int:
        return sum(
            int(np.count_nonzero(episode.old_task_values != episode.new_task_values)) for episode in self.episodes
        )

    @property
    def has_changes(self) -> bool:
        return self.metadata_changed or self.changed_parquet_files > 0


@dataclass(frozen=True)
class RootPlan:
    root: Path
    canonical_tasks: tuple[str, ...]
    splits: tuple[SplitPlan, ...]

    @property
    def has_changes(self) -> bool:
        return any(split.has_changes for split in self.splits)


@dataclass(frozen=True)
class RawJsonPlan:
    relative_path: Path
    source_sha256: str
    goal_count: int
    changed_goal_count: int

    @property
    def has_changes(self) -> bool:
        return self.changed_goal_count > 0


@dataclass(frozen=True)
class ProcessedRawPlan:
    root: Path
    files: tuple[RawJsonPlan, ...]

    @property
    def has_changes(self) -> bool:
        return any(file.has_changes for file in self.files)


NormalizationPlan = RootPlan | ProcessedRawPlan


@dataclass(frozen=True)
class ApplyResult:
    plan: NormalizationPlan
    backup_path: Path | None


def normalize_task_text(text: str) -> str:
    """Strip whitespace and correct the exact ``toothepaste`` token."""

    if not isinstance(text, str):
        raise NormalizationError(f"Task text must be a string, got {type(text).__name__}")
    normalized = MISSPELLED_TOKEN.sub("toothpaste", text.strip())
    if not normalized:
        raise NormalizationError("Task text becomes empty after stripping whitespace")
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NormalizationError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NormalizationError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
                if not isinstance(value, dict):
                    raise NormalizationError(f"Line {line_number} of {path} is not a JSON object")
                records.append(value)
    except OSError as exc:
        raise NormalizationError(f"Cannot read {path}: {exc}") from exc
    return records


def _is_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _discover_datasets(root: Path) -> list[tuple[str, Path, Path]]:
    """Return ``(logical name, absolute path, path relative to root)`` entries."""

    if _is_dataset(root):
        return [("train", root, Path("."))]

    validation_paths = [root / name for name in VALIDATION_NAMES if _is_dataset(root / name)]
    if len(validation_paths) > 1:
        raise NormalizationError(f"{root} contains both validation/ and validate/")

    discovered: dict[str, Path] = {}
    if _is_dataset(root / "train"):
        discovered["train"] = root / "train"
    if validation_paths:
        discovered["validation"] = validation_paths[0]
    if _is_dataset(root / "test"):
        discovered["test"] = root / "test"
    if not discovered:
        raise NormalizationError(f"{root} is not a LeRobot v2.1 dataset or split root")
    if "train" not in discovered:
        raise NormalizationError("A split root must contain train/ so it can define canonical task ordering")

    return [(name, discovered[name], discovered[name].relative_to(root)) for name in SPLIT_ORDER if name in discovered]


def _load_split(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    meta = path / "meta"
    missing = [filename for filename in REQUIRED_META_FILES if not (meta / filename).is_file()]
    if missing:
        raise NormalizationError(f"{path} is missing metadata: {', '.join(missing)}")

    info = _read_json(meta / "info.json")
    version = str(info.get("codebase_version", ""))
    if version not in {"v2.1", "2.1"}:
        raise NormalizationError(f"{path} is LeRobot {version or 'unknown'}, not v2.1")
    return (
        info,
        _read_jsonl(meta / "tasks.jsonl"),
        _read_jsonl(meta / "episodes.jsonl"),
        _read_jsonl(meta / "episodes_stats.jsonl"),
        _read_json(meta / "stats.json"),
    )


def _validate_task_records(tasks: list[dict[str, Any]], path: Path) -> dict[int, str]:
    indices = [record.get("task_index") for record in tasks]
    if indices != list(range(len(tasks))):
        raise NormalizationError(f"Task indices in {path}/meta/tasks.jsonl must be contiguous from zero")

    result: dict[int, str] = {}
    for record in tasks:
        task_index = int(record["task_index"])
        task = record.get("task")
        if not isinstance(task, str):
            raise NormalizationError(f"Task {task_index} in {path}/meta/tasks.jsonl has non-string text")
        result[task_index] = task
    return result


def _canonical_tasks(train_tasks: list[dict[str, Any]], train_path: Path) -> tuple[str, ...]:
    task_map = _validate_task_records(train_tasks, train_path)
    return tuple(dict.fromkeys(normalize_task_text(task_map[index]) for index in range(len(task_map))))


def _safe_episode_relative_path(info: dict[str, Any], episode_index: int) -> Path:
    try:
        chunks_size = int(info["chunks_size"])
        formatted = info["data_path"].format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NormalizationError(f"Invalid episode path metadata for episode {episode_index}: {exc}") from exc
    relative = Path(formatted)
    if relative.is_absolute() or ".." in relative.parts:
        raise NormalizationError(f"Episode path escapes the dataset root: {formatted}")
    return relative


def _numpy_stats(values: np.ndarray) -> dict[str, list[int | float]]:
    array = np.asarray(values).reshape(-1, 1)
    if not pa.types.is_integer(pa.array(array[:, 0]).type):
        raise NormalizationError("task_index values must be integers")
    if array.shape[0] == 0:
        raise NormalizationError("Cannot compute task statistics for an empty dataset")
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def _stable_normalized_tasks(values: Any, *, location: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise NormalizationError(f"{location} must be a list of task strings")
    return list(dict.fromkeys(normalize_task_text(value) for value in values))


def _build_split_plan(
    *,
    name: str,
    split_path: Path,
    relative_root: Path,
    canonical_tasks: tuple[str, ...],
) -> SplitPlan:
    info, tasks, episode_records, episode_stats, stats = _load_split(split_path)
    old_task_text = _validate_task_records(tasks, split_path)
    canonical_index = {task: index for index, task in enumerate(canonical_tasks)}
    if len(canonical_index) != len(canonical_tasks):
        raise NormalizationError("Canonical task table contains duplicate text")

    old_to_new: dict[int, int] = {}
    normalized_to_old: dict[str, list[int]] = {}
    for old_index, task in old_task_text.items():
        normalized = normalize_task_text(task)
        if normalized not in canonical_index:
            raise NormalizationError(f"{split_path}/meta/tasks.jsonl contains task absent from train: {normalized!r}")
        old_to_new[old_index] = canonical_index[normalized]
        normalized_to_old.setdefault(normalized, []).append(old_index)
    collisions = {task: tuple(indices) for task, indices in normalized_to_old.items() if len(indices) > 1}

    total_episodes = int(info.get("total_episodes", -1))
    expected_episode_indices = list(range(total_episodes))
    if [record.get("episode_index") for record in episode_records] != expected_episode_indices:
        raise NormalizationError(f"Episode records in {split_path} must be contiguous from zero")
    if [record.get("episode_index") for record in episode_stats] != expected_episode_indices:
        raise NormalizationError(f"Episode-stat records in {split_path} must be contiguous from zero")
    if sum(int(record.get("length", -1)) for record in episode_records) != int(info.get("total_frames", -1)):
        raise NormalizationError(f"Episode lengths in {split_path} do not equal info.json total_frames")

    new_episode_records = copy.deepcopy(episode_records)
    new_episode_stats = copy.deepcopy(episode_stats)
    episode_plans: list[EpisodePlan] = []
    all_new_task_values: list[np.ndarray] = []

    for offset, (episode, stat_record) in enumerate(zip(episode_records, episode_stats)):
        episode_index = int(episode["episode_index"])
        relative_parquet = _safe_episode_relative_path(info, episode_index)
        parquet_path = split_path / relative_parquet
        if not parquet_path.is_file():
            raise NormalizationError(f"Missing episode Parquet: {parquet_path}")
        try:
            table = pq.read_table(parquet_path, columns=["task_index"])
        except (OSError, pa.ArrowException) as exc:
            raise NormalizationError(f"Cannot read task_index from {parquet_path}: {exc}") from exc
        if table.num_rows != int(episode.get("length", -1)):
            raise NormalizationError(
                f"{parquet_path} has {table.num_rows} rows; episodes.jsonl says {episode.get('length')}"
            )
        field = table.schema.field("task_index")
        if not pa.types.is_integer(field.type):
            raise NormalizationError(f"{parquet_path} task_index must be an integer column, got {field.type}")
        column = table["task_index"]
        if column.null_count:
            raise NormalizationError(f"{parquet_path} task_index contains null values")
        old_values = column.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        try:
            new_values = np.asarray([old_to_new[int(value)] for value in old_values], dtype=np.int64)
        except KeyError as exc:
            raise NormalizationError(f"{parquet_path} refers to unknown task_index {exc.args[0]}") from exc

        decoded_tasks = list(dict.fromkeys(canonical_tasks[int(value)] for value in new_values))
        normalized_episode_tasks = _stable_normalized_tasks(
            episode.get("tasks"), location=f"episode {episode_index} tasks in {split_path}"
        )
        if set(decoded_tasks) != set(normalized_episode_tasks):
            raise NormalizationError(
                f"Episode {episode_index} in {split_path} has tasks {normalized_episode_tasks}, "
                f"but its Parquet decodes to {decoded_tasks}"
            )
        new_episode_records[offset]["tasks"] = normalized_episode_tasks

        if not isinstance(stat_record.get("stats"), dict):
            raise NormalizationError(f"Episode {episode_index} in {split_path} has no stats object")
        new_episode_stats[offset]["stats"]["task_index"] = _numpy_stats(new_values)
        episode_plans.append(
            EpisodePlan(
                episode_index=episode_index,
                relative_parquet_path=relative_parquet,
                old_task_values=old_values,
                new_task_values=new_values,
            )
        )
        all_new_task_values.append(new_values)

    if not episode_plans:
        raise NormalizationError(f"Refusing to normalize empty dataset {split_path}")

    new_info = copy.deepcopy(info)
    new_info["total_tasks"] = len(canonical_tasks)
    new_tasks = tuple({"task_index": task_index, "task": task} for task_index, task in enumerate(canonical_tasks))
    new_stats = copy.deepcopy(stats)
    new_stats["task_index"] = _numpy_stats(np.concatenate(all_new_task_values))

    metadata_changed = any(
        (
            info != new_info,
            tasks != list(new_tasks),
            episode_records != new_episode_records,
            episode_stats != new_episode_stats,
            stats != new_stats,
        )
    )
    return SplitPlan(
        name=name,
        relative_root=relative_root,
        old_to_new=old_to_new,
        collisions=collisions,
        episodes=tuple(episode_plans),
        new_info=new_info,
        new_tasks=new_tasks,
        new_episode_records=tuple(new_episode_records),
        new_episode_stats=tuple(new_episode_stats),
        new_stats=new_stats,
        metadata_changed=metadata_changed,
    )


def _build_lerobot_plan(resolved_root: Path) -> RootPlan:
    """Inspect a LeRobot dataset or split root without modifying it."""

    datasets = _discover_datasets(resolved_root)
    train_entry = next(entry for entry in datasets if entry[0] == "train")
    _, train_tasks, _, _, _ = _load_split(train_entry[1])
    canonical_tasks = _canonical_tasks(train_tasks, train_entry[1])
    split_plans = tuple(
        _build_split_plan(
            name=name,
            split_path=split_path,
            relative_root=relative_root,
            canonical_tasks=canonical_tasks,
        )
        for name, split_path, relative_root in datasets
    )
    return RootPlan(root=resolved_root, canonical_tasks=canonical_tasks, splits=split_plans)


def _looks_like_lerobot(root: Path) -> bool:
    return _is_dataset(root) or any(_is_dataset(root / name) for name in ("train", "test", *VALIDATION_NAMES))


def _guard_processed_raw_root(root: Path) -> None:
    known_source_raw = Path.home() / "Development" / "Datasets" / "raw"
    if root == known_source_raw or root in known_source_raw.parents:
        raise NormalizationError(f"Refusing a root broad enough to include source-raw data at {known_source_raw}")
    for candidate in (root, *root.parents):
        if candidate.name == "raw" and candidate.parent.name == "Datasets":
            raise NormalizationError(f"Refusing to touch source-raw data under {candidate}")
    if root.name == "Datasets" and (root / "raw").is_dir():
        raise NormalizationError(f"Refusing a /Datasets root that includes {root / 'raw'}")


def _normalize_goal_fields(value: Any, *, location: str = "$") -> tuple[int, int]:
    """Normalize every dictionary value whose key is exactly ``goal`` in place."""

    goal_count = 0
    changed_count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "goal":
                if not isinstance(child, str):
                    raise NormalizationError(f"{child_location} must be a string, got {type(child).__name__}")
                normalized = normalize_task_text(child)
                goal_count += 1
                if normalized != child:
                    value[key] = normalized
                    changed_count += 1
                continue
            nested_goals, nested_changes = _normalize_goal_fields(child, location=child_location)
            goal_count += nested_goals
            changed_count += nested_changes
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nested_goals, nested_changes = _normalize_goal_fields(child, location=f"{location}[{index}]")
            goal_count += nested_goals
            changed_count += nested_changes
    return goal_count, changed_count


def _load_and_normalize_raw_json(path: Path) -> tuple[dict[str, Any], str, int, int]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NormalizationError(f"{path} must contain a JSON object")
    if path.name == "data.json":
        text = value.get("text")
        if not isinstance(text, dict) or "goal" not in text:
            raise NormalizationError(f"{path} must contain text.goal")
    goal_count, changed_goal_count = _normalize_goal_fields(value)
    return value, hashlib.sha256(raw).hexdigest(), goal_count, changed_goal_count


def _build_processed_raw_plan(resolved_root: Path) -> ProcessedRawPlan:
    _guard_processed_raw_root(resolved_root)
    paths = sorted(
        {path for filename in RAW_JSON_FILENAMES for path in resolved_root.rglob(filename) if path.is_file()}
    )
    if not paths:
        raise NormalizationError(f"{resolved_root} has no LeRobot metadata or supported processed-raw JSON files")

    file_plans: list[RawJsonPlan] = []
    for path in paths:
        if path.is_symlink():
            raise NormalizationError(f"Refusing to follow supported JSON symlink: {path}")
        try:
            relative_path = path.relative_to(resolved_root)
        except ValueError as exc:
            raise NormalizationError(f"JSON file escapes processed-raw root: {path}") from exc
        _, source_sha256, goal_count, changed_goal_count = _load_and_normalize_raw_json(path)
        file_plans.append(
            RawJsonPlan(
                relative_path=relative_path,
                source_sha256=source_sha256,
                goal_count=goal_count,
                changed_goal_count=changed_goal_count,
            )
        )
    return ProcessedRawPlan(root=resolved_root, files=tuple(file_plans))


def build_plan(root: Path | str) -> NormalizationPlan:
    """Inspect a LeRobot or processed-raw root without modifying it."""

    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        raise NormalizationError(f"Dataset root does not exist: {resolved_root}")
    if _looks_like_lerobot(resolved_root):
        return _build_lerobot_plan(resolved_root)
    return _build_processed_raw_plan(resolved_root)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    mode = path.stat().st_mode if path.exists() else None
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=4, ensure_ascii=False, default=_json_default) + "\n")


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n" for record in records)
    _atomic_write_text(path, text)


def _clone_with_hardlinks(source: Path, destination: Path) -> None:
    def hardlink_or_copy(source_file: str, destination_file: str) -> str:
        try:
            os.link(source_file, destination_file)
            return destination_file
        except OSError:
            return shutil.copy2(source_file, destination_file)

    shutil.copytree(source, destination, copy_function=hardlink_or_copy, symlinks=True)


def _parquet_compression(path: Path) -> str | dict[str, str]:
    metadata = pq.read_metadata(path)
    codecs: dict[str, str] = {}
    for column_index in range(metadata.num_columns):
        codec_values = {
            metadata.row_group(row_group).column(column_index).compression.lower()
            for row_group in range(metadata.num_row_groups)
        }
        if len(codec_values) != 1:
            return "snappy"
        column_path = metadata.schema.column(column_index).path.split(".")[0]
        codec = next(iter(codec_values))
        codecs[column_path] = "none" if codec == "uncompressed" else codec
    return codecs


def _atomic_rewrite_task_column(path: Path, new_values: np.ndarray) -> None:
    mode = path.stat().st_mode
    temporary: Path | None = None
    try:
        table = pq.read_table(path)
        if table.num_rows != len(new_values):
            raise NormalizationError(
                f"{path} changed during normalization: expected {len(new_values)} rows, found {table.num_rows}"
            )
        column_index = table.schema.get_field_index("task_index")
        if column_index < 0:
            raise NormalizationError(f"{path} has no task_index column")
        field = table.schema.field(column_index)
        replacement = pa.array(new_values, type=field.type)
        updated = table.set_column(column_index, field, replacement)

        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(updated, temporary, compression=_parquet_compression(path))
        os.chmod(temporary, mode)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_stage(stage: Path, plan: RootPlan) -> None:
    for split in plan.splits:
        split_root = stage / split.relative_root
        for episode in split.episodes:
            if episode.task_ids_changed:
                _atomic_rewrite_task_column(split_root / episode.relative_parquet_path, episode.new_task_values)

        meta = split_root / "meta"
        _atomic_write_json(meta / "info.json", split.new_info)
        _atomic_write_jsonl(meta / "tasks.jsonl", split.new_tasks)
        _atomic_write_jsonl(meta / "episodes.jsonl", split.new_episode_records)
        _atomic_write_jsonl(meta / "episodes_stats.jsonl", split.new_episode_stats)
        _atomic_write_json(meta / "stats.json", split.new_stats)


def _plan_signature(plan: NormalizationPlan) -> str:
    if isinstance(plan, RootPlan):
        payload: dict[str, Any] = {
            "kind": "lerobot",
            "root": str(plan.root),
            "canonical_tasks": plan.canonical_tasks,
            "splits": [
                {
                    "name": split.name,
                    "relative_root": str(split.relative_root),
                    "old_to_new": split.old_to_new,
                    "collisions": split.collisions,
                    "episodes": [
                        {
                            "episode_index": episode.episode_index,
                            "relative_parquet_path": str(episode.relative_parquet_path),
                            "old_task_values": episode.old_task_values.tolist(),
                            "new_task_values": episode.new_task_values.tolist(),
                        }
                        for episode in split.episodes
                    ],
                    "new_info": split.new_info,
                    "new_tasks": split.new_tasks,
                    "new_episode_records": split.new_episode_records,
                    "new_episode_stats": split.new_episode_stats,
                    "new_stats": split.new_stats,
                    "metadata_changed": split.metadata_changed,
                }
                for split in plan.splits
            ],
        }
    else:
        payload = {
            "kind": "processed_raw",
            "root": str(plan.root),
            "files": [
                {
                    "relative_path": str(file.relative_path),
                    "source_sha256": file.source_sha256,
                    "goal_count": file.goal_count,
                    "changed_goal_count": file.changed_goal_count,
                }
                for file in plan.files
            ],
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _apply_lerobot_plan(plan: RootPlan, *, keep_backup: bool = True) -> ApplyResult:
    """Apply a previously inspected plan through a validated directory swap."""

    if not plan.has_changes:
        return ApplyResult(plan=plan, backup_path=None)

    current_plan = build_plan(plan.root)
    if _plan_signature(current_plan) != _plan_signature(plan):
        raise NormalizationError(f"{plan.root} changed after the normalization plan was built; rerun the dry run")

    root = plan.root
    stage = root.parent / f".{root.name}.normalize-{uuid.uuid4().hex}"
    backup = root.parent / f".{root.name}.pre-normalize-{uuid.uuid4().hex}"
    try:
        _clone_with_hardlinks(root, stage)
        _write_stage(stage, plan)

        staged_plan = build_plan(stage)
        if not isinstance(staged_plan, RootPlan) or staged_plan.has_changes:
            raise NormalizationError("Staged dataset did not pass the post-normalization consistency check")

        os.replace(root, backup)
        try:
            os.replace(stage, root)
            _fsync_directory(root.parent)
        except BaseException:
            os.replace(backup, root)
            _fsync_directory(root.parent)
            raise

        if not keep_backup:
            shutil.rmtree(backup)
            backup_path: Path | None = None
        else:
            backup_path = backup
        return ApplyResult(plan=plan, backup_path=backup_path)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _restore_backup_file(source: Path, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.restore-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _apply_processed_raw_plan(plan: ProcessedRawPlan, *, keep_backup: bool = True) -> ApplyResult:
    if not plan.has_changes:
        return ApplyResult(plan=plan, backup_path=None)

    current_plan = build_plan(plan.root)
    if _plan_signature(current_plan) != _plan_signature(plan):
        raise NormalizationError(f"{plan.root} changed after the normalization plan was built; rerun the dry run")

    changed_files = [file for file in plan.files if file.has_changes]
    backup = plan.root.parent / f".{plan.root.name}.pre-normalize-json-{uuid.uuid4().hex}"
    try:
        for file_plan in changed_files:
            source = plan.root / file_plan.relative_path
            destination = backup / file_plan.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        for file_plan in changed_files:
            path = plan.root / file_plan.relative_path
            normalized, source_sha256, _, changed_goal_count = _load_and_normalize_raw_json(path)
            if source_sha256 != file_plan.source_sha256:
                raise NormalizationError(f"{path} changed after the backup was made")
            if changed_goal_count != file_plan.changed_goal_count:
                raise NormalizationError(f"Goal fields in {path} changed after the plan was built")
            _atomic_write_json(path, normalized)

        validated = build_plan(plan.root)
        if not isinstance(validated, ProcessedRawPlan) or validated.has_changes:
            raise NormalizationError("Processed-raw JSON did not pass the post-normalization check")
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            for file_plan in changed_files:
                saved = backup / file_plan.relative_path
                if saved.is_file():
                    _restore_backup_file(saved, plan.root / file_plan.relative_path)
        except BaseException as restore_exc:
            rollback_error = restore_exc
        if rollback_error is not None:
            raise NormalizationError(
                f"Normalization failed and rollback also failed; backup remains at {backup}: {rollback_error}"
            ) from exc
        if backup.exists():
            shutil.rmtree(backup)
        raise

    if keep_backup:
        backup_path: Path | None = backup
    else:
        shutil.rmtree(backup)
        backup_path = None
    return ApplyResult(plan=plan, backup_path=backup_path)


def apply_plan(plan: NormalizationPlan, *, keep_backup: bool = True) -> ApplyResult:
    if isinstance(plan, RootPlan):
        return _apply_lerobot_plan(plan, keep_backup=keep_backup)
    return _apply_processed_raw_plan(plan, keep_backup=keep_backup)


def normalize_root(
    root: Path | str,
    *,
    apply: bool = False,
    keep_backup: bool = True,
) -> ApplyResult:
    """Build a normalization plan and optionally apply it."""

    plan = build_plan(root)
    if not apply:
        return ApplyResult(plan=plan, backup_path=None)
    return apply_plan(plan, keep_backup=keep_backup)


def format_plan(plan: NormalizationPlan) -> str:
    if isinstance(plan, RootPlan):
        lines = [
            f"LeRobot dataset: {plan.root}",
            f"Canonical tasks ({len(plan.canonical_tasks)}):",
        ]
        lines.extend(f"  {index}: {task}" for index, task in enumerate(plan.canonical_tasks))
        for split in plan.splits:
            lines.append(f"{split.name}:")
            lines.append(f"  task map: {split.old_to_new}")
            if split.collisions:
                lines.append(f"  normalized-text collisions: {split.collisions}")
            lines.append(
                f"  changes: {split.changed_parquet_files} Parquet file(s), "
                f"{split.changed_frames} frame task ID(s), metadata={'yes' if split.metadata_changed else 'no'}"
            )
        lines.append("Changes required." if plan.has_changes else "Dataset is already normalized.")
        return "\n".join(lines)

    total_goals = sum(file.goal_count for file in plan.files)
    changed_files = [file for file in plan.files if file.has_changes]
    changed_goals = sum(file.changed_goal_count for file in changed_files)
    unchanged_manifests = [
        file.relative_path
        for file in plan.files
        if file.relative_path.name in {"flatten_manifest.json", "split_manifest.json"}
        and file.goal_count
        and not file.has_changes
    ]
    lines = [
        f"Processed-raw dataset: {plan.root}",
        f"Supported JSON files: {len(plan.files)}",
        f"Goal fields: {total_goals}",
        f"Changes: {len(changed_files)} JSON file(s), {changed_goals} goal field(s)",
        "Directory names and non-JSON assets will not be changed.",
    ]
    if unchanged_manifests:
        lines.append("Manifest files already normalized and left unchanged:")
        lines.extend(f"  {path}" for path in unchanged_manifests)
    lines.append("Changes required." if plan.has_changes else "Dataset is already normalized.")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize task/goal text in a LeRobot v2.1 or processed-raw root, "
            "including the exact toothepaste-to-toothpaste correction."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v2.1 dataset/split root or processed-raw root",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the displayed plan. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--discard-backup",
        action="store_true",
        help="Delete the pre-normalization backup after a successful atomic swap.",
    )
    args = parser.parse_args(argv)
    if args.discard_backup and not args.apply:
        parser.error("--discard-backup requires --apply")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = normalize_root(args.root, apply=args.apply, keep_backup=not args.discard_backup)
    print(format_plan(result.plan))
    if not args.apply:
        print("Dry run only; rerun with --apply to write these changes.")
    elif result.backup_path is not None:
        print(f"Applied successfully. Original dataset retained at: {result.backup_path}")
    else:
        print("Applied successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NormalizationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
