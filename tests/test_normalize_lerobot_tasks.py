from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from unitree_lerobot.utils import normalize_lerobot_tasks as normalizer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _task_stats(values: np.ndarray) -> dict[str, list[int | float]]:
    return normalizer._numpy_stats(values)


def _make_split(root: Path, tasks: list[str], episode_task_ids: list[int]) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    meta.mkdir(parents=True)

    episode_records = []
    episode_stats = []
    all_task_values = []
    next_global_index = 0
    schema = pa.schema(
        [
            pa.field("task_index", pa.int64(), metadata={b"meaning": b"task id"}),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("sensor", pa.float32()),
        ],
        metadata={b"fixture": b"kept"},
    )
    for episode_index, task_index in enumerate(episode_task_ids):
        task_values = np.asarray([task_index, task_index], dtype=np.int64)
        table = pa.Table.from_arrays(
            [
                pa.array(task_values),
                pa.array([episode_index, episode_index], type=pa.int64()),
                pa.array([0, 1], type=pa.int64()),
                pa.array([next_global_index, next_global_index + 1], type=pa.int64()),
                pa.array([episode_index + 0.25, episode_index + 0.75], type=pa.float32()),
            ],
            schema=schema,
        )
        parquet_path = data / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path, compression="gzip")
        episode_records.append(
            {
                "episode_index": episode_index,
                "tasks": [tasks[task_index]],
                "length": len(task_values),
            }
        )
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "task_index": _task_stats(task_values),
                    "sensor": _task_stats(np.asarray([episode_index, episode_index + 1])),
                },
            }
        )
        all_task_values.append(task_values)
        next_global_index += len(task_values)

    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v2.1",
            "total_episodes": len(episode_records),
            "total_frames": next_global_index,
            "total_tasks": len(tasks),
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {"task_index": {"dtype": "int64", "shape": [1]}},
        },
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
    )
    _write_jsonl(meta / "episodes.jsonl", episode_records)
    _write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    _write_json(
        meta / "stats.json",
        {
            "task_index": _task_stats(np.concatenate(all_task_values)),
            "sensor": {"mean": [123.0]},
            "__fingerprints__": {"sensor": "sha256:unchanged"},
        },
    )
    _write_json(
        meta / "relative_stats.json",
        {"action": {"mean": [7.0]}, "__fingerprints__": {"action": "sha256:kept"}},
    )
    video = root / "videos" / "chunk-000" / "camera" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not-a-real-video-but-must-not-change")


def _make_lerobot_root(root: Path) -> None:
    train_tasks = [
        "  pick up cereal.  ",
        "put down cylinder toothepaste.",
        "pick red cup.",
        "pick red cup.\n",
    ]
    validation_tasks = [
        "pick red cup.\n",
        "pick up cereal.",
        "put down cylinder toothepaste.",
        "pick red cup.",
    ]
    test_tasks = [
        "put down cylinder toothepaste.",
        "pick red cup.\n",
        "pick up cereal.",
        "pick red cup.",
    ]
    _make_split(root / "train", train_tasks, [0, 1, 2, 3])
    _make_split(root / "validation", validation_tasks, [1, 2, 0, 3])
    _make_split(root / "test", test_tasks, [2, 0, 1, 3])


class LeRobotTaskNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "lerobot"
        _make_lerobot_root(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_dry_run_is_read_only_and_builds_train_ordered_canonical_tasks(self):
        before = _digest_tree(self.root)

        result = normalizer.normalize_root(self.root)

        self.assertIsInstance(result.plan, normalizer.RootPlan)
        self.assertEqual(
            result.plan.canonical_tasks,
            (
                "pick up cereal.",
                "put down cylinder toothpaste.",
                "pick red cup.",
            ),
        )
        self.assertEqual(result.plan.splits[0].old_to_new, {0: 0, 1: 1, 2: 2, 3: 2})
        self.assertEqual(before, _digest_tree(self.root))
        self.assertIsNone(result.backup_path)

    def test_apply_remaps_every_reference_and_preserves_non_task_data(self):
        old_sensor = {
            split: [
                pq.read_table(path)["sensor"].to_pylist()
                for path in sorted((self.root / split / "data").rglob("*.parquet"))
            ]
            for split in ("train", "validation", "test")
        }
        old_relative_stats = {
            split: (self.root / split / "meta" / "relative_stats.json").read_bytes()
            for split in ("train", "validation", "test")
        }

        result = normalizer.normalize_root(self.root, apply=True)

        self.assertIsNotNone(result.backup_path)
        self.assertTrue(result.backup_path.is_dir())
        expected_tasks = [
            {"task_index": 0, "task": "pick up cereal."},
            {"task_index": 1, "task": "put down cylinder toothpaste."},
            {"task_index": 2, "task": "pick red cup."},
        ]
        expected_ids = {
            "train": [0, 1, 2, 2],
            "validation": [0, 1, 2, 2],
            "test": [0, 1, 2, 2],
        }
        for split in ("train", "validation", "test"):
            split_root = self.root / split
            self.assertEqual(_read_jsonl(split_root / "meta" / "tasks.jsonl"), expected_tasks)
            self.assertEqual(json.loads((split_root / "meta" / "info.json").read_text())["total_tasks"], 3)
            tables = [pq.read_table(path) for path in sorted((split_root / "data").rglob("*.parquet"))]
            actual_ids = [int(table["task_index"][0].as_py()) for table in tables]
            self.assertEqual(actual_ids, expected_ids[split])
            self.assertEqual([table["sensor"].to_pylist() for table in tables], old_sensor[split])
            for table in tables:
                self.assertEqual(table.schema.metadata, {b"fixture": b"kept"})
                self.assertEqual(table.schema.field("task_index").metadata, {b"meaning": b"task id"})
            codec = (
                pq.read_metadata(sorted((split_root / "data").rglob("*.parquet"))[0]).row_group(0).column(0).compression
            )
            self.assertEqual(codec, "GZIP")
            self.assertEqual((split_root / "meta" / "relative_stats.json").read_bytes(), old_relative_stats[split])
            self.assertEqual(
                (split_root / "videos" / "chunk-000" / "camera" / "episode_000000.mp4").read_bytes(),
                b"not-a-real-video-but-must-not-change",
            )
            stats = json.loads((split_root / "meta" / "stats.json").read_text())
            all_ids = np.concatenate([np.asarray(table["task_index"].to_pylist(), dtype=np.int64) for table in tables])
            self.assertEqual(stats["task_index"], _task_stats(all_ids))
            self.assertEqual(stats["sensor"], {"mean": [123.0]})
            episode_stats = _read_jsonl(split_root / "meta" / "episodes_stats.jsonl")
            for table, record in zip(tables, episode_stats):
                values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
                self.assertEqual(record["stats"]["task_index"], _task_stats(values))

        checked = normalizer.build_plan(self.root)
        self.assertIsInstance(checked, normalizer.RootPlan)
        self.assertFalse(checked.has_changes)
        shutil.rmtree(result.backup_path)

    def test_stage_failure_leaves_original_tree_unchanged(self):
        before = _digest_tree(self.root)
        plan = normalizer.build_plan(self.root)

        with mock.patch.object(
            normalizer,
            "_atomic_rewrite_task_column",
            side_effect=RuntimeError("injected write failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                normalizer.apply_plan(plan)

        self.assertEqual(before, _digest_tree(self.root))
        self.assertEqual(list(self.root.parent.glob(f".{self.root.name}.normalize-*")), [])


class ProcessedRawTaskNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "processed_raw"
        episode = self.root / "atomic_toothepaste" / "episode_0000"
        _write_json(
            episode / "data.json",
            {
                "text": {
                    "goal": "  pick up the cylinder toothepaste.\n",
                    "desc": "toothepaste is deliberately not changed outside a goal field",
                },
                "data": [],
            },
        )
        _write_json(
            self.root / "atomic_toothepaste" / "split_manifest.json",
            {
                "episodes": [
                    {"goal": "put down toothepaste. "},
                    {"nested": {"goal": "already clean."}},
                ]
            },
        )
        _write_json(
            self.root / "legacy_toothepaste_2807" / "split_manifest.json",
            {"episodes": [{"goal": "already corrected toothpaste."}]},
        )
        asset = episode / "camera" / "frame.jpg"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"unchanged image")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_apply_changes_only_goal_fields_and_keeps_json_only_backup(self):
        before = _digest_tree(self.root)
        dry_run = normalizer.normalize_root(self.root)
        self.assertIsInstance(dry_run.plan, normalizer.ProcessedRawPlan)
        self.assertEqual(sum(file.changed_goal_count for file in dry_run.plan.files), 2)
        self.assertEqual(before, _digest_tree(self.root))

        result = normalizer.apply_plan(dry_run.plan)

        data_path = self.root / "atomic_toothepaste" / "episode_0000" / "data.json"
        data = json.loads(data_path.read_text())
        self.assertEqual(data["text"]["goal"], "pick up the cylinder toothpaste.")
        self.assertEqual(data["text"]["desc"], "toothepaste is deliberately not changed outside a goal field")
        manifest = json.loads((self.root / "atomic_toothepaste" / "split_manifest.json").read_text())
        self.assertEqual(manifest["episodes"][0]["goal"], "put down toothpaste.")
        self.assertEqual(
            (self.root / "atomic_toothepaste" / "episode_0000" / "camera" / "frame.jpg").read_bytes(),
            b"unchanged image",
        )
        self.assertTrue((self.root / "atomic_toothepaste").is_dir())
        self.assertTrue(result.backup_path.is_dir())
        backup_files = sorted(
            str(path.relative_to(result.backup_path)) for path in result.backup_path.rglob("*") if path.is_file()
        )
        self.assertEqual(
            backup_files,
            [
                "atomic_toothepaste/episode_0000/data.json",
                "atomic_toothepaste/split_manifest.json",
            ],
        )
        self.assertFalse(normalizer.build_plan(self.root).has_changes)
        shutil.rmtree(result.backup_path)

    def test_write_failure_rolls_back_every_json_byte_for_byte(self):
        before = _digest_tree(self.root)
        plan = normalizer.build_plan(self.root)
        real_write = normalizer._atomic_write_json
        call_count = 0

        def fail_second_write(path: Path, value: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected JSON failure")
            real_write(path, value)

        with mock.patch.object(normalizer, "_atomic_write_json", side_effect=fail_second_write):
            with self.assertRaisesRegex(RuntimeError, "injected JSON failure"):
                normalizer.apply_plan(plan)

        self.assertEqual(before, _digest_tree(self.root))
        self.assertEqual(list(self.root.parent.glob(f".{self.root.name}.pre-normalize-json-*")), [])

    def test_refuses_source_raw_and_a_broader_datasets_root(self):
        datasets = Path(self.tempdir.name) / "Datasets"
        raw = datasets / "raw" / "episode_0000"
        _write_json(raw / "data.json", {"text": {"goal": " bad "}})

        with self.assertRaisesRegex(normalizer.NormalizationError, "source-raw"):
            normalizer.build_plan(raw.parent)
        with self.assertRaisesRegex(normalizer.NormalizationError, "Datasets root"):
            normalizer.build_plan(datasets)


if __name__ == "__main__":
    unittest.main()
