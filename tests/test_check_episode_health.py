from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CHECKER_PATH = Path(__file__).parents[1] / "data_editor" / "check_episode_health.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_check_episode_health_under_test", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load checker from {CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve the defining module through sys.modules.
    import sys

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _snapshot(root: Path):
    return {
        path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_episode(root: Path, episode_id: str, document: dict) -> Path:
    episode = root / episode_id
    episode.mkdir(parents=True)
    data_json = episode / "data.json"
    data_json.write_text(json.dumps(document), encoding="utf-8")
    return data_json


class CheckEpisodeHealthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = _load_checker()

    def test_scan_reports_objective_frame_and_dds_evidence_without_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            problem_json = _write_episode(
                root,
                "episode_0063",
                {
                    "data": [
                        {"idx": 0, "timestamp_s": 1.0},
                        {"idx": 1, "timestamp_s": 1.033},
                        {"idx": 2, "timestamp_s": 1.153},
                    ],
                    "diagnostics": {
                        "inspire_ftp_state_subscribers": {
                            "metric": "valid_state_receive_gap",
                            "gap_threshold_s": 0.075,
                            "right": {
                                "topic": "rt/inspire_hand/state/r",
                                "gap_count": 1,
                                "recovered_gap_count": 1,
                                "open_gap_at_end": False,
                                "max_gap_duration_s": 0.075734794,
                                "gaps": [{"recovered": True}],
                            },
                            "total_side_gap_count": 1,
                        }
                    },
                },
            )
            unknown_json = _write_episode(root, "episode_0073", {})
            clean_json = _write_episode(root, "episode_0074", {})
            before = _snapshot(root)

            def fake_scan(path, *, max_frame_gap_s, min_measured_fps):
                self.assertEqual(Path(path), root)
                self.assertEqual(max_frame_gap_s, 0.075)
                self.assertEqual(min_measured_fps, 29.0)
                return [
                    SimpleNamespace(
                        episode="episode_0063",
                        data_json=str(problem_json),
                        status="reject",
                        reasons=[
                            "max_frame_gap_s=0.120000>0.075000",
                            "dds_gap:inspire_ftp_state_subscribers.right.gap_count=1",
                            "dds_gap:inspire_ftp_state_subscribers.total_side_gap_count=1",
                        ],
                    ),
                    SimpleNamespace(
                        episode="episode_0073",
                        data_json=str(unknown_json),
                        status="unknown",
                        reasons=["diagnostics_missing"],
                    ),
                    SimpleNamespace(
                        episode="episode_0074",
                        data_json=str(clean_json),
                        status="clean",
                        reasons=["all_required_metrics_within_limits"],
                    ),
                ]

            scan = self.checker.scan_episode_health(root, scanner=fake_scan)

            self.assertEqual(_snapshot(root), before)
            self.assertEqual(scan.total_episode_count, 3)
            self.assertEqual(scan.clean_count, 1)
            self.assertEqual(
                [finding.episode_id for finding in scan.possibly_problematic],
                ["episode_0063"],
            )
            self.assertEqual([finding.episode_id for finding in scan.warnings], ["episode_0063"])
            self.assertEqual([finding.episode_id for finding in scan.serious], ["episode_0073"])
            problem_reasons = "\n".join(scan.possibly_problematic[0].reasons)
            self.assertIn("recorded frame-loop gap: 120.000 ms", problem_reasons)
            self.assertIn("between frame IDs 1 and 2", problem_reasons)
            self.assertIn("DDS valid-state callback gap", problem_reasons)
            self.assertIn("max 75.735 ms", problem_reasons)
            self.assertIn("none open at episode end", problem_reasons)
            self.assertEqual(
                [finding.episode_id for finding in scan.unverified],
                ["episode_0073"],
            )

    def test_missing_data_json_is_unverified_and_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_episode = root / "episode_0008"
            missing_episode.mkdir()

            scan = self.checker.scan_episode_health(root, scanner=lambda *args, **kwargs: [])

            self.assertEqual(scan.total_episode_count, 1)
            self.assertEqual(scan.unverified[0].episode_id, "episode_0008")
            self.assertFalse((missing_episode / "data.json").exists())

    def test_empty_selection_is_not_reported_as_clean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "no episode"):
                self.checker.scan_episode_health(
                    temp_dir,
                    scanner=lambda *args, **kwargs: [],
                )

    def test_cli_lists_only_findings_and_explains_evidence_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_json = _write_episode(root, "episode_0012", {})

            def fake_scan(*args, **kwargs):
                return [
                    SimpleNamespace(
                        episode="episode_0012",
                        data_json=str(data_json),
                        status="reject",
                        reasons=["measured_fps=28.500<29.000"],
                    )
                ]

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = self.checker.main([str(root)], scanner=fake_scan)

            rendered = output.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertIn("episode_0012 [ORANGE WARNING]", rendered)
            self.assertIn("recorded frame-loop average: 28.500 FPS", rendered)
            self.assertIn("data[*].timestamp_s", rendered)
            self.assertIn("info.rgbd_pairing", rendered)
            self.assertIn("data[*].rgbd_pairing", rendered)
            self.assertIn("do not prove a camera-sensor drop", rendered)
            self.assertIn("No data was deleted, moved, renamed, or edited", rendered)

    def test_header_explicitly_reports_clean_and_warning_states(self):
        clean = self.checker.EpisodeHealthScan("/data", 2, 2, (), "shared.py")
        clean_text = self.checker.health_header_text(clean)
        self.assertIn("2 checked", clean_text)
        self.assertIn("Camera-sensor delivery is not verified", clean_text)

        finding = self.checker.EpisodeHealthFinding(
            "episode_0073",
            "possibly_problematic",
            ("recorded frame-loop gap",),
        )
        warning = self.checker.health_header_text(
            self.checker.EpisodeHealthScan("/data", 2, 1, (finding,), "shared.py")
        )
        self.assertIn("episode_0073", warning)
        self.assertIn("View Health Details", warning)
        self.assertIn("not proof of a camera-sensor drop", warning)

        serious_finding = self.checker.EpisodeHealthFinding(
            "episode_0099",
            "possibly_problematic",
            ("structural mismatch",),
            severity="serious",
        )
        serious = self.checker.health_header_text(
            self.checker.EpisodeHealthScan("/data", 2, 1, (serious_finding,), "shared.py")
        )
        self.assertIn("serious/structural", serious)
        self.assertIn("episode_0099", serious)

    def test_reject_with_missing_evidence_is_serious_and_unverified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_json = _write_episode(root, "episode_0005", {})
            result = SimpleNamespace(
                episode="episode_0005",
                data_json=str(data_json),
                status="reject",
                reasons=[
                    "max_frame_gap_s=0.100000>0.075000",
                    "diagnostics_missing",
                ],
            )

            scan = self.checker.scan_episode_health(
                root,
                scanner=lambda *args, **kwargs: [result],
            )

            self.assertEqual(scan.possibly_problematic, scan.unverified)
            self.assertEqual(scan.serious, scan.unverified)
            self.assertIn(
                "EVIDENCE INCOMPLETE",
                self.checker.render_report(scan),
            )

    def test_severity_thresholds_and_unrecovered_dds_are_red(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_json = _write_episode(
                root,
                "episode_0001",
                {
                    "diagnostics": {
                        "inspire_ftp_state_subscribers": {
                            "metric": "valid_state_receive_gap",
                            "left": {
                                "gap_count": 1,
                                "recovered_gap_count": 0,
                                "open_gap_at_end": True,
                                "max_gap_duration_s": 0.100,
                                "gaps": [{"recovered": False}],
                            },
                        }
                    }
                },
            )
            results = [
                SimpleNamespace(
                    episode="episode_0001",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["dds_gap:inspire_ftp_state_subscribers.left.gap_count=1"],
                ),
                SimpleNamespace(
                    episode="episode_0002",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["max_frame_gap_s=0.250000>0.075000"],
                ),
                SimpleNamespace(
                    episode="episode_0003",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["measured_fps=26.999<29.000"],
                ),
                SimpleNamespace(
                    episode="episode_0004",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["frame_count_mismatch:timing=10,data=9"],
                ),
            ]

            scan = self.checker.scan_episode_health(
                root,
                scanner=lambda *args, **kwargs: results,
            )

            self.assertEqual([finding.episode_id for finding in scan.serious], [
                "episode_0001",
                "episode_0002",
                "episode_0003",
                "episode_0004",
            ])
            self.assertFalse(scan.warnings)

    def test_severity_uses_exact_metrics_instead_of_rounded_reason_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_json = _write_episode(root, "episode_0001", {})
            results = [
                SimpleNamespace(
                    episode="episode_0001",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["max_frame_gap_s=0.250000>0.075000"],
                    max_frame_gap_s=0.2499996,
                ),
                SimpleNamespace(
                    episode="episode_0002",
                    data_json=str(data_json),
                    status="reject",
                    reasons=["measured_fps=27.000<29.000"],
                    measured_fps=26.9996,
                ),
            ]

            scan = self.checker.scan_episode_health(
                root,
                scanner=lambda *args, **kwargs: results,
            )

            self.assertEqual([finding.episode_id for finding in scan.warnings], ["episode_0001"])
            self.assertEqual([finding.episode_id for finding in scan.serious], ["episode_0002"])

    def test_unrecognized_reject_reasons_fail_closed_as_serious(self):
        structural_reasons = [
            "dds_streams_unexpected:source:extra",
            "dds_stream_not_observed:source.left",
            "dds_sample_count_zero:source.left",
            "dfx_sides_unexpected:extra",
            "dfx_sample_count_zero:left",
            "dfx_accepted_count_zero:right",
            "future_structural_failure",
        ]
        for structural_reason in structural_reasons:
            with self.subTest(reason=structural_reason), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                data_json = _write_episode(root, "episode_0001", {})
                result = SimpleNamespace(
                    episode="episode_0001",
                    data_json=str(data_json),
                    status="reject",
                    reasons=[
                        "max_frame_gap_s=0.100000>0.075000",
                        structural_reason,
                    ],
                    max_frame_gap_s=0.100,
                )

                scan = self.checker.scan_episode_health(
                    root,
                    scanner=lambda *args, **kwargs: [result],
                )

                self.assertEqual(scan.serious, scan.findings)
                self.assertFalse(scan.warnings)

    def test_explicit_xr_checkout_loads_shared_scanner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module_path = root / "teleop" / "utils" / "episode_quality.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_text(
                "def scan_task(path, **kwargs):\n    return []\n",
                encoding="utf-8",
            )

            module = self.checker.load_shared_episode_quality(root)

            self.assertEqual(Path(module.__file__).resolve(), module_path.resolve())
            self.assertEqual(module.scan_task(root), [])


if __name__ == "__main__":
    unittest.main()
