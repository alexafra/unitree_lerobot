#!/usr/bin/env python3
"""Read-only episode health report for a selected ``processed_raw`` folder.

The quality classification is owned by ``xr_teleoperate``.  This adapter keeps
the data editor and a small command-line report on that single implementation,
then adds concise, evidence-specific wording for humans.  It never writes to
the selected data directory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence


DEFAULT_MAX_FRAME_GAP_S = 0.075
DEFAULT_MIN_MEASURED_FPS = 29.0
DEFAULT_SERIOUS_GAP_S = 0.250
DEFAULT_SERIOUS_MIN_FPS = 27.0
CAMERA_EVIDENCE_NOTE = (
    "These are recorded frame-loop and valid DDS state-callback timing checks. "
    "They do not prove a camera-sensor drop, late exposure, or repeated image; "
    "that requires a camera capture sequence or hardware timestamp."
)
STORAGE_NOTE = (
    "Evidence is read from each episode's data.json: data[*].timestamp_s and "
    "timing describe the recorder frame loop; diagnostics/*_state_subscribers "
    "describe valid DDS callback inter-arrival gaps."
)
ATOMIC_RGBD_STORAGE_NOTE = (
    "Opt-in atomic RGB-D recordings additionally store the provenance contract "
    "at info.rgbd_pairing and per-row capture_sequence, server/client monotonic "
    "timestamps, and sample_held evidence at data[*].rgbd_pairing. Legacy "
    "recordings omit that per-frame camera provenance."
)
SEVERITY_NOTE = (
    "Orange means a fully evidenced, recovered timing delay below 250 ms (or recorder-loop "
    "rate from 27 to 29 FPS). Red means structural/incomplete evidence, an unrecovered/DFX "
    "anomaly, a gap of at least 250 ms, or recorder-loop rate below 27 FPS."
)

_EPISODE_NAME = re.compile(r"^episode_(\d+)$")
_FRAME_GAP_REASON = re.compile(r"^max_frame_gap_s=(?P<observed>[0-9.eE+-]+)>(?P<limit>[0-9.eE+-]+)$")
_FPS_REASON = re.compile(r"^measured_fps=(?P<observed>[0-9.eE+-]+)<(?P<limit>[0-9.eE+-]+)$")
_SHARED_MODULE_NAME = "_xr_teleoperate_episode_quality_for_data_editor"


class EpisodeHealthCheckError(RuntimeError):
    """The shared, read-only quality checker could not be loaded or run."""


@dataclass(frozen=True)
class EpisodeHealthFinding:
    episode_id: str
    classification: str
    reasons: tuple[str, ...]
    evidence_incomplete: bool = False
    severity: str = "warning"


@dataclass(frozen=True)
class EpisodeHealthScan:
    selected_path: str
    total_episode_count: int
    clean_count: int
    findings: tuple[EpisodeHealthFinding, ...]
    checker_source: str

    @property
    def possibly_problematic(self) -> tuple[EpisodeHealthFinding, ...]:
        return tuple(finding for finding in self.findings if finding.classification == "possibly_problematic")

    @property
    def unverified(self) -> tuple[EpisodeHealthFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.classification == "unverified" or finding.evidence_incomplete
        )

    @property
    def warnings(self) -> tuple[EpisodeHealthFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "warning")

    @property
    def serious(self) -> tuple[EpisodeHealthFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "serious")


def _module_file_from_root(root: Path) -> Path:
    root = Path(root).expanduser()
    if root.name == "episode_quality.py":
        return root
    return root / "teleop" / "utils" / "episode_quality.py"


def load_shared_episode_quality(explicit_root: Path | None = None) -> ModuleType:
    """Load ``teleop.utils.episode_quality`` without copying its implementation."""

    if explicit_root is None:
        try:
            installed_module = importlib.import_module("teleop.utils.episode_quality")
        except ModuleNotFoundError as error:
            if error.name not in {"teleop", "teleop.utils", "teleop.utils.episode_quality"}:
                raise EpisodeHealthCheckError(f"xr_teleoperate quality checker import failed: {error}") from error
        except Exception as error:
            raise EpisodeHealthCheckError(f"xr_teleoperate quality checker import failed: {error}") from error
        else:
            if not callable(getattr(installed_module, "scan_task", None)):
                raise EpisodeHealthCheckError("installed teleop.utils.episode_quality has no scan_task()")
            return installed_module

    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(_module_file_from_root(explicit_root))
    configured_root = os.environ.get("XR_TELEOPERATE_ROOT")
    if configured_root:
        candidates.append(_module_file_from_root(Path(configured_root)))
    # Normal development layout: Development/unitree_lerobot and
    # Development/xr_teleoperate are sibling repositories.
    candidates.append(
        Path(__file__).resolve().parents[2] / "xr_teleoperate" / "teleop" / "utils" / "episode_quality.py"
    )

    for module_file in candidates:
        if not module_file.is_file():
            continue
        resolved = module_file.resolve()
        cached = sys.modules.get(_SHARED_MODULE_NAME)
        if cached is not None and Path(getattr(cached, "__file__", "")).resolve() == resolved:
            return cached
        spec = importlib.util.spec_from_file_location(_SHARED_MODULE_NAME, resolved)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_SHARED_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            sys.modules.pop(_SHARED_MODULE_NAME, None)
            raise EpisodeHealthCheckError(
                f"could not load xr_teleoperate quality checker at {resolved}: {error}"
            ) from error
        if not callable(getattr(module, "scan_task", None)):
            sys.modules.pop(_SHARED_MODULE_NAME, None)
            raise EpisodeHealthCheckError(f"xr_teleoperate quality checker has no scan_task(): {resolved}")
        return module

    searched = ", ".join(str(path) for path in candidates)
    raise EpisodeHealthCheckError(
        "teleop.utils.episode_quality is unavailable. Install xr_teleoperate, "
        "set XR_TELEOPERATE_ROOT, or pass --xr-teleoperate-root. "
        f"Searched: {searched}"
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_document(path: Any) -> dict[str, Any] | None:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None
    return document if isinstance(document, dict) else None


def _recorded_frame_gap_reason(
    document: dict[str, Any] | None,
    observed_s: float,
    limit_s: float,
) -> str:
    frame_pair = ""
    frames = document.get("data") if document is not None else None
    if isinstance(frames, list):
        previous: tuple[Any, float] | None = None
        largest: tuple[float, Any, Any] | None = None
        for frame in frames:
            if not isinstance(frame, dict):
                previous = None
                continue
            timestamp = _finite_number(frame.get("timestamp_s"))
            if timestamp is None:
                previous = None
                continue
            current = (frame.get("idx"), timestamp)
            if previous is not None:
                gap = timestamp - previous[1]
                if largest is None or gap > largest[0]:
                    largest = (gap, previous[0], current[0])
            previous = current
        if largest is not None and largest[1] is not None and largest[2] is not None:
            frame_pair = f" between frame IDs {largest[1]} and {largest[2]}"
    return f"recorded frame-loop gap: {observed_s * 1000:.3f} ms{frame_pair} exceeds the {limit_s * 1000:.3f} ms limit"


def _dds_callback_reasons(document: dict[str, Any] | None) -> list[str]:
    if document is None or not isinstance(document.get("diagnostics"), dict):
        return []
    details = []
    for source_name, source in document["diagnostics"].items():
        if not isinstance(source, dict) or source.get("metric") != "valid_state_receive_gap":
            continue
        threshold_s = _finite_number(source.get("gap_threshold_s"))
        for stream_name, stream in source.items():
            if not isinstance(stream, dict):
                continue
            gap_count = stream.get("gap_count")
            if isinstance(gap_count, bool) or not isinstance(gap_count, int) or gap_count <= 0:
                continue
            topic = stream.get("topic")
            stream_label = str(stream_name)
            if isinstance(topic, str) and topic:
                stream_label += f" ({topic})"
            max_gap_s = _finite_number(stream.get("max_gap_duration_s"))
            timing = f"; max {max_gap_s * 1000:.3f} ms" if max_gap_s is not None else ""
            if threshold_s is not None:
                timing += f"; stored threshold >{threshold_s * 1000:.3f} ms"
            recovered_count = stream.get("recovered_gap_count")
            recovery = ""
            if isinstance(recovered_count, int) and not isinstance(recovered_count, bool):
                recovery = f"; {recovered_count} recovered"
            if stream.get("open_gap_at_end") is True:
                recovery += "; a gap remained open at episode end"
            elif stream.get("open_gap_at_end") is False:
                recovery += "; none open at episode end"
            details.append(
                f"DDS valid-state callback gap: {source_name}.{stream_label} recorded "
                f"{gap_count} event{'s' if gap_count != 1 else ''}{timing}{recovery}"
            )
    return details


def _format_result_reasons(result: Any) -> tuple[str, ...]:
    raw_reasons = [str(reason) for reason in getattr(result, "reasons", ())]
    document = _read_document(getattr(result, "data_json", None))
    formatted: list[str] = []
    saw_dds_gap = False

    for reason in raw_reasons:
        frame_gap = _FRAME_GAP_REASON.fullmatch(reason)
        if frame_gap:
            formatted.append(
                _recorded_frame_gap_reason(
                    document,
                    float(frame_gap.group("observed")),
                    float(frame_gap.group("limit")),
                )
            )
            continue
        fps = _FPS_REASON.fullmatch(reason)
        if fps:
            formatted.append(
                "recorded frame-loop average: "
                f"{float(fps.group('observed')):.3f} FPS is below the "
                f"{float(fps.group('limit')):.3f} FPS limit"
            )
            continue
        if reason.startswith("dds_gap:"):
            saw_dds_gap = True
            continue
        if reason.startswith("dfx:"):
            formatted.append(f"DFX hand lost-counter diagnostic: {reason.removeprefix('dfx:')}")
            continue
        if reason == "all_required_metrics_within_limits":
            continue
        formatted.append(f"quality evidence: {reason}")

    if saw_dds_gap:
        dds_reasons = _dds_callback_reasons(document)
        formatted.extend(
            dds_reasons
            or ["DDS valid-state callback gap: " + reason for reason in raw_reasons if reason.startswith("dds_gap:")]
        )

    # Keep the report stable if several aggregate counters describe the same event.
    return tuple(dict.fromkeys(formatted))


def _has_incomplete_evidence(raw_reasons: Sequence[Any]) -> bool:
    markers = (
        "missing",
        "invalid",
        "unavailable",
        "unknown",
        "unsupported",
        "unreadable",
        "malformed",
        "not_object",
    )
    return any(marker in str(reason).lower() for reason in raw_reasons for marker in markers)


def _dds_gap_is_serious(
    document: dict[str, Any] | None,
    *,
    serious_gap_s: float,
) -> bool:
    """Return whether a recorded DDS gap is unrecovered, unverifiable, or large."""

    if document is None or not isinstance(document.get("diagnostics"), dict):
        return True
    found_positive_gap = False
    for source in document["diagnostics"].values():
        if not isinstance(source, dict) or source.get("metric") != "valid_state_receive_gap":
            continue
        for stream in source.values():
            if not isinstance(stream, dict):
                continue
            gap_count = stream.get("gap_count")
            if isinstance(gap_count, bool) or not isinstance(gap_count, int) or gap_count <= 0:
                continue
            found_positive_gap = True
            max_gap_s = _finite_number(stream.get("max_gap_duration_s"))
            if max_gap_s is None or max_gap_s >= serious_gap_s:
                return True
            recovered_count = stream.get("recovered_gap_count")
            if (
                isinstance(recovered_count, bool)
                or not isinstance(recovered_count, int)
                or recovered_count < gap_count
            ):
                return True
            if stream.get("open_gap_at_end") is not False:
                return True
            events = stream.get("gaps")
            if not isinstance(events, list) or len(events) != gap_count:
                return True
            if any(not isinstance(event, dict) or event.get("recovered") is not True for event in events):
                return True
    return not found_positive_gap


def _finding_severity(
    result: Any,
    document: dict[str, Any] | None,
    *,
    serious_gap_s: float,
    serious_min_fps: float,
) -> str:
    """Map shared evidence to a display severity without changing its verdict."""

    status = str(getattr(result, "status", "unknown"))
    raw_reasons = tuple(str(reason) for reason in getattr(result, "reasons", ()))
    if status != "reject" or _has_incomplete_evidence(raw_reasons):
        return "serious"

    exact_frame_gap_s = _finite_number(getattr(result, "max_frame_gap_s", None))
    exact_measured_fps = _finite_number(getattr(result, "measured_fps", None))
    saw_dds_gap = False
    for reason in raw_reasons:
        frame_gap = _FRAME_GAP_REASON.fullmatch(reason)
        if frame_gap:
            observed_frame_gap_s = (
                exact_frame_gap_s
                if exact_frame_gap_s is not None
                else float(frame_gap.group("observed"))
            )
            if observed_frame_gap_s >= serious_gap_s:
                return "serious"
            continue
        fps = _FPS_REASON.fullmatch(reason)
        if fps:
            observed_measured_fps = (
                exact_measured_fps
                if exact_measured_fps is not None
                else float(fps.group("observed"))
            )
            if observed_measured_fps < serious_min_fps:
                return "serious"
            continue
        if reason.startswith("dds_gap:"):
            saw_dds_gap = True
            continue
        # Only the three explicitly understood timing findings above can be
        # orange. Fail closed when the shared checker adds a new reject reason,
        # or when it reports any structural/DFX inconsistency.
        return "serious"

    if not raw_reasons or (
        saw_dds_gap and _dds_gap_is_serious(document, serious_gap_s=serious_gap_s)
    ):
        return "serious"
    return "warning"


def _episode_sort_key(episode_id: str) -> tuple[int, int | str]:
    match = _EPISODE_NAME.fullmatch(episode_id)
    if match:
        return 0, int(match.group(1))
    return 1, episode_id


def _missing_episode_data_json(selected_path: Path) -> list[Path]:
    if not selected_path.is_dir():
        return []
    if _EPISODE_NAME.fullmatch(selected_path.name):
        return [] if (selected_path / "data.json").is_file() else [selected_path]
    episode_dirs = [
        child for child in selected_path.iterdir() if child.is_dir() and _EPISODE_NAME.fullmatch(child.name)
    ]
    return sorted(
        (child for child in episode_dirs if not (child / "data.json").is_file()),
        key=lambda child: _episode_sort_key(child.name),
    )


def scan_episode_health(
    selected_path: Path | str,
    *,
    max_frame_gap_s: float = DEFAULT_MAX_FRAME_GAP_S,
    min_measured_fps: float = DEFAULT_MIN_MEASURED_FPS,
    serious_gap_s: float = DEFAULT_SERIOUS_GAP_S,
    serious_min_fps: float = DEFAULT_SERIOUS_MIN_FPS,
    xr_teleoperate_root: Path | None = None,
    scanner: Callable[..., Sequence[Any]] | None = None,
) -> EpisodeHealthScan:
    """Classify episodes without modifying the selected folder or its contents."""

    path = Path(selected_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"selected episode folder does not exist: {path}")
    if not math.isfinite(max_frame_gap_s) or max_frame_gap_s < 0:
        raise ValueError("max_frame_gap_s must be finite and non-negative")
    if not math.isfinite(min_measured_fps) or min_measured_fps < 0:
        raise ValueError("min_measured_fps must be finite and non-negative")
    if not math.isfinite(serious_gap_s) or serious_gap_s < max_frame_gap_s:
        raise ValueError("serious_gap_s must be finite and at least max_frame_gap_s")
    if not math.isfinite(serious_min_fps) or not 0 <= serious_min_fps <= min_measured_fps:
        raise ValueError("serious_min_fps must be finite and between zero and min_measured_fps")

    if scanner is None:
        quality_module = load_shared_episode_quality(xr_teleoperate_root)
        scanner = quality_module.scan_task
        module_file = getattr(quality_module, "__file__", None)
        checker_source = str(Path(module_file).resolve()) if module_file is not None else quality_module.__name__
    else:
        checker_source = getattr(scanner, "__module__", "injected scanner")

    try:
        results = list(
            scanner(
                path,
                max_frame_gap_s=max_frame_gap_s,
                min_measured_fps=min_measured_fps,
            )
        )
    except (OSError, ValueError):
        raise
    except Exception as error:
        raise EpisodeHealthCheckError(f"shared xr_teleoperate quality scan failed: {error}") from error
    missing_episode_dirs = _missing_episode_data_json(path)
    if not results and not missing_episode_dirs:
        raise ValueError(f"no episode_*/data.json files or episode directories found at {path}")
    findings: list[EpisodeHealthFinding] = []
    clean_count = 0
    for result in results:
        status = str(getattr(result, "status", "unknown"))
        episode_id = str(getattr(result, "episode", "unknown_episode"))
        if status == "clean":
            clean_count += 1
            continue
        classification = "possibly_problematic" if status == "reject" else "unverified"
        reasons = _format_result_reasons(result)
        if not reasons:
            reasons = (f"quality evidence: shared checker status={status}",)
        findings.append(
            EpisodeHealthFinding(
                episode_id,
                classification,
                reasons,
                evidence_incomplete=(status != "clean" and _has_incomplete_evidence(getattr(result, "reasons", ()))),
                severity=_finding_severity(
                    result,
                    _read_document(getattr(result, "data_json", None)),
                    serious_gap_s=serious_gap_s,
                    serious_min_fps=serious_min_fps,
                ),
            )
        )

    for episode_dir in missing_episode_dirs:
        findings.append(
            EpisodeHealthFinding(
                episode_dir.name,
                "unverified",
                (f"quality evidence: data.json is missing from {episode_dir}",),
                evidence_incomplete=True,
                severity="serious",
            )
        )

    findings.sort(key=lambda finding: _episode_sort_key(finding.episode_id))
    return EpisodeHealthScan(
        selected_path=str(path.resolve()),
        total_episode_count=len(results) + len(missing_episode_dirs),
        clean_count=clean_count,
        findings=tuple(findings),
        checker_source=checker_source,
    )


def _compact_episode_ids(findings: Sequence[EpisodeHealthFinding], limit: int = 8) -> str:
    ids = [finding.episode_id for finding in findings]
    if len(ids) <= limit:
        return ", ".join(ids)
    return ", ".join(ids[:limit]) + f", +{len(ids) - limit} more"


def health_header_text(scan: EpisodeHealthScan) -> str:
    """Return explicit status text for the editor's top health banner."""

    parts = []
    if scan.serious:
        parts.append(
            f"{len(scan.serious)} serious/structural issue(s): " + _compact_episode_ids(scan.serious)
        )
    if scan.warnings:
        parts.append(
            f"{len(scan.warnings)} recovered timing warning(s): " + _compact_episode_ids(scan.warnings)
        )
    if not parts:
        return (
            f"✓ Episode health — {scan.total_episode_count} checked; no recorder-loop/DDS "
            "warnings or missing required evidence. Camera-sensor delivery is not verified by these checks."
        )
    symbol = "⛔" if scan.serious else "⚠"
    return (
        f"{symbol} Episode health — " + "; ".join(parts) + ". Open View Health Details for objective reasons. "
        "Recorder-loop/DDS timing is not proof of a camera-sensor drop."
    )


def render_report(scan: EpisodeHealthScan) -> str:
    lines = [
        f"READ-ONLY EPISODE HEALTH: {scan.selected_path}",
        (
            f"Scanned {scan.total_episode_count}: {scan.clean_count} clean, "
            f"{len(scan.warnings)} orange timing warnings, "
            f"{len(scan.serious)} red serious/structural issues, "
            f"{len(scan.unverified)} with incomplete evidence"
        ),
        STORAGE_NOTE,
        ATOMIC_RGBD_STORAGE_NOTE,
        SEVERITY_NOTE,
        f"Shared classifier: {scan.checker_source}",
    ]
    if not scan.findings:
        lines.append("No measured problems or missing required evidence were found.")
    for finding in scan.findings:
        label = "RED SERIOUS" if finding.severity == "serious" else "ORANGE WARNING"
        if finding.classification == "unverified" or finding.evidence_incomplete:
            label += "; EVIDENCE INCOMPLETE"
        lines.append(f"{finding.episode_id} [{label}]")
        lines.extend(f"  - {reason}" for reason in finding.reasons)
    lines.append(CAMERA_EVIDENCE_NOTE)
    lines.append("No data was deleted, moved, renamed, or edited.")
    return "\n".join(lines)


def _nonnegative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a processed_raw episode folder and list measured health warnings "
            "or missing evidence. This command never changes episode data."
        )
    )
    parser.add_argument(
        "episode_folder",
        type=Path,
        help="Folder containing episode_*/data.json, one episode folder, or one data.json.",
    )
    parser.add_argument(
        "--max-frame-gap-s",
        type=_nonnegative_float,
        default=DEFAULT_MAX_FRAME_GAP_S,
        help=f"Warn above this recorded frame-loop gap (default: {DEFAULT_MAX_FRAME_GAP_S}).",
    )
    parser.add_argument(
        "--min-measured-fps",
        type=_nonnegative_float,
        default=DEFAULT_MIN_MEASURED_FPS,
        help=f"Warn below this recorder-loop frame rate (default: {DEFAULT_MIN_MEASURED_FPS}).",
    )
    parser.add_argument(
        "--serious-gap-s",
        type=_nonnegative_float,
        default=DEFAULT_SERIOUS_GAP_S,
        help=f"Mark frame/DDS gaps at or above this value red (default: {DEFAULT_SERIOUS_GAP_S}).",
    )
    parser.add_argument(
        "--serious-min-fps",
        type=_nonnegative_float,
        default=DEFAULT_SERIOUS_MIN_FPS,
        help=f"Mark recorder-loop rates below this value red (default: {DEFAULT_SERIOUS_MIN_FPS}).",
    )
    parser.add_argument(
        "--xr-teleoperate-root",
        type=Path,
        help="xr_teleoperate checkout root if it is not installed or next to this repository.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    scanner: Callable[..., Sequence[Any]] | None = None,
) -> int:
    args = parse_args(argv)
    try:
        scan = scan_episode_health(
            args.episode_folder,
            max_frame_gap_s=args.max_frame_gap_s,
            min_measured_fps=args.min_measured_fps,
            serious_gap_s=args.serious_gap_s,
            serious_min_fps=args.serious_min_fps,
            xr_teleoperate_root=args.xr_teleoperate_root,
            scanner=scanner,
        )
    except (OSError, ValueError, EpisodeHealthCheckError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render_report(scan))
    return 1 if scan.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
