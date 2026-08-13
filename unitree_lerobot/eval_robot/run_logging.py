"""Small process-safe logging helpers for G1 deployment runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


LOG_FORMAT = (
    "%(asctime)s.%(msecs)03dZ %(levelname)s process=%(processName)s pid=%(process)d "
    "thread=%(threadName)s %(name)s: %(message)s"
)
TERMINAL_YELLOW = "\x1b[33m"
TERMINAL_RESET = "\x1b[0m"


class UtcFormatter(logging.Formatter):
    converter = __import__("time").gmtime


class TerminalFormatter(UtcFormatter):
    """Add opt-in terminal colour without mutating the shared log record."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if getattr(record, "terminal_yellow", False):
            return f"{TERMINAL_YELLOW}{rendered}{TERMINAL_RESET}"
        return rendered


def create_run_directory(root: str | Path, *, prefix: str = "eval_groot_g1") -> Path:
    """Create a collision-safe per-run directory and return its absolute path."""

    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{prefix}_{stamp}_pid{os.getpid()}"
    for suffix in range(1_000):
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        candidate = root_path / name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a unique run directory below {root_path}")


def configure_process_logging(log_file: str | Path | None) -> None:
    """Configure stderr plus one process-owned UTF-8 log file."""

    terminal_formatter = TerminalFormatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    file_formatter = UtcFormatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(terminal_formatter)
    handlers: list[logging.Handler] = [stream]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    for handler in handlers:
        root.addHandler(handler)


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write strict JSON, never emitting NaN or Infinity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def diagnostic_json_path(run_directory: str | Path, trigger: str) -> Path:
    """Return a collision-safe path for a child timing artifact."""

    directory = Path(run_directory)
    safe_trigger = re.sub(r"[^a-z0-9_-]+", "_", trigger.lower()).strip("_") or "event"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"active_timing_{safe_trigger}_{stamp}_pid{os.getpid()}"
    for suffix in range(1_000):
        name = f"{stem}.json" if suffix == 0 else f"{stem}_{suffix:02d}.json"
        candidate = directory / name
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique timing artifact below {directory}")
