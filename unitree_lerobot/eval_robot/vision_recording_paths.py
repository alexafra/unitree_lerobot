"""Readable model/checkpoint recording folders; no inference or robot dependencies."""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re


def _component(value: object, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not name:
        return fallback
    if len(name) > 240:
        suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
        name = name[:229] + "-" + suffix
    return name


def recording_model_label(model_name: object) -> str:
    """Short display label; the manifest retains the full server model identity."""
    if not isinstance(model_name, str) or not model_name.startswith("inspire_c_"):
        return _component(model_name, "unknown-model")
    architecture = model_name.removeprefix("inspire_c_").split("_patch_", 1)[0]
    if architecture == "rgb":
        label = "rgb"
    else:
        if architecture.startswith(("rgb_surface_normals_", "normals_6ch_")):
            label = "normals"
        elif architecture.startswith("rgbd_turbo_"):
            label = "turbo"
        elif architecture.startswith(("rgbd_", "d1_4ch_")):
            label = "depth"
        else:
            return _component(model_name, "unknown-model")
        for marker, suffix in (
            ("late_fusion_pre_adapter", "latepre"),
            ("late_fusion_post_adapter", "latepost"),
            ("separate_views", "separate"),
            ("early_fusion", "early"),
        ):
            if marker in architecture:
                label += "_" + suffix
                break
        else:
            return _component(model_name, "unknown-model")
    if "_final_" in model_name:
        label += "_final"
    training = re.search(r"(?:^|_)train([0-9]+)eps(?:_|$)", model_name)
    if training:
        label += f"_{training.group(1)}ep"
    return label


def recording_output_path(
    root: Path, model_identity: object, *, now: datetime | None = None
) -> Path:
    """Prepare model/checkpoint parents and choose an unused minute-level run name.

    The recorder still atomically creates the final directory with exist_ok=False;
    a concurrent name collision therefore cannot overwrite another recording.
    Older servers remain usable, but are explicitly labelled unknown.
    """
    identity = model_identity if isinstance(model_identity, dict) else {}
    model = recording_model_label(identity.get("model_name"))
    checkpoint = _component(identity.get("checkpoint"), "unknown-checkpoint")
    parent = Path(root).resolve(strict=True)
    for component in (model, checkpoint):
        parent = parent / component
        if parent.is_symlink():
            raise ValueError(f"Recording parent must not be a symbolic link: {parent}")
        parent.mkdir(mode=0o700, exist_ok=True)
    stamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M")
    for number in range(1, 10000):
        name = stamp if number == 1 else f"run{number:02d}_{stamp}"
        candidate = parent / name
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Too many recordings for this model in minute {stamp}")
