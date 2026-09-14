"""Versioned RealSense calibration records shared by conversion and deployment."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np


REALSENSE_RGBD_CALIBRATION_SCHEMA = "realsense_rgbd_calibration.v1"
RECORDED_CAMERA_CALIBRATION_SOURCE = "episode.info.depth.calibration"
D435I_254322071415_PROFILE = "d435i-254322071415"
D435I_254322071415_PROFILE_SOURCE = f"converter.profile.{D435I_254322071415_PROFILE}"

_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROFILE_FIELDS = {
    "width",
    "height",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion",
    "coeffs",
    "format",
    "fps",
}


@dataclass(frozen=True)
class CameraCalibrationIdentity:
    """Small checkpoint-safe identity for a complete camera calibration."""

    source: str
    schema: str
    model: str
    serial: str
    product_id: str
    firmware: str
    fingerprint: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "schema": self.schema,
            "camera": {
                "model": self.model,
                "serial": self.serial,
                "product_id": self.product_id,
                "firmware": self.firmware,
            },
            "fingerprint": self.fingerprint,
        }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("camera calibration must contain only finite JSON values") from exc
    return encoded.encode("utf-8")


def calibration_fingerprint(calibration_without_fingerprint: Mapping[str, Any]) -> str:
    """Return the canonical v1 fingerprint for a calibration payload."""

    if not isinstance(calibration_without_fingerprint, Mapping):
        raise TypeError("camera calibration payload must be an object")
    if "fingerprint" in calibration_without_fingerprint:
        raise ValueError("fingerprint input must omit the fingerprint field")
    return f"sha256:{hashlib.sha256(_canonical_json(calibration_without_fingerprint)).hexdigest()}"


def with_calibration_fingerprint(calibration_without_fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a calibration payload and append its canonical fingerprint."""

    result = copy.deepcopy(dict(calibration_without_fingerprint))
    result["fingerprint"] = calibration_fingerprint(result)
    return result


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"camera calibration {field} must be a non-empty string")
    return value


def _require_finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"camera calibration {field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"camera calibration {field} must be {qualifier}")
    return result


def _validate_profile(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PROFILE_FIELDS:
        raise ValueError(f"camera calibration {field} must contain exactly {sorted(_PROFILE_FIELDS)!r}")
    for dimension in ("width", "height"):
        item = value[dimension]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"camera calibration {field}.{dimension} must be a positive integer")
    for name in ("fx", "fy"):
        _require_finite_number(value[name], field=f"{field}.{name}", positive=True)
    for name in ("cx", "cy"):
        _require_finite_number(value[name], field=f"{field}.{name}")
    if not (0.0 <= float(value["cx"]) < value["width"]):
        raise ValueError(f"camera calibration {field}.cx must lie inside the image")
    if not (0.0 <= float(value["cy"]) < value["height"]):
        raise ValueError(f"camera calibration {field}.cy must lie inside the image")
    _require_nonempty_string(value["distortion"], field=f"{field}.distortion")
    _require_nonempty_string(value["format"], field=f"{field}.format")
    _require_finite_number(value["fps"], field=f"{field}.fps", positive=True)
    coeffs = value["coeffs"]
    if not isinstance(coeffs, list) or len(coeffs) != 5:
        raise ValueError(f"camera calibration {field}.coeffs must contain exactly 5 numbers")
    for index, coefficient in enumerate(coeffs):
        _require_finite_number(coefficient, field=f"{field}.coeffs[{index}]")
    return value


def validate_realsense_rgbd_calibration(value: Any) -> dict[str, Any]:
    """Validate a full v1 calibration and its content-addressed fingerprint."""

    expected_fields = {
        "schema",
        "camera",
        "color",
        "depth",
        "depth_to_color",
        "fingerprint",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"camera calibration must contain exactly {sorted(expected_fields)!r}")
    if value["schema"] != REALSENSE_RGBD_CALIBRATION_SCHEMA:
        raise ValueError(
            f"unsupported camera calibration schema {value['schema']!r}; expected {REALSENSE_RGBD_CALIBRATION_SCHEMA!r}"
        )
    camera = value["camera"]
    camera_fields = {"model", "serial", "product_id", "firmware"}
    if not isinstance(camera, dict) or set(camera) != camera_fields:
        raise ValueError("camera calibration camera must contain exactly model, serial, product_id, and firmware")
    for field in sorted(camera_fields):
        _require_nonempty_string(camera[field], field=f"camera.{field}")
    _validate_profile(value["color"], field="color")
    _validate_profile(value["depth"], field="depth")

    extrinsics = value["depth_to_color"]
    if not isinstance(extrinsics, dict) or set(extrinsics) != {"rotation", "translation_m"}:
        raise ValueError("camera calibration depth_to_color must contain exactly rotation and translation_m")
    for name, length in (("rotation", 9), ("translation_m", 3)):
        numbers = extrinsics[name]
        if not isinstance(numbers, list) or len(numbers) != length:
            raise ValueError(f"camera calibration depth_to_color.{name} must contain exactly {length} numbers")
        for index, number in enumerate(numbers):
            _require_finite_number(number, field=f"depth_to_color.{name}[{index}]")

    fingerprint = value["fingerprint"]
    if not isinstance(fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("camera calibration fingerprint must be sha256:<64 lowercase hex digits>")
    fingerprint_payload = {key: item for key, item in value.items() if key != "fingerprint"}
    expected_fingerprint = calibration_fingerprint(fingerprint_payload)
    if fingerprint != expected_fingerprint:
        raise ValueError(
            "camera calibration fingerprint does not match its canonical JSON payload: "
            f"got {fingerprint!r}, expected {expected_fingerprint!r}"
        )
    return value


def calibration_identity(
    calibration: Any,
    *,
    source: str = RECORDED_CAMERA_CALIBRATION_SOURCE,
) -> CameraCalibrationIdentity:
    """Extract checkpoint-safe identity/provenance from a full calibration."""

    validated = validate_realsense_rgbd_calibration(calibration)
    provenance = _require_nonempty_string(source, field="source")
    return CameraCalibrationIdentity(
        source=provenance,
        schema=validated["schema"],
        model=validated["camera"]["model"],
        serial=validated["camera"]["serial"],
        product_id=validated["camera"]["product_id"],
        firmware=validated["camera"]["firmware"],
        fingerprint=validated["fingerprint"],
    )


def validate_calibration_identity(value: Any) -> CameraCalibrationIdentity:
    """Parse the compact calibration identity carried by a checkpoint contract."""

    if not isinstance(value, dict) or set(value) != {
        "source",
        "schema",
        "camera",
        "fingerprint",
    }:
        raise ValueError("camera_calibration must contain exactly source, schema, camera, and fingerprint")
    source = _require_nonempty_string(value["source"], field="source")
    if value["schema"] != REALSENSE_RGBD_CALIBRATION_SCHEMA:
        raise ValueError(
            f"unsupported camera_calibration schema {value['schema']!r}; expected {REALSENSE_RGBD_CALIBRATION_SCHEMA!r}"
        )
    camera = value["camera"]
    camera_fields = {"model", "serial", "product_id", "firmware"}
    if not isinstance(camera, dict) or set(camera) != camera_fields:
        raise ValueError("camera_calibration.camera must contain exactly model, serial, product_id, and firmware")
    model = _require_nonempty_string(camera["model"], field="camera.model")
    serial = _require_nonempty_string(camera["serial"], field="camera.serial")
    product_id = _require_nonempty_string(camera["product_id"], field="camera.product_id")
    firmware = _require_nonempty_string(camera["firmware"], field="camera.firmware")
    fingerprint = value["fingerprint"]
    if not isinstance(fingerprint, str) or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("camera_calibration.fingerprint must be sha256:<64 lowercase hex digits>")
    return CameraCalibrationIdentity(
        source=source,
        schema=value["schema"],
        model=model,
        serial=serial,
        product_id=product_id,
        firmware=firmware,
        fingerprint=fingerprint,
    )


_D435I_254322071415_PAYLOAD = {
    "schema": REALSENSE_RGBD_CALIBRATION_SCHEMA,
    "camera": {
        "model": "Intel RealSense D435I",
        "serial": "254322071415",
        "product_id": "0B3A",
        "firmware": "5.15.1.55",
    },
    "color": {
        "width": 640,
        "height": 480,
        "fx": 609.3858642578125,
        "fy": 609.4705200195312,
        "cx": 325.95001220703125,
        "cy": 247.26507568359375,
        "distortion": "distortion.inverse_brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "format": "bgr8",
        "fps": 30,
    },
    "depth": {
        "width": 640,
        "height": 480,
        "fx": 397.5912170410156,
        "fy": 397.5912170410156,
        "cx": 315.6465148925781,
        "cy": 244.2028350830078,
        "distortion": "distortion.brown_conrady",
        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "format": "z16",
        "fps": 30,
    },
    "depth_to_color": {
        "rotation": [
            0.9999486207962036,
            0.0033009203616529703,
            0.009585974738001823,
            -0.0033925294410437346,
            0.9999485611915588,
            0.009556086733937263,
            -0.009553938172757626,
            -0.009588115848600864,
            0.9999083876609802,
        ],
        "translation_m": [
            0.014800711534917355,
            0.0008831368759274483,
            0.0007359444862231612,
        ],
    },
}

D435I_254322071415_CALIBRATION = with_calibration_fingerprint(_D435I_254322071415_PAYLOAD)
validate_realsense_rgbd_calibration(D435I_254322071415_CALIBRATION)

NAMED_CAMERA_CALIBRATIONS = {
    D435I_254322071415_PROFILE: D435I_254322071415_CALIBRATION,
}
