from __future__ import annotations

import copy

import pytest

from unitree_lerobot.utils.camera_calibration import (
    D435I_254322071415_CALIBRATION,
    RECORDED_CAMERA_CALIBRATION_SOURCE,
    calibration_identity,
    validate_calibration_identity,
    validate_realsense_rgbd_calibration,
)


EXPECTED_FINGERPRINT = "sha256:f7860e3e2be34af131e214c74d417217c889eff3a069dbec543be3fba15b027b"


def test_replacement_d435i_profile_has_pinned_identity_and_fingerprint():
    calibration = validate_realsense_rgbd_calibration(D435I_254322071415_CALIBRATION)

    assert calibration["fingerprint"] == EXPECTED_FINGERPRINT
    assert calibration["camera"] == {
        "model": "Intel RealSense D435I",
        "serial": "254322071415",
        "product_id": "0B3A",
        "firmware": "5.15.1.55",
    }
    assert calibration["color"]["fx"] == 609.3858642578125
    assert calibration["color"]["fy"] == 609.4705200195312


def test_compact_identity_round_trips_all_provenance_fields():
    expected = calibration_identity(
        D435I_254322071415_CALIBRATION,
        source=RECORDED_CAMERA_CALIBRATION_SOURCE,
    )

    assert validate_calibration_identity(expected.to_metadata()) == expected


def test_full_calibration_rejects_payload_tampering():
    tampered = copy.deepcopy(D435I_254322071415_CALIBRATION)
    tampered["color"]["fx"] += 1.0

    with pytest.raises(ValueError, match="fingerprint does not match"):
        validate_realsense_rgbd_calibration(tampered)


def test_compact_identity_rejects_missing_firmware():
    identity = calibration_identity(D435I_254322071415_CALIBRATION).to_metadata()
    del identity["camera"]["firmware"]

    with pytest.raises(ValueError, match="product_id, and firmware"):
        validate_calibration_identity(identity)
