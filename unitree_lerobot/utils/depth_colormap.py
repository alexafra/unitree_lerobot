# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned model-side colour maps for encoded depth views."""

from __future__ import annotations

import base64
import hashlib

import numpy as np


FIXED_TURBO_DEPTH_COLORMAP = "fixed_turbo_v1"
FIXED_TURBO_DEPTH_LUT_SHA256 = "sha256:296e8c9e48ca54a9574132044221c8fcb3cdeb666030550d6c7e66ecff4efb5c"

# Effective 256-entry gray-code -> RGB table. Code 0 remains exact black for
# invalid sensor depth. Valid codes 1..255 are spread across the complete
# Matplotlib Turbo reference table using:
#   normalized = (gray - 1) / 254
#   turbo_index = min(floor(normalized * 256), 255)
# Embedding the resulting bytes makes the representation independent of the
# installed Matplotlib/OpenCV version.
_FIXED_TURBO_DEPTH_LUT_BASE64 = (
    "AAAAMBI7MRVCMhhKNBtRNR5YNiFfNyNlOCZsOSlyOix5Oy9/PDKFPDWLPTeRPjqWPz2cQEChQEOmQUWrQUiwQku1"
    "Q066Q1C+Q1PCRFbHRFjLRVvORV7SRWDWRWPZRmbdRmjgRmvjRm3mRnDoRnPrRnXtRnjwRnryRn30Rn/2RoL4RYT5"
    "RYf7RYn8RIz9Q479QpH+QZP+QJb+P5j+Ppv+PJ39O6D8OaL8OKX7Nqj5NKr4M6z2Ma/1L7HzLbTxK7bvKrntKLvr"
    "Jr3pJcDmI8LkIcThIMbfHsncHcvaHM3XG8/UGtHSGdPPGNXMGNfKF9nHF9rEF9zCF96/GOC9GOG6GeO4GuS2G+W0"
    "HeexHuivIOmsIuupJOymJ+2jKe6gLO+dL/CaMvGXNfOUOPSRO/SNP/WKQvaHRveDSviATfl8Ufl5Vfp2WftyXftv"
    "YfxsZfxoaf1lbf1icf1fdP5ceP5ZfP5WgP5ThP5Qh/5Ni/5Ljv5Ikv5Glf5EmP5Cm/1Anv0+pPw7pvs6qfs5rPo3"
    "rvk3sfg2s/g1tvc1ufU0u/Q0vvM0wPIzw/Ezxe8zyO4zyu0zzes0z+o00eg01Oc11uU12OM12uI23eA239424dw3"
    "49o35dg459c46NU46tM57NE57c8578058Ms68sg688Y69MQ69sI698A5+L45+bw5+bo4+rc3+7U3+7M2/LA1/K40"
    "/asz/aky/aYx/aMw/qEv/p4u/pst/pgs/ZUr/ZIp/Y8o/Ywn/Ikm/IYk+4Mj+4Ai+n0g+nof+Xce+HQc93Eb924a"
    "9msY9WgX9GUW82MV8mAU8V0T71oR7lgQ7VUP7FIO6lAN6U0N6EsM5kkL5UYK40QK4kIJ4EAI3j4I3TwH2zoH2TgG"
    "1zYG1jQF1DIF0jAF0C8Ezi0EyysDySkDxygDxSYCwyQCwCMCviECux8BuR4BthwBtBsBsRkBrhgBrBYBqRUBphQB"
    "oxIBoBEBnRABmg4Blw0BlAwBkQsBjgoBiwkBhwgBhAcBgQYCfQUCegQC"
)


def _load_fixed_turbo_depth_lut() -> np.ndarray:
    raw = base64.b64decode(_FIXED_TURBO_DEPTH_LUT_BASE64, validate=True)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != FIXED_TURBO_DEPTH_LUT_SHA256:
        raise RuntimeError(
            f"Pinned fixed-Turbo depth LUT digest mismatch: got {digest}, expected {FIXED_TURBO_DEPTH_LUT_SHA256}"
        )
    lut = np.frombuffer(raw, dtype=np.uint8).reshape(256, 3)
    if not np.array_equal(lut[0], np.zeros(3, dtype=np.uint8)):
        raise RuntimeError("Pinned fixed-Turbo depth LUT must reserve code 0 as black")
    return lut


FIXED_TURBO_DEPTH_LUT = _load_fixed_turbo_depth_lut()


def fixed_turbo_depth_colormap_contract() -> dict[str, object]:
    """Return the serialized contract for the pinned gray-depth lookup."""

    return {
        "name": FIXED_TURBO_DEPTH_COLORMAP,
        "lut_sha256": FIXED_TURBO_DEPTH_LUT_SHA256,
        "input_view": "depth_gray_view",
        "input_encoding": "linear_grayscale_replicated_rgb",
        "invalid_input_value": 0,
        "invalid_output_rgb": [0, 0, 0],
        "valid_input_range": [1, 255],
    }


def apply_fixed_turbo_depth_colormap_array(encoded: np.ndarray) -> np.ndarray:
    """Map replicated-gray depth arrays with arbitrary leading dimensions."""

    encoded = np.asarray(encoded)
    if encoded.dtype != np.uint8 or encoded.ndim < 3 or encoded.shape[-1] != 3:
        raise ValueError(
            "fixed_turbo_v1 expects a uint8 depth array ending in HxWx3; "
            f"got shape={encoded.shape}, dtype={encoded.dtype}"
        )
    gray = encoded[..., 0]
    if not np.array_equal(gray, encoded[..., 1]) or not np.array_equal(gray, encoded[..., 2]):
        raise ValueError("fixed_turbo_v1 expects linear_grayscale_replicated_rgb input with three identical channels")
    return FIXED_TURBO_DEPTH_LUT[gray]
