#!/usr/bin/env python3
"""Render the same aligned-depth frame with several fixed metric ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_lerobot.utils.depth_encoding import (  # noqa: E402
    DEFAULT_DEPTH_SCALE_M_PER_UNIT,
    encode_depth_gray_rgb,
)


DEFAULT_NEAR_M = 0.3
DEFAULT_FAR_VALUES_M = (1.0, 1.5, 2.5, 3.0)


def _load_episode_depth(episode_path: Path, frame_index: int) -> tuple[Path, str]:
    data_path = episode_path / "data.json" if episode_path.is_dir() else episode_path
    if data_path.name != "data.json" or not data_path.is_file():
        raise ValueError(f"Expected an episode directory or data.json, got {episode_path}")

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"Episode has no data list: {data_path}")
    matches = [row for row in rows if row.get("idx") == frame_index]
    if len(matches) != 1:
        raise ValueError(
            f"Expected frame idx={frame_index} exactly once in {data_path}; found {len(matches)}"
        )
    relative_depth = matches[0].get("depths", {}).get("depth_0")
    if not isinstance(relative_depth, str) or not relative_depth:
        raise ValueError(f"Frame idx={frame_index} has no depths.depth_0 in {data_path}")
    return data_path.parent / relative_depth, data_path.parent.name


def resolve_depth_source(source: Path, frame_index: int) -> tuple[Path, str]:
    """Resolve a direct PNG or an episode's indexed aligned-depth frame."""

    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".png":
        return source, source.stem
    return _load_episode_depth(source, frame_index)


def _put_label(image: np.ndarray, text: str, y: int, *, scale: float = 0.72) -> None:
    cv2.putText(
        image,
        text,
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 220, 40),
        2,
        cv2.LINE_AA,
    )


def render_comparison(
    depth_u16: np.ndarray,
    *,
    near_m: float = DEFAULT_NEAR_M,
    far_values_m: tuple[float, ...] = DEFAULT_FAR_VALUES_M,
) -> np.ndarray:
    """Return a labelled two-column montage using the production encoder."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Expected HxW uint16 depth, got shape={depth.shape}, dtype={depth.dtype}")
    if not np.isfinite(near_m) or near_m <= 0:
        raise ValueError(f"near_m must be positive and finite, got {near_m!r}")
    if not far_values_m:
        raise ValueError("At least one far distance is required")
    if any(not np.isfinite(far) or far <= near_m for far in far_values_m):
        raise ValueError(f"Every far distance must be finite and greater than {near_m}: {far_values_m}")

    depth_m = depth.astype(np.float32) * np.float32(DEFAULT_DEPTH_SCALE_M_PER_UNIT)
    valid = depth != 0
    valid_count = int(np.count_nonzero(valid))
    total_count = depth.size
    invalid_pct = 100.0 * (total_count - valid_count) / total_count

    tiles: list[np.ndarray] = []
    for far_m in far_values_m:
        encoded_rgb = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=DEFAULT_DEPTH_SCALE_M_PER_UNIT,
            near_m=near_m,
            far_m=far_m,
        )
        encoded_bgr = cv2.cvtColor(encoded_rgb, cv2.COLOR_RGB2BGR)
        header = np.full((78, depth.shape[1], 3), 24, dtype=np.uint8)
        far_pct = (
            100.0 * np.count_nonzero(valid & (depth_m >= far_m)) / valid_count
            if valid_count
            else 0.0
        )
        near_pct = (
            100.0 * np.count_nonzero(valid & (depth_m <= near_m)) / valid_count
            if valid_count
            else 0.0
        )
        _put_label(header, f"{near_m:g} m to {far_m:g} m", 30)
        _put_label(
            header,
            f"black <= near {near_pct:.2f}% | white >= far {far_pct:.2f}% | invalid {invalid_pct:.2f}%",
            62,
            scale=0.43,
        )
        tiles.append(np.vstack((header, encoded_bgr)))

    columns = 2
    rows: list[np.ndarray] = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start : start + columns]
        if len(row_tiles) < columns:
            row_tiles.extend([blank] * (columns - len(row_tiles)))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed metric encodings of one aligned uint16 depth frame. "
            "The processing scale is always the canonical 0.001 m/unit."
        )
    )
    parser.add_argument("source", type=Path, help="Aligned depth PNG, episode directory, or data.json")
    parser.add_argument("--frame-index", type=int, default=0, help="Episode frame idx (default: 0)")
    parser.add_argument("--near-m", type=float, default=DEFAULT_NEAR_M)
    parser.add_argument(
        "--far-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_FAR_VALUES_M),
        help="Far bounds to compare (default: 1.0 1.5 2.5 3.0)",
    )
    parser.add_argument("--output", type=Path, help="Output montage PNG")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV preview window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_index < 0:
        raise SystemExit("--frame-index must be non-negative")
    depth_path, source_name = resolve_depth_source(args.source, args.frame_index)
    depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise SystemExit(f"Could not read aligned depth image: {depth_path}")

    montage = render_comparison(
        depth_u16,
        near_m=args.near_m,
        far_values_m=tuple(args.far_m),
    )
    output = args.output or (
        Path.cwd() / f"{source_name}_depth_ranges_frame_{args.frame_index:06d}.png"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), montage):
        raise SystemExit(f"Could not write comparison image: {output}")
    print(f"source_depth={depth_path}")
    print(f"depth_scale_m_per_unit={DEFAULT_DEPTH_SCALE_M_PER_UNIT}")
    print(f"comparison={output}")

    if args.show:
        cv2.imshow("Depth range comparison", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
