from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "data_editor" / "compare_depth_ranges.py"
SPEC = importlib.util.spec_from_file_location("compare_depth_ranges", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareDepthRangesTest(unittest.TestCase):
    def test_episode_frame_resolves_aligned_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = Path(temp_dir) / "episode_0012"
            depth_path = episode / "depths" / "000007_depth_0.png"
            depth_path.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(depth_path), np.full((4, 5), 700, dtype=np.uint16)))
            (episode / "data.json").write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "idx": 7,
                                "depths": {"depth_0": "depths/000007_depth_0.png"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            resolved, source_name = MODULE.resolve_depth_source(episode, 7)

            self.assertEqual(resolved, depth_path)
            self.assertEqual(source_name, "episode_0012")

    def test_default_montage_compares_four_ranges_at_canonical_scale(self):
        depth = np.full((12, 16), 800, dtype=np.uint16)

        montage = MODULE.render_comparison(depth)

        tile_height = depth.shape[0] + 78
        self.assertEqual(montage.shape, (tile_height * 2, depth.shape[1] * 2, 3))
        # The same 0.8 m surface becomes progressively darker as the far bound
        # grows, which is the comparison this tool is intended to make visible.
        image_means = [
            montage[78:tile_height, : depth.shape[1]].mean(),
            montage[78:tile_height, depth.shape[1] :].mean(),
            montage[tile_height + 78 :, : depth.shape[1]].mean(),
            montage[tile_height + 78 :, depth.shape[1] :].mean(),
        ]
        self.assertGreater(image_means[0], image_means[1])
        self.assertGreater(image_means[1], image_means[2])
        self.assertGreater(image_means[2], image_means[3])
        self.assertEqual(MODULE.DEFAULT_DEPTH_SCALE_M_PER_UNIT, 0.001)
        self.assertEqual(MODULE.DEFAULT_NEAR_M, 0.3)
        self.assertEqual(MODULE.DEFAULT_FAR_VALUES_M, (1.0, 1.5, 2.5, 3.0))

    def test_rejects_non_uint16_depth(self):
        with self.assertRaisesRegex(ValueError, "HxW uint16"):
            MODULE.render_comparison(np.zeros((4, 5), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
