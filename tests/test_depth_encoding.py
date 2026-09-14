from __future__ import annotations

import unittest

import numpy as np

from unitree_lerobot.utils.depth_encoding import (
    CANONICAL_DEPTH_SCALE_M_PER_UNIT,
    DEFAULT_DEPTH_NEAR_M,
    canonicalize_depth_scale_m_per_unit,
    encode_depth_gray_rgb,
)


class DepthEncodingTest(unittest.TestCase):
    def test_production_default_range_remains_025_to_1m(self):
        self.assertEqual(DEFAULT_DEPTH_NEAR_M, 0.25)

    def test_realsense_float32_scale_spelling_canonicalizes_to_exact_constant(self):
        reported = 0.0010000000474974513

        canonical = canonicalize_depth_scale_m_per_unit(reported)

        self.assertIs(canonical, CANONICAL_DEPTH_SCALE_M_PER_UNIT)
        self.assertEqual(canonical, 0.001)

    def test_canonical_scale_does_not_change_encoded_bytes(self):
        generator = np.random.default_rng(7)
        depth = generator.integers(0, 5001, size=(31, 47), dtype=np.uint16)
        reported = 0.0010000000474974513

        reported_bytes = encode_depth_gray_rgb(depth, scale_m_per_unit=reported)
        canonical_bytes = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=canonicalize_depth_scale_m_per_unit(reported),
        )

        np.testing.assert_array_equal(canonical_bytes, reported_bytes)

    def test_canonical_scale_rejects_a_different_sensor_unit(self):
        with self.assertRaisesRegex(ValueError, "canonical 0.001"):
            canonicalize_depth_scale_m_per_unit(0.0005)

    def test_fixed_metric_mapping_and_invalid_value(self):
        depth = np.array([[0, 1, 250, 625, 1000, 2000]], dtype=np.uint16)

        encoded = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=0.001,
            near_m=0.25,
            far_m=1.0,
        )

        expected_gray = np.array([[0, 1, 1, 128, 255, 255]], dtype=np.uint8)
        self.assertEqual(encoded.shape, (1, 6, 3))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue(encoded.flags.c_contiguous)
        for channel in range(3):
            np.testing.assert_array_equal(encoded[..., channel], expected_gray)

    def test_matches_the_previous_converter_expression_byte_for_byte(self):
        generator = np.random.default_rng(42)
        depth = generator.integers(0, 5001, size=(37, 53), dtype=np.uint16)
        depth[::5, ::7] = 0
        scale = 0.001
        near = 0.25
        far = 1.0

        depth_m = depth.astype(np.float32) * scale
        valid = depth != 0
        normalized = np.clip((depth_m - near) / (far - near), 0.0, 1.0)
        legacy_gray = np.zeros_like(depth, dtype=np.uint8)
        legacy_gray[valid] = 1 + np.round(254 * normalized[valid]).astype(np.uint8)
        expected = np.repeat(legacy_gray[..., None], 3, axis=-1)

        actual = encode_depth_gray_rgb(
            depth,
            scale_m_per_unit=scale,
            near_m=near,
            far_m=far,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_sensor_scales_that_represent_the_same_metres_encode_identically(self):
        millimetre_units = np.array([[500, 750]], dtype=np.uint16)
        half_millimetre_units = np.array([[1000, 1500]], dtype=np.uint16)

        millimetre_encoded = encode_depth_gray_rgb(
            millimetre_units,
            scale_m_per_unit=0.001,
        )
        half_millimetre_encoded = encode_depth_gray_rgb(
            half_millimetre_units,
            scale_m_per_unit=0.0005,
        )

        np.testing.assert_array_equal(millimetre_encoded, half_millimetre_encoded)

    def test_rejects_invalid_arrays_and_parameters(self):
        invalid_arrays = (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.uint16),
        )
        for invalid in invalid_arrays:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaisesRegex(ValueError, "HxW uint16"):
                    encode_depth_gray_rgb(invalid, scale_m_per_unit=0.001)

        valid = np.zeros((2, 3), dtype=np.uint16)
        for scale in (0, -0.001, np.inf, np.nan):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    encode_depth_gray_rgb(valid, scale_m_per_unit=scale)
        for near, far in ((1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (0.0, np.inf)):
            with self.subTest(near=near, far=far):
                with self.assertRaises(ValueError):
                    encode_depth_gray_rgb(
                        valid,
                        scale_m_per_unit=0.001,
                        near_m=near,
                        far_m=far,
                    )


if __name__ == "__main__":
    unittest.main()
