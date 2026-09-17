from __future__ import annotations

import hashlib
import unittest

import numpy as np

from unitree_lerobot.utils.camera_calibration import (
    D435I_254322071415_CALIBRATION,
    calibration_identity,
)
from unitree_lerobot.utils.surface_normal_encoding import (
    DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
    LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
    PinholeIntrinsics,
    REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
    encode_surface_normals_rgb,
    pinhole_intrinsics_from_metadata,
    surface_normals_encoding_metadata,
)


class SurfaceNormalEncodingTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = PinholeIntrinsics(
            width=7,
            height=5,
            fx=100.0,
            fy=100.0,
            cx=3.0,
            cy=2.0,
        )

    def test_flat_plane_is_camera_facing_and_border_is_invalid(self):
        depth = np.full((5, 7), 1000, dtype=np.uint16)

        encoded = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=self.intrinsics,
        )

        self.assertEqual(encoded.shape, (5, 7, 3))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue(encoded.flags.c_contiguous)
        expected = np.broadcast_to(np.array([128, 128, 1], dtype=np.uint8), (3, 5, 3))
        np.testing.assert_array_equal(encoded[1:-1, 1:-1], expected)
        self.assertTrue(np.all(encoded[0] == 0))
        self.assertTrue(np.all(encoded[-1] == 0))
        self.assertTrue(np.all(encoded[:, 0] == 0))
        self.assertTrue(np.all(encoded[:, -1] == 0))

    def test_back_projection_recovers_a_sloped_plane_normal(self):
        intrinsics = PinholeIntrinsics(
            width=9,
            height=7,
            fx=200.0,
            fy=180.0,
            cx=4.0,
            cy=3.0,
        )
        expected_normal = np.array([0.4472136, 0.0, -0.8944272], dtype=np.float32)
        x_over_z = (np.arange(intrinsics.width, dtype=np.float32) - intrinsics.cx) / intrinsics.fx
        plane_offset = -expected_normal[2]
        depth_m = -plane_offset / (expected_normal[0] * x_over_z + expected_normal[2])
        depth_m = np.repeat(depth_m[None, :], intrinsics.height, axis=0)
        depth_u16 = np.rint(depth_m / 2e-5).astype(np.uint16)

        encoded = encode_surface_normals_rgb(
            depth_u16,
            scale_m_per_unit=2e-5,
            intrinsics=intrinsics,
            encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
        )

        decoded_center = (encoded[3, 4].astype(np.float32) - 1.0) / 127.0 - 1.0
        np.testing.assert_allclose(decoded_center, expected_normal, atol=0.015)

    def test_equivalent_metric_depth_scales_encode_identically(self):
        millimetres = np.full((5, 7), 1000, dtype=np.uint16)
        half_millimetres = np.full((5, 7), 2000, dtype=np.uint16)

        first = encode_surface_normals_rgb(
            millimetres,
            scale_m_per_unit=0.001,
            intrinsics=self.intrinsics,
        )
        second = encode_surface_normals_rgb(
            half_millimetres,
            scale_m_per_unit=0.0005,
            intrinsics=self.intrinsics,
        )

        np.testing.assert_array_equal(first, second)

    def test_canonical_scale_spelling_does_not_change_normal_bytes(self):
        depth = np.full((5, 7), 1000, dtype=np.uint16)
        depth[2, :] = np.arange(980, 1050, 10, dtype=np.uint16)

        canonical = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=self.intrinsics,
        )
        sdk_reported = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.0010000000474974513,
            intrinsics=self.intrinsics,
        )

        np.testing.assert_array_equal(canonical, sdk_reported)

    def test_missing_depth_and_discontinuities_are_invalid(self):
        depth = np.full((5, 7), 1000, dtype=np.uint16)
        depth[2, 1] = 0
        depth[2, 4] = 1200

        encoded = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            intrinsics=self.intrinsics,
            max_neighbor_depth_delta_m=0.05,
        )

        # Missing/stepped pixels and centres that depend on them are rejected.
        np.testing.assert_array_equal(encoded[2, 1], [0, 0, 0])
        np.testing.assert_array_equal(encoded[2, 2], [0, 0, 0])
        np.testing.assert_array_equal(encoded[2, 3], [0, 0, 0])
        np.testing.assert_array_equal(encoded[2, 4], [0, 0, 0])
        np.testing.assert_array_equal(encoded[2, 5], [0, 0, 0])
        np.testing.assert_array_equal(encoded[1, 3], [128, 128, 1])

    def test_rejects_invalid_inputs_and_contract_values(self):
        with self.assertRaisesRegex(ValueError, "HxW uint16"):
            encode_surface_normals_rgb(
                np.zeros((5, 7), dtype=np.float32),
                scale_m_per_unit=0.001,
                intrinsics=self.intrinsics,
            )
        with self.assertRaisesRegex(ValueError, "does not match intrinsics"):
            encode_surface_normals_rgb(
                np.zeros((4, 7), dtype=np.uint16),
                scale_m_per_unit=0.001,
                intrinsics=self.intrinsics,
            )
        for scale in (False, np.bool_(True), 0, -0.001, np.inf, np.nan):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    encode_surface_normals_rgb(
                        np.zeros((5, 7), dtype=np.uint16),
                        scale_m_per_unit=scale,
                        intrinsics=self.intrinsics,
                    )
        for discontinuity in (False, np.bool_(True), 0, -0.1, np.inf, np.nan):
            with self.subTest(discontinuity=discontinuity):
                with self.assertRaises(ValueError):
                    encode_surface_normals_rgb(
                        np.zeros((5, 7), dtype=np.uint16),
                        scale_m_per_unit=0.001,
                        intrinsics=self.intrinsics,
                        max_neighbor_depth_delta_m=discontinuity,
                    )
        for range_kwargs in ({"depth_near_m": False}, {"depth_far_m": np.bool_(True)}):
            with self.subTest(range_kwargs=range_kwargs):
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    encode_surface_normals_rgb(
                        np.zeros((5, 7), dtype=np.uint16),
                        scale_m_per_unit=0.001,
                        intrinsics=self.intrinsics,
                        **range_kwargs,
                    )
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    surface_normals_encoding_metadata(**range_kwargs)

    def test_metadata_is_versioned_and_serializes_the_exact_transform(self):
        metadata = surface_normals_encoding_metadata()

        self.assertEqual(metadata["source_key"], "depth_0")
        self.assertEqual(metadata["aligned_to"], "color_0")
        self.assertEqual(metadata["feature_key"], "observation.images.surface_normals_view")
        self.assertEqual(metadata["encoding"], "camera_xyz_uint8")
        self.assertEqual(metadata["encoding_version"], 2)
        self.assertEqual(
            metadata["depth_valid_range_m"],
            {
                "near_m": 0.25,
                "far_m": 1.0,
                "inclusive": True,
                "required_samples": ["center", "left", "right", "up", "down"],
            },
        )
        self.assertEqual(metadata["axis_order"], ["x", "y", "z"])
        self.assertEqual(metadata["orientation"], "camera_facing_dot_normal_point_lte_zero")
        self.assertEqual(metadata["method"], "central_difference_3d")
        self.assertEqual(metadata["invalid_value"], [0, 0, 0])
        self.assertEqual(metadata["valid_component_range"], [1, 255])
        self.assertEqual(
            metadata["intrinsics"],
            {
                "model": "pinhole",
                "width": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.width,
                "height": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.height,
                "fx": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.fx,
                "fy": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.fy,
                "cx": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.cx,
                "cy": DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480.cy,
            },
        )

    def test_new_d435i_intrinsics_are_named_without_replacing_legacy_default(self):
        self.assertNotEqual(
            REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
            DEFAULT_REALSENSE_COLOR_INTRINSICS_640X480,
        )
        self.assertEqual(
            REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
            PinholeIntrinsics(
                width=640,
                height=480,
                fx=609.3858642578125,
                fy=609.4705200195312,
                cx=325.95001220703125,
                cy=247.26507568359375,
            ),
        )

    def test_calibration_tagged_metadata_round_trips_intrinsics_and_identity(self):
        identity = calibration_identity(
            D435I_254322071415_CALIBRATION,
            source="converter.profile.d435i-254322071415",
        )
        metadata = surface_normals_encoding_metadata(
            intrinsics=REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
            camera_calibration=identity,
        )

        self.assertEqual(metadata["camera_calibration"], identity.to_metadata())
        self.assertEqual(
            pinhole_intrinsics_from_metadata(metadata["intrinsics"]),
            REALSENSE_D435I_254322071415_COLOR_INTRINSICS_640X480,
        )

    def test_v2_masks_out_of_range_center_or_neighbor_with_inclusive_boundaries(self):
        expected_normal = np.array([128, 128, 1], dtype=np.uint8)
        for depth_mm in (250, 1_000):
            with self.subTest(boundary_mm=depth_mm):
                depth = np.full((5, 7), depth_mm, dtype=np.uint16)
                encoded = encode_surface_normals_rgb(
                    depth,
                    scale_m_per_unit=0.001,
                    intrinsics=self.intrinsics,
                )
                np.testing.assert_array_equal(encoded[2, 3], expected_normal)

        for row, column, depth_mm in (
            (2, 3, 249),
            (2, 3, 1_001),
            (2, 2, 249),
            (2, 4, 1_001),
            (1, 3, 249),
            (3, 3, 1_001),
        ):
            with self.subTest(row=row, column=column, depth_mm=depth_mm):
                depth = np.full((5, 7), 500, dtype=np.uint16)
                depth[row, column] = depth_mm
                encoded = encode_surface_normals_rgb(
                    depth,
                    scale_m_per_unit=0.001,
                    intrinsics=self.intrinsics,
                    max_neighbor_depth_delta_m=1.0,
                )
                np.testing.assert_array_equal(encoded[2, 3], [0, 0, 0])

    def test_v1_retains_unmasked_bytes_and_metadata(self):
        for depth_mm in (100, 1_200):
            with self.subTest(depth_mm=depth_mm):
                depth = np.full((5, 7), depth_mm, dtype=np.uint16)
                legacy = encode_surface_normals_rgb(
                    depth,
                    scale_m_per_unit=0.001,
                    intrinsics=self.intrinsics,
                    encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
                )
                np.testing.assert_array_equal(legacy[2, 3], [128, 128, 1])

                current = encode_surface_normals_rgb(
                    depth,
                    scale_m_per_unit=0.001,
                    intrinsics=self.intrinsics,
                )
                np.testing.assert_array_equal(current[2, 3], [0, 0, 0])

        metadata = surface_normals_encoding_metadata(
            encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION
        )
        self.assertEqual(metadata["encoding_version"], 1)
        self.assertNotIn("depth_valid_range_m", metadata)

    def test_v1_randomized_bytes_match_the_pre_v2_encoder_fixture(self):
        generator = np.random.default_rng(17092026)
        depth = generator.integers(0, 2_001, size=(480, 640), dtype=np.uint16)

        encoded = encode_surface_normals_rgb(
            depth,
            scale_m_per_unit=0.001,
            encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
        )

        self.assertEqual(
            hashlib.sha256(encoded.tobytes()).hexdigest(),
            "1ac86e8c9a42410db9ee57ae83d1105fc0d28f9f27134482fa48d618ed3931b1",
        )


if __name__ == "__main__":
    unittest.main()
