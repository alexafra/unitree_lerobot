import unittest

import numpy as np

from unitree_lerobot.eval_robot.robot_control.policy_reference import PolicyReferenceSampler


class PolicyReferenceSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = PolicyReferenceSampler()

    def install(self, start=(0.0,), end=(1.0,), starts_at=0.0, period_s=1.0):
        self.sampler.start_segment(
            np.asarray(start), np.asarray(end), starts_at=starts_at, period_s=period_s
        )

    def test_endpoints_and_outside_times_hold_exact_values(self):
        self.install(start=(-2.0, 4.0), end=(3.0, -1.0), starts_at=10.0, period_s=0.5)
        for now in (-100.0, 10.0):
            np.testing.assert_array_equal(self.sampler.sample(now), (-2.0, 4.0))
        for now in (10.5, 100.0):
            np.testing.assert_array_equal(self.sampler.sample(now), (3.0, -1.0))

    def test_30hz_knots_sample_at_100hz_without_frame_lag(self):
        self.install(start=(2.0,), end=(3.0,), period_s=1.0 / 30.0)
        for tick in range(4):
            np.testing.assert_allclose(self.sampler.sample(tick / 100.0), (2.0 + 0.3 * tick,))
        self.install(start=(3.0,), end=(4.0,), starts_at=1.0 / 30.0, period_s=1.0 / 30.0)
        # The first publisher tick after a knot samples its real phase, not an
        # extra full frame at the prior endpoint.
        np.testing.assert_allclose(self.sampler.sample(0.04), (3.2,))
        np.testing.assert_allclose(self.sampler.sample(2.0 / 30.0), (4.0,))

    def test_multidof_direction_flatness_and_no_overshoot(self):
        start = np.array([-2.0, 5.0, 0.4, -0.1])
        end = np.array([4.0, -3.0, 0.4, -0.8])
        self.install(start, end)
        references = np.stack([self.sampler.sample(now) for now in np.linspace(-1.0, 2.0, 301)])
        self.assertTrue(np.all(references >= np.minimum(start, end)))
        self.assertTrue(np.all(references <= np.maximum(start, end)))
        self.assertTrue(np.all(np.diff(references[:, 0]) >= 0.0))
        self.assertTrue(np.all(np.diff(references[:, 1]) <= 0.0))
        np.testing.assert_array_equal(references[:, 2], np.full(301, 0.4))
        np.testing.assert_allclose(self.sampler.sample(0.25), start * 0.75 + end * 0.25)

    def test_constant_segment_holds_final_authorized_reference(self):
        self.install(start=(0.2, -0.8), end=(0.2, -0.8), period_s=1.0 / 30.0)
        for now in (0.0, 0.01, 0.02, 0.03, 100.0):
            np.testing.assert_allclose(self.sampler.sample(now), (0.2, -0.8))

    def test_copies_endpoints_and_returned_references(self):
        plan = np.array([[0.0, 2.0], [1.0, 4.0]])
        self.install(plan[0], plan[1])
        plan[:] = 100.0
        np.testing.assert_array_equal(self.sampler.sample(0.5), (0.5, 3.0))
        for now in (0.0, 0.5, 1.0):
            reference = self.sampler.sample(now)
            reference[:] = -100.0
        np.testing.assert_array_equal(self.sampler.sample(0.5), (0.5, 3.0))

    def test_new_segment_replaces_old_only_on_explicit_install(self):
        self.install(start=(0.0,), end=(1.0,))
        future_start, future_end = np.array([1.0]), np.array([-1.0])
        np.testing.assert_array_equal(self.sampler.sample(0.5), (0.5,))
        self.sampler.start_segment(future_start, future_end, starts_at=1.0, period_s=1.0)
        np.testing.assert_array_equal(self.sampler.sample(1.5), (0.0,))

    def test_reset_requires_a_fresh_segment(self):
        with self.assertRaisesRegex(RuntimeError, "no active segment"):
            self.sampler.sample(0.0)
        self.install()
        self.sampler.reset()
        with self.assertRaisesRegex(RuntimeError, "no active segment"):
            self.sampler.sample(0.5)
        self.install(start=(2.0,), end=(4.0,))
        np.testing.assert_array_equal(self.sampler.sample(0.5), (3.0,))

    def test_invalid_endpoints(self):
        invalid = [[], [[0.0]], [np.nan], [np.inf], [-np.inf], [True], [1j], ["0"]]
        for values in invalid:
            for side in ("start", "end"):
                with self.subTest(values=values, side=side):
                    kwargs = {"start": (0.0,), "end": (1.0,), side: values}
                    with self.assertRaises(ValueError):
                        self.install(**kwargs)
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.install(start=(0.0,), end=(1.0, 2.0))

    def test_invalid_timestamps_and_periods(self):
        invalid_times = [np.nan, np.inf, -np.inf, True, "1", None, np.array([1.0])]
        for value in invalid_times:
            for field in ("starts_at", "period_s"):
                with self.subTest(value=value, field=field):
                    with self.assertRaises(ValueError):
                        self.install(**{field: value})
        for value in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                self.install(period_s=value)
        self.install()
        for value in invalid_times:
            with self.subTest(now=value):
                with self.assertRaises(ValueError):
                    self.sampler.sample(value)

    def test_failed_install_keeps_existing_segment(self):
        self.install(start=(0.0,), end=(4.0,))
        with self.assertRaises(ValueError):
            self.install(start=(10.0,), end=(np.nan,))
        np.testing.assert_array_equal(self.sampler.sample(0.5), (2.0,))

    def test_backward_sample_time_is_clamped_not_stateful(self):
        self.install(starts_at=10.0)
        np.testing.assert_array_equal(self.sampler.sample(10.75), (0.75,))
        np.testing.assert_array_equal(self.sampler.sample(9.0), (0.0,))
        np.testing.assert_array_equal(self.sampler.sample(10.25), (0.25,))

    def test_large_finite_endpoints_do_not_overflow_the_difference(self):
        largest = np.finfo(np.float64).max
        self.install(start=(-largest, largest), end=(largest, largest))
        for now in (0.1, 0.5, 0.9):
            with np.errstate(over="raise", invalid="raise"):
                reference = self.sampler.sample(now)
            self.assertTrue(np.all(np.isfinite(reference)))
            self.assertTrue(np.all(np.abs(reference) <= largest))


if __name__ == "__main__":
    unittest.main()
