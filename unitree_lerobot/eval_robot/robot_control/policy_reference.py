"""Pure timestamped policy-reference sampling, before command conditioning.

This helper does not schedule actions, validate robot limits, or publish commands.
The caller owns the action clock, permitted lookahead, and final safety checks.
An active segment owns copies of its endpoints so replacing a future plan cannot
change an interval that is already being sampled.
"""

from __future__ import annotations

import numpy as np


def _finite_time(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _endpoint(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional real numeric array")
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


class PolicyReferenceSampler:
    """Sample one immutable linear segment on the caller's action timeline.

    Before the segment starts, return its first endpoint; after its period,
    hold its final endpoint. Sampling does not advance or otherwise mutate the
    segment, and does not require calls at a particular frequency.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard all previously planned references."""
        self._start: np.ndarray | None = None
        self._end: np.ndarray | None = None
        self._starts_at: float | None = None
        self._period_s: float | None = None

    def start_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        starts_at: float,
        period_s: float,
    ) -> None:
        """Install validated endpoint copies without modifying a failed install."""
        candidate_start = _endpoint(start, "start")
        candidate_end = _endpoint(end, "end")
        if candidate_start.shape != candidate_end.shape:
            raise ValueError("start and end must have the same shape")
        candidate_starts_at = _finite_time(starts_at, "starts_at")
        candidate_period_s = _finite_time(period_s, "period_s")
        if candidate_period_s <= 0.0:
            raise ValueError("period_s must be greater than zero")
        self._start = candidate_start
        self._end = candidate_end
        self._starts_at = candidate_starts_at
        self._period_s = candidate_period_s

    def sample(self, now: float) -> np.ndarray:
        """Return an independent reference array, clipped to the endpoint hull."""
        if self._start is None or self._end is None:
            raise RuntimeError("Policy reference sampler has no active segment")
        now = _finite_time(now, "now")
        assert self._starts_at is not None and self._period_s is not None
        if now <= self._starts_at:
            return self._start.copy()
        elapsed = now - self._starts_at
        if elapsed >= self._period_s:
            return self._end.copy()
        alpha = elapsed / self._period_s
        # A convex sum avoids overflowing end - start for opposite-sign finite
        # endpoints. The clip also bounds any floating-point roundoff.
        reference = (1.0 - alpha) * self._start + alpha * self._end
        return np.clip(
            reference,
            np.minimum(self._start, self._end),
            np.maximum(self._start, self._end),
        )
