"""
Productivity timer: Tracks time counted as productive (attentive).

Only advances when update(is_attentive=True) is called. Used together
with SessionTimer to compute productivity_score = productive / total.

Usage: Call update(is_attentive(frame)) periodically from the app,
e.g. in a background task that captures frames and runs attention detection.
"""

import time
from typing import Optional


class ProductivityTimer:
    """
    Tracks productive seconds - time during which the user was attentive.

    Call update(is_attentive) periodically (e.g. every frame or poll interval).
    productive_seconds increases only when is_attentive is True.
    total_elapsed mirrors the wall-clock time since start (for ratio calculation).
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._productive_seconds: float = 0.0
        self._last_update_time: Optional[float] = None

    def start(self) -> None:
        """Start the productivity timer."""
        now = time.monotonic()
        self._start_time = now
        self._last_update_time = now
        self._stop_time = None
        self._productive_seconds = 0.0

    def stop(self) -> None:
        """Stop the productivity timer."""
        self._stop_time = time.monotonic()

    def update(self, is_attentive: bool) -> None:
        """
        Update productive time based on current attention state.

        Call this periodically (e.g. every 0.5s) with update(is_attentive(frame)).
        If is_attentive is True, the elapsed time since the last update
        is added to productive_seconds.
        """
        if self._start_time is None or self._stop_time is not None:
            return
        now = time.monotonic()
        delta = now - (self._last_update_time or now)
        self._last_update_time = now
        if is_attentive and delta > 0:
            self._productive_seconds += delta

    @property
    def productive_seconds(self) -> float:
        """Seconds counted as productive (user was attentive)."""
        return self._productive_seconds

    @property
    def total_elapsed(self) -> float:
        """
        Total wall-clock seconds since start.

        If stopped, returns duration from start to stop.
        If running, returns duration from start to now.
        """
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return max(0, end - self._start_time)
