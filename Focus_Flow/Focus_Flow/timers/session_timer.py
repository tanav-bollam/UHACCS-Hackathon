"""
Session timer: Tracks total elapsed time for a study session.

Runs from start() to stop() regardless of attention state.
"""

import time
from typing import Optional


class SessionTimer:
    """
    Tracks total session duration.

    Use start() when session begins and stop() when it ends.
    elapsed_seconds reflects time since start (or total if stopped).
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self) -> None:
        """Start the session timer."""
        self._start_time = time.monotonic()
        self._stop_time = None

    def stop(self) -> None:
        """Stop the session timer."""
        self._stop_time = time.monotonic()

    @property
    def is_running(self) -> bool:
        """True if timer has been started and not stopped."""
        return self._start_time is not None and self._stop_time is None

    def elapsed(self) -> float:
        """
        Return total elapsed seconds since start.

        If stopped, returns duration from start to stop.
        If running, returns duration from start to now.
        """
        return self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        """
        Total elapsed seconds since start.

        If stopped, returns duration from start to stop.
        If running, returns duration from start to now.
        """
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return max(0, end - self._start_time)
