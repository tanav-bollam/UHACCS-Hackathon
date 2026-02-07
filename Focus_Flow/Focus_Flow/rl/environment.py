"""
Study environment: RL environment for the tutor agent.

Provides discrete state representation and step dynamics for the study session.
"""

from typing import Any, Optional


class StudyEnvironment:
    """
    Environment representing the study session state for RL.

    Actions: continue, take_break, encourage.
    take_break resets attention; continue may decay it over time.
    """

    def __init__(self):
        self._attention_level: float = 1.0
        self._time_in_session_seconds: float = 0.0
        self._productive_ratio: float = 0.0

    def get_state(self) -> dict[str, Any]:
        """
        Return current state for the RL agent.

        Values are raw floats; agent will discretize for tabular Q-learning.
        """
        return {
            "attention_level": self._attention_level,
            "time_in_session_seconds": self._time_in_session_seconds,
            "productive_ratio": self._productive_ratio,
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool]:
        """
        Simulate one step given an action.

        Dynamics:
        - take_break: resets attention to 1.0, small positive reward
        - encourage: boosts attention slightly, small positive reward
        - continue: attention may decay slightly based on session length
        """
        reward = 0.0
        if action == "take_break":
            self._attention_level = 1.0
            reward = 0.1
        elif action == "encourage":
            self._attention_level = min(1.0, self._attention_level + 0.1)
            reward = 0.05
        elif action == "continue":
            decay = 0.01 * (self._time_in_session_seconds / 60.0)
            self._attention_level = max(0.0, self._attention_level - decay * 0.1)
            reward = 0.02 if self._attention_level > 0.5 else -0.02
        done = False
        return self.get_state(), reward, done

    def set_state(
        self,
        attention_level: Optional[float] = None,
        time_in_session_seconds: Optional[float] = None,
        productive_ratio: Optional[float] = None,
    ) -> None:
        """Sync environment with live session metrics."""
        if attention_level is not None:
            self._attention_level = attention_level
        if time_in_session_seconds is not None:
            self._time_in_session_seconds = time_in_session_seconds
        if productive_ratio is not None:
            self._productive_ratio = productive_ratio
