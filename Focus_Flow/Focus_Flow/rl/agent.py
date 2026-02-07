"""
RL Agent: Tabular reinforcement learning agent for tutor recommendations.

Q-learning with discrete state space. Selects actions (continue, take_break,
encourage) based on study state.
"""

import random
from typing import Any, Optional

ACTIONS = ["continue", "take_break", "encourage"]


def _bin(value: float, bins: tuple[float, ...]) -> int:
    """Bin a float into discrete index (0, 1, 2, ...)."""
    for i, bound in enumerate(bins):
        if value < bound:
            return i
    return len(bins)


def _state_key(state: dict[str, Any]) -> str:
    """
    Discretize state dict into a hashable string key for tabular Q-learning.

    Bins: attention_level (3), time_in_session (3), productive_ratio (3).
    """
    att = state.get("attention_level", 0.5)
    time_s = state.get("time_in_session_seconds", 0)
    prod = state.get("productive_ratio", 0.5)
    b_att = _bin(att, (0.33, 0.66))
    b_time = _bin(time_s / 3600.0, (0.25, 0.5))
    b_prod = _bin(prod, (0.33, 0.66))
    return f"{b_att}_{b_time}_{b_prod}"


class RLAgent:
    """
    Tabular Q-learning agent for tutor recommendations.

    Uses epsilon-greedy exploration. State is discretized into bins.
    TODO: Persist Q-table for learning across sessions.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self._q: dict[str, dict[str, float]] = {}

    def _get_q(self, state_key: str, action: str) -> float:
        """Get Q(s,a), initializing to 0 if unseen."""
        if state_key not in self._q:
            self._q[state_key] = {a: 0.0 for a in ACTIONS}
        return self._q[state_key].get(action, 0.0)

    def _max_q_next(self, next_state_key: str) -> float:
        """Max over actions for next state."""
        if next_state_key not in self._q:
            return 0.0
        return max(self._q[next_state_key].values())

    def select_action(self, state: dict[str, Any]) -> tuple[str, float]:
        """
        Select action via epsilon-greedy policy.

        Returns:
            (action_name, confidence) where confidence is max Q for state.
        """
        key = _state_key(state)
        if key not in self._q:
            self._q[key] = {a: 0.0 for a in ACTIONS}
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(ACTIONS, key=lambda a: self._get_q(key, a))
        q_val = self._get_q(key, action)
        confidence = min(1.0, max(0.0, 0.5 + q_val * 0.1))
        return action, confidence

    def update(
        self,
        state: dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Q-learning update: Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a)).

        If next_state is None, uses target = r (simplified one-step).
        """
        s_key = _state_key(state)
        if s_key not in self._q:
            self._q[s_key] = {a: 0.0 for a in ACTIONS}
        q_old = self._get_q(s_key, action)
        if next_state is not None:
            s_next = _state_key(next_state)
            target = reward + self.gamma * self._max_q_next(s_next)
        else:
            target = reward
        self._q[s_key][action] = q_old + self.learning_rate * (target - q_old)
