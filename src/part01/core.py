"""Core tabular RL helpers shared across Part 01 modules."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class StateDiscretizer:
    """Map continuous state coordinates to discrete bin indices."""

    low: np.ndarray
    high: np.ndarray
    bins: Tuple[int, int]

    @classmethod
    def from_env(cls, env, bins: Tuple[int, int]):
        return cls(
            low=np.asarray(env.observation_space.low, dtype=float),
            high=np.asarray(env.observation_space.high, dtype=float),
            bins=tuple(int(value) for value in bins),
        )

    def encode(self, state: Iterable[float]) -> Tuple[int, int]:
        state_array = np.asarray(state, dtype=float)
        scaled = (state_array - self.low) / (self.high - self.low)
        discrete = (scaled * np.asarray(self.bins, dtype=float)).astype(int)
        clipped = np.clip(discrete, 0, np.asarray(self.bins) - 1)
        return int(clipped[0]), int(clipped[1])

    def bin_centers(self):
        position_edges = np.linspace(self.low[0], self.high[0], self.bins[0] + 1)
        velocity_edges = np.linspace(self.low[1], self.high[1], self.bins[1] + 1)
        position_centers = 0.5 * (position_edges[:-1] + position_edges[1:])
        velocity_centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
        return position_centers, velocity_centers


@dataclass
class ActionAdapter:
    """Convert tabular action indices to the format expected by the environment."""

    values: np.ndarray
    continuous: bool

    @classmethod
    def from_config(cls, env, action_values=None):
        if action_values is None:
            return cls(
                values=np.arange(env.action_space.n, dtype=int),
                continuous=False,
            )
        return cls(
            values=np.asarray(action_values, dtype=np.float32),
            continuous=True,
        )

    @property
    def action_count(self) -> int:
        return int(self.values.shape[0])

    def to_env(self, action_index: int):
        if self.continuous:
            return np.asarray([float(self.values[action_index])], dtype=np.float32)
        return int(self.values[action_index])

    def force_value(self, action_index: int) -> float:
        if self.continuous:
            return float(self.values[action_index])
        action = int(self.values[action_index])
        if action == 0:
            return -1.0
        if action == 2:
            return 1.0
        return 0.0


def rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute a moving average for plotting."""

    if values.size < window:
        return values.copy()
    weights = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, weights, mode="valid")
