"""Environment helpers and wrappers for the Part 01 experiments."""

from typing import Dict

import gymnasium as gym
import numpy as np

from .config import ExperimentConfig


class RewardInfoWrapper(gym.Wrapper):
    """Expose reward bookkeeping fields in ``info`` for downstream analysis."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.setdefault("raw_reward", float(reward))
        info.setdefault("reward_penalty", 0.0)
        info.setdefault("shaped_reward", float(reward))
        return obs, reward, terminated, truncated, info


class UniformActionCostWrapper(gym.Wrapper):
    """Apply the same penalty to every non-neutral discrete action."""

    def __init__(self, env: gym.Env, action_cost: float = 0.1):
        super().__init__(env)
        self.action_cost = float(action_cost)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty = self.action_cost if action in (0, 2) else 0.0
        shaped_reward = float(reward - penalty)
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["reward_penalty"] = float(penalty)
        info["shaped_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info


class DirectionalActionCostWrapper(gym.Wrapper):
    """Apply different penalties to left and right thrust actions."""

    def __init__(self, env: gym.Env, left_cost: float = 0.2, right_cost: float = 0.1):
        super().__init__(env)
        self.left_cost = float(left_cost)
        self.right_cost = float(right_cost)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty = 0.0
        if action == 0:
            penalty = self.left_cost
        elif action == 2:
            penalty = self.right_cost
        shaped_reward = float(reward - penalty)
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["reward_penalty"] = float(penalty)
        info["shaped_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info


class ContinuousEnergyShapingWrapper(gym.Wrapper):
    """Reward increases in mechanical energy and rightward progress during training."""

    def __init__(self, env: gym.Env, energy_scale: float = 180.0, progress_scale: float = 8.0):
        super().__init__(env)
        self.energy_scale = float(energy_scale)
        self.progress_scale = float(progress_scale)

    @staticmethod
    def _height(position: float) -> float:
        return float(np.sin(3.0 * position) * 0.45 + 0.55)

    def _energy(self, state) -> float:
        position = float(state[0])
        velocity = float(state[1])
        return self._height(position) + 0.5 * velocity * velocity

    def step(self, action):
        previous_state = np.asarray(self.unwrapped.state, dtype=float)
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = np.asarray(obs, dtype=float)

        energy_bonus = self.energy_scale * (self._energy(next_state) - self._energy(previous_state))
        progress_bonus = self.progress_scale * max(0.0, float(next_state[0] - previous_state[0]))
        shaping = float(energy_bonus + progress_bonus)
        shaped_reward = float(reward + shaping)

        info = dict(info)
        info["raw_reward"] = float(reward)
        info["reward_penalty"] = float(-shaping)
        info["shaped_reward"] = shaped_reward
        info["energy_bonus"] = float(energy_bonus)
        info["progress_bonus"] = float(progress_bonus)
        return obs, shaped_reward, terminated, truncated, info


def _apply_training_wrapper(env: gym.Env, config: ExperimentConfig) -> gym.Env:
    if config.training_wrapper == "uniform_cost":
        kwargs: Dict[str, float] = dict(config.wrapper_kwargs)
        return UniformActionCostWrapper(env, action_cost=kwargs.get("action_cost", 0.1))
    if config.training_wrapper == "directional_cost":
        kwargs = dict(config.wrapper_kwargs)
        return DirectionalActionCostWrapper(
            env,
            left_cost=kwargs.get("left_cost", 0.2),
            right_cost=kwargs.get("right_cost", 0.1),
        )
    if config.training_wrapper == "continuous_energy":
        kwargs = dict(config.wrapper_kwargs)
        return ContinuousEnergyShapingWrapper(
            env,
            energy_scale=kwargs.get("energy_scale", 180.0),
            progress_scale=kwargs.get("progress_scale", 8.0),
        )
    return RewardInfoWrapper(env)


def build_env(
    config: ExperimentConfig,
    reward_mode: str = "objective",
    render_mode: str = None,
) -> gym.Env:
    """Construct an environment for training, objective evaluation, or rendering."""

    env = gym.make(config.env_id, render_mode=render_mode)
    if reward_mode == "training" and config.has_engineered_reward:
        return _apply_training_wrapper(env, config)
    return RewardInfoWrapper(env)
