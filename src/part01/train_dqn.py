from __future__ import annotations
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class _LogCallback(BaseCallback):
    def __init__(self, slug: str, log_interval: int = 10_000):
        super().__init__()
        self.slug = slug
        self.log_interval = log_interval
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.log_interval == 0 and self._ep_rewards:
            avg = np.mean(self._ep_rewards[-50:])
            print(f"[{self.slug}] steps {self.num_timesteps:>8,} | avg reward (last 50 eps) {avg:8.2f}")
        return True


def train_dqn(
    slug: str,
    env,
    *,
    total_timesteps: int = 200_000,
    model_path: Path,
    log_path: Path,
    seed: int = 42,
) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    callback = _LogCallback(slug)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,     # fill replay buffer before learning
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,  # fraction of training spent decaying epsilon
        exploration_final_eps=0.05,
        train_freq=4,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(str(model_path))
    np.save(str(log_path), np.array(callback._ep_rewards))

    return {"model": model, "episode_rewards": callback._ep_rewards}