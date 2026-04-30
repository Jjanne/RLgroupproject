from __future__ import annotations
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
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
            print(
                f"[{self.slug}] steps {self.num_timesteps:>8,} "
                f"| avg reward (last 50 eps) {avg:8.2f}"
            )
        return True


def train_ppo(
    slug: str,
    env,
    *,
    total_timesteps: int = 200_000,
    model_path: Path,
    log_path: Path,
    seed: int = 42,
) -> dict:
    """Train a PPO agent on a continuous-action MountainCar environment.

    Hyperparameters are tuned for MountainCarContinuous-v0:
    - A larger n_steps rollout buffer gives PPO longer trajectories to
      learn from, which matters in a sparse-reward environment where
      the car rarely reaches the goal early in training.
    - A small but non-zero ent_coef encourages exploration of the
      continuous action space without destabilising the policy.
    - clip_range=0.2 and n_epochs=10 are standard PPO settings that
      prevent overly large policy updates.
    - gae_lambda=0.95 reduces variance in the advantage estimates.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    callback = _LogCallback(slug)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        use_sde=True,           
        sde_sample_freq=4,      
        ent_coef=0.0,           
        n_steps=512,            
        batch_size=64,          
        n_epochs=10,
        learning_rate=1e-3,    
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
        )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(str(model_path))
    np.save(str(log_path), np.array(callback._ep_rewards))

    return {"model": model, "episode_rewards": callback._ep_rewards}