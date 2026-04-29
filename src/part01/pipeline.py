"""Training and evaluation pipelines for Assignment 22.00 Part 01."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import (
    LOGS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    TENSORBOARD_DIR,
    ExperimentConfig,
    PART01_EXPERIMENTS,
    ensure_result_directories,
)
from .core import ActionAdapter, StateDiscretizer
from .envs import build_env

try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


def _artifact_paths(config: ExperimentConfig) -> Dict[str, Path]:
    ensure_result_directories()
    return {
        "model": MODELS_DIR / f"{config.slug}_q_table.npy",
        "history": LOGS_DIR / f"{config.slug}_training_history.npz",
        "metadata": LOGS_DIR / f"{config.slug}_metadata.json",
        "tensorboard": TENSORBOARD_DIR / config.slug,
    }


def _choose_action(
    rng: np.random.Generator,
    q_table: np.ndarray,
    state,
    epsilon: float,
) -> int:
    if float(rng.random()) < float(epsilon):
        return int(rng.integers(q_table.shape[-1]))
    return int(np.argmax(q_table[state]))


def _initial_q_table(config: ExperimentConfig, action_count: int) -> np.ndarray:
    shape = (int(config.state_bins[0]), int(config.state_bins[1]), int(action_count))
    if config.q_init_low == 0.0 and config.q_init_high == 0.0:
        return np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(config.seed)
    return rng.uniform(config.q_init_low, config.q_init_high, size=shape).astype(np.float32)


def _serialise_history(history: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    serialised = {}
    for key, values in history.items():
        dtype = np.float32
        if key in ("steps", "success", "non_null_actions", "left_actions", "right_actions"):
            dtype = np.int32
        serialised[key] = np.asarray(values, dtype=dtype)
    return serialised


def _load_history(history_path: Path) -> Dict[str, np.ndarray]:
    archive = np.load(history_path, allow_pickle=False)
    return {key: archive[key] for key in archive.files}


def load_artifacts(config: ExperimentConfig) -> Dict[str, object]:
    """Load a previously trained Q-table and its training history."""

    paths = _artifact_paths(config)
    return {
        "config": config,
        "paths": paths,
        "q_table": np.load(paths["model"], allow_pickle=False),
        "history": _load_history(paths["history"]),
    }


def train_experiment(
    config: ExperimentConfig,
    overwrite: bool = False,
    tensorboard: bool = False,
) -> Dict[str, object]:
    """Train one tabular controller and persist the resulting artifacts."""

    paths = _artifact_paths(config)
    if (
        not overwrite
        and paths["model"].exists()
        and paths["history"].exists()
        and paths["metadata"].exists()
    ):
        return load_artifacts(config)

    env = build_env(config, reward_mode="training")
    discretizer = StateDiscretizer.from_env(env, config.state_bins)
    action_adapter = ActionAdapter.from_config(env, config.action_values)
    q_table = _initial_q_table(config, action_adapter.action_count)
    rng = np.random.default_rng(config.seed)

    history: Dict[str, List[float]] = {
        "reward": [],
        "raw_reward": [],
        "penalty": [],
        "steps": [],
        "success": [],
        "non_null_actions": [],
        "left_actions": [],
        "right_actions": [],
        "squared_action_sum": [],
        "max_position": [],
        "epsilon": [],
    }

    epsilon = float(config.epsilon_start)
    writer = None
    if tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(logdir=str(paths["tensorboard"]))

    for episode in range(config.episodes):
        obs, _ = env.reset(seed=config.seed + episode)
        state = discretizer.encode(obs)
        action_index = None
        if config.algorithm == "sarsa":
            action_index = _choose_action(rng, q_table, state, epsilon)

        done = False
        total_reward = 0.0
        total_raw_reward = 0.0
        total_penalty = 0.0
        steps = 0
        non_null_actions = 0
        left_actions = 0
        right_actions = 0
        squared_action_sum = 0.0
        max_position = float(obs[0])
        success = 0

        while not done:
            if config.algorithm == "q_learning":
                action_index = _choose_action(rng, q_table, state, epsilon)

            env_action = action_adapter.to_env(action_index)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            next_state = discretizer.encode(next_obs)

            if config.algorithm == "sarsa" and not done:
                next_action_index = _choose_action(rng, q_table, next_state, epsilon)
                td_target = reward + config.gamma * q_table[next_state + (next_action_index,)]
            elif config.algorithm == "sarsa":
                next_action_index = None
                td_target = reward
            elif done:
                td_target = reward
            else:
                td_target = reward + config.gamma * np.max(q_table[next_state])

            q_table[state + (action_index,)] += config.alpha * (
                td_target - q_table[state + (action_index,)]
            )

            force = action_adapter.force_value(action_index)
            if abs(force) > 1e-9:
                non_null_actions += 1
            if force < 0.0:
                left_actions += 1
            if force > 0.0:
                right_actions += 1
            squared_action_sum += force * force
            max_position = max(max_position, float(next_obs[0]))

            raw_reward = float(info.get("raw_reward", reward))
            penalty = float(info.get("reward_penalty", 0.0))
            total_reward += float(reward)
            total_raw_reward += raw_reward
            total_penalty += penalty
            steps += 1
            if terminated:
                success = 1

            state = next_state
            if config.algorithm == "sarsa" and next_action_index is not None:
                action_index = next_action_index

        epsilon = max(float(config.epsilon_end), float(epsilon * config.epsilon_decay))
        history["reward"].append(total_reward)
        history["raw_reward"].append(total_raw_reward)
        history["penalty"].append(total_penalty)
        history["steps"].append(steps)
        history["success"].append(success)
        history["non_null_actions"].append(non_null_actions)
        history["left_actions"].append(left_actions)
        history["right_actions"].append(right_actions)
        history["squared_action_sum"].append(squared_action_sum)
        history["max_position"].append(max_position)
        history["epsilon"].append(epsilon)

        if writer is not None:
            step = episode + 1
            writer.add_scalar("train/reward", total_reward, step)
            writer.add_scalar("train/raw_reward", total_raw_reward, step)
            writer.add_scalar("train/steps", steps, step)
            writer.add_scalar("train/success", success, step)
            writer.add_scalar("train/non_null_actions", non_null_actions, step)
            writer.add_scalar("train/max_position", max_position, step)
            writer.add_scalar("train/epsilon", epsilon, step)

        if (episode + 1) % config.log_every == 0:
            recent = history["reward"][-config.log_every :]
            print(
                f"[{config.slug}] episode {episode + 1:>4} "
                f"| avg shaped reward {np.mean(recent):>8.2f} "
                f"| epsilon {epsilon:.4f}"
            )

    if writer is not None:
        writer.close()
    env.close()

    np.save(paths["model"], q_table)
    serialised_history = _serialise_history(history)
    np.savez_compressed(paths["history"], **serialised_history)
    paths["metadata"].write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    return {
        "config": config,
        "paths": paths,
        "q_table": q_table,
        "history": serialised_history,
    }


def run_all_experiments(
    experiments: Optional[Iterable[ExperimentConfig]] = None,
    overwrite: bool = False,
    tensorboard: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Train or load all configured experiments."""

    selected_experiments = list(experiments or PART01_EXPERIMENTS)
    artifacts: Dict[str, Dict[str, object]] = {}
    for config in selected_experiments:
        artifacts[config.slug] = train_experiment(
            config,
            overwrite=overwrite,
            tensorboard=tensorboard,
        )
    return artifacts


def evaluate_experiment(
    config: ExperimentConfig,
    q_table: np.ndarray,
    episodes: Optional[int] = None,
    reward_mode: str = "objective",
    seed_offset: int = 10_000,
) -> Dict[str, object]:
    """Evaluate one learned policy and summarise the resulting episodes."""

    env = build_env(config, reward_mode=reward_mode)
    discretizer = StateDiscretizer.from_env(env, config.state_bins)
    action_adapter = ActionAdapter.from_config(env, config.action_values)
    episode_count = int(episodes or config.eval_episodes)

    rewards = []
    raw_rewards = []
    penalties = []
    steps_taken = []
    successes = []
    non_null_actions = []
    left_actions = []
    right_actions = []
    squared_action_sum = []
    max_positions = []
    final_positions = []

    for episode in range(episode_count):
        obs, _ = env.reset(seed=config.seed + seed_offset + episode)
        state = discretizer.encode(obs)
        done = False

        total_reward = 0.0
        total_raw_reward = 0.0
        total_penalty = 0.0
        steps = 0
        active_actions = 0
        left_count = 0
        right_count = 0
        force_square_sum = 0.0
        max_position = float(obs[0])
        success = 0

        while not done:
            action_index = int(np.argmax(q_table[state]))
            env_action = action_adapter.to_env(action_index)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            force = action_adapter.force_value(action_index)
            if abs(force) > 1e-9:
                active_actions += 1
            if force < 0.0:
                left_count += 1
            if force > 0.0:
                right_count += 1
            force_square_sum += force * force
            max_position = max(max_position, float(next_obs[0]))

            raw_reward = float(info.get("raw_reward", reward))
            penalty = float(info.get("reward_penalty", 0.0))
            total_reward += float(reward)
            total_raw_reward += raw_reward
            total_penalty += penalty
            steps += 1
            if terminated:
                success = 1

            state = discretizer.encode(next_obs)

        rewards.append(total_reward)
        raw_rewards.append(total_raw_reward)
        penalties.append(total_penalty)
        steps_taken.append(steps)
        successes.append(success)
        non_null_actions.append(active_actions)
        left_actions.append(left_count)
        right_actions.append(right_count)
        squared_action_sum.append(force_square_sum)
        max_positions.append(max_position)
        final_positions.append(float(next_obs[0]))

    env.close()

    reward_array = np.asarray(rewards, dtype=np.float32)
    raw_reward_array = np.asarray(raw_rewards, dtype=np.float32)
    penalty_array = np.asarray(penalties, dtype=np.float32)
    steps_array = np.asarray(steps_taken, dtype=np.int32)
    success_array = np.asarray(successes, dtype=np.int32)
    non_null_array = np.asarray(non_null_actions, dtype=np.int32)
    left_array = np.asarray(left_actions, dtype=np.int32)
    right_array = np.asarray(right_actions, dtype=np.int32)
    squared_array = np.asarray(squared_action_sum, dtype=np.float32)
    max_position_array = np.asarray(max_positions, dtype=np.float32)
    final_position_array = np.asarray(final_positions, dtype=np.float32)

    return {
        "slug": config.slug,
        "title": config.title,
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "reward_mode": reward_mode,
        "episodes": episode_count,
        "mean_reward": float(np.mean(reward_array)),
        "std_reward": float(np.std(reward_array)),
        "mean_raw_reward": float(np.mean(raw_reward_array)),
        "mean_penalty": float(np.mean(penalty_array)),
        "mean_steps": float(np.mean(steps_array)),
        "success_rate": float(np.mean(success_array)),
        "mean_non_null_actions": float(np.mean(non_null_array)),
        "mean_left_actions": float(np.mean(left_array)),
        "mean_right_actions": float(np.mean(right_array)),
        "mean_squared_action_sum": float(np.mean(squared_array)),
        "mean_max_position": float(np.mean(max_position_array)),
        "mean_final_position": float(np.mean(final_position_array)),
    }


def evaluate_all_experiments(
    artifacts: Dict[str, Dict[str, object]],
    experiments: Optional[Iterable[ExperimentConfig]] = None,
    objective_episodes: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Evaluate every experiment in objective mode and, when relevant, training mode."""

    selected_experiments = list(experiments or PART01_EXPERIMENTS)
    rows: List[Dict[str, object]] = []

    for config in selected_experiments:
        q_table = artifacts[config.slug]["q_table"]
        rows.append(
            evaluate_experiment(
                config,
                q_table=q_table,
                episodes=objective_episodes,
                reward_mode="objective",
            )
        )
        if config.has_engineered_reward:
            rows.append(
                evaluate_experiment(
                    config,
                    q_table=q_table,
                    episodes=objective_episodes,
                    reward_mode="training",
                    seed_offset=20_000,
                )
            )

    return rows


def collect_trajectory(
    config: ExperimentConfig,
    q_table: np.ndarray,
    reward_mode: str = "objective",
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Collect one greedy rollout for trajectory and phase-portrait plots."""

    env = build_env(config, reward_mode=reward_mode)
    discretizer = StateDiscretizer.from_env(env, config.state_bins)
    action_adapter = ActionAdapter.from_config(env, config.action_values)

    obs, _ = env.reset(seed=seed)
    state = discretizer.encode(obs)
    positions = [float(obs[0])]
    velocities = [float(obs[1])]
    rewards = []
    raw_rewards = []
    actions = []
    forces = []

    done = False
    while not done:
        action_index = int(np.argmax(q_table[state]))
        env_action = action_adapter.to_env(action_index)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = bool(terminated or truncated)

        positions.append(float(next_obs[0]))
        velocities.append(float(next_obs[1]))
        rewards.append(float(reward))
        raw_rewards.append(float(info.get("raw_reward", reward)))
        actions.append(action_index)
        forces.append(action_adapter.force_value(action_index))

        state = discretizer.encode(next_obs)

    env.close()

    return {
        "position": np.asarray(positions, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "raw_reward": np.asarray(raw_rewards, dtype=np.float32),
        "action_index": np.asarray(actions, dtype=np.int32),
        "force": np.asarray(forces, dtype=np.float32),
    }


def save_evaluation_rows(rows: List[Dict[str, object]], filename: str = "evaluation_summary.json") -> Path:
    """Persist evaluation summaries for downstream notebook/report use."""

    ensure_result_directories()
    output_path = TABLES_DIR / filename
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return output_path
