"""Analysis, visualisation, and explanation helpers for Part 01."""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import PLOTS_DIR, TABLES_DIR, ExperimentConfig, PART01_EXPERIMENTS, ensure_result_directories

os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR.parent / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import ActionAdapter, StateDiscretizer, rolling_mean
from .envs import build_env
from .pipeline import collect_trajectory


def build_summary_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Convert evaluation summaries to a sorted dataframe."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["success_rate_pct"] = frame["success_rate"] * 100.0
    return frame.sort_values(["reward_mode", "env_id", "slug"]).reset_index(drop=True)


def save_summary_tables(summary_frame: pd.DataFrame) -> Dict[str, Path]:
    """Persist the evaluation summaries to CSV and Markdown."""

    ensure_result_directories()
    csv_path = TABLES_DIR / "evaluation_summary.csv"
    markdown_path = TABLES_DIR / "evaluation_summary.md"
    summary_frame.to_csv(csv_path, index=False)
    markdown_path.write_text(summary_frame.to_markdown(index=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def _policy_array(config: ExperimentConfig, q_table: np.ndarray) -> Tuple[np.ndarray, StateDiscretizer]:
    env = build_env(config, reward_mode="objective")
    discretizer = StateDiscretizer.from_env(env, config.state_bins)
    env.close()
    policy = np.argmax(q_table, axis=2)
    return policy, discretizer


def plot_training_dashboard(
    artifacts,
    output_path=None,
    window=100,
):
    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "training_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    for slug, artifact in artifacts.items():
        rewards = _extract_rewards(artifact)
        if rewards is None:
            continue

        label = artifact["config"].title
        axes[0].plot(rolling_mean(rewards, window), label=label)

        if "history" in artifact:
            history = artifact["history"]
            axes[1].plot(rolling_mean(history["raw_reward"], window), label=label)
            axes[2].plot(rolling_mean(history["success"].astype(float), window), label=label)
            axes[3].plot(rolling_mean(history["max_position"], window), label=label)

    axes[0].set_title("Rolling reward")
    axes[1].set_title("Raw reward (tabular only)")
    axes[2].set_title("Success rate (tabular only)")
    axes[3].set_title("Max position (tabular only)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, output_path


def plot_discrete_policy_heatmaps(artifacts, experiments=None, output_path=None):
    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "discrete_policy_heatmaps.png"

    selected = [
        config for config in (experiments or PART01_EXPERIMENTS)
        if config.env_id == "MountainCar-v0"
        and "q_table" in artifacts[config.slug]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()

    for axis, config in zip(axes, selected):
        artifact = artifacts[config.slug]
        q_table = artifact["q_table"]
        policy, _ = _policy_array(config, q_table)

        im = axis.imshow(policy.T, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
        axis.set_title(config.title)

    fig.colorbar(im, ax=axes)
    fig.savefig(output_path, dpi=160)
    return fig, output_path




def plot_continuous_policy_heatmap(
    artifact: Dict[str, object],
    output_path: Path = None,
):
    """Visualise the greedy thrust chosen by the continuous-controller experiment."""

    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "continuous_policy_heatmap.png"

    config = artifact["config"]
    env = build_env(config, reward_mode="objective")
    action_adapter = ActionAdapter.from_config(env, config.action_values)
    env.close()

    policy_index = np.argmax(artifact["q_table"], axis=2)
    thrust_map = np.zeros_like(policy_index, dtype=np.float32)
    for index in range(action_adapter.action_count):
        thrust_map[policy_index == index] = action_adapter.force_value(index)

    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(
        thrust_map.T,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axis.set_title(config.title)
    axis.set_xlabel("Position bin")
    axis.set_ylabel("Velocity bin")
    fig.colorbar(image, ax=axis, label="Greedy thrust")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, output_path


def plot_policy_disagreement(
    artifacts: Dict[str, Dict[str, object]],
    reference_slug: str = "discrete_q_learning",
    output_path: Path = None,
):
    """Highlight where the alternative discrete policies disagree with the baseline."""

    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "policy_disagreement.png"

    reference_artifact = artifacts[reference_slug]
    reference_policy = np.argmax(reference_artifact["q_table"], axis=2)

    comparison_slugs = [
        slug
        for slug in artifacts
        if slug != reference_slug and artifacts[slug]["config"].env_id == "MountainCar-v0"
    ]

    fig, axes = plt.subplots(1, len(comparison_slugs), figsize=(5 * len(comparison_slugs), 5), sharey=True)
    if len(comparison_slugs) == 1:
        axes = [axes]

    for axis, slug in zip(axes, comparison_slugs):
        artifact = artifacts[slug]
        other_policy = np.argmax(artifact["q_table"], axis=2)
        disagreement = (reference_policy != other_policy).astype(float)
        image = axis.imshow(disagreement.T, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(f"{artifact['config'].title}\nvs baseline")
        axis.set_xlabel("Position bin")
        axis.set_ylabel("Velocity bin")
        agreement_pct = 100.0 * (1.0 - disagreement.mean())
        axis.text(
            0.02,
            0.97,
            f"Agreement: {agreement_pct:.1f}%",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.colorbar(image, ax=axes, label="1 = different greedy action")
    fig.subplots_adjust(top=0.88, wspace=0.25)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, output_path


def plot_phase_trajectories(
    artifacts: Dict[str, Dict[str, object]],
    slugs: Iterable[str] = None,
    output_path: Path = None,
):
    """Plot one greedy trajectory per selected experiment in phase space."""

    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "phase_trajectories.png"

    selected_slugs = list(
        slugs
        or (
            "discrete_q_learning",
            "discrete_directional_cost",
            "discrete_non_null_cost",
            "continuous_q_learning",
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for axis, slug in zip(axes, selected_slugs):
        artifact = artifacts[slug]
        trajectory = collect_trajectory(artifact["config"], artifact["q_table"])
        axis.plot(trajectory["position"], trajectory["velocity"], marker="o", markersize=2, linewidth=1.0)
        axis.set_title(artifact["config"].title)
        axis.set_xlabel("Position")
        axis.set_ylabel("Velocity")
        axis.grid(True, alpha=0.3)

    fig.suptitle("Greedy policy trajectories in phase space", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, output_path


def plot_objective_vs_engineered_reward(
    summary_frame: pd.DataFrame,
    output_path: Path = None,
):
    """Compare objective and engineered rewards for the adapted experiments."""

    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "objective_vs_engineered_reward.png"

    filtered = summary_frame[summary_frame["slug"].isin(["discrete_directional_cost", "discrete_non_null_cost"])]
    if filtered.empty:
        raise ValueError("No shaped-reward experiments found in the summary frame.")

    pivot = filtered.pivot(index="slug", columns="reward_mode", values="mean_reward")
    pivot = pivot.rename(
        index={
            "discrete_directional_cost": "Directional cost",
            "discrete_non_null_cost": "Non-null cost",
        }
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Objective reward vs engineered reward")
    axis.set_xlabel("Adapted environment")
    axis.set_ylabel("Mean episodic reward")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(title="Evaluation reward mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, output_path


def build_policy_dataset(config: ExperimentConfig, q_table: np.ndarray) -> pd.DataFrame:
    """Create a feature table from the greedy policy for explanation models."""

    policy, discretizer = _policy_array(config, q_table)
    position_centers, velocity_centers = discretizer.bin_centers()
    rows = []

    for position_index, position_value in enumerate(position_centers):
        for velocity_index, velocity_value in enumerate(velocity_centers):
            action_index = int(policy[position_index, velocity_index])
            rows.append(
                {
                    "position": float(position_value),
                    "velocity": float(velocity_value),
                    "abs_velocity": float(abs(velocity_value)),
                    "momentum_proxy": float(position_value * velocity_value),
                    "kinetic_energy": float(0.5 * velocity_value * velocity_value),
                    "height_proxy": float(np.sin(3.0 * position_value)),
                    "action": action_index,
                }
            )

    return pd.DataFrame(rows)


def explain_policy_with_random_forest(
    config: ExperimentConfig,
    q_table: np.ndarray,
    output_path: Path = None,
):
    """Train a simple surrogate model and visualise its feature importances."""

    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / f"{config.slug}_feature_importance.png"

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "scikit-learn is required for the policy explanation cell. "
            "Install the project requirements first."
        ) from exc

    dataset = build_policy_dataset(config, q_table)
    feature_columns = [
        "position",
        "velocity",
        "abs_velocity",
        "momentum_proxy",
        "kinetic_energy",
        "height_proxy",
    ]
    X = dataset[feature_columns]
    y = dataset["action"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=config.seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=4,
        random_state=config.seed,
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)

    fig, axis = plt.subplots(figsize=(9, 5))
    importance.plot(kind="bar", ax=axis, color="#2b8cbe")
    axis.set_title(f"Policy feature importance - {config.title}\nSurrogate accuracy: {accuracy:.3f}")
    axis.set_ylabel("Random-forest importance")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    return {
        "dataset": dataset,
        "accuracy": float(accuracy),
        "importance": importance,
        "figure": fig,
        "output_path": output_path,
    }


def generate_all_figures(
    artifacts: Dict[str, Dict[str, object]],
    summary_rows: List[Dict[str, object]],
) -> Dict[str, Path]:
    """Generate the main assignment figures in one call."""

    summary_frame = build_summary_frame(summary_rows)
    outputs = {}
    _, outputs["training_dashboard"] = plot_training_dashboard(artifacts)
    _, outputs["discrete_policies"] = plot_discrete_policy_heatmaps(artifacts)
    _, outputs["continuous_policy"] = plot_continuous_policy_heatmap(artifacts["continuous_q_learning"])
    _, outputs["policy_disagreement"] = plot_policy_disagreement(artifacts)
    _, outputs["phase_trajectories"] = plot_phase_trajectories(artifacts)
    _, outputs["objective_vs_engineered"] = plot_objective_vs_engineered_reward(summary_frame)
    explanation = explain_policy_with_random_forest(
        artifacts["discrete_q_learning"]["config"],
        artifacts["discrete_q_learning"]["q_table"],
    )
    outputs["feature_importance"] = explanation["output_path"]
    return outputs

def plot_sb3_phase_trajectories(artifacts, output_path=None):
    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "sb3_phase_trajectories.png"

    fig, ax = plt.subplots(figsize=(8, 6))

    for slug, artifact in artifacts.items():
        if not _is_sb3_artifact(artifact):
            continue

        traj = collect_trajectory_sb3(
            artifact["config"],
            artifact["model"]
        )

        ax.plot(traj["position"], traj["velocity"], label=artifact["config"].title)

    ax.set_title("SB3 Phase Trajectories")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.legend()
    ax.grid(True)

    fig.savefig(output_path, dpi=160)
    return fig, output_path

def plot_sb3_training_curves(artifacts, output_path=None):
    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "sb3_training_curves.png"

    fig, ax = plt.subplots(figsize=(10, 5))

    for slug, artifact in artifacts.items():
        if "episode_rewards" not in artifact:
            continue

        rewards = np.array(artifact["episode_rewards"])
        ax.plot(rolling_mean(rewards, 50), label=artifact["config"].title)

    ax.set_title("SB3 Training Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)

    fig.savefig(output_path, dpi=160)
    return fig, output_path



def plot_dqn_policy_heatmap(model, resolution=100):
    positions = np.linspace(-1.2, 0.6, resolution)
    velocities = np.linspace(-0.07, 0.07, resolution)

    grid = np.zeros((resolution, resolution))

    for i, p in enumerate(positions):
        for j, v in enumerate(velocities):
            obs = np.array([p, v])
            action, _ = model.predict(obs, deterministic=True)
            grid[i, j] = action

    plt.imshow(grid.T, origin="lower")
    plt.colorbar(label="Action")
    plt.title("DQN Policy Heatmap")



def plot_ppo_policy_heatmap(model, resolution=100):
    positions = np.linspace(-1.2, 0.6, resolution)
    velocities = np.linspace(-0.07, 0.07, resolution)

    grid = np.zeros((resolution, resolution))

    for i, p in enumerate(positions):
        for j, v in enumerate(velocities):
            obs = np.array([p, v])
            action, _ = model.predict(obs, deterministic=True)
            grid[i, j] = action

    plt.imshow(grid.T, origin="lower", cmap="coolwarm")
    plt.colorbar(label="Force")
    plt.title("PPO Policy Heatmap")



def plot_algorithm_comparison(summary_frame, output_path=None):
    ensure_result_directories()
    if output_path is None:
        output_path = PLOTS_DIR / "algorithm_comparison.png"

    fig, ax = plt.subplots(figsize=(10, 5))

    summary_frame.groupby("slug")["mean_reward"].mean().plot(kind="bar", ax=ax)

    ax.set_title("Algorithm Comparison")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, axis="y")

    fig.savefig(output_path, dpi=160)
    return fig, output_path