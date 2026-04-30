"""Generate the standalone Part 01 notebook."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent / "mountain_car.ipynb"


def markdown_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_notebook():
    cells = [
        markdown_cell(
            """# Assignment 22.00 - Part 01

## MountainCar testbed, analysis, and interpretability notebook

This notebook is the **standalone Part 01 deliverable** for the group assignment.
Running it from top to bottom will:

1. train or reload all Part 01 MountainCar experiments,
2. evaluate them under both **objective** and **engineered** rewards when relevant,
3. generate the figures and summary tables used for reporting,
4. analyse the learned policies numerically, visually, and with a simple explanation model.

The notebook is intentionally thin: the reusable logic lives in `src/part01/`,
while this document orchestrates and explains the full workflow.
"""
        ),
        markdown_cell(
            """## Assignment mapping

The implementation below addresses the main Part 01 requirements:

- **Design choices**: state discretisation, action representations, reward variants, hyperparameters, and RL algorithm selection.
- **Custom environment adaptations**: reward wrappers for discrete action-cost scenarios.
- **Framework / modularisation**: the notebook imports a reusable package instead of duplicating logic.
- **Evaluation**: mean reward, variability, success rate, step count, control effort, and physical reach.
- **Policy analysis**: heatmaps, policy disagreement maps, phase trajectories, and feature-importance explanations.
- **Comparative policy analysis**: baseline discrete, on-policy discrete, two adapted discrete environments, and continuous control.
"""
        ),
        code_cell(
            """from pathlib import Path

import pandas as pd

from part01.analysis import (
    build_summary_frame,
    explain_policy_with_random_forest,
    generate_all_figures,
    plot_algorithm_comparison,
    plot_continuous_policy_heatmap,
    plot_dqn_policy_heatmap,
    plot_discrete_policy_heatmaps,
    plot_objective_vs_engineered_reward,
    plot_phase_trajectories,
    plot_policy_disagreement,
    plot_ppo_policy_heatmap,
    plot_sb3_phase_trajectories,
    plot_sb3_training_curves,
    plot_training_dashboard,
    save_summary_tables,
)
from part01.config import PART01_EXPERIMENTS, PLOTS_DIR, TABLES_DIR
from part01.pipeline import evaluate_all_experiments, run_all_experiments, save_evaluation_rows

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
"""
        ),
        markdown_cell(
            """## Experiment catalogue

These are the scenarios included in the Part 01 framework:

- **Discrete baseline (Q-learning)**: minimum-step objective on `MountainCar-v0`
- **Discrete baseline (SARSA)**: on-policy comparison on `MountainCar-v0`
- **Discrete directional-cost adaptation**: different costs for left vs right thrust
- **Discrete non-null-action adaptation**: linear penalty for every non-idle action
- **Continuous MountainCar**: discretised thrust controller on `MountainCarContinuous-v0`
"""
        ),
        code_cell(
            """experiment_frame = pd.DataFrame([config.to_dict() for config in PART01_EXPERIMENTS])
experiment_frame[[
    "slug",
    "title",
    "env_id",
    "algorithm",
    "state_bins",
    "episodes",
    "training_wrapper",
    "description",
]]
"""
        ),
        markdown_cell(
            """## Reproducibility controls

- `OVERWRITE_MODELS = False` means cached artifacts are reused if they already exist.
- `USE_TENSORBOARD = False` keeps the notebook lightweight by default.
- Set `OVERWRITE_MODELS = True` if you want to retrain everything from scratch.
"""
        ),
        code_cell(
            """OVERWRITE_MODELS = False
USE_TENSORBOARD = False

artifacts = run_all_experiments(
    experiments=PART01_EXPERIMENTS,
    overwrite=OVERWRITE_MODELS,
    tensorboard=USE_TENSORBOARD,
)
"""
        ),
        markdown_cell(
            """## Evaluation tables

The next cells evaluate each learned policy on:

- its **objective/native reward** environment, and
- when the training reward was shaped, also on the **engineered reward** environment.

This is important because Part 01 asks us to compare actual performance against
the reward signal used during optimisation.
"""
        ),
        code_cell(
            """evaluation_rows = evaluate_all_experiments(artifacts, experiments=PART01_EXPERIMENTS)
save_evaluation_rows(evaluation_rows)

summary_frame = build_summary_frame(evaluation_rows)
save_summary_tables(summary_frame)

summary_frame
"""
        ),
        code_cell(
            """objective_summary = summary_frame[summary_frame["reward_mode"] == "objective"].copy()
objective_summary = objective_summary.sort_values(
    ["success_rate", "mean_reward", "mean_steps"],
    ascending=[False, False, True],
)
objective_summary
"""
        ),
        markdown_cell(
            """## Overall algorithm comparison

This bar chart compares the **objective-evaluation mean reward** of every
available experiment. If `stable-baselines3` is installed, the DQN and PPO
agents are included automatically; otherwise the notebook keeps the original
tabular experiments and skips the SB3 baselines gracefully.
"""
        ),
        code_cell(
            """fig, algorithm_comparison_path = plot_algorithm_comparison(summary_frame)
fig
"""
        ),
        markdown_cell(
            """## Training behaviour and convergence

The dashboard below compares:

- shaped reward,
- native/objective reward,
- rolling success rate,
- maximum position reached during training.
"""
        ),
        code_cell(
            """fig, training_dashboard_path = plot_training_dashboard(artifacts)
fig
"""
        ),
        markdown_cell(
            """## SB3 training curves

When the DQN and PPO baselines are available, the next plot shows their
episode-reward learning curves separately from the tabular dashboard.
"""
        ),
        code_cell(
            """if any(slug in artifacts for slug in ("discrete_dqn", "continuous_ppo")):
    fig, sb3_training_path = plot_sb3_training_curves(artifacts)
    fig
else:
    print("SB3 artifacts are not available in this environment.")
"""
        ),
        markdown_cell(
            """## Policy structure

Discrete experiments use the same 3-action interface, so their greedy policies can
be compared directly. The continuous controller is shown separately as a thrust map.
"""
        ),
        code_cell(
            """fig, discrete_policy_path = plot_discrete_policy_heatmaps(artifacts)
fig
"""
        ),
        code_cell(
            """fig, continuous_policy_path = plot_continuous_policy_heatmap(artifacts["continuous_q_learning"])
fig
"""
        ),
        code_cell(
            """fig, disagreement_path = plot_policy_disagreement(artifacts)
fig
"""
        ),
        markdown_cell(
            """## Neural policy maps

The next cells visualise the learned policies of the neural baselines when they are
available:

- **DQN** predicts a discrete action directly on `MountainCar-v0`
- **PPO** predicts a continuous force on `MountainCarContinuous-v0`
"""
        ),
        code_cell(
            """if "discrete_dqn" in artifacts:
    fig, dqn_policy_path = plot_dqn_policy_heatmap(artifacts["discrete_dqn"]["model"])
    fig
else:
    print("DQN artifact is not available in this environment.")
"""
        ),
        code_cell(
            """if "continuous_ppo" in artifacts:
    fig, ppo_policy_path = plot_ppo_policy_heatmap(artifacts["continuous_ppo"]["model"])
    fig
else:
    print("PPO artifact is not available in this environment.")
"""
        ),
        markdown_cell(
            """## Objective reward vs engineered reward

For the adapted discrete environments, we explicitly compare the reward used during
training with the original environment reward. This reveals whether a shaped reward
helps the agent optimise the true task or simply a proxy objective.
"""
        ),
        code_cell(
            """fig, reward_tradeoff_path = plot_objective_vs_engineered_reward(summary_frame)
fig
"""
        ),
        markdown_cell(
            """## Physical interpretation through trajectories

Phase-space trajectories show whether the learned control law follows the expected
MountainCar strategy: first build momentum by moving away from the goal, then
convert that momentum into enough energy to climb the right hill.
"""
        ),
        code_cell(
            """fig, trajectory_path = plot_phase_trajectories(artifacts)
fig
"""
        ),
        markdown_cell(
            """## Neural trajectories

If the SB3 agents are available, the next figure shows their deterministic rollouts
in phase space so they can be compared against the tabular trajectories above.
"""
        ),
        code_cell(
            """if any(slug in artifacts for slug in ("discrete_dqn", "continuous_ppo")):
    fig, sb3_trajectory_path = plot_sb3_phase_trajectories(artifacts)
    fig
else:
    print("SB3 artifacts are not available in this environment.")
"""
        ),
        markdown_cell(
            """## Explanation model

To make the baseline policy more interpretable, we train a **random-forest surrogate**
that predicts the greedy action from engineered physical descriptors:

- position,
- velocity,
- absolute velocity,
- momentum proxy,
- kinetic energy,
- height proxy.

High surrogate accuracy means the learned tabular policy can be approximated by a
simple explanatory model, while the feature importances reveal which state concepts
drive the action choice most strongly.
"""
        ),
        code_cell(
            """explanation = explain_policy_with_random_forest(
    artifacts["discrete_q_learning"]["config"],
    artifacts["discrete_q_learning"]["q_table"],
)
explanation["importance"]
"""
        ),
        code_cell(
            """explanation["figure"]
"""
        ),
        markdown_cell(
            """## Deliverable outputs

The generated artifacts are saved under `results/part01/`:

- `models/` - learned Q-tables
- `logs/` - training histories and metadata
- `plots/` - figures for the report/presentation
- `tables/` - evaluation summaries in JSON, CSV, and Markdown
"""
        ),
        code_cell(
            """sorted(path.name for path in PLOTS_DIR.glob("*.png"))
"""
        ),
        markdown_cell(
            """## Final interpretation prompts

Use the tables and figures above to support the written discussion:

- Which controller gives the best **objective** performance?
- Which reward shaping strategy best balances **goal-reaching** and **control effort**?
- How does the continuous controller differ structurally from the discrete ones?
- Which state variables dominate the learned policy according to the surrogate model?

These questions align with the Part 01 requirement to justify the solution both
numerically and from the physical perspective of the MountainCar dynamics.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
