# Assignment 22.00 - Part 01

This branch contains a **single reproducible MountainCar testbed** for the Part 01 group assignment deliverable.
It is designed to satisfy the notebook/framework requirements from the PDF:

- multiple MountainCar scenarios,
- explicit design choices for state/action representations and reward design,
- modular training/evaluation code,
- policy comparison and interpretability analysis,
- presentation-ready figures and summary tables.

## What is included

The Part 01 framework covers five experiments:

1. `discrete_q_learning`
   Standard `MountainCar-v0` with Q-learning.
2. `discrete_sarsa`
   Standard `MountainCar-v0` with SARSA for algorithmic comparison.
3. `discrete_directional_cost`
   Adapted `MountainCar-v0` with different penalties for left and right thrust.
4. `discrete_non_null_cost`
   Adapted `MountainCar-v0` with a linear penalty on every non-idle action.
5. `continuous_q_learning`
   `MountainCarContinuous-v0` solved with discretised continuous thrust values.

## Repository structure

```text
RLgroupproject/
├── main.py
├── requirements.txt
├── results/
│   └── part01/
│       ├── logs/
│       ├── models/
│       ├── plots/
│       ├── tables/
│       └── tensorboard/
└── src/
    ├── generate_part01_notebook.py
    ├── mountain_car.ipynb
    ├── run_part01.py
    ├── train_qlearning.py
    ├── train_sarsa.py
    ├── train_qlearning_action_cost.py
    ├── train_qlearning_directional_cost.py
    ├── train_qlearning_continuous.py
    ├── evaluate.py
    ├── evaluate_all.py
    ├── compare_results.py
    ├── plots.py
    └── part01/
        ├── __init__.py
        ├── analysis.py
        ├── config.py
        ├── core.py
        ├── envs.py
        └── pipeline.py
```

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main ways to run Part 01

Run the full pipeline:

```bash
python src/run_part01.py
```

This will:

- train or reload all Part 01 experiments,
- evaluate them on objective and engineered rewards where relevant,
- write summary tables to `results/part01/tables/`,
- generate figures in `results/part01/plots/`.

Regenerate the notebook file if needed:

```bash
python src/generate_part01_notebook.py
```

Open and run the structured notebook:

```text
src/mountain_car.ipynb
```

## Experiment-specific entry points

```bash
python src/train_qlearning.py
python src/train_sarsa.py
python src/train_qlearning_directional_cost.py
python src/train_qlearning_action_cost.py
python src/train_qlearning_continuous.py
python src/evaluate_all.py
python src/compare_results.py
python src/plots.py
```

## Outputs used for the assignment

The notebook and scripts generate the artifacts required for the Part 01 discussion:

- training histories,
- learned policies,
- objective vs engineered reward comparisons,
- success-rate and convergence plots,
- policy heatmaps,
- phase portraits / trajectories,
- a surrogate-model feature-importance explanation of the baseline policy.

## Notes for the final submission

- `src/mountain_car.ipynb` is the main notebook deliverable.
- The reusable framework lives in `src/part01/` so the notebook stays readable.
- Generated artifacts under `results/part01/` are intentionally ignored by git because the notebook and scripts reproduce them.
- If you want a presentation-ready zip, export the selected plots/tables after running the notebook or pipeline.
