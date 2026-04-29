"""Print ranked Part 01 evaluation tables."""

import pandas as pd

from part01.analysis import build_summary_frame
from part01.pipeline import evaluate_all_experiments, run_all_experiments


def main():
    artifacts = run_all_experiments(overwrite=False, tensorboard=False)
    summary = build_summary_frame(evaluate_all_experiments(artifacts))

    objective_only = summary[summary["reward_mode"] == "objective"].copy()
    objective_only = objective_only.sort_values(
        ["success_rate", "mean_reward", "mean_steps"],
        ascending=[False, False, True],
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(objective_only.to_string(index=False))


if __name__ == "__main__":
    main()

