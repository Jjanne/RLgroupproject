"""Run the full Part 01 MountainCar pipeline."""

import pandas as pd

from part01.analysis import build_summary_frame, generate_all_figures, save_summary_tables
from part01.pipeline import evaluate_all_experiments, run_all_experiments, save_evaluation_rows


def main():
    artifacts = run_all_experiments(overwrite=False, tensorboard=False)
    evaluation_rows = evaluate_all_experiments(artifacts)
    save_evaluation_rows(evaluation_rows)

    summary_frame = build_summary_frame(evaluation_rows)
    save_summary_tables(summary_frame)
    generate_all_figures(artifacts, evaluation_rows)

    objective_frame = summary_frame[summary_frame["reward_mode"] == "objective"].copy()
    objective_frame = objective_frame.sort_values(["success_rate", "mean_reward"], ascending=[False, False])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nObjective-evaluation ranking")
    print(objective_frame.to_string(index=False))


if __name__ == "__main__":
    main()

