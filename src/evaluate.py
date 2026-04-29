"""Evaluate the baseline Q-learning solution on MountainCar-v0."""

import pandas as pd

from part01.analysis import build_summary_frame
from part01.config import PART01_EXPERIMENTS
from part01.pipeline import evaluate_experiment, run_all_experiments


def main():
    baseline = next(config for config in PART01_EXPERIMENTS if config.slug == "discrete_q_learning")
    artifacts = run_all_experiments([baseline], overwrite=False, tensorboard=False)
    row = evaluate_experiment(baseline, artifacts[baseline.slug]["q_table"], reward_mode="objective")
    frame = build_summary_frame([row])

    pd.set_option("display.width", 160)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

