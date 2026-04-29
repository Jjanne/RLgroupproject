"""Evaluate every Part 01 experiment using the stored models."""

import pandas as pd

from part01.analysis import build_summary_frame, save_summary_tables
from part01.pipeline import evaluate_all_experiments, run_all_experiments, save_evaluation_rows


def main():
    artifacts = run_all_experiments(overwrite=False, tensorboard=False)
    rows = evaluate_all_experiments(artifacts)
    save_evaluation_rows(rows)

    frame = build_summary_frame(rows)
    save_summary_tables(frame)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

