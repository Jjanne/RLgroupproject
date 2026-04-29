"""Generate all Part 01 plots from the stored experiments."""

from part01.analysis import generate_all_figures
from part01.pipeline import evaluate_all_experiments, run_all_experiments


def main():
    artifacts = run_all_experiments(overwrite=False, tensorboard=False)
    rows = evaluate_all_experiments(artifacts)
    outputs = generate_all_figures(artifacts, rows)

    print("Generated figures:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

