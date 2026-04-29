"""Train the adapted discrete environment with a uniform non-null action cost."""

from part01.config import PART01_EXPERIMENTS
from part01.pipeline import train_experiment


def main():
    config = next(config for config in PART01_EXPERIMENTS if config.slug == "discrete_non_null_cost")
    train_experiment(config, overwrite=True, tensorboard=False)


if __name__ == "__main__":
    main()

