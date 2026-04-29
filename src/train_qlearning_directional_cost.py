"""Train the adapted discrete environment with left/right-specific action costs."""

from part01.config import PART01_EXPERIMENTS
from part01.pipeline import train_experiment


def main():
    config = next(config for config in PART01_EXPERIMENTS if config.slug == "discrete_directional_cost")
    train_experiment(config, overwrite=True, tensorboard=False)


if __name__ == "__main__":
    main()

