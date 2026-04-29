"""
Simple comparison of the RL methods we tested on MountainCar.

In this script, we look at how three agents performed:
- Q-learning
- SARSA
- Q-learning with an added action cost

For each one, we check:
- the average reward over the last 500 episodes
- how much the results vary (standard deviation)
- the best and worst episodes

"""

import numpy as np


def summarize(name, rewards):
    last_rewards = rewards[-500:]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Mean reward last 500 episodes: {np.mean(last_rewards):.2f}")
    print(f"Std reward last 500 episodes: {np.std(last_rewards):.2f}")
    print(f"Best reward: {np.max(rewards):.2f}")
    print(f"Worst reward: {np.min(rewards):.2f}")


def main():
    q_rewards = np.load("results/logs/rewards.npy")
    sarsa_rewards = np.load("results/logs/sarsa_rewards.npy")
    action_cost_rewards = np.load("results/logs/qlearning_action_cost_rewards.npy")

    summarize("Q-learning baseline", q_rewards)
    summarize("SARSA", sarsa_rewards)
    summarize("Q-learning with action cost", action_cost_rewards)

distance_rewards = np.load("results/logs/qlearning_distance_rewards.npy")
summarize("Q-learning (distance reward)", distance_rewards)

if __name__ == "__main__":
    main()