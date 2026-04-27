import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(values, window=100):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_rewards():
    rewards = np.load("results/logs/rewards.npy")

    avg_rewards = moving_average(rewards, window=100)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.title("Q-learning Training Progress on MountainCar-v0")
    plt.xlabel("Episode")
    plt.ylabel("Average reward over 100 episodes")
    plt.grid(True)
    plt.savefig("results/plots/qlearning_rewards.png")
    plt.show()


if __name__ == "__main__":
    plot_rewards()