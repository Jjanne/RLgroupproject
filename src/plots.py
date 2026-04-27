import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gymnasium as gym


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


def plot_policy():
    env = gym.make("MountainCar-v0")

    with open("results/models/q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    policy = np.argmax(q_table, axis=2)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(policy.T, origin="lower", aspect="auto")
    plt.title("Learned Q-learning Policy for MountainCar-v0")
    plt.xlabel("Position bins")
    plt.ylabel("Velocity bins")
    plt.colorbar(label="Action: 0=left, 1=no push, 2=right")
    plt.savefig("results/plots/qlearning_policy.png")
    plt.show()

    env.close()


if __name__ == "__main__":
    plot_rewards()
    plot_policy()