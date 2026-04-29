"""
Plots and comparison

This file is used to visualize the results of the different experiments.

It plots:
- how the rewards evolve during training
- the learned policy
- a comparison between Q-learning, SARSA, and the action-cost version

This helps to clearly see how each method behaves.
"""

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
    plt.close()


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
    plt.close()

    env.close()

def plot_comparison():
    q_rewards = np.load("results/logs/rewards.npy")
    sarsa_rewards = np.load("results/logs/sarsa_rewards.npy")

    q_avg = moving_average(q_rewards, window=100)
    sarsa_avg = moving_average(sarsa_rewards, window=100)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(q_avg, label="Q-learning")
    plt.plot(sarsa_avg, label="SARSA")
    plt.title("Q-learning vs SARSA on MountainCar-v0")
    plt.xlabel("Episode")
    plt.ylabel("Average reward over 100 episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/qlearning_vs_sarsa.png")
    plt.close()

def plot_all_training_curves():
    q_rewards = np.load("results/logs/rewards.npy")
    sarsa_rewards = np.load("results/logs/sarsa_rewards.npy")
    action_cost_rewards = np.load("results/logs/qlearning_action_cost_rewards.npy")

    q_avg = moving_average(q_rewards, window=100)
    sarsa_avg = moving_average(sarsa_rewards, window=100)
    action_cost_avg = moving_average(action_cost_rewards, window=100)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(q_avg, label="Q-learning baseline")
    plt.plot(sarsa_avg, label="SARSA")
    plt.plot(action_cost_avg, label="Q-learning with action cost")
    plt.title("MountainCar-v0 Training Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average reward over 100 episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/all_training_curves.png")
    plt.close()

if __name__ == "__main__":
    plot_rewards()
    plot_policy()
    plot_comparison()
    plot_all_training_curves()