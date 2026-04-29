"""
Plots for the MountainCar experiments.

This file creates the visualizations used to compare the different RL methods:
- Q-learning baseline
- SARSA
- Q-learning with action cost
- Q-learning with distance-based reward

The plots help show how rewards change during training and how each method
performs compared to the others.

Main outputs:
- Q-learning reward curve
- Learned policy heatmap
- Q-learning vs SARSA comparison
- Full comparison of all methods
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

def plot_all_training_curves():
    q_rewards = np.load("results/logs/rewards.npy")
    sarsa_rewards = np.load("results/logs/sarsa_rewards.npy")
    action_cost_rewards = np.load("results/logs/qlearning_action_cost_rewards.npy")
    distance_rewards = np.load("results/logs/qlearning_distance_rewards.npy")

    q_avg = moving_average(q_rewards, window=100)
    sarsa_avg = moving_average(sarsa_rewards, window=100)
    action_cost_avg = moving_average(action_cost_rewards, window=100)
    distance_avg = moving_average(distance_rewards, window=100)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(q_avg, label="Q-learning baseline")
    plt.plot(sarsa_avg, label="SARSA")
    plt.plot(action_cost_avg, label="Q-learning with action cost")
    plt.plot(distance_avg, label="Q-learning with distance reward")
    plt.title("MountainCar-v0 Training Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average reward over 100 episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/all_training_curves.png")
    plt.close()

def get_discrete_state_for_plot(state, env, bins):
    low = env.observation_space.low
    high = env.observation_space.high

    scaled = (state - low) / (high - low)
    discrete = (scaled * bins).astype(int)
    discrete = np.clip(discrete, 0, bins - 1)

    return tuple(discrete)

def plot_trajectory():
    env = gym.make("MountainCar-v0")

    with open("results/models/q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    bins = np.array([40, 40])

    state, _ = env.reset()
    discrete_state = get_discrete_state_for_plot(state, env, bins)

    positions = []
    velocities = []

    done = False

    while not done:
        positions.append(state[0])
        velocities.append(state[1])

        action = np.argmax(q_table[discrete_state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        discrete_state = get_discrete_state_for_plot(state, env, bins)

    env.close()

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(positions, velocities, marker="o", markersize=2)
    plt.title("Q-learning Trajectory in Phase Space")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.savefig("results/plots/qlearning_trajectory.png")
    plt.close()

if __name__ == "__main__":
    plot_rewards()
    plot_policy()
    plot_comparison()
    plot_all_training_curves()
    plot_trajectory()