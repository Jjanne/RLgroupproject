"""
Q-learning with a distance-based reward on MountainCar.

This script trains a Q-learning agent, but instead of using the default
reward (-1 per step), it uses the cars position as the reward signal.

The idea is to give the agent more useful feedback. In the standard setup,
the agent only gets negative rewards, which makes learning slow. Here,
we reward the agent for getting closer to the goal.

Key idea:
- reward = position of the car
- higher position means better reward

This encourages the agent to move toward the right side of the hill
more directly, instead of only learning through trial and error.

Outputs:
- Q-table saved in results/models/
- training rewards saved in results/logs/
"""

import gymnasium as gym
import numpy as np
import pickle
import os


def get_discrete_state(state, env, bins):
    low = env.observation_space.low
    high = env.observation_space.high

    scaled = (state - low) / (high - low)
    discrete = (scaled * bins).astype(int)
    discrete = np.clip(discrete, 0, bins - 1)

    return tuple(discrete)


def train():
    env = gym.make("MountainCar-v0")

    bins = np.array([20, 20])
    q_table = np.random.uniform(low=-2, high=0, size=(bins[0], bins[1], env.action_space.n))

    episodes = 5000
    alpha = 0.1
    gamma = 0.99

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, env, bins)

        total_reward = 0

        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[discrete_state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_discrete = get_discrete_state(next_state, env, bins)

            # 🔥 NEW REWARD: distance-based
            position = next_state[0]
            reward = position  # closer to goal → higher reward

            max_future_q = np.max(q_table[next_discrete])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_table[discrete_state + (action,)] = new_q

            discrete_state = next_discrete
            total_reward += reward

        rewards.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 500 == 0:
            print(f"Episode {ep}, avg reward: {np.mean(rewards[-100:]):.2f}")

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    with open("results/models/q_table_distance.pkl", "wb") as f:
        pickle.dump(q_table, f)

    np.save("results/logs/qlearning_distance_rewards.npy", rewards)

    print("Done training (distance reward)")


if __name__ == "__main__":
    train()