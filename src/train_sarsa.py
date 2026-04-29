"""
SARSA (on-policy)

This file implements SARSA on the same MountainCar problem.

Unlike Q-learning, SARSA updates based on the action the agent actually takes,
so it tends to be a bit more cautious.

In practice, this usually means it learns a bit slower, but more steadily.
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


def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 3)
    return np.argmax(q_table[state])


def train():
    env = gym.make("MountainCar-v0")

    bins = np.array([40, 40])
    q_table = np.random.uniform(low=-2, high=0, size=(40, 40, 3))

    episodes = 5000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.05

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = get_discrete_state(state, env, bins)

        action = choose_action(state, q_table, epsilon)

        done = False
        total_reward = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_discrete = get_discrete_state(next_state, env, bins)
            next_action = choose_action(next_state_discrete, q_table, epsilon)

            # SARSA update
            q_table[state][action] += alpha * (
                reward
                + gamma * q_table[next_state_discrete][next_action]
                - q_table[state][action]
            )

            state = next_state_discrete
            action = next_action

            total_reward += reward

        rewards.append(total_reward)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-100:])
            print(f"Episode {ep+1}, avg reward: {avg:.2f}, epsilon: {epsilon:.3f}")

    env.close()

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    with open("results/models/sarsa_q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    np.save("results/logs/sarsa_rewards.npy", rewards)

    print("Done training SARSA")


if __name__ == "__main__":
    train()