"""
Q-learning with action cost

Here we modified the reward function by adding a small penalty when the agent
accelerates left or right.

The idea was to simulate a cost for using force (like fuel).

Since MountainCar actually needs a lot of back-and-forth movement to build
momentum, this makes the task harder, and the agent performs worse.
"""

import gymnasium as gym
import numpy as np
import pickle
import os
from wrappers import ActionCostWrapper


def get_discrete_state(state, env, bins):
    low = env.observation_space.low
    high = env.observation_space.high

    scaled = (state - low) / (high - low)
    discrete = (scaled * bins).astype(int)
    discrete = np.clip(discrete, 0, bins - 1)

    return tuple(discrete)


def train():
    env = gym.make("MountainCar-v0")
    env = ActionCostWrapper(env, action_cost=0.1)

    bins = np.array([40, 40])
    q_table = np.zeros((bins[0], bins[1], env.action_space.n))

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    episodes = 5000

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = get_discrete_state(state, env, bins)

        done = False
        total_reward = 0

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_discrete = get_discrete_state(next_state, env, bins)

            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state_discrete])

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state + (action,)] = new_value

            state = next_state_discrete
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-100:])
            print(f"Episode {ep+1}, avg reward: {avg:.2f}, epsilon: {epsilon:.3f}")

    env.close()

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    with open("results/models/q_table_action_cost.pkl", "wb") as f:
        pickle.dump(q_table, f)

    np.save("results/logs/qlearning_action_cost_rewards.npy", rewards)

    print("Done training Q-learning with action cost")


if __name__ == "__main__":
    train()