"""
Evaluation of the trained RL agents on MountainCar.

This script runs a trained agent using its Q-table and measures how well it performs.

Instead of learning, the agent follows its learned policy and we track:
- average reward
- number of steps per episode
- success rate (reaching the goal)

This helps us understand how effective the learned policy is,
and allows us to compare different methods objectively.
"""

import gymnasium as gym
import numpy as np
import pickle


def get_discrete_state(state, env, bins):
    low = env.observation_space.low
    high = env.observation_space.high

    scaled = (state - low) / (high - low)
    discrete = (scaled * bins).astype(int)
    discrete = np.clip(discrete, 0, bins - 1)

    return tuple(discrete)


def evaluate_agent(episodes=100):
    env = gym.make("MountainCar-v0")

    bins = np.array([40, 40])

    with open("results/models/q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    rewards = []
    steps = []
    successes = 0

    for ep in range(episodes):
        state, _ = env.reset()
        state = get_discrete_state(state, env, bins)

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated:
                successes += 1

            state = get_discrete_state(next_state, env, bins)

            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)

    env.close()

    print("Evaluation results")
    print("------------------")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average steps: {np.mean(steps):.2f}")
    print(f"Success rate: {successes / episodes * 100:.2f}%")


if __name__ == "__main__":
    evaluate_agent()