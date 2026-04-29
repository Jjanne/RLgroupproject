"""
Evaluate all trained agents on the same standard MountainCar-v0 environment.

This is important because some training methods use different reward designs.
To compare them fairly, we test all saved Q-tables in the original environment.

Metrics:
- average reward
- average steps
- success rate
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


def evaluate_model(name, model_path, bins, episodes=100):
    env = gym.make("MountainCar-v0")

    with open(model_path, "rb") as f:
        q_table = pickle.load(f)

    rewards = []
    steps = []
    successes = 0

    for _ in range(episodes):
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

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Average steps: {np.mean(steps):.2f}")
    print(f"Success rate: {successes / episodes * 100:.2f}%")


def main():
    evaluate_model(
        "Q-learning baseline",
        "results/models/q_table.pkl",
        bins=np.array([40, 40])
    )

    evaluate_model(
        "SARSA",
        "results/models/sarsa_q_table.pkl",
        bins=np.array([40, 40])
    )

    evaluate_model(
        "Q-learning with action cost",
        "results/models/q_table_action_cost.pkl",
        bins=np.array([40, 40])
    )

    evaluate_model(
        "Q-learning with distance reward",
        "results/models/q_table_distance.pkl",
        bins=np.array([20, 20])
    )


if __name__ == "__main__":
    main()