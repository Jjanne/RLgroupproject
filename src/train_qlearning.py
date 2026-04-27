import gymnasium as gym
import numpy as np
import pickle
import os


# Convert continuous state to discrete bins
def get_discrete_state(state, env, bins):
    low = env.observation_space.low
    high = env.observation_space.high

    scaled = (state - low) / (high - low)
    discrete = (scaled * bins).astype(int)

    # just in case it goes out of bounds
    discrete = np.clip(discrete, 0, bins - 1)

    return tuple(discrete)


def train():
    env = gym.make("MountainCar-v0")

    # number of bins for position and velocity
    bins = np.array([40, 40])

    # Q-table initialization
    q_table = np.zeros((bins[0], bins[1], env.action_space.n))

    # hyperparameters 
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
            # exploration vs exploitation
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_discrete = get_discrete_state(next_state, env, bins)

            # Q-learning update
            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state_discrete])

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state + (action,)] = new_value

            state = next_state_discrete
            total_reward += reward

        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        # print progress 
        if (ep + 1) % 500 == 0:
            avg_last = np.mean(rewards[-500:])
            print(f"Episode {ep+1}, avg reward: {avg_last:.2f}, epsilon: {epsilon:.3f}")

    env.close()

    return q_table, rewards


if __name__ == "__main__":
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    q_table, rewards = train()

    with open("results/models/q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    np.save("results/logs/rewards.npy", np.array(rewards))

    print("Done training")