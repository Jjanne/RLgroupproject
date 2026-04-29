# Reinforcement Learning Project: Q-learning vs SARSA on MountainCar

## Overview

In this project, we implemented and compared two reinforcement learning algorithms:

- Q-learning (off-policy)
- SARSA (on-policy)
- Q-learning with an action cost (reward penalty)
- Q-learning with a distance-based reward (reward shaping)

The agents are trained on the MountainCar-v0 environment from Gymnasium.

The goal is to learn a policy that allows the car to reach the top of the hill. Since the car does not have enough power to go straight up, it must learn to build momentum by moving back and forth.

To explore how reward design impacts learning, we introduced two variations:
- Action cost: adds a penalty when the agent accelerates, making the task harder and encouraging more efficient actions
- Distance reward: rewards progress toward the goal, helping the agent learn faster by guiding exploration

These variations allow us to compare not only different algorithms, but also how modifying the reward function affects performance.

## Project Structure

```

RLgroupproject/
├── src/
│   ├── train_qlearning.py
│   ├── train_sarsa.py
│   ├── train_qlearning_action_cost.py
│   ├── train_qlearning_distance_reward.py
│   ├── evaluate.py
│   ├── evaluate_all.py
│   ├── compare_results.py
│   ├── plots.py
│   ├── utils.py
│   └── wrappers.py
│
├── results/
│   ├── logs/
│   ├── models/
│   └── plots/
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the Q-learning agent

python src/train_qlearning.py

### 3. Train the SARSA agent

python src/train_sarsa.py

### 4. Train the action-cost variation

python src/train_qlearning_action_cost.py

This version adds a small penalty when the agent accelerates, making the task harder.

### 5. Evaluate the trained agent

python src/evaluate.py

This runs the agent and prints performance metrics in the terminal.

### 6. Generate plots

python src/plots.py

This will create visualizations of the training progress and comparisons between the different methods.

