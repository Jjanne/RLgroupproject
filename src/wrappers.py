import gymnasium as gym


class ActionCostWrapper(gym.Wrapper):
    def __init__(self, env, action_cost=0.1):
        super().__init__(env)
        self.action_cost = action_cost

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Penalize active acceleration actions: left or right
        if action in [0, 2]:
            reward -= self.action_cost

        return next_state, reward, terminated, truncated, info