import gymnasium as gym
import numpy as np

from src.util.vecnormalize import VecNormalizeGymEnv


class ActionRescaleWrapper(gym.ActionWrapper):
    def __init__(self, env, new_low=-1.0, new_high=1.0):
        super().__init__(env)
        self.old_low = env.action_space.low
        self.old_high = env.action_space.high
        self.new_low = new_low
        self.new_high = new_high
        self.action_space = gym.spaces.Box(low=new_low, high=new_high, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        # Rescale from [-1,1] to [old_low, old_high]
        scaled_action = self.old_low + (action - self.new_low) * (self.old_high - self.old_low) / (self.new_high - self.new_low)
        return np.clip(scaled_action, self.old_low, self.old_high)

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")

    print("Original action space:", env.action_space)

    # Wrap the environment with both wrappers
    env = ActionRescaleWrapper(env, new_low=-1, new_high=1)
    env = VecNormalizeGymEnv(env)

    print("Rescaled action space:", env.action_space)
    print("Original action space:", env.norm_obs)

    obs, info = env.reset()
    for _ in range(3):
        action = env.action_space.sample()  # From [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break