import numpy as np
import gymnasium as gym
import pickle
from copy import deepcopy
from typing import Optional, Union

class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's algorithm.
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class VecNormalizeGymEnv(gym.Wrapper):
    """
    Gym Wrapper for normalizing observations.
    Works with gymnasium.Env with ndarray or Dict observation spaces.

    :param env: gymnasium environment
    :param norm_obs: whether to normalize observations
    :param clip_obs: clipping value for normalized observations
    :param epsilon: small value to avoid division by zero
    :param norm_obs_keys: for Dict observations, which keys to normalize. If None, normalize all keys.
    """
    def __init__(
        self,
        env: gym.Env,
        norm_obs: bool = True,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
        norm_obs_keys: Optional[list[str]] = None,
    ):
        super().__init__(env)
        self.norm_obs = norm_obs
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.norm_obs_keys = norm_obs_keys

        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_spaces = obs_space.spaces
            if self.norm_obs_keys is None:
                self.norm_obs_keys = list(self.obs_spaces.keys())

            # Initialize RunningMeanStd per key
            self.obs_rms = {
                key: RunningMeanStd(shape=self.obs_spaces[key].shape)
                for key in self.norm_obs_keys
            }
        elif isinstance(obs_space, gym.spaces.Box):
            if self.norm_obs_keys is not None:
                raise ValueError("norm_obs_keys only valid with Dict observation spaces")
            self.obs_rms = RunningMeanStd(shape=obs_space.shape)
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

        self.training = True
        self.last_obs = None

    def normalize_obs(self, obs: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
        if not self.norm_obs:
            return obs
        obs_ = deepcopy(obs)
        if isinstance(obs, dict):
            for key in self.norm_obs_keys:
                obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
        else:
            obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
        return obs_

    def _normalize_obs(self, obs: np.ndarray, rms: RunningMeanStd) -> np.ndarray:
        return np.clip((obs - rms.mean) / np.sqrt(rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def update_obs_rms(self, obs: Union[np.ndarray, dict]) -> None:
        if not self.training or not self.norm_obs:
            return

        if isinstance(obs, dict):
            for key in self.norm_obs_keys:
                self.obs_rms[key].update(np.array(obs[key])[None, ...])  # shape (1, ...)
        else:
            self.obs_rms.update(np.array(obs)[None, ...])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.update_obs_rms(obs)
        self.last_obs = obs
        return self.normalize_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_obs_rms(obs)
        self.last_obs = obs
        return self.normalize_obs(obs), reward, terminated, truncated, info

    def save(self, filepath: str):
        """Save normalization statistics."""
        with open(filepath, "wb") as f:
            pickle.dump(self.obs_rms, f)

    def load(self, filepath: str):
        """Load normalization statistics."""
        with open(filepath, "rb") as f:
            self.obs_rms = pickle.load(f)

    def get_original_obs(self) -> Union[np.ndarray, dict]:
        """Return unnormalized last observation."""
        return deepcopy(self.last_obs)

if __name__ == "__main__":
    env_id = "Walker2d-v5"
    env = gym.make(env_id)
    env = VecNormalizeGymEnv(env, norm_obs=True)

    obs_, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs_, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs_, info = env.reset()

    # Save stats
    env.save("obs_norm_stats.pkl")

    # Later or in eval
    env = gym.make(env_id)
    env = VecNormalizeGymEnv(env, norm_obs=True)
    env.load("obs_norm_stats.pkl")

    obs_ = env.reset()  # normalized obs
