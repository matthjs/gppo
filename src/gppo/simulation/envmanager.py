from __future__ import annotations

import abc
import logging
import multiprocessing as mp
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
import wandb
import numpy as np
import gymnasium as gym
from ale_py.vector_env import AtariVectorEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EnvManager:
    """Create and manage vectorized environments with optional normalization.

    Parameters
    ----------
    env_fns:
        Sequence of callables that return a fresh gym.Env when called. Usually
        created with a lambda: ``[lambda: make_env(i) for i in range(n_envs)]``.
    n_envs:
        If env_fns is None, env_fn is duplicated n_envs times.
    use_subproc:
        If True and SubprocVecEnv is available, use SubprocVecEnv. Otherwise use DummyVecEnv.
    norm_obs, norm_reward:
        Whether to wrap with VecNormalize (if available).
    """

    def __init__(
        self,
        env_id: str,
        training: bool = True,
        env_fns: Optional[Sequence[Callable[[], Any]]] = None,
        env_fn: Optional[Callable[[], Any]] = None,
        n_envs: int = 1,
        use_subproc: bool = True,
        norm_obs: bool = False,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
        norm_obs_keys: Optional[list[str]] = None,
        gamma: float = 0.99,
        norm_reward: bool = False,
    ) -> None:
        self.env_id = env_id

        if env_fns is None and env_fn is None:
            raise ValueError(
                "Provide env_fns (sequence) or env_fn (single callable)")

        if env_fns is None:
            env_fns = [env_fn for _ in range(n_envs)]
        else:
            n_envs = len(env_fns)

        # e.g. used to determine whether to update moving averages for VecNormalize
        self.training = training

        self.env_fns = env_fns
        self.n_envs = n_envs
        self.use_subproc = use_subproc and SubprocVecEnv is not None
        self.gamma = gamma

        # Normalize observations settings
        self.norm_obs = norm_obs
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.norm_obs_keys = norm_obs_keys

        self.norm_reward = norm_reward

        self.vec_env = None
        self._create_vec_env()

    def _create_vec_env(self) -> None:
        # If env_fns contains an AtariVectorEnv already, use it directly
        if len(self.env_fns) == 1 and isinstance(self.env_fns[0](), AtariVectorEnv):
            logger.info("Using native AtariVectorEnv directly")
            self.vec_env = self.env_fns[0]()
            return

        # Otherwise, use standard SB3 VecEnv logic
        if self.use_subproc:
            logger.info(
                "Creating SubprocVecEnv with %d processes", self.n_envs)
            self.vec_env = SubprocVecEnv(list(self.env_fns))
        else:
            logger.info("Creating DummyVecEnv with %d envs", self.n_envs)
            self.vec_env = DummyVecEnv(list(self.env_fns))

        # Optional normalization
        if (self.norm_obs or self.norm_reward) and VecNormalize is not None:
            logger.info("Wrapping vec_env with VecNormalize (obs=%s, reward=%s)",
                        self.norm_obs, self.norm_reward)
            self.vec_env = VecNormalize(
                self.vec_env,
                training=self.training,
                norm_obs=self.norm_obs,
                clip_obs=self.clip_obs,
                norm_reward=self.norm_reward,
                gamma=self.gamma,
                epsilon=self.epsilon,
                norm_obs_keys=self.norm_obs_keys,
            )

    def reset(self) -> np.ndarray:
        return self.vec_env.reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        return self.vec_env.step(actions)

    def close(self) -> None:
        self.vec_env.close()

    def render(self, **kwargs) -> None:
        # Delegate rendering to first env if supported
        return self.vec_env.render(**kwargs)


def make_env():
    return gym.make("CartPole-v1")  # simple test environment


def test_sb3():
    manager = EnvManager(env_fn=make_env, n_envs=4, norm_obs=True)

    # SB3 expects a vectorized environment
    model = PPO("MlpPolicy", manager.vec_env, verbose=1)
    model.learn(total_timesteps=1000)

    obs = manager.reset()
    done = np.array([False])
    while not done.any():
        action, _ = model.predict(obs)
        obs, reward, done, info = manager.step(action)

    manager.close()

# Simple main script to test out functionality


def main():
    print("=== Single Environment ===")
    manager = EnvManager(env_fn=make_env, n_envs=1)
    obs = manager.reset()
    print("Reset output:", obs)

    done = False
    while not done:
        action = np.array([manager.vec_env.action_space.sample()])
        obs, reward, done, info = manager.step(action)
    manager.close()

    print("\n=== Multiple Environments ===")
    manager_multi = EnvManager(
        env_fn=make_env, n_envs=3, use_subproc=True, norm_obs=True)
    obs_multi = manager_multi.reset()
    print("Reset output shape:", obs_multi.shape)

    actions = np.array([manager_multi.vec_env.action_space.sample()
                       for _ in range(3)])
    obs_multi, rewards, dones, infos = manager_multi.step(actions)
    print("Step outputs:")
    print("Observations:", obs_multi)
    print("Rewards:", rewards)
    manager_multi.close()


if __name__ == "__main__":
    test_sb3()
