from __future__ import annotations

import abc
import logging
import multiprocessing as mp
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional imports
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.base_class import BaseAlgorithm
    SB3_AVAILABLE = True
except Exception:
    SubprocVecEnv = None
    DummyVecEnv = None
    VecNormalize = None
    BaseCallback = object
    BaseAlgorithm = object
    SB3_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

import numpy as np

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
        if env_fns is None and env_fn is None:
            raise ValueError("Provide env_fns (sequence) or env_fn (single callable)")

        if env_fns is None:
            env_fns = [env_fn for _ in range(n_envs)]
        else:
            n_envs = len(env_fns)

        self.training = training   # e.g. used to determine whether to update moving averages for VecNormalize

        self.env_fns = env_fns
        self.n_envs = n_envs
        self.use_subproc = use_subproc and SB3_AVAILABLE and SubprocVecEnv is not None
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
        if self.use_subproc:
            logger.info("Creating SubprocVecEnv with %d processes", self.n_envs)
            self.vec_env = SubprocVecEnv(list(self.env_fns))
        else:
            logger.info("Creating DummyVecEnv with %d envs", self.n_envs)
            self.vec_env = DummyVecEnv(list(self.env_fns))

        if (self.norm_obs or self.norm_reward) and VecNormalize is not None:
            logger.info("Wrapping vec_env with VecNormalize (obs=%s, reward=%s)", self.norm_obs, self.norm_reward)
            # TODO: Double check that SB3 VecNormalize behaves the same way as custom implementation
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
        try:
            self.vec_env.close()
        except Exception:
            pass

    def render(self, **kwargs) -> None:
        # Delegate rendering to first env if supported
        try:
            return self.vec_env.render(**kwargs)
        except Exception:
            return None

    @contextmanager
    def context(self):
        try:
            yield self
        finally:
            self.close()