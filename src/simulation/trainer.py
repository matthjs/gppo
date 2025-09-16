import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import wandb
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulatorRL:
    """High-level trainer supporting SB3 agents or custom agents."""

    def __init__(
        self,
        env_manager: Any,
        agent: Any,
        callbacks: Optional[List[Any]] = None,
        metric_trackers: Optional[List[Any]] = None,
        device: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_run: Optional[Any] = None,
        wandb_log_interval: int = 1000,
        wandb_checkpoint_freq: int = 0,
        wandb_checkpoint_dir: str = "./checkpoints",
    ):
        self.env_manager = env_manager
        self.agent = agent
        self.callbacks = callbacks or []
        self.metric_trackers = metric_trackers or []
        self.device = device
        self.total_steps = 0
        self.obs = None

        # WandB setup
        self.wandb_config = wandb_config
        self.wandb_run = wandb_run
        self.wandb_log_interval = wandb_log_interval
        self.wandb_checkpoint_freq = wandb_checkpoint_freq
        self.wandb_checkpoint_dir = wandb_checkpoint_dir
        self._owns_wandb = False

        if wandb_config or wandb_run:
            self._init_wandb()

        # SB3-specific setup
        if isinstance(agent, BaseAlgorithm):
            logger.info("Detected SB3 agent; attaching environment and callbacks")
            agent.set_env(env_manager.vec_env)
            self.sb3_adapter = SB3CallbackAdapter(
                metric_callbacks=self.callbacks,
                metric_trackers=self.metric_trackers,
            )
        else:
            self.sb3_adapter = None

    def _init_wandb(self):
        pass

    def _call_callbacks(self, fn_name: str, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, fn_name)(self, *args, **kwargs)

    def train(self, total_timesteps: int, batch_size: Optional[int] = None, rollout_length: Optional[int] = None):
        if isinstance(self.agent, BaseAlgorithm):
            self._train_sb3(total_timesteps)
        else:
            self._train_custom(total_timesteps, batch_size, rollout_length)

    def _train_sb3(self, total_timesteps: int):
        logger.info("Delegating training to SB3 agent.learn")
        self._call_callbacks("on_train_begin")
        callbacks = [self.sb3_adapter]
        # if getattr(self, "_sb3_wandb_cb", None):
            # callbacks.append(self._sb3_wandb_cb)

        self.agent.learn(total_timesteps=total_timesteps, callback=callbacks)
        self._call_callbacks("on_train_end")

        if self._owns_wandb and self.wandb_run:
            self.wandb_run.finish()

    def _train_custom(self, total_timesteps: int, rollout_length: Optional[int]):
        logger.info(f"Running custom training loop: total_timesteps={total_timesteps}")
        obs = self.env_manager.reset()
        self.obs = obs
        self._call_callbacks("on_train_begin")

        n_envs = self.env_manager.n_envs
        rollout_length = rollout_length or 128
            # Update agent
            #self.agent.learn()

            # self._log_metrics()
            # self._maybe_checkpoint()
            # self._call_callbacks("on_epoch_end", self.total_steps)

        self._call_callbacks("on_train_end")
        if self._owns_wandb and self.wandb_run:
            self.wandb_run.finish()

    def _log_metrics(self):
        pass

    def _maybe_checkpoint(self):
        pass

    def get_metrics(self) -> Dict[str, float]:
        pass