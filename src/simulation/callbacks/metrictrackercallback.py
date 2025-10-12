import logging
import os
from typing import List
from git import Union
import numpy as np
from loguru import logger
from src.metrics.metrictrackerNEW import MetricsTracker
from src.simulation.callbacks.abstractcallback import AbstractCallback
from src.simulation.simulatorldata import SimulatorRLData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetricTrackerCallback(AbstractCallback):
    def __init__(self, run_id: Union[str, int], metric_tracker: MetricsTracker = None) -> None:
        super().__init__()
        if isinstance(run_id, int):
            run_id = str(run_id)
        self.run_id = run_id
        self.n_env = None
        self.episode_returns = None
        self.completed_returns = None
        self.episodes_finished = None
        self.num_episodes = None
        self.metrics_tracker = metric_tracker
        self.experiment_id = None
        self.agent_id = None

        # this is a bit sketchy
        self.save_path = self.metrics_tracker.save_path if self.metrics_tracker else "./"

    def init_callback(self, data: SimulatorRLData) -> None:
        super().init_callback(data)
        # self.episode_reward = 0
        self.n_env = data.n_env
        self.agent_id = data.agent_id
        self.num_episodes = data.num_episodes
        # Track ongoing returns for each environment
        self.episode_returns = np.zeros(self.n_env, dtype=np.float32)
        self.completed_returns: List[float] = []
        self.episodes_finished = 0
        self.experiment_id = data.experiment_id
        self.env_id = data.env_id


    def on_step(self, action, reward, next_obs, done) -> bool:
        """
        Callback for each step of the environment.

        :param action: The action taken by the agent.
        :param reward: The reward received from the environment.
        :param new_obs: The new observation from the environment.
        :param done: Whether the episode is done.
        :return: Whether to continue the experiment.
        """
        super().on_step(action, reward, next_obs, done)
        self.episode_returns += reward
        # Handle finished episodes
        for i in range(self.n_env):
            if done[i]:
                episode_return = self.episode_returns[i]
                self.completed_returns.append(self.episode_returns[i])
                self.episodes_finished += 1
                logger.info(f"Episode {self.episodes_finished} finished with return {self.episode_returns[i]:.2f}")
                self.episode_returns[i] = 0.0  # reset for next episode
                if self.metrics_tracker:
                    self.metrics_tracker.record_metric("return", agent_id=self.agent_id,
                                            episode_idx=self.episodes_finished, value=episode_return, run_id=self.run_id)
        return True
    
    def on_rollout_start(self, learning_info = None) -> None:
        super().on_rollout_start(learning_info)
        if learning_info and self.metrics_tracker:
            for key, value in learning_info.items():
                self.metrics_tracker.record_metric(key, self.agent_id, self.episodes_finished, value, run_id=self.run_id)
    
    def on_learn(self, learning_info) -> None:
        super().on_learn(learning_info)
        if learning_info and self.metrics_tracker:
            for key, value in learning_info.items():
                self.metrics_tracker.record_metric(key, self.agent_id, self.episodes_finished, value, run_id=self.run_id)

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        super().on_episode_end()

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        super().on_training_end()
        if self.metrics_tracker:
            # self.metrics_tracker.plot_all_metrics(num_episodes=self.num_episodes)
            self.metrics_tracker.save_all_runs(os.path.join(self.save_path, self.experiment_id), self.agent_id, self.env_id)
            self.metrics_tracker.save_env_aggregated_plots(os.path.join(self.save_path, self.experiment_id), self.env_id)
