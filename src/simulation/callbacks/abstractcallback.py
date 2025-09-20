from abc import ABC
from types import SimpleNamespace
from typing import Any, Optional
import pandas as pd
from src.agents.agent import Agent
from src.simulation.simulatorldata import SimulatorRLData


class AbstractCallback(ABC):
    """
    Abstract base class for callbacks in reinforcement learning experiments.
    """

    def __init__(self) -> None:
        """
        Constructor for AbstractCallback.
        """
        self.mode: Optional[str] = None
        self.num_steps: int = 0
        self.num_episodes: int = 0
        self.logging: bool = False
        self.extra: Optional[Any] = None
        self.agent: Optional[Agent] = None
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[Any] = None
        self.metrics_tracker_registry: Optional[Any] = None
        self.df: Optional[pd.DataFrame] = None
        self.experiment_id: Optional[str] = None
        self.old_obs: Optional[Any] = None

    def init_callback(self, data: SimulatorRLData) -> None:
        pass

    def on_step(self, action: Any, reward: float, next_obs: Any, done: bool) -> bool:
        """
        Callback for each step of the environment.

        :param action: The action taken by the agent.
        :param reward: The reward received from the environment.
        :param new_obs: The new observation from the environment.
        :param done: Whether the episode is done.
        :return: Whether to continue the run.
        """
        self.num_steps += 1
        self.old_obs = next_obs
        return True
    
    def on_learn(self, learning_info) -> None:
        pass

    def on_episode_end(self) -> None:
        """
        Callback at the end of an episode.
        """
        self.num_episodes += 1

    def on_training_start(self) -> None:
        """
        Callback at the start of training.
        """
        pass

    def on_training_end(self) -> None:
        """
        Callback at the end of training.
        """
        pass

    def on_update_start(self) -> None:
        """
        Callback at the start of an update.
        """
        pass

    def on_update_end(self) -> None:
        """
        Callback at the end of an update.
        """
        pass
