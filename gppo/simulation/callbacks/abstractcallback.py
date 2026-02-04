from abc import ABC
from types import SimpleNamespace
from typing import Any, Optional
import pandas as pd
from gppo.agents.agent import Agent
from gppo.simulation.simulatorldata import SimulatorRLData


class AbstractCallback(ABC):
    """
    Abstract base class for callbacks in reinforcement learning experiments.
    """

    def __init__(self) -> None:
        """
        Constructor for AbstractCallback.
        """
        self.num_steps = 0
        self.old_obs = None

    def init_callback(self, data: SimulatorRLData) -> None:
        pass

    def on_step(self, action, reward, next_obs, done) -> bool:
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

    def on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        learning_info: optional learning information from previous rollout.
        """
        pass

    def on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
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
