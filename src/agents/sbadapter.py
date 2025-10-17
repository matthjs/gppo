from abc import abstractmethod
from typing import Dict, Union
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import torch
from traitlets import Any
from src.agents.agent import Agent


class StableBaselinesAdapter(Agent):
    """
    Adapter class to use Stable Baselines models as agents.
    """

    def __init__(self, model: BaseAlgorithm) -> None:
        """
        Constructor for StableBaselinesAdapter.

        :param model: Stable Baselines model to adapt.
        """
        self._sb_model = model
        # self.entropy_values = []
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store_transition(
            self,
            state: np.ndarray,
            action: Union[int, np.ndarray],
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        pass

    def learn(self) -> Dict[str, Any]:
        """!
        Perform learning update.
        This method should be implemented by the child class.
        Can return a dictionary e.g. {'loss': 0.1}.
        """
        rollout_buffer = self._sb_model.rollout_buffer
        if not rollout_buffer.full:
            return {}

        # entropy = np.mean(self.entropy_values)
        # self.entropy_values.clear()
        return {}

    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
        action = self._sb_model.predict(observation)[0] # [0]
        # values, log_prob, entropy = self._sb_model.policy.evaluate_actions(
        #                torch.from_numpy(observation),
        #                torch.from_numpy(action)
        #            )
        #self.entropy_values.append(entropy)
        return action
    
    def full_buffer(self) -> bool:
        return self._sb_model.rollout_buffer.full

    def is_stable_baselines_wrapper(self) -> bool:
        """
        Check if the agent is a wrapper for a Stable Baselines model.

        :return: True if it is, False otherwise.
        """
        return True

    def stable_baselines_unwrapped(self) -> BaseAlgorithm:
        """
        Get the unwrapped Stable Baselines model.

        :return: The unwrapped model.
        """
        return self._sb_model
    
    def save(self, path: str) -> None:
        """
        Save the Stable Baselines model to a file.

        :param path: File path (without extension) to save the model.
        """
        self._sb_model.save(path)

    def load(self, path: str) -> None:
        """
        Load a Stable Baselines model from a file and update the adapter.

        :param path: File path (without extension) from which to load the model.
        """
        model_class = type(self._sb_model)
        self._sb_model = model_class.load(path)
