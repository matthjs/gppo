from abc import abstractmethod, ABC
import numpy as np
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.base_class import BaseAlgorithm


class Agent(ABC):
    # True interface, no implemented methods or constructor.
    @abstractmethod
    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
        """
        Based on state/observation, choose an action.
        """
        pass

    def update(self, params: Optional[Dict[str, Any]] = None):
        """
        An abstract update method that can be used to update some parts
        of the agent that may not have to do with updating model parameters.
        I.e., scheduling for epsilon greedy.
        Not required to implement this method.
        """
        pass

    @abstractmethod
    def store_transition(
            self,
            state: np.ndarray,
            action: Union[int, np.ndarray],
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        pass

    @abstractmethod
    def learn(self) -> Dict[str, Any]:
        """!
        Perform learning update.
        This method should be implemented by the child class.
        Can return a dictionary e.g. {'loss': 0.1}.
        """
        pass

    @abstractmethod
    def is_stable_baselines_wrapper(self) -> bool:
        pass

    @abstractmethod
    def stable_baselines_unwrapped(self) -> BaseAlgorithm:
        pass

    def save(self, path: str) -> None:
        raise NotImplementedError("save method not implemented!")

    def load(self, path: str) -> None:
        raise NotImplementedError("load method not implemented!")
