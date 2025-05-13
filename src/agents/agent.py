from abc import abstractmethod, ABC
import torch
import numpy as np
from typing import Tuple, Callable, Any, Dict, Optional, Union
from src.util.exploration import ExplorationPolicy
from src.util.replaybuffer import ReplayBuffer


# MOVE THIS CLASS TO A SEPERATE FILE
class Agent(ABC):
    # True interface, no implemented methods or constructor.
    @abstractmethod
    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
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
            action: Union[int, np.ndarray],  # Is this always an int?
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        pass

    @abstractmethod
    def learn(self) -> Dict[str, Any]:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        Can return a dictionary e.g. {'loss': 0.1}.
        """

        pass
