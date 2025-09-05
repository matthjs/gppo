from typing import Optional, Dict, Any, Tuple, Callable, Union
import numpy as np
from src.agents.onpolicyagent import OnPolicyAgent
import torch
from src.util.exploration import ExplorationPolicy


class RandomAgent(OnPolicyAgent):
    def __init__(self,
                 env_action_space,
                 memory_size: int = None,
                 state_dimensions = None,
                 action_dimensions = None,
                 batch_size: int = None,
                 learning_rate: float = None,
                 gamma: float = None,
                 device: torch.device = None):
        """
        Wrapper agent for running the default environment random policy which
        samples uniformly from the action space.
        """
        super().__init__(
            memory_size,
            state_dimensions,
            action_dimensions,
            batch_size,
            learning_rate,
            gamma,
            device,
        )
        self.env_action_space = env_action_space

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        pass

    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
        return self.env_action_space.sample()

    def update(self, params: Optional[Dict[str, Any]] = None):
        pass

    def learn(self) -> None:
        pass
