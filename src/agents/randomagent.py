from typing import Optional, Dict, Any, Tuple, Callable
import numpy as np
import torch
from src.agents.agent import Agent
from src.util.exploration import ExplorationPolicy


class RandomAgent(Agent):
    def __init__(self,
                 env_action_space,
                 memory_size: int = None,
                 state_dimensions: Tuple[int, int, int] = None,
                 n_actions: int = None,
                 batch_size: int = None,
                 learning_rate: float = None,
                 discount_factor: float = None,
                 expl_policy_factory: Callable[[], ExplorationPolicy] = lambda: -1,
                 device: torch.device = None):
        """
        Wrapper agent for running the default environment random policy.
        """
        super().__init__(memory_size, state_dimensions, n_actions, batch_size, learning_rate, discount_factor,
                         expl_policy_factory, device)
        self.env_action_space = env_action_space

    def store_transition(
            self,
            state: np.ndarray,
            action: int,  # Is this always an int?
            reward: float,
            new_state: np.ndarray,
            done: bool
    ) -> None:
        pass

    def choose_action(
            self,
            observation: np.ndarray
    ) -> int:
        return self.env_action_space.sample()

    def update(self, params: Optional[Dict[str, Any]] = None):
        pass

    def learn(self) -> None:
        pass
