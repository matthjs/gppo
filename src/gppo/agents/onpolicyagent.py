from abc import ABC
from typing import Callable, Union
import torch
from gppo.agents.agent import Agent
from gppo.util.rolloutbuffer import RolloutBuffer


class OnPolicyAgent(Agent, ABC):
    """
    Abstract base class for on-policy RL algorithms.
    Instantiates a rolloutbuffer that can be used by the child class.
    """
    def __init__(self,
                 memory_size: int,
                 state_dimensions,
                 action_dimensions,
                 batch_size: int,
                 learning_rate: Union[float, Callable[[float], float]],
                 discount_factor: float,
                 device: torch.device):
        super().__init__()
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.memory_size = memory_size
        self.rollout_buffer = RolloutBuffer(capacity=self.memory_size, obs_shape=state_dimensions, action_shape=action_dimensions, device=device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.device = device

    # Implement store_transition() in child class

    # Implement choose_action() in child class

    # Implement learn() in child class
