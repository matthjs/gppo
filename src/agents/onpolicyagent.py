from abc import ABC
from typing import Union

import numpy as np
import torch

from src.agents.agent import Agent
from src.util.rolloutbuffer import RolloutBuffer


class OnPolicyAgent(Agent, ABC):
    def __init__(self,
                 memory_size: int,
                 state_dimensions,
                 action_dimensions,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 device: torch.device):
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimensions
        self.memory_size = memory_size
        self.rollout_buffer = RolloutBuffer(capacity=self.memory_size, device=device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate,
        self.discount_factor = discount_factor
        self.device = device

    # Implement store_transition() in child class

    # Implement choose_action() in child class

    # Implement learn() in child class
