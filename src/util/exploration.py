import random
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from typing import Callable


class ExplorationPolicy(ABC):
    """
    Abstract base class for discrete action selection
    """

    @abstractmethod
    def select_action(self,
                      state: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


def make_policy(name: str, **params) -> Callable[[], ExplorationPolicy]:
    """
    Factory function for exploration policy instantiation.

    :param name: Name of the exploration strategy.
    :param params: Parameters required for policy construction.
    :return: Callable that returns a new ExplorationPolicy instance.
    """
    if name == "epsilon_greedy":
        return lambda: EpsilonGreedy(**params)
    elif name == "softmax":
        raise NotImplementedError("Not Implemented :(")
    else:
        raise ValueError(f"Unknown exploration policy: {name}")


class EpsilonGreedy(ExplorationPolicy):
    def __init__(
            self,
            q_net: nn.Module,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay_steps: int,
            action_dim: int,
            device: torch.device,
    ):
        """
        Epsilon greedy policy with a schedule for the epsilon.
        With probability epsilon select a random action,
        otherwise select the action that maximizes the Q value
        estimated by `q_net`.
        :param q_net: Q-network used to select greedy actions.
        :param epsilon_start: Initial epsilon value.
        :param epsilon_end: Final epsilon value.
        :param epsilon_decay_steps: Number of steps over which to decay epsilon.
        :param action_dim: Number of possible actions.
        :param device: Torch device used for inference.
        """

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.action_dim = action_dim
        self.q_net = q_net
        self.device = device
        self.steps = 0

    def update(self) -> None:
        """
        Update epsilon according to a linear schedule.
        """
        self.steps += 1
        fraction = min(self.steps / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action based on epsilon-greedy rule.

        :param state: Current environment state (as NumPy array).
        :return: Selected action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_net(state)
                return int(torch.argmax(q_values))
