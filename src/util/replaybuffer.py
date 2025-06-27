from abc import ABC, abstractmethod
import random
from collections import deque
import torch


class ReplayBufferBase(ABC):
    @abstractmethod
    def push(self, transition: tuple) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> tuple:
        pass


class ReplayBuffer(ReplayBufferBase):
    """
    A simple replay buffer for storing and sampling transitions.

    Each transition is expected to be a tuple of:
    (state, action, reward, next_state, done),
    where states are NumPy arrays or tensors.
    """

    def __init__(self, capacity: int, device: torch.device):
        """
        Initialize the replay buffer.

        :param capacity: Maximum number of transitions to store.
        :param device: Device to place sampled tensors on.
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, transition: tuple) -> None:
        """
        Add a transition to the buffer.

        :param transition: A tuple (state, action, reward, next_state, done).
        """
        # Convert the transition to tensors and store them directly in the buffer
        state, action, reward, next_state, done = transition
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions.

        :param batch_size: Number of samples to return.
        :return: A tuple of stacked tensors: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )

    def __len__(self):
        """
        :return: Number of transitions currently stored.
        """
        return len(self.buffer)
