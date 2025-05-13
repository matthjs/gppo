from typing import Tuple
import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for PPO or similar on-policy algorithms.
    Stores transitions for one batch of on-policy updates.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.device = device
        self.capacity = capacity

    def push(
            self,
            state,
            action,
            reward,
            done,
            log_prob=None,
            value=None
    ) -> None:
        def ensure_tensor(x, dtype, device):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=device)
            return torch.tensor(x, dtype=dtype, device=device)

        self.states.append(ensure_tensor(state, torch.float32, self.device))
        self.actions.append(ensure_tensor(action, torch.long, self.device))
        self.rewards.append(ensure_tensor(reward, torch.float32, self.device))
        self.dones.append(ensure_tensor(done, torch.float32, self.device))
        if log_prob:
            self.log_probs.append(ensure_tensor(log_prob, torch.float32, self.device))
        if value:
            self.values.append(ensure_tensor(value, torch.float32, self.device))

    def get(self, batch_size: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns mini-batches of transitions as tensors, without shuffling the buffer.
        Typically used after computing GAE and returns.
        Usage:
            for epoch in range(epochs):
                for minibatch in buffer.get(batch_size=64):  # Sample minibatches in order
                    states, actions, rewards, dones, log_probs, values = minibatch
                    # Do the policy update with the minibatch here
        :param batch_size: The number of samples to return per batch.
        """
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        if self.log_probs:
            log_probs = torch.stack(self.log_probs)
        if self.values:
            values = torch.stack(self.values)

        # Yield mini-batches in the original order
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            yield (
                states[start:end],
                actions[start:end],
                rewards[start:end],
                dones[start:end],
                log_probs[start:end] if self.log_probs else None,
                values[start:end] if self.values else None
            )

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        """
        :return: Number of transitions currently stored.
        """
        return len(self.states)
