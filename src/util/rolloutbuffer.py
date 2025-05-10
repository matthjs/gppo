from typing import Tuple
import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for PPO or similar on-policy algorithms.
    Stores transitions for one batch of on-policy updates.
    """

    def __init__(self, buffer_size: int, device: torch.device) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.device = device
        self.buffer_size = buffer_size

    def push(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            log_prob: float,
            value: float
    ) -> None:
        self.states.append(torch.tensor(state, dtype=torch.float32, device=self.device))
        self.actions.append(torch.tensor(action, dtype=torch.long, device=self.device))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
        self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32, device=self.device))
        self.values.append(torch.tensor(value, dtype=torch.float32, device=self.device))

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
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        # Yield mini-batches in the original order
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            yield (
                states[start:end],
                actions[start:end],
                rewards[start:end],
                dones[start:end],
                log_probs[start:end],
                values[start:end]
            )

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
