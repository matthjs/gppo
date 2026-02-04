import torch
import numpy as np
from torch import Tensor
from typing import Generator, Optional, Tuple, Union

def ensure_tensor(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)

class RolloutBuffer:
    """
    Rollout buffer for PPO or similar on-policy algorithms.
    Stores transitions for one batch of on-policy updates.
    Provides support for computing Generalized Advantage Estimate (GAE).
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int], action_shape: int, device: torch.device):
        """
        Initialize empty buffer.

        :param capacity: Max number of transitions to store.
        :param obs_shape: Dimensionality of the state or observation space.
        :param action_shape: Dimensionality of the action space.
        :param device: Torch device to store tensors on.
        """
        self.capacity = capacity
        self.device = device
        self.pos = 0   # 'pointer'

        # Pre-allocate tensors
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)

    def push(
        self,
        obs: Union[Tensor, np.ndarray],
        action: Union[Tensor, np.ndarray],
        reward: Union[Tensor, np.ndarray],
        done: Union[Tensor, np.ndarray],
        log_prob: Union[Tensor, np.ndarray] = None,
        value: Union[Tensor, np.ndarray] = None,
    ) -> None:
        """
        Add a transition to the buffer.
        Assumes that if batched inputs are given that
        the batch dimension refers to the number of parallel environments
        so e.g. action (n_envs, action_dim)

        :param state: Observed state.
        :param action: Action taken.
        :param reward: Reward received.
        :param done: Done flag.
        :param log_prob: Log-probability of action.
        :param value: Estimated value of state.
        """
        n_envs = reward.shape[0]
        idxs = slice(self.pos, self.pos + n_envs)
        
        self.obs[idxs] = ensure_tensor(obs, torch.float32, self.device)
        self.actions[idxs] = ensure_tensor(action, torch.float32, self.device)
        self.rewards[idxs] = ensure_tensor(reward, torch.float32, self.device)
        self.dones[idxs] = ensure_tensor(done, torch.float32, self.device)
        self.values[idxs] = ensure_tensor(value, torch.float32, self.device)
        self.log_probs[idxs] = ensure_tensor(log_prob, torch.float32, self.device)

        self.pos += n_envs

    def compute_returns_and_advantages(
        self,
        last_value: Tensor,
        last_done: Tensor,
        gamma: float,
        gae_lambda: float
    ) -> None:
        """
        Compute GAE and returns for all stored transitions in a multi-environment rollout.

        :param last_value: Value estimate for final states, shape [n_envs].
        :param last_done: Done flags for final states, shape [n_envs].
        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda parameter.
        """
        # Number of environments
        n_envs = last_value.shape[0]
        # Number of timesteps collected
        T = self.pos // n_envs
        last_done = ensure_tensor(last_done, torch.float32, self.device)

        # Reshape flattened buffer into [T, N]
        rewards = self.rewards[:self.pos].view(T, n_envs)
        values = torch.cat([self.values[:self.pos].view(T, n_envs), last_value.unsqueeze(0).to(self.device)], dim=0)
        dones = torch.cat([self.dones[:self.pos].view(T, n_envs), last_done.unsqueeze(0).to(self.device)], dim=0)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(n_envs, device=self.device)

        # Compute GAE backwards
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Flatten back to match the buffer layout
        self.advantages[:self.pos] = advantages.view(-1)
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

        # Normalize advantages
        adv = self.advantages[:self.pos]
        self.advantages[:self.pos] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get(self, batch_size: int) -> Generator:
        """
        Yield random mini-batches of data
        
        :param batch_size: Size of each mini-batch.
        :return: Generator of (state, action, log_prob, return, advantage) all of shape [batch_size, dim].
        """
        indices = torch.randperm(self.pos, device=self.device)
        for i in range(0, self.pos, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield (
                self.obs[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices].unsqueeze(-1),   # to ensure [batch_size] -> [batch_size, 1]
                self.returns[batch_indices].unsqueeze(-1),
                self.advantages[batch_indices].unsqueeze(-1),
            )

    def clear(self) -> None:
        """Reset buffer position"""
        self.pos = 0

    def __len__(self) -> int:
        return self.pos
