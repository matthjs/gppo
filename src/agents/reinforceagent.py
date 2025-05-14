from typing import Dict, Any, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.agents.onpolicyagent import OnPolicyAgent


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_dim))  # outputs mean
        self.net = nn.Sequential(*layers)
        # learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ReinforceAgent(OnPolicyAgent):
    def __init__(
        self,
        memory_size: int,
        state_dimensions,
        action_dimensions,
        batch_size: int,
        learning_rate: float,
        discount_factor: float,
        num_epochs: int,
        device: torch.device,
    ):
        super().__init__(
            memory_size,
            state_dimensions,
            action_dimensions,
            batch_size,
            learning_rate,
            discount_factor,
            device,
        )
        input_dim = int(np.prod(state_dimensions))
        output_dim = action_dimensions if isinstance(action_dimensions, int) else int(np.prod(action_dimensions))

        # Define MLP policy
        self.policy = MLPPolicy(
            input_dim=input_dim,
            hidden_sizes=[64, 64],
            output_dim=output_dim,
        ).to(self.device)

        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def choose_action(
        self,
        observation: np.ndarray,
    ) -> Union[int, np.ndarray]:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.policy(state)
        action = dist.sample()
        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
        )

    def _compute_rewards_to_go(self, rewards: List[float]) -> List[float]:
        rewards_to_go: List[float] = []
        cumulative = 0.0
        for r in reversed(rewards):
            cumulative = r + self.discount_factor * cumulative
            rewards_to_go.insert(0, cumulative)
        return rewards_to_go

    def learn(self) -> Dict[str, Any]:
        if len(self.rollout_buffer) < self.batch_size:
            return {}

        total_loss = 0.0
        for _ in range(self.num_epochs):
            for minibatch in self.rollout_buffer.get(batch_size=len(self.rollout_buffer)):
                states, actions, rewards, dones, _, _ = minibatch
                # flatten states if needed
                states = states.view(states.size(0), -1)

                # Compute rewards-to-go
                rewards_to_go = self._compute_rewards_to_go(rewards.cpu().numpy())
                returns = torch.tensor(rewards_to_go, dtype=torch.float32, device=self.device)

                dist = self.policy(states)
                log_probs = dist.log_prob(actions).sum(-1)
                loss = -(log_probs * returns).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        self.rollout_buffer.clear()
        return {"loss": total_loss}
