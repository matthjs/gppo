"""
Some inspiration taken from the StableBaselines3 implementation:
https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
"""

from gymnasium import spaces

from src.agents.onpolicyagent import OnPolicyAgent
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Dict, Any, Tuple, Optional, List


class ActorCritic(nn.Module):
    def __init__(
            self,
            input_dim: int,
            action_space: spaces.Space,
            hidden_sizes: List[int] = [64, 64],
    ):
        super().__init__()
        # build MLP
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        # actor head
        self.action_space = action_space
        if isinstance(action_space, spaces.Discrete):
            self.action_head = nn.Linear(last_dim, action_space.n)
        else:
            self.action_mean = nn.Linear(last_dim, int(np.prod(action_space.shape)))
            self.action_log_std = nn.Parameter(torch.zeros(1, int(np.prod(action_space.shape))))
        # critic head
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.shared(x)
        # value
        value = self.value_head(x).squeeze(-1)
        # action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_head(x)
            dist = Categorical(logits=logits)
            log_std = None
        else:
            mean = self.action_mean(x)
            std = torch.exp(self.action_log_std)
            dist = Normal(mean, std)
            log_std = self.action_log_std
        return dist, value, log_std


class PPOAgent(OnPolicyAgent):
    def __init__(
            self,
            state_dimensions,
            action_space: spaces.Space,
            memory_size: int = 2048,
            batch_size: int = 64,
            learning_rate: float = 3e-4,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            memory_size,
            state_dimensions,
            action_space,
            batch_size,
            learning_rate,
            gamma,
            device,
        )
        input_dim = int(np.prod(state_dimensions))
        self.policy = ActorCritic(
            input_dim=input_dim,
            action_space=action_space,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.last_log_prob = None
        self.last_value = None

    def choose_action(self, observation: np.ndarray) -> Any:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value, _ = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.last_log_prob = log_prob
        self.last_value = value
        return action.cpu().numpy().squeeze(0)

    def store_transition(
            self,
            state: np.ndarray,
            action: Any,
            reward: float,
            new_state: np.ndarray,  # Not used but needed for interface compatibility
            done: bool,
    ) -> None:
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
            self.last_log_prob,
            self.last_value,
        )

    def learn(self) -> Dict[str, Any]:
        # Get last state value for GAE
        if len(self.rollout_buffer) == 0:
            return {}
        # Compute last value
        last_state = self.rollout_buffer.states[-1].unsqueeze(0)
        with torch.no_grad():
            _, last_value, _ = self.policy(last_state)
        # compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            torch.tensor(last_value.cpu().item(), dtype=torch.float32, device=self.device),
            self.discount_factor,
            self.gae_lambda,
        )
        # optimize policy
        info: Dict[str, Any] = {}
        losses = []
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get(self.batch_size):
                dist, values, _ = self.policy(states.view(states.size(0), -1))
                log_probs = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().mean()

                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    value_pred = values
                else:
                    value_pred = self.rollout_buffer.values_tensor(states) + torch.clamp(
                        values - self.rollout_buffer.values_tensor(states),
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                value_loss = nn.HuberLoss(delta=1)(returns, value_pred.squeeze(-1))

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        info["loss"] = float(np.mean(losses))
        return info
