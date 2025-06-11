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
            action_dim: int,
            hidden_sizes: List[int] = [64, 64],
    ):
        super().__init__()
        # build MLP
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        # actor head
        # self.action_space = action_space
        # if isinstance(action_space, spaces.Discrete):
        #     self.action_head = nn.Linear(last_dim, action_space.n)
        # else:
        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        # critic head
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.shared(x)
        # value
        value = self.value_head(x).squeeze(-1)
        # action distribution
        # if isinstance(self.action_space, spaces.Discrete):
        #     logits = self.action_head(x)
        #     dist = Categorical(logits=logits)
        #    log_std = None
        # else:
        mean = self.action_mean(x)
        std = torch.exp(self.action_log_std)
        dist = Normal(mean, std)
        log_std = self.action_log_std
        return dist, value, log_std


class PPOAgent(OnPolicyAgent):
    def __init__(
            self,
            state_dimensions,
            action_dimensions,
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
            target_kl: Optional[float] = None,
            device: torch.device = torch.device("cpu"),
            **kwargs
    ):
        """
        :param state_dimensions:
        :param action_dim
        :memory: amount of steps in the environment to store before updating. Equal to the n_steps param
        in the StableBaselines3 implementation
        :param batch_size: Minibatch size
        :param learning_rate: learning rate for SGD based optimizer
        :param n_epochs: Number of epoch when optimizing surrogate loss
        :param gamma: discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param clip_range: Clipping parameter, it can be a function of the remaining progress.
        :param clip_range_vf: Clipping parameter for the value function, None means no clipping
        will be performed for the value function.
        :param ent_coef: Entropy coefficient for the loss calculation (c_1).
        :param vf_coef: Value function coefficient for the loss calculation (c_2).
        :param max_grad_norm: The maximum value for the gradient clipping
        :param device: Device.
        """
        super().__init__(
            memory_size,
            state_dimensions,
            action_dimensions,
            batch_size,
            learning_rate,
            gamma,
            device,
        )
        input_dim = int(np.prod(state_dimensions))
        action_dim = int(np.prod(action_dimensions))
        self.policy = ActorCritic(
            input_dim=input_dim,
            action_dim=action_dim,
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
        self.next_state = None

    def choose_action(self, observation: np.ndarray) -> Any:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value, _ = self.policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
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
        self.next_state = new_state
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
            self.last_log_prob.detach(),
            self.last_value.detach(),
        )

    def _early_kl_stop(self, log_prob, old_log_prob) -> bool:
        """
        NOTE TO SELF: THIS IS NOT FINISHED SO DO NOT USE.
        Determines whether optimization loop should stop early using
        approximate form of reverse KL divergence.
        """
        approx_kl_divs = []

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            approx_kl_divs.append(approx_kl_div)

        return True

    def learn(self) -> Dict[str, Any]:
        """
        Usage: run the learn() method at every timestep in the environment,
        it will only update the agent once the rollout_buffer has been filled.
        """
        # Get last state value for GAE
        if len(self.rollout_buffer) < self.memory_size:
            return {}
        # Compute last value
        last_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_value, _ = self.policy(last_state)

        # compute returns and advantages
        # implementation choice: \hat{R} and \hat{A} are computed at this stage instead of
        # one at a time at every step.
        self.rollout_buffer.compute_returns_and_advantages(
            last_value.detach(),  # Directly use detached tensor,
            self.discount_factor,
            self.gae_lambda,
        )
        # optimize policy
        info: Dict[str, Any] = {}
        losses = []
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get(self.batch_size):
                dist, values, _ = self.policy(states)
                log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()  # Sum over actions before mean

                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs)  # Note: Subtraction is division in log space.
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    value_pred = values
                else:
                    # Clip the difference between old and new value.,
                    old_values = returns - advantages  # Directly use buffer's returns/advantages
                    value_pred = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )

                # Value loss --> TD(gae_lambda) target.
                value_loss = nn.HuberLoss(delta=1)(returns, value_pred.unsqueeze(-1))

                # Full surrogate loss: L_clip - c1 * L_VF + c_2 S[pi](s_t)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        info["loss"] = float(np.mean(losses))
        return info

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dimensions': self.state_dimensions,
                'action_dimensions': self.action_dimensions,
                'learning_rate': self.learning_rate,
                'n_epochs': self.n_epochs,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'clip_range_vf': self.clip_range_vf,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
            }
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

