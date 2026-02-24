"""
Some inspiration taken from the StableBaselines3 implementation:
https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
"""
from gppo.agents.onpolicyagent import OnPolicyAgent
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.nn import functional as F
from gppo.util.network import ActorCriticMLP
from gppo.util.resolve import resolve_optimizer_cls


class PPOAgent(OnPolicyAgent):
    """
    Implementation of PPO with support for parallel environment execution.

    Note: Does not support discrete action spaces.
    """

    def __init__(
            self,
            state_dimensions,
            action_dimensions,
            n_steps: int = 2048,
            n_envs: int = 1,
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
            torch_compile: bool = False,
            optimizer_cfg: dict = None,
            **kwargs
    ):
        """
        :param state_dimensions: Shape of the input state.
        :param action_dimensions: Shape of the actions.
        :param n_steps: Number of steps per environment before updating.
        :param n_envs: Number of parallel environments.
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
            n_steps * n_envs,
            state_dimensions,
            action_dimensions,
            batch_size,
            learning_rate,
            gamma,
            device,
        )

        self.n_steps = n_steps
        self.n_envs = n_envs

        input_dim = int(np.prod(state_dimensions))
        action_dim = int(np.prod(action_dimensions))
        self.policy = ActorCriticMLP(
            input_dim=input_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.torch_compile = torch_compile
        if self.torch_compile:
            self.policy = torch.compile(self.policy)

        self.optimizer = None
        if optimizer_cfg:
            cls, kwargs = resolve_optimizer_cls(optimizer_cfg)
            self.optimizer = cls(self.parameters(), **kwargs)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                lr=self.learning_rate)
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.last_log_prob = None
        self.last_value = None
        self.last_done = None
        self.next_state = None

        if self.torch_compile:
            self._compute_ppo_loss = torch.compile(self._compute_ppo_loss)

    def choose_action(self, observation: np.ndarray) -> Any:
        self.policy.eval()
        with torch.no_grad():
            # Support both single and parallel environments
            state = torch.tensor(
                observation, dtype=torch.float32, device=self.device)    # [n_env, dim]
            dist, value, _ = self.policy(state)    # value: [n_env]
            action = dist.sample()    # [n_env, dim]
            log_prob = dist.log_prob(action).sum(
                dim=-1)   # log_prob: [n_env]
            self.last_log_prob = log_prob
            self.last_value = value
            return action.cpu().numpy()

    def store_transition(
            self,
            state: np.ndarray,
            action: Any,
            reward: float,
            new_state: np.ndarray,
            done: bool,
    ) -> None:
        self.next_state = new_state
        self.last_done = done
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
            self.last_log_prob.detach(),
            self.last_value.detach(),
        )

    def full_buffer(self) -> bool:
        return len(self.rollout_buffer) >= self.rollout_buffer.capacity

    def compute_returns_and_advantages(self) -> None:
        # Compute last value
        last_state = torch.tensor(
            self.next_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_value, _ = self.policy(last_state)

        # compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            last_value.detach(),
            self.last_done,
            self.discount_factor,
            self.gae_lambda,
        )

    @staticmethod
    def _compute_ppo_loss(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        clip_range: float,
        vf_coef: float,
        ent_coef: float,
        entropy_loss: torch.Tensor,
        clip_range_vf: Optional[float] = None,
    ):
        # Policy loss
        # Note: Subtraction is division in log space.
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if clip_range_vf is None:
            value_pred = values
        else:
            # Clip the difference between old and new value.,
            old_values = returns - advantages # Directly use buffer's returns/advantages
            value_pred = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
        # Value loss --> TD(gae_lambda) target.
        value_loss = F.mse_loss(returns, value_pred.unsqueeze(-1))
        entropy_loss_term = ent_coef * entropy_loss
        # Full surrogate loss: L_clip - c1 * L_VF + c_2 S[pi](s_t)
        loss = policy_loss + vf_coef * value_loss + entropy_loss_term
        return loss, policy_loss, value_loss

    def learn(self) -> Dict[str, Any]:
        """
        Usage: run the learn() method at every timestep in the environment,
        it will only update the agent once the rollout_buffer has been filled.

        :return: Dictionary with loss and diagnostic statistics.
        """
        self.policy.train()
        if len(self.rollout_buffer) < self.rollout_buffer.capacity:
            return {}

        self.compute_returns_and_advantages()

        # optimize policy
        info: Dict[str, Any] = {}
        info["value_loss"] = 0.0
        info["policy_loss"] = 0.0
        info["entropy"] = 0.0
        cnt = 0
        losses = []
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get(self.batch_size):
                cnt += 1
                dist, values, _ = self.policy(states)
                log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy_loss = -dist.entropy().sum(dim=-1).mean()  # Sum over actions before mean

                loss, policy_loss, value_loss = self._compute_ppo_loss(
                    log_probs, old_log_probs, advantages, returns,
                    values, self.clip_range, self.vf_coef,
                    self.ent_coef, entropy_loss
                )
                losses.append(loss.item())

                info["value_loss"] += value_loss.item()
                info["policy_loss"] += policy_loss.item()
                info["entropy"] += -entropy_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        info["loss"] = float(np.mean(losses))
        info["value_loss"] = info["value_loss"] / cnt
        info["policy_loss"] = info["policy_loss"] / cnt
        info["entropy"] = info["entropy"] / cnt
        return info

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dimensions': self.state_dimensions,
                'action_dimensions': self.action_dimensions,
                'learning_rate': self.learning_rate,
                'n_steps': self.n_steps,
                'n_envs': self.n_envs,
                'n_epochs': self.n_epochs,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
            }
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
