from typing import Optional, List, Dict, Union, Any

import numpy as np
import torch

from src.agents.onpolicyagent import OnPolicyAgent
from src.agents.ppoagent import PPOAgent
from src.gp.acdeepsigma import ActorCriticDGP
from src.gp.mll.actorcriticmll import ActorCriticMLL
from src.agents.onpolicyagent import OnPolicyAgent
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Dict, Any, Tuple, Optional, List


class GPPOAgent(OnPolicyAgent):
    def __init__(
            self,
            state_dimensions,
            action_dimensions,
            # DGP params
            hidden_layers_config: List[Dict],
            policy_hidden_config: List[Dict],
            value_hidden_config: List[Dict],
            beta: float = 1.0,
            num_inducing_points: int = 128,
            num_quad_sites: int = 8,
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
        super().__init__(
            memory_size,
            state_dimensions,
            action_dimensions,
            batch_size,
            learning_rate,
            gamma,
            device,
        )

        self.policy = ActorCriticDGP(
            input_dim=state_dimensions[-1],
            hidden_layers_config=hidden_layers_config,
            policy_hidden_config=policy_hidden_config,
            value_hidden_config=value_hidden_config,
            num_inducing_points=num_inducing_points,
            Q=num_quad_sites
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.objective = ActorCriticMLL(self.policy,
                                        self.policy.likelihood,
                                        num_data=self.batch_size,
                                        clip_range=self.clip_range,
                                        vf_coef=self.vf_coef,
                                        ent_coef=self.ent_coef,
                                        beta=beta)

        self.last_log_prob = None
        self.last_value = None
        self.next_state = None

    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_dist, value_dist = self.policy(state)
        # NOTE: Do something smarter here with action selection
        action = action_dist.sample()

        # log_prob calculation (specific to DSPPs)
        base_log_marginal = self.policy.likelihood.log_marginal(action, action_dist)
        deep_log_marginal = self.policy.quad_weights.unsqueeze(-1) + base_log_marginal
        log_prob = deep_log_marginal.logsumexp(dim=0).sum(-1) # action_dist.log_prob(action)  # NOTE TO SELF THIS IS PROBABLY NOT CORRECT

        self.last_log_prob = log_prob
        self.last_value = value_dist.sample().mean(0)   # value_dist.mean.mean(0)   mean or sample?

        action_t = action.mean(0)
        if action_t.dim() > 1:
            return action_t.squeeze(0).cpu().numpy()

        return action_t.cpu().numpy()

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

    def learn(self) -> Dict[str, Any]:
        """
        Usage: run the learn() method at every timestep in the environment,
        it will only update the agent once the rollout-buffer has been filled.
        """
        if len(self.rollout_buffer) < self.batch_size:
            return {}

        # Compute last value (this is akward can this be changed?)
        last_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value_dist = self.policy(last_state)
            last_value = value_dist.mean.mean(0)

        # compute returns and advantages
        # implementation choice: \hat{R} and \hat{A} are computed at this stage instead of
        # one at a time at every step.
        self.rollout_buffer.compute_returns_and_advantages(
            last_value.detach(),
            self.discount_factor,
            self.gae_lambda
        )

        # Perform update rule
        info: Dict[str, Any] = {}
        losses = []
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get(self.batch_size):
                loss = self.objective(states, actions, advantages, returns, old_log_probs)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        info["loss"] = float(np.mean(losses))
        return info
