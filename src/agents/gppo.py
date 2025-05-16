from typing import Optional, List, Dict, Union, Any

import numpy as np
import torch

from src.agents.onpolicyagent import OnPolicyAgent
from src.agents.ppoagent import PPOAgent
from src.gp.acdeepsigma import ActorCriticDGP


class GPPOAgent(OnPolicyAgent):
    def __init__(
            self,
            state_dimensions,
            action_dimensions,
            # DGP params
            hidden_layers_config: List[Dict],
            policy_hidden_config: List[Dict],
            value_hidden_config: List[Dict],
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
        log_prob = action_dist.log_prob(action)
        self.last_log_prob = log_prob
        self.last_value = value_dist.mean.mean(0)
        return action.mean(0).cpu().numpy().squeeze(0)

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
