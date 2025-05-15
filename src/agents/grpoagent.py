from typing import Union

import numpy as np
import torch

from src.agents.onpolicyagent import OnPolicyAgent
from src.agents.reinforceagent import MLPPolicy


class GRPO(OnPolicyAgent):
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
        group_size: int = 64,
        relative_clip: float = 0.2,
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
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        self.rollout_buffer.push(
            state,
            action,
            reward,
            done,
            log_prob
        )

