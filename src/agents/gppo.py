from typing import Optional

import torch

from src.agents.ppoagent import PPOAgent


class GPPOAgent(PPOAgent):
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
    ):
        super().__init__(
            state_dimensions,
            action_dimensions,
            memory_size,
            batch_size,
            learning_rate,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            ent_coef,
            vf_coef,
            max_grad_norm,
            target_kl,
            device
        )