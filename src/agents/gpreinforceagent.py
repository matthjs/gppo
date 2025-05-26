from typing import Dict, Any, Union, List

import gpytorch
import numpy as np
import torch

from src.agents.onpolicyagent import OnPolicyAgent
from src.gp.deepsigma import DSPPModel
# from src.gp.mll.deep_predictive_log_likelihood_rl import DeepPredictiveLogLikelihoodRL


class GPReinforceAgent(OnPolicyAgent):
    def __init__(self,
                 memory_size: int,
                 state_dimensions,
                 action_dimensions,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 num_epochs: int,
                 device: torch.device,
                 ):
        super().__init__(memory_size, state_dimensions, action_dimensions, batch_size, learning_rate, discount_factor,
                         device)

        # Use CNN feature extractor if image data
        self.policy = DSPPModel(
            input_dim=state_dimensions[0],
            hidden_layers_config=[
                # {"output_dims": 2, "mean_type": "linear"},
                #{"output_dims": 2, "mean_type": "linear"},
                {"output_dims": None, "mean_type": "constant"},
            ]
        ).to(self.device)
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        # self.mll = DeepPredictiveLogLikelihoodRL(self.policy.likelihood, self.policy, self.batch_size)

    def choose_action(
            self,
            observation: np.ndarray
    ) -> Union[int, np.ndarray]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Could also use posterior() method
            state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred = self.policy(state_tensor)
            action_dist = self.policy.likelihood(pred)
            action_t = action_dist.sample()
            action_t_ = action_t.mean(dim=(0))
            action = action_t_.cpu().numpy()

        return action

    def store_transition(self,
                         state: np.ndarray,
                         action: Union[int, np.ndarray],
                         reward: float,
                         new_state: np.ndarray,
                         done: bool) -> None:
        self.rollout_buffer.push(
            state,
            action,
            reward,
            done
        )

    def _compute_rewards_to_go(self, rewards: List[float]) -> List[float]:
        rewards_to_go = []
        cumulative = 0
        for r in reversed(rewards):
            cumulative = r + self.discount_factor * cumulative
            rewards_to_go.insert(0, cumulative)
        return rewards_to_go

    def learn(self) -> Dict[str, Any]:
        if len(self.rollout_buffer) < self.batch_size:
            return {}

        total_loss = 0.0

        for epoch in range(self.num_epochs):
            for minibatch in self.rollout_buffer.get(batch_size=len(self.rollout_buffer)):

                states, actions, rewards, dones, _, _ = minibatch
                # Conditionally squeeze
                if actions.dim() > 1 and actions.shape[1] == 1:
                    actions = actions.squeeze(1)

                # Compute rewards-to-go
                rewards_to_go = self._compute_rewards_to_go(rewards.cpu().numpy())
                rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32, device=self.device).unsqueeze(-1)
                # Forward pass to get the predicted distribution of actions
                pred = self.policy(states)
                # mll = DeepPredictiveLogLikelihoodRL(self.policy.likelihood, self.policy, len(self.rollout_buffer))
                loss = -mll(pred, actions, adv=rewards_to_go).mean() # -self.mll(pred, actions, adv=rewards_to_go).mean()

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        self.rollout_buffer.clear()
        return {"loss": total_loss}


