from typing import Tuple
import torch


class RolloutBuffer:
    """
    Rollout buffer for PPO or similar on-policy algorithms.
    Stores transitions for one batch of on-policy updates.
    Provides support for computing Generalized Advantage Estimate (GAE).
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.device = device
        self.capacity = capacity

    def push(
        self,
        state,
        action,
        reward,
        done,
        log_prob=None,
        value=None,
    ) -> None:
        def ensure_tensor(x, dtype, device):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=device)
            return torch.tensor(x, dtype=dtype, device=device)

        if state is not None:
            self.states.append(ensure_tensor(state, torch.float32, self.device))
        if action is not None:
            self.actions.append(ensure_tensor(action, torch.float32, self.device))
        if reward is not None:
            self.rewards.append(ensure_tensor(reward, torch.float32, self.device))
        if done is not None:
            self.dones.append(ensure_tensor(done, torch.float32, self.device))
        if log_prob is not None:
            self.log_probs.append(ensure_tensor(log_prob, torch.float32, self.device))
        if value is not None:
            self.values.append(ensure_tensor(value, torch.float32, self.device))

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        # Generalized advantage estimation
        self.advantages = []
        gae = 0
        values = [v.detach() for v in self.values] + [last_value.detach()]
        # values = self.values
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            self.advantages.insert(0, gae)

        # Normalize advantages
        advantages_tensor = torch.stack(self.advantages)
        if len(advantages_tensor.shape) > 1:    # Normalization can only be done if mini batchsize >= 1
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        self.advantages = list(advantages_tensor)
        # compute returns (TD_target)
        self.returns = [adv.detach() + val.detach() for adv, val in zip(self.advantages, self.values)]

    def get(self, batch_size: int) -> Tuple:
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = torch.stack(self.returns)
        advantages = torch.stack(self.advantages)
        # yield minibatches
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            yield (
                states[start:end],
                actions[start:end],
                old_log_probs[start:end],
                returns[start:end],
                advantages[start:end],
            )

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.returns.clear()
        self.advantages.clear()

    def __len__(self) -> int:
        return len(self.states)