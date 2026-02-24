from typing import Dict, Any, Tuple, Union
import torch

from gppo.agents.dqvagent import DQVAgent
from gppo.util.network import ConvNetEstimator, DuelingConvNetEstimator


class DQVMaxAgent(DQVAgent):
    """
    Deep Quality-Value-Max (DQV-Max) agent.
    """
    def __init__(self,
                 memory_size: int,
                 state_dimensions: Tuple[int, int, int],
                 n_actions: int,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 target_update_freq: int,
                 expl_policy_name: str,
                 expl_policy_params: dict, device: torch.device,
                 dueling_architecture: bool = False,
                 **kwargs):
        """
        :param memory_size: Max number of transitions in replay buffer.
        :param state_dimensions: Shape of input observations.
        :param n_actions: Number of possible actions.
        :param batch_size: Number of samples per training step.
        :param learning_rate: Optimizer learning rate (shared for Q and V).
        :param discount_factor: Gamma, discount for future rewards.
        :param target_update_freq: Frequency of updating target network.
        :param expl_policy_name: Name of exploration policy (e.g., epsilon greedy).
        :param expl_policy_params: Parameters for exploration strategy.
        :param device: Torch device (CPU or CUDA).
        :param delta: Delta value for Huber loss.
        :param dueling_architecture: Whether to use dueling network for Q.
        :param kwargs: Extra arguments.
        """
        super().__init__(memory_size, state_dimensions, n_actions, batch_size, learning_rate, discount_factor,
                         target_update_freq, expl_policy_name, expl_policy_params, device, **kwargs)
        self.constr = DuelingConvNetEstimator if dueling_architecture else ConvNetEstimator
        self.target_net = self.constr(num_out_features=n_actions).to(device=device)

    def _target_net_update(self):
        self.step += 1
        if self.step >= self.target_update_freq:
            self.step = 0
            self.target_net.load_state_dict(self.q_net.state_dict())

    def compute_td_target(self,
                          rewards,
                          next_states,
                          dones) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute separate TD targets for Q and V networks.

        :param rewards: Batch of rewards from environment.
        :param next_states: Batch of next states.
        :param dones: Binary tensor indicating episode termination (0 or 1).
        :return: Tuple of (V-target, Q-target).
        """
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            value_target = rewards + self.discount_factor * next_q_values * (1 - dones)

            action_value_target = rewards + self.discount_factor * self.v_net(next_states).squeeze() * (1 - dones)
            return value_target, action_value_target

    def learn(self) -> Dict[str, Any]:
        """
        Perform one Q-learning and V-learning update step.

        :return: Dictionary containing Q and V losses.
        """
        # NOTE: Minor code duplication w.r.t. parent class
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._target_net_update()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Get the q_values corresponding to the taken actions given state
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        v_values = self.v_net(states).squeeze(1)

        value_target, action_value_target = self.compute_td_target(rewards, next_states, dones)

        q_loss = self.loss_fn(q_values, action_value_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        v_loss = self.loss_fn(v_values, value_target)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return {"q_loss": q_loss, "v_loss":  v_loss}
