from typing import Tuple

import torch
from torch import optim

from src.agents.dqnagent import DQNAgent


class DDQNAgent(DQNAgent):
    """
    Double DQN implementation. Inherits from DQN and simply replaces the TD target computation
    with the one from Double DQN.
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
                 expl_policy_params: dict,
                 device: torch.device,
                 dueling_architecture: bool = False,
                 **kwargs):
        """
        Double Deep Q Network (DDQN) agent with optional dueling architecture.

        :param memory_size: Max number of transitions in replay buffer.
        :param state_dimensions: Shape of the input state.
        :param n_actions: Number of possible actions.
        :param batch_size: Number of samples per training update.
        :param learning_rate: Optimizer learning rate (OPTIMIZER FIXED TO RMSProp).
        :param discount_factor: Gamma, discount for future rewards.
        :param target_update_freq: Update frequency for target network.
        :param expl_policy_name: Name of exploration strategy (e.g., epsilon_greedy).
        :param expl_policy_params: Parameters for exploration strategy.
        :param device: Torch device (e.g., CPU or CUDA).
        :param delta: Delta parameter for Huber loss.
        :param dueling_architecture: Whether to use dueling DQN.
        :param kwargs: Extra parameters.
        """
        super().__init__(memory_size, state_dimensions, n_actions, batch_size, learning_rate, discount_factor,
                         target_update_freq, expl_policy_name, expl_policy_params, device, dueling_architecture,
                         **kwargs)
        """
        """

    def compute_td_target(self,
                          rewards,
                          next_states,
                          dones) -> torch.Tensor:
        """
        Compute TD(0) target:
        Action is selected via q_net but evaluated via target_net.

        :param rewards: Batch of rewards from environment
        :param next_states: Batch of next states
        :param dones: Binary tensor indicating episode termination (0 or 1) for each sample.

        :return: One-step TD target.
        """
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1)
            q_values = self.target_net(next_states)
            selected_q_values = q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # Compute TD target
            target = rewards + self.discount_factor * selected_q_values * (1 - dones)
        return target





