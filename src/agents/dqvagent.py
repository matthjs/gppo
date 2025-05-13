from typing import Optional, Dict, Any, Tuple, Union
import torch
from torch import nn, optim
from src.agents.agent import Agent
from src.agents.offpolicyagent import OffPolicyAgent
from src.util.exploration import make_policy
from src.util.network import DuelingConvNetEstimator, ConvNetEstimator


class DQVAgent(OffPolicyAgent):
    """
    Deep Quality Value (DQV) Agent.
    """
    def __init__(self, memory_size: int,
                 state_dimensions: Tuple[int, int, int],
                 n_actions: int,
                 batch_size: int,
                 learning_rate: float,
                 discount_factor: float,
                 target_update_freq: int,
                 expl_policy_name: str,
                 expl_policy_params: dict,
                 device: torch.device,
                 delta: float = 1.0,
                 dueling_architecture: bool = False,
                 **kwargs):
        """
        DQV agent: learns separate Q and V networks using TD targets from V.

        :param memory_size: Max number of transitions in replay buffer.
        :param state_dimensions: Shape of the input state.
        :param n_actions: Number of possible actions.
        :param batch_size: Number of samples per training update.
        :param learning_rate: Optimizer learning rate (RMSProp fixed).
        :param discount_factor: Gamma, discount for future rewards.
        :param target_update_freq: Frequency of target V-network update.
        :param expl_policy_name: Exploration policy name (e.g., epsilon_greedy).
        :param expl_policy_params: Parameters for exploration strategy.
        :param device: Torch device (e.g., CPU or CUDA).
        :param delta: Delta parameter for Huber loss.
        :param dueling_architecture: Whether to use dueling architecture for Q.
        :param kwargs: Extra parameters.
        """
        self.constr = DuelingConvNetEstimator if dueling_architecture else ConvNetEstimator
        self.q_net = self.constr(num_out_features=n_actions).to(device=device)
        self.v_net = ConvNetEstimator(num_out_features=1).to(device=device)
        self.target_net = ConvNetEstimator(num_out_features=1).to(device=device)

        self.loss_fn = nn.HuberLoss(delta=delta)
        self.step = 0    # Internal counter
        self.target_update_freq = target_update_freq
        self.q_optimizer = optim.RMSprop(self.q_net.parameters(), lr=learning_rate)  # For simplicity use same
        # learning rate for Q and V.
        self.v_optimizer = optim.RMSprop(self.v_net.parameters(), lr=learning_rate)

        expl_policy_params["q_net"] = self.q_net
        expl_policy_params["action_dim"] = n_actions
        expl_policy_params["device"] = device
        policy_factory = make_policy(expl_policy_name, **expl_policy_params)
        super().__init__(memory_size,
                         state_dimensions,
                         n_actions,
                         batch_size,
                         learning_rate,
                         discount_factor,
                         policy_factory,
                         device)

    def _target_net_update(self):
        # Maybe use soft update but should not be necessary
        self.step += 1
        if self.step >= self.target_update_freq:
            self.step = 0
            self.target_net.load_state_dict(self.v_net.state_dict())

    def update(self, params: Optional[Dict[str, Any]] = None):
        """
        Update the exploration policy (e.g., decay epsilon).
        """
        self.expl_policy.update()

    def compute_td_target(self,
                          rewards,
                          next_states,
                          dones) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute TD(0) target using target V-network.

        :param rewards: Batch of rewards.
        :param next_states: Batch of next states.
        :param dones: Binary tensor indicating episode termination.

        :return: One-step TD target.
        """
        with torch.no_grad():
            return rewards + self.discount_factor * self.target_net(next_states).squeeze() * (1 - dones)

    def learn(self) -> Dict[str, Any]:
        """
        Perform one learning step for both Q and V networks using TD target from V.

        :return: Dict with Q and V loss.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._target_net_update()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Get the q_values corresponding to the taken actions given state
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        v_values = self.v_net(states).squeeze(1)

        target = self.compute_td_target(rewards, next_states, dones)

        q_loss = self.loss_fn(q_values, target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        v_loss = self.loss_fn(v_values, target)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return {"q_loss": q_loss, "v_loss":  v_loss}
