from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn, optim
from src.agents.agent import Agent
from src.agents.offpolicyagent import OffPolicyAgent
from src.util.exploration import make_policy
from src.util.network import ConvNetEstimator, DuelingConvNetEstimator


class DQNAgent(OffPolicyAgent):
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
                 delta: float = 1.0,
                 dueling_architecture: bool = False,
                 **kwargs
                 ):
        """
        Deep Q Network (DQN) agent with optional dueling architecture.

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
        # e.g., "epsilon_greedy", "epsilon: 0.01"
        self.constr = DuelingConvNetEstimator if dueling_architecture else ConvNetEstimator

        self.q_net = self.constr(num_out_features=n_actions).to(device=device)
        self.target_net = self.constr(num_out_features=n_actions).to(device=device)
        self.loss_fn = nn.HuberLoss(delta=delta)
        self.step = 0   # Internal counter
        self.target_update_freq = target_update_freq
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=learning_rate)

        expl_policy_params["q_net"] = self.q_net
        expl_policy_params["action_dim"] = n_actions
        expl_policy_params["device"] = device
        policy_factory = make_policy(expl_policy_name, **expl_policy_params)
        super().__init__(
            memory_size=memory_size,
            state_dimensions=state_dimensions,
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            expl_policy_factory=policy_factory,
            batch_size=batch_size,
            device=device
        )

    def _target_net_update(self) -> None:
        self.step += 1
        if self.step >= self.target_update_freq:
            self.step = 0
            self.target_net.load_state_dict(self.q_net.state_dict())

    def update(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the exploration policy (e.g., decay epsilon).
        """
        self.expl_policy.update()

    def compute_td_target(self,
                          rewards,
                          next_states,
                          dones) -> torch.Tensor:
        """
        Compute TD(0) target using max Q from target network.

        :param rewards: Batch of rewards from environment
        :param next_states: Batch of next states
        :param dones: Binary tensor indicating episode termination (0 or 1) for each sample.

        :return: One-step TD target.
        """
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            # Construct TD(0) target
            target = rewards + self.discount_factor * next_q_values * (1 - dones)
        return target

    def learn(self) -> Dict[str, Any]:
        """
        Perform one Q-learning update step using replay buffer.

        :return: Dictionary containing training loss (if any).
        Assumption is that this function is called every timestep of the environment.
        """
        # Not enough trajectories to sample
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._target_net_update()

        # Sample trajectories (batched!)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Get the q_values corresponding to the taken actions given state (batched)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        target = self.compute_td_target(rewards, next_states, dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
