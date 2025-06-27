from typing import Union
from src.gp.actorcriticdgp import ActorCriticDGP
from src.gp.deepsigma import sample_from_gmm
from src.gp.mll.actorcriticmll import ActorCriticMLL
from src.agents.onpolicyagent import OnPolicyAgent
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List


class GPPOAgent(OnPolicyAgent):
    """
    Gaussian Process Proximal Policy Optimization (GPPO) agent using a Deep Gaussian Process (DGP) to estimate
    the policy and value function.

    The algorithm extends PPO by replacing neural network function approximators with Deep Sigma Point Processes (DSPPs),
    enabling uncertainty-aware learning and well-calibrated policy/value predictions. Advantage estimates can be
    uncertainty-aware by sampling from the posterior value distribution.
    """
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
            memory_size: int = 512,
            batch_size: int = 128,
            learning_rate: float = 3e-4,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            sample_vf: bool = True,
            device: torch.device = torch.device("cpu"),
            **kwargs
    ):
        """
        :param state_dimensions: Shape of the state input.
        :param action_dimensions: Shape of the action output.
        :param hidden_layers_config: Shared DGP layer config.
        :param policy_hidden_config: Policy head config.
        :param value_hidden_config: Value head config.
        :param beta: KL divergence regularization weight.
        :param num_inducing_points: Number of inducing points per layer.
        :param num_quad_sites: Number of sigma points (quadrature sites) used.
        :param memory_size: Rollout buffer size.
        :param batch_size: Batch size for updates.
        :param learning_rate: Learning rate for optimizer.
        :param n_epochs: Number of epochs per policy update.
        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda. Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        :param clip_range: PPO clipping range.
        :param ent_coef: Entropy coefficient.
        :param vf_coef: Value function loss coefficient.
        :param max_grad_norm: Max gradient norm for clipping.
        :param sample_vf: Whether to sample from the value function posterior.
        :param device: PyTorch device.
        :param kwargs: Additional arguments (ignored).
        """
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
            Q=num_quad_sites,
            num_actions=action_dimensions[-1]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.sample_vf = sample_vf

        self.objective = ActorCriticMLL(self.policy,
                                        self.policy.policy_likelihood,
                                        self.policy.value_likelihood,
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
        """
        Select an action from the policy given an observation.

        Samples from the DGP posterior predictive distribution over actions.
        Stores log-probability and sampled value for use in learning.

        :param observation: Current environment state.
        :return: Sampled action.
        """
        self.policy.eval()
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_dist, value_dist = self.policy(state)

            quad_weights_log = self.policy.quad_weights.unsqueeze(-1)
            quad_weights = quad_weights_log.exp()

            action = sample_from_gmm(quad_weights, action_dist.mean, action_dist.variance, 1)
            # log_prob calculation (specific to DSPPs)
            base_log_marginal = self.policy.policy_likelihood.log_marginal(action, action_dist)
            deep_log_marginal = quad_weights_log + base_log_marginal
            log_prob = deep_log_marginal.logsumexp(dim=0)

            self.last_log_prob = log_prob
            self.last_value = sample_from_gmm(quad_weights, value_dist.mean, value_dist.variance, 1) if \
                self.sample_vf else (quad_weights * value_dist.mean).sum(0)

            if action.dim() > 1:
                return action.squeeze(0).cpu().numpy()

            return action.cpu().numpy()

    def store_transition(
            self,
            state: np.ndarray,
            action: Any,
            reward: float,
            new_state: np.ndarray,
            done: bool,
    ) -> None:
        """
        Store a transition in the rollout buffer.

        Uses previously computed log-probability and value sample from policy.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param new_state: Next state.
        :param done: Whether episode terminated.
        """
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

        Computes advantage estimates using sampled value predictions,
        then performs multiple gradient updates using the clipped PPO style objective
        with DSPP log-likelihoods and KL regularization.

        :return: Dictionary with loss and diagnostic statistics.
        """
        self.policy.train()
        if len(self.rollout_buffer) < self.batch_size:
            return {}

        # Compute last value (this is awkward can this be changed?)
        last_state = torch.tensor(self.next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value_dist = self.policy(last_state)
            quad_weights = self.policy.quad_weights.unsqueeze(-1).exp()
            last_value = sample_from_gmm(quad_weights, value_dist.mean, value_dist.variance, 1) if \
                self.sample_vf else (quad_weights * value_dist.mean).sum(0)

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
        info["value_loss"] = 0.0
        info["policy_loss"] = 0.0
        info["entropy"] = 0.0
        cnt = 0
        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get(self.batch_size):
                cnt += 1
                loss, policy_loss, value_loss, entropy = self.objective(states, actions, advantages, returns, old_log_probs)
                losses.append(loss.item())
                info["value_loss"] += value_loss.item()
                info["policy_loss"] += policy_loss.item()
                info["entropy"] += entropy.item()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.rollout_buffer.clear()
        info["loss"] = float(np.mean(losses))
        info["value_loss"] = info["value_loss"] / cnt
        info["policy_loss"] = info["policy_loss"] / cnt
        info["entropy"] = info["entropy"] / cnt
        return info

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dimensions': self.state_dimensions,
                'action_dimensions': self.action_dimensions,
                'learning_rate': self.learning_rate,
                'n_epochs': self.n_epochs,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
                'num_inducing_points': self.policy.num_inducing_points,
                'num_quad_sites': self.policy.Q,
            }
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

