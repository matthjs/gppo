from typing import List, Dict, Tuple, Any

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from torch import Tensor

from src.gp.deepgplayers import DSPPHiddenLayer
from src.gp.deepsigma import DSPPModel


class ActorCriticDGP(DSPPModel):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_layers_config: List[Dict[str, Any]],
        Q,
        num_inducing_points,
        policy_hidden_config: List[Dict],
        value_hidden_config: List[Dict] = None,
        **kwargs
    ):
        """
        DGP-based policy and value network for RL

        :param action_dim: Dimension of action space
        :param value_dim: Dimension of value output (default 1)
        :param policy_hidden_config: Configuration for policy head layers
        :param value_hidden_config: Configuration for value head layers
        """
        super().__init__(input_dim, hidden_layers_config, Q, num_inducing_points, **kwargs)

        # Postcondition self.layers() has base GP
        # out_dim got the output dimensions of latent var
        # policy_hidden_config take in out_dim and output action_dim
        # value_hidden_config take in out_dim and output dim=1
        self.policy_likelihood = MultitaskGaussianLikelihood(num_actions)
        self.value_likelihood = GaussianLikelihood()

        # Policy head configuration
        self.policy_head = torch.nn.ModuleList()
        last_dim = self.out_dim

        # Build policy head layers
        for layer_config in policy_hidden_config:
            self.policy_head.append(DSPPHiddenLayer(
                input_dims=last_dim,
                output_dims=layer_config['output_dims'],
                mean_type=layer_config['mean_type'],
                num_inducing=layer_config.get('num_inducing', 128),
                Q=self.Q,
            ))
            last_dim = layer_config['output_dims']

        # Value head configuration
        self.value_head = torch.nn.ModuleList()
        last_dim = self.out_dim
        value_hidden_config = value_hidden_config or [
            {'output_dims': 1, 'mean_type': 'constant', 'num_inducing': self.num_inducing_points}
        ]

        # Build value head layers
        for layer_config in value_hidden_config:
            self.value_head.append(DSPPHiddenLayer(
                input_dims=last_dim,
                output_dims=layer_config['output_dims'],
                mean_type=layer_config['mean_type'],
                num_inducing=layer_config.get('num_inducing', 128),
                Q=self.Q
            ))
            last_dim = layer_config['output_dims']

    def forward(self, inputs: Tensor, **kwargs) -> Tuple[MultivariateNormal, MultivariateNormal]:
        """
        This method will output TWO distributions, one for the policy and one
        for the value function.
        """
        latent_dist = super().forward(inputs, **kwargs)
        # Pass base features to policy and value function head
        policy_x = latent_dist
        value_x = latent_dist

        for layer in self.policy_head:
            policy_x = layer(policy_x, **kwargs)

        for layer in self.value_head:
            value_x = layer(value_x, **kwargs)

        return policy_x, value_x

    def get_policy(self, inputs: Tensor) -> MultivariateNormal:
        """Get policy distribution only"""
        return self(inputs)[0]

    def get_value(self, inputs: Tensor) -> MultivariateNormal:
        """Get value estimate only"""
        return self(inputs)[1]

    def posterior(
            self,
            X: Tensor,
            apply_likelihood: bool = False,  # Renamed from observation_noise
            *args, **kwargs) -> Tuple[MultivariateNormal, MultivariateNormal]:
        # This is the same as __call__ but will automatically apply the likelihood
        # Does not implement self.input_transform or self.outcome_transform features.
        self.eval()

        with torch.no_grad():
            policy_dist, value_dist = self(X, mean_input=X)
            if apply_likelihood:
                policy_dist = self.likelihood(policy_dist)
                value_dist = self.likelihood(value_dist)

        return policy_dist, value_dist