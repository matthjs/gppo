"""
Taken from: Taken from: https://github.com/matthjs/xai-gp/tree/main/xai_gp/models/gp
"""
import json
from typing import Any, Dict, List, Union
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from gpytorch.models.deep_gps.dspp import DSPP
from pandas import Categorical
from torch import Tensor
import torch.nn as nn
from src.gp.deepgplayers import DSPPHiddenLayer
from src.gp.gpbase import GPytorchModel
import numpy as np

import torch
from torch import Tensor

import torch

import torch

def sample_from_gmm(weights: torch.Tensor,
                    means: torch.Tensor,
                    variances: torch.Tensor,
                    num_samples: int) -> torch.Tensor:
    """
    Sample from a Gaussian mixture model with diagonal covariance.

    Args:
        weights: tensor of shape [Q, 1] or [Q], mixture weights (sum to 1)
        means: tensor of shape [Q, 1, D] or [Q, 1] (univariate)
        variances: tensor of shape [Q, 1, D] or [Q, 1]
        num_samples: number of samples to draw

    Returns:
        samples: tensor of shape [num_samples, D] or [num_samples] (univariate)
    """
    # Flatten weights to 1D if needed
    weights = weights.squeeze(-1)  # shape [Q]

    # Sample component indices
    component_indices = torch.multinomial(weights, num_samples, replacement=True)  # [num_samples]

    # Squeeze means/variances dims of 1 if present
    means = means.squeeze(1)      # [Q, D] or [Q]
    variances = variances.squeeze(1)  # [Q, D] or [Q]

    # Gather the selected components' means and variances
    selected_means = means[component_indices]       # [num_samples, D] or [num_samples]
    selected_vars = variances[component_indices]    # [num_samples, D] or [num_samples]

    std_devs = torch.sqrt(selected_vars)

    # Sample from Normal distributions
    eps = torch.randn_like(std_devs)  # same shape as selected_means
    samples = selected_means + eps * std_devs

    return samples



class DSPPModel(DSPP, GPytorchModel):

    def __init__(self,
                 input_dim: int,
                 hidden_layers_config: List[Dict[str, Any]],
                 Q: int = 8,
                 num_inducing_points: int = 256,
                 input_transform: Any = None,
                 outcome_transform: Any = None,
                 classification: bool = False,
                 num_classes: int = 2,
                 **kwargs):
        """
        Initialize a Deep Sigma Point Process.

        :param input_dim: Dimensionality of input features.
        :param train_x_shape: Shape of the training data.
        :param hidden_layers_config: List of dictionaries where each dictionary contains the configuration
                                     for a hidden layer. Each dictionary should have the keys:
                                     - "output_dims": Number of output dimensions.
                                     - "mean_type": Type of mean function ("linear" or "constant").
                                     NOTE: The last layer should always have mean_type as "constant".
        :param Q: Number of quadrature sites.
        :param num_inducing_points: Number of inducing points (per unit) for the variational strategy. Default is 128.
        :param input_transform: Transformation to be applied to the inputs. Default is None.
        :param outcome_transform: Transformation to be applied to the outputs. Default is None.
        :param classification: Whether to use Softmax likelihood or standard Gaussian likelihood.
        :param num_classes: Number of classes in case of classification.
        """
        super().__init__(num_quad_sites=Q)
        input_dims = input_dim
        self.layers = []
        self.Q = Q
        self.num_inducing_points = num_inducing_points

        hidden_layers_config = json.loads(hidden_layers_config) if isinstance(hidden_layers_config, str) else \
            hidden_layers_config

        # Build hidden layers
        if hidden_layers_config is None:
            print("[DSPP Constructor] WARNING: No hidden layers configuration is provided. Forward()"
                  " will call the identity function.")
            self.layers.append(nn.Identity())
        else:
            for layer_config in hidden_layers_config:
                hidden_layer = DSPPHiddenLayer(
                    input_dims=input_dims,
                    output_dims=layer_config['output_dims'],
                    num_inducing=num_inducing_points,
                    mean_type=layer_config['mean_type'],
                    Q=Q
                )
                self.layers.append(hidden_layer)
                input_dims = hidden_layer.output_dims if hidden_layer.output_dims else 1

        self.out_dim = input_dims
        self.layers = torch.nn.ModuleList(self.layers)
        # mixing_weight is currently false which means self.out_dim == num_classes is required.
        self.likelihood = SoftmaxLikelihood(num_classes=num_classes, num_features=self.out_dim, mixing_weights=False) \
            if classification else GaussianLikelihood()
        self._num_outputs = 1
        # self.double()
        self.intermediate_outputs = None

        # Initialize transforms
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def forward(self, inputs: Tensor, **kwargs) -> MultivariateNormal:
        """
        Forward pass through the model.
        Side effect: stores intermediate output representations in a list.

        :param inputs: Input tensor.
        :return: Output distribution (with mean, variance) after passing through the hidden layers.
        """
        x = inputs
        self.intermediate_outputs = []
        for layer in self.layers:
            x = layer(x, **kwargs)
            self.intermediate_outputs.append(x)
        return x

    def posterior(
            self,
            X: Tensor,
            apply_likelihood: bool = False,  # Renamed from observation_noise
            *args, **kwargs
    ) -> Union[MultivariateNormal, Categorical]:
        """
        Compute the posterior distribution.

        :param X: Input tensor.
        :param apply_likelihood: Whether to apply the likelihood transformation.
                                 For classification, this returns class probabilities.
                                 For regression, this adds observation noise.
        :return: Posterior distribution.
        """
        self.eval()
        if self.input_transform is not None:
            X = self.input_transform(X)

        with torch.no_grad():
            dist = self(X, mean_input=X)
            if apply_likelihood:
                dist = self.likelihood(dist)

        if self.outcome_transform is not None:
            dist = self.outcome_transform.untransform(dist)
        return dist

    def get_intermediate_outputs(self) -> List[Tensor]:
        return self.intermediate_outputs
