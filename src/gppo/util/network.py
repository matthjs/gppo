from typing import List, Tuple, Optional
import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, conv: nn.Module):
        """
        Wraps a conv block with an optional residual connection.
        """
        super().__init__()
        self.conv = conv

    def forward(self, x):
        out = self.conv(x)
        if out.shape == x.shape:
            return out + x
        return out


class ConvNetEstimator(nn.Module):
    def __init__(
            self,
            input_channels: int = 4,
            num_out_features: int = 6,
            use_residual: bool = False
    ):
        """
        Convolutional neural network for estimating Q-values for discrete actions.

        :param input_channels: Number of channels in the input state image.
        :param num_out_features: Number of discrete actions (output size).
        :param use_residual: Whether to use residual connections in the conv layers.
        """

        super(ConvNetEstimator, self).__init__()

        self.input_channel = input_channels

        def conv_block(in_channels, out_channels, kernel_size, stride) -> nn.Sequential:
            layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                nn.ReLU()
            )
            return ResidualBlock(layers) if use_residual else layers

        self.feature_extractor = nn.Sequential(
            # Convolutional layers
            conv_block(input_channels, 32, kernel_size=8, stride=4),  # [B, 32, 20, 20]
            conv_block(32, 64, kernel_size=4, stride=2),  # [B, 64, 9, 9]
            conv_block(64, 64, kernel_size=3, stride=1),  # [B, 64, 7, 7]
            # Flattening
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.action_stream = nn.Linear(512, num_out_features)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the CNN.

        :param state: Input image as a NumPy array [C, H, W] or [B, C, H, W].
        :return: Output logits corresponding to action values.
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension if missing

        # Transpose if the channels are in the wrong dimension: [B, H, W, C] -> [B, C, H, W]
        if state.shape[1] != self.input_channel:  # If the channels are not in the second position
            state = state.permute(0, 3, 1, 2)  # Convert from [B, H, W, C] -> [B, C, H, W]

        x = self.feature_extractor(state)
        return self.action_stream(x)    # Get estimated Q-values for each action


class DuelingConvNetEstimator(ConvNetEstimator):
    def __init__(self,
                 input_channels: int = 4,
                 num_out_features: int = 6,
                 use_residual: bool = False):
        """
        CNN with dueling architecture: separates value and advantage streams.

        :param input_channels: Number of channels in the input state image.
        :param num_out_features: Number of discrete actions (output size).
        :param use_residual: Whether to use residual connections in the conv layers.
        """
        super().__init__(input_channels, num_out_features, use_residual)
        # action_stream acts as the advantage stream here
        self.value_stream = nn.Linear(512, 1)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension if missing

        # Transpose if the channels are in the wrong dimension: [B, H, W, C] -> [B, C, H, W]
        if state.shape[1] != self.input_channel:  # If the channels are not in the second position
            state = state.permute(0, 3, 1, 2)  # Convert from [B, H, W, C] -> [B, C, H, W]

        x = self.feature_extractor(state)
        value = self.value_stream(x)
        advantage = self.action_stream(x)

        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals


class ActorCriticMLP(nn.Module):
    """
    MLP that approximates both policy and value function. Assumes continuous actions
    for the policy. Parameterizes the policy as a Multivariate Gaussian with a
    diagonal covariance matrix.
    """
    def __init__(
            self,
            input_dim: int,
            action_dim: int,
            hidden_sizes: List[int] = [64, 64],
            ortho_init: bool = True,
            discrete: bool = False
    ):
        super().__init__()
        self.discrete = discrete
        # build MLP
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        # actor head
        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        # critic head
        self.value_head = nn.Linear(last_dim, 1)
        if ortho_init:
            self._init_weights()

    def _init_weights(self):
        # Shared trunk: orthogonal with sqrt(2) is the standard gain for ReLU activations
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        # Small gain keeps the initial policy close to uniform across actions
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

        # Standard gain for the value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.shared(x)
        # value
        value = self.value_head(x).squeeze(-1)
        # action distribution

        if self.discrete:
            logits = self.action_mean(x)
            dist = Categorical(logits=logits)
            return dist, value, None
        else:
            mean = self.action_mean(x)
            std = torch.exp(self.action_log_std)
            return Normal(mean, std), value, self.action_log_std


class ActorCriticCNN(nn.Module):
    """
    CNN-based Actor-Critic for image-based observation spaces.
    Reuses the convolutional feature extractor from ConvNetEstimator,
    then splits into actor (policy) and critic (value) heads.
    Suitable for PPO and similar on-policy algorithms.
    """
    def __init__(
            self,
            input_channels: int = 4,
            action_dim: int = 6,
            use_residual: bool = False,
            feature_dim: int = 512,
            ortho_init: bool = True,
            discrete: bool = False
    ):
        super().__init__()
        self.discrete = discrete
        self.input_channels = input_channels

        # Reuse the proven conv backbone — borrow it from a throwaway estimator instance
        # so we don't duplicate the architecture definition.
        _backbone = ConvNetEstimator(input_channels, action_dim, use_residual)
        self.feature_extractor = _backbone.feature_extractor  # Sequential up to the 512-d ReLU

        # Actor head: outputs mean of a Gaussian policy
        self.action_mean = nn.Linear(feature_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head: outputs a scalar state-value estimate
        self.value_head = nn.Linear(feature_dim, 1)

        if ortho_init:
            self._init_weights()

    def _init_weights(self):
        """Orthogonal init is standard practice for PPO heads."""
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _preprocess(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Shared input handling: numpy conversion, batch dim, channel ordering."""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if state.shape[1] != self.input_channels:           # [B, H, W, C] -> [B, C, H, W]
            state = state.permute(0, 3, 1, 2)
        return state

    def forward(self, state: np.ndarray | torch.Tensor) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        :param state: [C, H, W] or [B, C, H, W] image (numpy or tensor).
        :return: (action_distribution, state_value, action_log_std)
        """
        x = self.feature_extractor(self._preprocess(state))
        # Critic
        value = self.value_head(x).squeeze(-1)

        # Actor
        if self.discrete:
            dist = Categorical(logits=self.action_mean(x))
            return dist, value, None
        else:
            mean = self.action_mean(x)
            std = torch.exp(self.action_log_std)
            return Normal(mean, std.expand_as(mean)), value, self.action_log_std
