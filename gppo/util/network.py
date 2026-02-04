from typing import List, Tuple, Optional
import torch
from torch.distributions import Normal
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
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.shared(x)
        # value
        value = self.value_head(x).squeeze(-1)
        # action distribution
        mean = self.action_mean(x)
        std = torch.exp(self.action_log_std)
        dist = Normal(mean, std)
        log_std = self.action_log_std
        return dist, value, log_std

