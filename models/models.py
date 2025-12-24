# models.py
# ----------
# This file defines neural network models used by the DQN agent.
# These models take an observation from the trading environment
# and output Q-values for each possible action (Buy / Close / Skip).

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Noisy Linear Layer (for exploration)
# =========================================================
class NoisyLinear(nn.Linear):
    """
    A linear (fully-connected) layer with parameterized noise.

    Why this exists:
    ----------------
    In standard DQN, exploration is usually done with epsilon-greedy.
    NoisyNet replaces that by injecting noise directly into the network
    weights, encouraging exploration automatically.

    This helps:
    - Reduce dependence on epsilon schedules
    - Encourage more consistent exploration
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        # Initialize as a normal Linear layer
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        # Learnable noise scale for weights
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )

        # Random noise (not learnable)
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features)
        )

        if bias:
            # Learnable noise scale for bias
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init)
            )
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize base weights and bias uniformly.
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        """
        Forward pass with noise injected into weights and bias.
        """
        # Sample fresh noise every forward pass
        self.epsilon_weight.normal_()

        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias

        # Apply linear transformation with noisy weights
        return F.linear(
            input,
            self.weight + self.sigma_weight * self.epsilon_weight,
            bias
        )


# =========================================================
# Fully-Connected Dueling DQN
# =========================================================
class SimpleFFDQN(nn.Module):
    """
    A simple fully-connected Dueling DQN architecture.

    Used when observations are flat vectors (e.g., shape = (42,)).
    """

    def __init__(self, obs_len, actions_n):
        """
        obs_len   : length of observation vector
        actions_n : number of discrete actions
        """
        super(SimpleFFDQN, self).__init__()

        # -------------------------
        # Value stream
        # -------------------------
        # Estimates how good the state is (V(s))
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)   # Outputs a single value V(s)
        )

        # -------------------------
        # Advantage stream
        # -------------------------
        # Estimates how good each action is relative to the state
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)  # One value per action
        )

    def forward(self, x):
        """
        Forward pass.

        Combines value and advantage streams using the dueling formula:
            Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
        """
        val = self.fc_val(x)
        adv = self.fc_adv(x)

        return val + adv - adv.mean(dim=1, keepdim=True)


# =========================================================
# 1D Convolutional DQN (medium size)
# =========================================================
class DQNConv1D(nn.Module):
    """
    Dueling DQN using 1D convolutions.

    Used when observations are shaped as:
        (channels, time_steps)
    This is suitable for CNN-based agents (State1D).
    """

    def __init__(self, shape, actions_n):
        """
        shape     : observation shape (channels, bars_count)
        actions_n : number of actions
        """
        super(DQNConv1D, self).__init__()

        # -------------------------
        # Convolutional feature extractor
        # -------------------------
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5),
            nn.ReLU(),
        )

        # Compute size after convolutions
        out_size = self._get_conv_out(shape)

        # -------------------------
        # Value stream
        # -------------------------
        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # -------------------------
        # Advantage stream
        # -------------------------
        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        """
        Helper function to compute the flattened size
        after the convolution layers.
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass through CNN + dueling heads.
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)

        return val + adv - adv.mean(dim=1, keepdim=True)


# =========================================================
# Larger 1D Convolutional DQN
# =========================================================
class DQNConv1DLarge(nn.Module):
    """
    A deeper and more expressive 1D CNN DQN.

    Suitable for longer time windows or more complex patterns.
    """

    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        # -------------------------
        # Deep convolutional stack
        # -------------------------
        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        # -------------------------
        # Value stream
        # -------------------------
        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # -------------------------
        # Advantage stream
        # -------------------------
        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        """
        Compute output size after all convolution layers.
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass through deep CNN + dueling heads.
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)

        return val + adv - adv.mean(dim=1, keepdim=True)
