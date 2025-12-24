# finance_rl/ppo_models.py
from __future__ import annotations

import torch
import torch.nn as nn


class ActorCriticMLP(nn.Module):
    """
    PPO Actor-Critic for DISCRETE actions.

    Input:  (B, obs_dim)
    Output: logits (B, n_actions), value (B,)
      - logits are unnormalized scores for a Categorical distribution
      - value is V(s) baseline for advantage estimation
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden, n_actions)  # actor logits
        self.value_head = nn.Linear(hidden, 1)           # critic value

        self._init_weights()

    def _init_weights(self):
        """
        PPO is usually more stable with small initial weights.
        Orthogonal init is a common default, but we keep it simple and safe here.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

        # Slightly smaller init for final policy head can help early training
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x: float tensor (B, obs_dim)
        returns: logits (B, n_actions), value (B,)
        """
        z = self.shared(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


class ActorCriticConv1D(nn.Module):
    """
    PPO Actor-Critic for State1D observations (channels, time).

    Your State1D returns obs shape: (C, bars_count)
    Gymnasium will give you obs as numpy array with that shape,
    and in training you'll typically batch it to (B, C, T).

    Input:  (B, C, T)
    Output: logits (B, n_actions), value (B,)
    """

    def __init__(self, in_channels: int, n_actions: int, bars_count: int, hidden: int = 256):
        super().__init__()

        # A small 1D CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute conv output size = 64 * bars_count
        conv_out = 64 * bars_count

        self.shared = nn.Sequential(
            nn.Linear(conv_out, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x: float tensor (B, C, T)
        returns: logits (B, n_actions), value (B,)
        """
        feat = self.conv(x)
        z = self.shared(feat)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value
