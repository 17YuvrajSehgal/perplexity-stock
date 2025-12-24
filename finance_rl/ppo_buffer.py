# finance_rl/ppo_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch


@dataclass
class PPOBatch:
    """
    One minibatch used during PPO updates.
    """
    obs: torch.Tensor         # (B, obs_dim) or (B, C, T) for CNN
    actions: torch.Tensor     # (B,)
    old_logprobs: torch.Tensor  # (B,)
    advantages: torch.Tensor  # (B,)
    returns: torch.Tensor     # (B,)
    old_values: torch.Tensor  # (B,)


class RolloutBuffer:
    """
    Stores a fixed-length rollout of experience and computes:
      - GAE(lambda) advantages
      - returns (targets for value function)

    This buffer is designed for PPO:
      - store old_logprobs and old_values from the behavior policy
      - normalize advantages for stability
    """

    def __init__(
        self,
        obs_shape,
        size: int,
        device: str = "cpu",
        dtype=np.float32,
    ):
        """
        obs_shape: int or tuple
          - if MLP: obs_shape = obs_dim (int) or (obs_dim,)
          - if CNN: obs_shape = (C, T)
        size: rollout length (T)
        """
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.obs_shape = tuple(obs_shape)
        self.size = int(size)
        self.device = device

        # Storage
        self.obs = np.zeros((self.size, *self.obs_shape), dtype=dtype)
        self.actions = np.zeros((self.size,), dtype=np.int64)
        self.rewards = np.zeros((self.size,), dtype=np.float32)

        # done flags: 1.0 when episode ended (terminated or truncated), else 0.0
        self.dones = np.zeros((self.size,), dtype=np.float32)

        # Behavior policy outputs (at collection time)
        self.values = np.zeros((self.size,), dtype=np.float32)
        self.logprobs = np.zeros((self.size,), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((self.size,), dtype=np.float32)
        self.returns = np.zeros((self.size,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self):
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        logprob: float,
    ):
        """
        Add one transition (s_t, a_t, r_t, done_t, V(s_t), log pi(a_t|s_t)).
        """
        if self.ptr >= self.size:
            raise RuntimeError("RolloutBuffer is full. Call reset() before adding more.")

        self.obs[self.ptr] = np.asarray(obs, dtype=self.obs.dtype)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if bool(done) else 0.0
        self.values[self.ptr] = float(value)
        self.logprobs[self.ptr] = float(logprob)

        self.ptr += 1
        if self.ptr == self.size:
            self.full = True

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
    ):
        """
        Compute advantages and returns using GAE(lambda).

        last_value:
          - V(s_{T}) at the final observation after collecting rollout
          - used to bootstrap if last step was not terminal

        dones[t] == 1.0 means episode ended at step t.
        """
        if not self.full:
            raise RuntimeError("RolloutBuffer not full. Collect 'size' steps before compute_gae().")

        adv = 0.0
        for t in reversed(range(self.size)):
            mask = 1.0 - self.dones[t]  # 0 if terminal else 1

            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv

            self.advantages[t] = adv

        self.returns = self.advantages + self.values

        if normalize_adv:
            m = float(self.advantages.mean())
            s = float(self.advantages.std()) + 1e-8
            self.advantages = (self.advantages - m) / s

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[PPOBatch]:
        """
        Yield mini-batches as torch tensors on the correct device.

        Returns PPOBatch objects.
        """
        if not self.full:
            raise RuntimeError("RolloutBuffer not full. Collect rollout before batching.")

        idxs = np.arange(self.size)
        if shuffle:
            np.random.shuffle(idxs)

        for start in range(0, self.size, batch_size):
            b_idx = idxs[start:start + batch_size]

            # Use torch.as_tensor to avoid unnecessary copies (NumPy2 safe)
            obs_t = torch.as_tensor(self.obs[b_idx], device=self.device, dtype=torch.float32)
            actions_t = torch.as_tensor(self.actions[b_idx], device=self.device, dtype=torch.long)
            old_logp_t = torch.as_tensor(self.logprobs[b_idx], device=self.device, dtype=torch.float32)
            adv_t = torch.as_tensor(self.advantages[b_idx], device=self.device, dtype=torch.float32)
            ret_t = torch.as_tensor(self.returns[b_idx], device=self.device, dtype=torch.float32)
            old_v_t = torch.as_tensor(self.values[b_idx], device=self.device, dtype=torch.float32)

            yield PPOBatch(
                obs=obs_t,
                actions=actions_t,
                old_logprobs=old_logp_t,
                advantages=adv_t,
                returns=ret_t,
                old_values=old_v_t,
            )

    def __len__(self):
        return self.ptr
