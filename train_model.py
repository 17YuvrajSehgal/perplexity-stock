# train_model.py (Fix B: no ptan, Gymnasium-native)

import os
import argparse
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from tensorboardX import SummaryWriter

from finance_rl import environ, data_yf as data, common, validation
from models import models

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
BATCH_SIZE = 32
BARS_COUNT = 10
GAMMA = 0.99
LEARNING_RATE = 1e-4

REPLAY_SIZE = 100_000
REPLAY_INITIAL = 10_000
TARGET_NET_SYNC = 1_000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1_000_000

VALIDATION_EVERY_STEP = 100_000

# ---------------------------------------------------------
# Simple Experience container (matches common.calc_loss)
# ---------------------------------------------------------
Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "last_state"]
)

# ---------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", required=True, help="Run name")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("--data", default="yf_data", help="Directory with CSV files")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------
    prices = data.load_many_from_dir(args.data)

    env = environ.StocksEnv(
        prices,
        bars_count=BARS_COUNT,
        volumes=True,
        reset_on_close=False
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # -----------------------------------------------------
    # Networks
    # -----------------------------------------------------
    net = models.SimpleFFDQN(obs_size, n_actions).to(device)
    tgt_net = models.SimpleFFDQN(obs_size, n_actions).to(device)
    tgt_net.load_state_dict(net.state_dict())

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # -----------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------
    buffer = ReplayBuffer(REPLAY_SIZE)

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    writer = SummaryWriter(comment="-fixB-" + args.run)

    step_idx = 0
    epsilon = EPSILON_START

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0

    with common.RewardTracker(writer, stop_reward=np.inf, group_rewards=10) as tracker:
        while True:
            step_idx += 1

            # Epsilon decay
            epsilon = max(
                EPSILON_STOP,
                EPSILON_START - step_idx / EPSILON_STEPS
            )

            # Action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_v = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)

                    action = net(obs_v).max(1)[1].item()

            # Environment step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            exp = Experience(
                state=obs,
                action=action,
                reward=reward,
                last_state=None if done else next_obs
            )
            buffer.append(exp)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # Episode end
            if done:
                if tracker.reward((episode_reward, episode_steps), step_idx, epsilon):
                    break
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0

            # Wait until buffer is populated
            if len(buffer) < REPLAY_INITIAL:
                continue

            # Train
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss = common.calc_loss(batch, net, tgt_net, GAMMA, device=device)
            loss.backward()
            optimizer.step()

            # Sync target network
            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.load_state_dict(net.state_dict())

            # Periodic validation
            if step_idx % VALIDATION_EVERY_STEP == 0:
                res = validation.validation_run(env, net, device=device)
                for k, v in res.items():
                    writer.add_scalar("val/" + k, v, step_idx)
