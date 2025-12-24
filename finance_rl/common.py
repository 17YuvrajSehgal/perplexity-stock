import sys
import time
import numpy as np

import torch
import torch.nn as nn


# =========================================================
# RewardTracker
# =========================================================
class RewardTracker:
    """
    A helper class used during training to:
      - keep track of episode rewards and episode lengths
      - print training progress to the console
      - log useful metrics to TensorBoard (through `writer`)
      - optionally stop training once a target mean reward is reached

    Think of it as a "training dashboard":
    it continuously reports how well the agent is doing.
    """

    def __init__(self, writer, stop_reward, group_rewards=1):
        """
        writer        : TensorBoard SummaryWriter (or compatible logger)
        stop_reward   : if mean reward over last 100 episodes exceeds this, training stops
        group_rewards : print/log only after this many episodes are collected
                        (useful to smooth noisy results)
        """
        self.writer = writer
        self.stop_reward = stop_reward

        # Buffers to collect a small group of episode results (for smoothing)
        self.reward_buf = []
        self.steps_buf = []

        self.group_rewards = group_rewards

    def __enter__(self):
        """
        Called when you use:
            with RewardTracker(...) as tracker:
                ...

        Initializes timers and storage for rewards/steps.
        """
        self.ts = time.time()     # time stamp for speed calculation
        self.ts_frame = 0         # frame index at last speed measurement
        self.total_rewards = []   # stores mean reward per episode
        self.total_steps = []     # stores steps per episode
        return self

    def __exit__(self, *args):
        """
        Called automatically at the end of the 'with' block.
        Closes the TensorBoard writer.
        """
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None):
        """
        Add one finished episode result to the tracker.

        reward_steps : tuple (episode_reward, episode_steps)
        frame        : current frame count (global training step counter)
        epsilon      : epsilon for epsilon-greedy exploration (optional)

        Returns:
            True  -> training should stop (solved)
            False -> keep training
        """
        reward, steps = reward_steps

        # Store this episode in a buffer (for grouping/smoothing)
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)

        # If we haven't collected enough episodes for one group, do nothing yet
        if len(self.reward_buf) < self.group_rewards:
            return False

        # Average over the group (smooth out randomness)
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)

        # Clear buffers now that we consumed them
        self.reward_buf.clear()
        self.steps_buf.clear()

        # Store grouped results for global statistics
        self.total_rewards.append(reward)
        self.total_steps.append(steps)

        # Calculate training speed: frames per second
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()

        # Mean reward/steps over last 100 episodes (typical RL reporting)
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])

        # Build a nice console print message
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame,
            len(self.total_rewards) * self.group_rewards,
            mean_reward,
            mean_steps,
            speed,
            epsilon_str
        ))
        sys.stdout.flush()

        # Log metrics to TensorBoard so you can plot them later
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)

        # Stop training condition: agent is "good enough"
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True

        return False


# =========================================================
# Evaluate the value of states
# =========================================================
def calc_values_of_states(states, net, device="cpu"):
    """
    Given a list/array of states, estimate how "valuable" they are.

    We do this by:
      - feeding states into the Q-network
      - taking max Q-value across actions for each state
      - averaging those max values

    Why this is useful:
      - It gives a rough sense of how confident/optimistic the network is.
    """
    mean_vals = []

    # Split into chunks so we don't run out of GPU memory
    for batch in np.array_split(states, 64):
        states_v = torch.as_tensor(batch, device=device)

        action_values_v = net(states_v)                 # Q(s, a) for all actions
        best_action_values_v = action_values_v.max(1)[0]  # max_a Q(s, a)
        mean_vals.append(best_action_values_v.mean().item())

    return np.mean(mean_vals)


# =========================================================
# Convert a batch of experience objects into arrays
# =========================================================
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.asarray(exp.state)   # NumPy 2.0 safe
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)

        if exp.last_state is None:
            last_states.append(state)   # masked anyway
        else:
            last_states.append(np.asarray(exp.last_state))  # NumPy 2.0 safe

    return (
        np.asarray(states),
        np.asarray(actions),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.uint8),
        np.asarray(last_states),
    )

def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    """
    Compute the DQN loss for a batch of experiences.

    This implements Double-DQN style target calculation:
      - choose next action using the ONLINE network (net)
      - evaluate that action using the TARGET network (tgt_net)

    Parameters
    ----------
    batch : list[Experience]
        Each item must have: state, action, reward, last_state (None if terminal)
    net : torch.nn.Module
        Online Q-network (being trained)
    tgt_net : torch.nn.Module
        Target Q-network (slow-moving copy)
    gamma : float
        Discount factor
    device : str or torch.device
        "cpu" or "cuda"

    Returns
    -------
    torch.Tensor
        Scalar loss tensor
    """
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # Convert to tensors
    states_v = torch.as_tensor(states, device=device)
    next_states_v = torch.as_tensor(next_states, device=device)
    actions_v = torch.as_tensor(actions, device=device, dtype=torch.long)
    rewards_v = torch.as_tensor(rewards, device=device, dtype=torch.float32)

    # bool mask for terminal transitions
    done_mask = torch.as_tensor(dones, device=device, dtype=torch.bool)

    # Q(s,a) for taken actions
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Double DQN:
    # 1) choose best next action from ONLINE net
    next_state_actions = net(next_states_v).max(1)[1]

    # 2) evaluate that action using TARGET net
    next_state_values = tgt_net(next_states_v).gather(
        1, next_state_actions.unsqueeze(-1)
    ).squeeze(-1)

    # no bootstrap value for terminal states
    next_state_values = next_state_values.masked_fill(done_mask, 0.0)

    # Bellman target
    expected_state_action_values = rewards_v + gamma * next_state_values.detach()

    return nn.MSELoss()(state_action_values, expected_state_action_values)
