import numpy as np
import torch

from finance_rl import environ


# Import the trading environment (used for action definitions)


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02, comission=0.1):
    """
    Run a validation (evaluation) phase for a trained agent.

    This function:
    - Runs the agent for a fixed number of episodes
    - Uses mostly greedy actions (small epsilon for randomness)
    - Tracks rewards, episode length, and individual trade profits
    - Returns averaged statistics across episodes

    Parameters
    ----------
    env : StocksEnv
        Trading environment
    net : torch.nn.Module
        Trained DQN network
    episodes : int
        Number of evaluation episodes
    device : str
        'cpu' or 'cuda'
    epsilon : float
        Small probability of random action (for robustness)
    comission : float
        Transaction cost percentage (e.g., 0.1 = 0.1%)

    Returns
    -------
    dict
        Mean statistics across all evaluation episodes
    """

    # Dictionary to collect statistics
    stats = {
        'episode_reward': [],   # total reward per episode
        'episode_steps': [],    # number of steps per episode
        'order_profits': [],    # profit of each completed trade
        'order_steps': [],      # how long each trade was held
    }

    # -----------------------------------------------------
    # Run multiple evaluation episodes
    # -----------------------------------------------------
    for episode in range(episodes):

        # Reset environment at start of episode
        obs, _ = env.reset()

        total_reward = 0.0
        position = None          # price at which current position was opened
        position_steps = None    # number of steps the position has been held
        episode_steps = 0

        # -------------------------------------------------
        # Step through one episode
        # -------------------------------------------------
        while True:
            # Convert observation to a torch tensor
            obs_v = torch.tensor([obs]).to(device)

            # Get Q-values for all actions
            out_v = net(obs_v)

            # Greedy action selection (argmax Q-value)
            action_idx = out_v.max(dim=1)[1].item()

            # Small epsilon-greedy exploration (for robustness)
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()

            action = environ.Actions(action_idx)

            # Current market price (used to compute trade profit)
            close_price = env._state._cur_close()

            # ---------------------------------------------
            # Track individual trades (Buy / Close)
            # ---------------------------------------------
            if action == environ.Actions.Buy and position is None:
                # Open a new position
                position = close_price
                position_steps = 0

            elif action == environ.Actions.Close and position is not None:
                # Close the position and calculate profit
                profit = close_price - position
                profit -= (close_price + position) * comission / 100
                profit = 100.0 * profit / position

                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)

                position = None
                position_steps = None

            # ---------------------------------------------
            # Step environment forward
            # ---------------------------------------------
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            total_reward += reward
            episode_steps += 1

            # Increase holding duration if a position is open
            if position_steps is not None:
                position_steps += 1

            # ---------------------------------------------
            # Episode termination
            # ---------------------------------------------
            if done:
                # If episode ends with an open position, close it virtually
                if position is not None:
                    profit = close_price - position
                    profit -= (close_price + position) * comission / 100
                    profit = 100.0 * profit / position

                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)

                break

        # Store episode-level statistics
        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    # -----------------------------------------------------
    # Return mean statistics across all episodes
    # -----------------------------------------------------
    return {key: np.mean(vals) for key, vals in stats.items()}
