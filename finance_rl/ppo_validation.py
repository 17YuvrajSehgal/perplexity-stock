import numpy as np
import torch
from torch.distributions import Categorical

from . import environ


@torch.no_grad()
def validation_run_ppo(
    env,
    model,
    episodes: int = 50,
    device: str = "cpu",
    greedy: bool = True,
):
    """
    Evaluate a PPO policy on the given environment.

    Parameters
    ----------
    env : gymnasium.Env
        Your StocksEnv wrapped or unwrapped. Must use Gymnasium API:
        reset() -> (obs, info)
        step(a) -> (obs, reward, terminated, truncated, info)
    model : torch.nn.Module
        Actor-Critic model. Forward: logits, value = model(obs_batch)
    episodes : int
        Number of evaluation episodes.
    device : str
        "cpu" or "cuda"
    greedy : bool
        If True, pick argmax(logits). If False, sample from policy distribution.

    Returns
    -------
    dict
        Mean metrics across episodes.
    """
    stats = {
        # episode-level
        "episode_reward": [],
        "episode_steps": [],

        # trade-level
        "num_trades": [],
        "win_rate": [],
        "avg_trade_return": [],   # percent return per trade
        "avg_hold_steps": [],
        "sum_trade_return": [],   # sum of trade returns per episode (rough "equity" proxy)
    }

    for _ in range(episodes):
        obs, info = env.reset()
        done = False

        total_reward = 0.0
        steps = 0

        # Track trades manually (single-position assumption)
        in_pos = False
        entry_price = None
        hold_steps = 0

        trade_returns = []
        trade_hold_steps = []

        while not done:
            obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(obs_v)

            if greedy:
                action = int(torch.argmax(logits, dim=1).item())
            else:
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

            # Current close (uses internal state, which is fine for evaluation)
            cur_close = float(env._state._cur_close())

            # Update trade bookkeeping BEFORE step (action applies at this bar)
            act_enum = environ.Actions(action)

            if act_enum == environ.Actions.Buy and not in_pos:
                in_pos = True
                entry_price = cur_close
                hold_steps = 0

            elif act_enum == environ.Actions.Close and in_pos:
                # close trade and compute percent return (no commission here;
                # env reward already includes commission if configured)
                if entry_price and entry_price > 0:
                    tr = 100.0 * (cur_close - entry_price) / entry_price
                else:
                    tr = 0.0
                trade_returns.append(tr)
                trade_hold_steps.append(hold_steps)

                in_pos = False
                entry_price = None
                hold_steps = 0

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            if in_pos:
                hold_steps += 1

        # If episode ends while holding, close "virtually" for reporting
        if in_pos and entry_price and entry_price > 0:
            last_close = float(env._state._cur_close())
            tr = 100.0 * (last_close - entry_price) / entry_price
            trade_returns.append(tr)
            trade_hold_steps.append(hold_steps)

        # Episode metrics
        stats["episode_reward"].append(total_reward)
        stats["episode_steps"].append(steps)

        # Trade metrics
        n_trades = len(trade_returns)
        stats["num_trades"].append(n_trades)

        if n_trades > 0:
            wins = sum(1 for x in trade_returns if x > 0)
            stats["win_rate"].append(wins / n_trades)
            stats["avg_trade_return"].append(float(np.mean(trade_returns)))
            stats["avg_hold_steps"].append(float(np.mean(trade_hold_steps)))
            stats["sum_trade_return"].append(float(np.sum(trade_returns)))
        else:
            stats["win_rate"].append(0.0)
            stats["avg_trade_return"].append(0.0)
            stats["avg_hold_steps"].append(0.0)
            stats["sum_trade_return"].append(0.0)

    # return means
    return {k: float(np.mean(v)) for k, v in stats.items()}
