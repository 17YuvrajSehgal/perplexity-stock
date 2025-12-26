import numpy as np
import torch
from torch.distributions import Categorical

from finance_rl import environ


def _base_env(env):
    # Unwrap wrappers (TimeLimit, RecordEpisodeStatistics, etc.)
    while hasattr(env, "env"):
        env = env.env
    return env


@torch.no_grad()
def validation_run_ppo(
    env,
    model,
    episodes: int = 50,
    device: str = "cpu",
    greedy: bool = True,
):
    """
    Validation that matches the UPDATED environment semantics:

    - Action at time t executes at OPEN(t+1)
    - Therefore trade entry/exit prices should be taken from base._state._prices.open[next_offset]
      where next_offset = current_offset + 1 (because env advances the offset before executing).

    This keeps trade stats (win_rate/avg_trade_return) aligned with env reward.
    """
    base = _base_env(env)

    stats = {
        "episode_reward": [],
        "episode_steps": [],
        "num_trades": [],
        "win_rate": [],
        "avg_trade_return": [],
        "avg_hold_steps": [],
        "sum_trade_return": [],
    }

    for _ in range(episodes):
        obs, info = env.reset()
        done = False

        total_reward = 0.0
        steps = 0

        # Trade bookkeeping (execution-aligned)
        in_pos = False
        entry_price = None
        hold_steps = 0

        trade_returns = []
        trade_hold_steps = []

        while not done:
            obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _value = model(obs_v)

            if greedy:
                action = int(torch.argmax(logits, dim=1).item())
            else:
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

            act_enum = environ.Actions(action)

            # We want to record trade entry/exit at the SAME execution price as the env:
            # env will advance offset to t+1, then execute at OPEN(t+1).
            cur_ofs = int(base._state._offset)
            next_ofs = cur_ofs + 1

            # If next_ofs is out of bounds, the env will terminate on step; don't try to read prices.
            prices = base._state._prices
            n = int(prices.open.shape[0])
            can_exec = next_ofs < n

            exec_open = float(prices.open[next_ofs]) if can_exec else None

            # Bookkeeping before stepping (so it matches which action we sent)
            if act_enum == environ.Actions.Buy and (not in_pos) and can_exec:
                in_pos = True
                entry_price = exec_open
                hold_steps = 0

            elif act_enum == environ.Actions.Close and in_pos and can_exec:
                if entry_price and entry_price > 0:
                    tr = 100.0 * (exec_open - entry_price) / entry_price
                else:
                    tr = 0.0
                trade_returns.append(tr)
                trade_hold_steps.append(hold_steps)

                in_pos = False
                entry_price = None
                hold_steps = 0

            # Step the env
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            if in_pos:
                hold_steps += 1

        # If episode ends while holding, "close" virtually at the last available close
        # (This is just for stats; env may not have forced close.)
        if in_pos and entry_price and entry_price > 0:
            # base._state._offset should be a valid index here (env protects terminal obs)
            last_close = float(base._state._prices.close[int(base._state._offset)])
            tr = 100.0 * (last_close - entry_price) / entry_price
            trade_returns.append(tr)
            trade_hold_steps.append(hold_steps)

        stats["episode_reward"].append(total_reward)
        stats["episode_steps"].append(steps)

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

    return {k: float(np.mean(v)) for k, v in stats.items()}
