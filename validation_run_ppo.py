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

        # Trade bookkeeping
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

            # Use the *base* env internal state
            cur_close = float(base._state._cur_close())

            act_enum = environ.Actions(action)

            if act_enum == environ.Actions.Buy and not in_pos:
                in_pos = True
                entry_price = cur_close
                hold_steps = 0

            elif act_enum == environ.Actions.Close and in_pos:
                if entry_price and entry_price > 0:
                    tr = 100.0 * (cur_close - entry_price) / entry_price
                else:
                    tr = 0.0
                trade_returns.append(tr)
                trade_hold_steps.append(hold_steps)

                in_pos = False
                entry_price = None
                hold_steps = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            if in_pos:
                hold_steps += 1

        # If episode ends while holding, close virtually for stats
        if in_pos and entry_price and entry_price > 0:
            last_close = float(base._state._cur_close())
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
