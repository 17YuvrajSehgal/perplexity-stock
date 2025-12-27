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

    IMPORTANT:
    This validation now matches the environment's execution model in environ.State.step():
      - action chosen at time t
      - env advances to t+1
      - execution occurs at OPEN(t+1)

    Returns mean metrics across episodes.
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
        "sum_trade_return": [],   # sum of trade returns per episode
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

            act_enum = environ.Actions(action)

            # ---- Match execution model: trades execute at OPEN(t+1) ----
            st = env._state
            next_idx = st._offset + 1

            exec_open = None
            if 0 <= next_idx < st._prices.open.shape[0]:
                exec_open = float(st._prices.open[next_idx])

            # Keep a copy to detect forced close after env.step()
            prev_have_pos_env = bool(st.have_position)

            # --- bookkeeping BEFORE step (because env will execute at OPEN(t+1)) ---
            if exec_open is not None:
                if act_enum == environ.Actions.Buy and not in_pos:
                    in_pos = True
                    entry_price = exec_open
                    hold_steps = 0

                elif act_enum == environ.Actions.Close and in_pos:
                    if entry_price and entry_price > 0:
                        tr = 100.0 * (exec_open - entry_price) / entry_price
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

            # --- detect forced close (terminal liquidation / reset_on_close etc.) ---
            # If we thought we were in a position, but env position ended without an explicit Close,
            # record the trade using the best available execution price.
            now_have_pos_env = bool(env._state.have_position)

            if in_pos and prev_have_pos_env and (not now_have_pos_env) and act_enum != environ.Actions.Close:
                # env ended position; use exec_open if available, otherwise current close as fallback
                exit_price = exec_open
                if exit_price is None:
                    exit_price = float(env._state._cur_close())

                if entry_price and entry_price > 0:
                    tr = 100.0 * (exit_price - entry_price) / entry_price
                else:
                    tr = 0.0
                trade_returns.append(tr)
                trade_hold_steps.append(hold_steps)

                in_pos = False
                entry_price = None
                hold_steps = 0

            if in_pos:
                hold_steps += 1

        # If episode ends while holding (should be rare after Bug 2 fix), close "virtually" for reporting
        if in_pos and entry_price and entry_price > 0:
            # Use last close as a safe fallback
            last_close = float(env._state._cur_close())
            tr = 100.0 * (last_close - entry_price) / entry_price
            trade_returns.append(tr)
            trade_hold_steps.append(hold_steps)

        # Episode metrics
        stats["episode_reward"].append(total_reward)
        stats["episode_steps"].append(steps)

        # Trade metrics
        n_trades = len(trade_returns)
        stats["num_trades"].append(float(n_trades))

        if n_trades > 0:
            wins = sum(1 for x in trade_returns if x > 0.0)
            stats["win_rate"].append(float(wins / n_trades))
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
