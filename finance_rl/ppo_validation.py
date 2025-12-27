import numpy as np
import torch
from torch.distributions import Categorical

from . import environ


@torch.no_grad()
def validation_run_ppo(
    env,
    model,
    episodes: int = 50,
    device="cpu",
    greedy: bool = True,
):
    """
    Evaluate a PPO policy on the given environment.

    IMPORTANT:
    Matches execution model in environ.State.step():
      - action chosen at time t
      - env advances to t+1
      - execution occurs at OPEN(t+1)

    Returns mean metrics across episodes.
    """
    # Make device robust (accept torch.device or str)
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # Always use unwrapped env for internal state access
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

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

        # Manual trade tracking (single-position assumption)
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

            # --- determine execution open(t+1) from the *current* state before step ---
            st = base_env._state
            next_idx = st._offset + 1

            exec_open = None
            if 0 <= next_idx < st._prices.open.shape[0]:
                exec_open = float(st._prices.open[next_idx])

            # Keep a copy to detect env-forced closes after env.step()
            prev_have_pos_env = bool(st.have_position)

            # --- Bookkeeping BEFORE step, because env executes at OPEN(t+1) ---
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

            # Step environment (could be wrapper; state access is via base_env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

            # --- detect forced close (terminal liquidation / reset_on_close etc.) ---
            now_have_pos_env = bool(base_env._state.have_position)

            if in_pos and prev_have_pos_env and (not now_have_pos_env) and act_enum != environ.Actions.Close:
                # env ended position without explicit Close in our bookkeeping
                exit_price = exec_open
                if exit_price is None:
                    exit_price = float(base_env._state._cur_close())

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

        # If episode ends while holding, close virtually for reporting
        if in_pos and entry_price and entry_price > 0:
            last_close = float(base_env._state._cur_close())
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

    return {k: float(np.mean(v)) for k, v in stats.items()}
