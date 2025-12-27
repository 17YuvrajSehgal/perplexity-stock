# train_ppo.py
"""
PPO training script for your finance_rl trading environment.

Key improvements in this version
--------------------------------
✅ Separate train vs validation env instances (no shared state)
✅ Optional chronological train/val split (no leakage): --train_ratio / --min_train / --min_val
✅ Training terminates by default (finite --max_rollouts, default=500)
✅ Optional early stopping based on validation episode_reward (patience + min_delta)
✅ Still saves BEST model by validation episode_reward: saves/ppo_<run>_best.pt
✅ Rollout reward heartbeat + full TensorBoard logs as before

Usage
-----
CPU:
python -u train_ppo.py -r ppo_aapl --data yf_data

GPU:
python -u train_ppo.py -r ppo_aapl --data yf_data --cuda

Recommended (finite + early-stop + split):
python -u train_ppo.py -r ppo_aapl_fixed --data yf_data --cuda \
  --train_ratio 0.8 --min_train 200 --min_val 200 \
  --max_rollouts 500 --early_stop --patience 20 --min_rollouts 50 --val_every_rollouts 10
"""
from __future__ import annotations

import os
import time
import argparse
from typing import Dict, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from finance_rl import environ, data_yf
from finance_rl.ppo_models import ActorCriticMLP, ActorCriticConv1D
from finance_rl.ppo_buffer import RolloutBuffer
from finance_rl.ppo_validation import validation_run_ppo


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    1 - Var[y_true - y_pred] / Var[y_true]
    Diagnostic: is the critic predicting returns well?
    """
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-12))


@torch.no_grad()
def policy_act(model, obs, device, greedy: bool = False):
    """
    Sample (or greedy-select) action from current policy.
    Returns: action(int), logprob(float), value(float)
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = model(obs_t)
    dist = Categorical(logits=logits)

    if greedy:
        action_t = torch.argmax(logits, dim=1)
    else:
        action_t = dist.sample()

    logprob_t = dist.log_prob(action_t)
    return int(action_t.item()), float(logprob_t.item()), float(value.item())


def build_env(
    prices,
    *,
    bars: int,
    volumes: bool,
    extra_features: bool,
    reward_on_close: bool,
    reward_mode: str,
    state_1d: bool,
) -> gym.Env:
    """
    Build a fresh StocksEnv instance.
    """
    return environ.StocksEnv(
        prices,
        bars_count=bars,
        volumes=volumes,
        extra_features=extra_features,
        reset_on_close=False,
        reward_on_close=reward_on_close,
        reward_mode=reward_mode,
        state_1d=state_1d,
    )


def maybe_split_prices(
    prices_all: Dict[str, data_yf.Prices],
    *,
    do_split: bool,
    train_ratio: float,
    min_train: int,
    min_val: int,
) -> Tuple[Dict[str, data_yf.Prices], Dict[str, data_yf.Prices]]:
    """
    If do_split is True, split each instrument chronologically using data_yf.split_many_by_ratio.
    Otherwise, return the same dict for both train and val (in-sample validation).
    """
    if not do_split:
        return prices_all, prices_all

    prices_train, prices_val = data_yf.split_many_by_ratio(
        prices_all,
        train_ratio=train_ratio,
        min_train=min_train,
        min_val=min_val,
    )
    return prices_train, prices_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", required=True, help="Run name for logs/checkpoints")
    parser.add_argument("--data", default="yf_data", help="Directory with yfinance CSVs")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_limit", type=int, default=1000, help="Max steps per episode (TimeLimit truncation)")

    # PPO hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout_steps", type=int, default=1024)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.02)

    # Environment options
    parser.add_argument("--bars", type=int, default=10)
    parser.add_argument("--volumes", action="store_true", help="Include volume features")
    parser.add_argument("--no_extra", action="store_true", help="Disable extra features (vol/time/atr)")
    parser.add_argument("--reward_mode", choices=["close_pnl", "step_logret"], default="close_pnl")
    parser.add_argument("--reward_on_close", action="store_true", help="Backward-compat flag (usually keep OFF for PPO)")

    # Model choices
    parser.add_argument("--state_1d", action="store_true", help="Use State1D (CNN) instead of flat state")

    # Logging + checkpoints
    parser.add_argument("--val_every_rollouts", type=int, default=10)
    parser.add_argument("--save_every_rollouts", type=int, default=10)

    # IMPORTANT: don't run forever by default
    parser.add_argument("--max_rollouts", type=int, default=500, help="Stop after N rollouts (default: 500)")

    # Optional early stopping on validation
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping based on val episode_reward")
    parser.add_argument("--patience", type=int, default=20, help="Validations without improvement before stopping")
    parser.add_argument("--min_rollouts", type=int, default=50, help="Do not early-stop before this many rollouts")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="Min improvement to reset patience")

    # NEW: Data split options (no leakage)
    parser.add_argument("--split", action="store_true", help="Chronological train/val split (recommended)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train fraction when --split is enabled")
    parser.add_argument("--min_train", type=int, default=200, help="Min train points per instrument when --split")
    parser.add_argument("--min_val", type=int, default=200, help="Min val points per instrument when --split")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs("runs", exist_ok=True)
    os.makedirs("saves", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # -------------------------
    # Load prices + (optional) split
    # -------------------------
    prices_all = data_yf.load_many_from_dir(args.data)

    prices_train, prices_val = maybe_split_prices(
        prices_all,
        do_split=args.split,
        train_ratio=args.train_ratio,
        min_train=args.min_train,
        min_val=args.min_val,
    )

    if args.split:
        print(f"[data] instruments: all={len(prices_all)} train={len(prices_train)} val={len(prices_val)}")
    else:
        print(f"[data] instruments: all={len(prices_all)} (no split; validation is in-sample)")

    extra_features = (not args.no_extra)

    # Separate env instances to avoid state coupling between train and validation
    env_train_base = build_env(
        prices_train,
        bars=args.bars,
        volumes=args.volumes,
        extra_features=extra_features,
        reward_on_close=args.reward_on_close,
        reward_mode=args.reward_mode,
        state_1d=args.state_1d,
    )
    env_train = gym.wrappers.TimeLimit(env_train_base, max_episode_steps=args.time_limit)

    env_val = build_env(
        prices_val,
        bars=args.bars,
        volumes=args.volumes,
        extra_features=extra_features,
        reward_on_close=args.reward_on_close,
        reward_mode=args.reward_mode,
        state_1d=args.state_1d,
    )

    obs_shape = env_train.observation_space.shape
    n_actions = env_train.action_space.n

    # -------------------------
    # Build model
    # -------------------------
    if args.state_1d:
        C, T = obs_shape  # (channels, time)
        model = ActorCriticConv1D(in_channels=C, n_actions=n_actions, bars_count=T).to(device)
    else:
        obs_dim = obs_shape[0]
        model = ActorCriticMLP(obs_dim=obs_dim, n_actions=n_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(comment=f"-ppo-{args.run}")

    # -------------------------
    # Training bookkeeping
    # -------------------------
    obs, info = env_train.reset(seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    global_step = 0
    rollout_idx = 0
    t0 = time.time()

    best_val_reward = -1e9  # save best by val episode_reward
    no_improve = 0          # for early stopping

    print(f"[PPO] device={device} obs_shape={obs_shape} actions={n_actions}")
    print(f"[PPO] reward_mode={args.reward_mode} volumes={args.volumes} extra_features={extra_features}")
    print(f"[PPO] split={args.split} max_rollouts={args.max_rollouts} early_stop={args.early_stop}")
    print(f"[PPO] logs: runs/  checkpoints: saves/")
    print("------------------------------------------------------------")

    # -------------------------
    # Main loop: rollout -> update
    # -------------------------
    while True:
        rollout_idx += 1
        # IMPORTANT: device should be torch.device, not str
        buf = RolloutBuffer(obs_shape=obs_shape, size=args.rollout_steps, device=device)

        # ===== Collect rollout =====
        for _ in range(args.rollout_steps):
            global_step += 1

            action, logprob, value = policy_act(model, obs, device=device, greedy=False)

            next_obs, reward, terminated, truncated, info = env_train.step(action)
            done = bool(terminated or truncated)

            buf.add(
                obs=obs,
                action=action,
                reward=float(reward),
                done=done,
                value=value,
                logprob=logprob,
            )

            # Episode stats
            episode_reward += float(reward)
            episode_steps += 1
            obs = next_obs

            if done:
                episode_count += 1
                writer.add_scalar("train/episode_reward", episode_reward, global_step)
                writer.add_scalar("train/episode_steps", episode_steps, global_step)

                obs, info = env_train.reset()
                episode_reward = 0.0
                episode_steps = 0

        # Rollout reward heartbeat
        roll_sum = float(buf.rewards.sum())
        roll_mean = float(buf.rewards.mean())
        writer.add_scalar("train/rollout_reward_sum", roll_sum, global_step)
        writer.add_scalar("train/rollout_reward_mean", roll_mean, global_step)

        # Bootstrap last value for GAE
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_v = model(obs_t)
            last_value = float(last_v.item())

        buf.compute_gae(last_value=last_value, gamma=args.gamma, lam=args.gae_lambda, normalize_adv=True)

        # ===== PPO Update =====
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clipfracs = []

        for _epoch in range(args.epochs):
            for batch in buf.get_batches(batch_size=args.minibatch, shuffle=True):
                logits, values = model(batch.obs)
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratio = torch.exp(new_logp - batch.old_logprobs)

                # Policy loss (clipped surrogate)
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * batch.advantages
                loss_pi = -torch.min(unclipped, clipped).mean()

                # Value loss
                loss_v = (batch.returns - values).pow(2).mean()

                # Total loss
                loss = loss_pi + args.value_coef * loss_v - args.entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = float((batch.old_logprobs - new_logp).mean().item())
                    clipfrac = float((torch.abs(ratio - 1.0) > args.clip_eps).float().mean().item())

                policy_losses.append(float(loss_pi.item()))
                value_losses.append(float(loss_v.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(approx_kl)
                clipfracs.append(clipfrac)

            # Early stop PPO epoch loop if KL too big
            if args.target_kl > 0 and (len(approx_kls) > 0) and (np.mean(approx_kls) > args.target_kl):
                break

        # ===== Logging =====
        fps = global_step / max(1e-9, (time.time() - t0))
        writer.add_scalar("ppo/policy_loss", float(np.mean(policy_losses) if policy_losses else 0.0), global_step)
        writer.add_scalar("ppo/value_loss", float(np.mean(value_losses) if value_losses else 0.0), global_step)
        writer.add_scalar("ppo/entropy", float(np.mean(entropies) if entropies else 0.0), global_step)
        writer.add_scalar("ppo/approx_kl", float(np.mean(approx_kls) if approx_kls else 0.0), global_step)
        writer.add_scalar("ppo/clipfrac", float(np.mean(clipfracs) if clipfracs else 0.0), global_step)
        writer.add_scalar("train/fps", float(fps), global_step)
        writer.add_scalar("train/episodes", float(episode_count), global_step)

        # Value function explained variance
        # Ensure numpy arrays
        ev = explained_variance(np.asarray(buf.values), np.asarray(buf.returns))
        writer.add_scalar("ppo/explained_variance", ev, global_step)

        # Console heartbeat every rollout
        print(
            f"[rollout {rollout_idx}] step={global_step} "
            f"roll_sum={roll_sum:.3f} roll_mean={roll_mean:.5f} "
            f"pi_loss={np.mean(policy_losses) if policy_losses else 0.0:.4f} "
            f"v_loss={np.mean(value_losses) if value_losses else 0.0:.4f} "
            f"ent={np.mean(entropies) if entropies else 0.0:.3f} "
            f"kl={np.mean(approx_kls) if approx_kls else 0.0:.4f} "
            f"clip={np.mean(clipfracs) if clipfracs else 0.0:.3f} "
            f"eps_done={episode_count} fps={fps:.1f}"
        )

        # ===== Validation + Best model saving + Early stop =====
        if args.val_every_rollouts > 0 and (rollout_idx % args.val_every_rollouts == 0):
            model.eval()
            val = validation_run_ppo(env_val, model, episodes=20, device=str(device), greedy=True)
            model.train()

            for k, v in val.items():
                writer.add_scalar("val/" + k, v, global_step)

            print(f"  [val] {val}")

            cur_val = float(val.get("episode_reward", -1e9))
            if cur_val > best_val_reward + args.min_delta:
                best_val_reward = cur_val
                no_improve = 0
                best_path = os.path.join("saves", f"ppo_{args.run}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  [best] new best val episode_reward={best_val_reward:.4f} -> {best_path}")
            else:
                no_improve += 1

            if args.early_stop and (rollout_idx >= args.min_rollouts) and (no_improve >= args.patience):
                print(f"[PPO] early stopping: no val improvement for {no_improve} validations.")
                break

        # ===== Save periodic checkpoint =====
        if args.save_every_rollouts > 0 and (rollout_idx % args.save_every_rollouts == 0):
            ckpt_path = os.path.join("saves", f"ppo_{args.run}_rollout{rollout_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [save] {ckpt_path}")

        # Hard stop condition
        if args.max_rollouts and rollout_idx >= args.max_rollouts:
            print("[PPO] reached max_rollouts, stopping.")
            break

    writer.close()


if __name__ == "__main__":
    main()
