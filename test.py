# eval_trading_month.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from finance_rl import environ
from finance_rl.ppo_models import ActorCriticMLP, ActorCriticConv1D


# ---------- Data loading with dates ----------
@dataclass
class PricesWithDates:
    dates: np.ndarray  # dtype datetime64[ns]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def _pick_col(df: pd.DataFrame, key: str) -> str:
    key_l = key.lower()
    cols = list(df.columns)

    for c in cols:
        if str(c).strip().lower() == key_l:
            return c

    hits = [c for c in cols if key_l in str(c).lower()]
    if not hits:
        raise KeyError(f"Could not find column containing '{key}'. Columns={cols[:30]}")

    if key_l == "close":
        for c in hits:
            if "adj" not in str(c).lower():
                return c
    return hits[0]


def load_yf_csv_with_dates(csv_path: str | Path, fill_volume: float = 0.0) -> PricesWithDates:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # find date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV has no Date column. Need Date to select 'this month'.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    open_c = _pick_col(df, "open")
    high_c = _pick_col(df, "high")
    low_c  = _pick_col(df, "low")
    close_c = _pick_col(df, "close")

    # volume might be missing
    try:
        vol_c = _pick_col(df, "volume")
    except KeyError:
        vol_c = None

    def as_float(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

    o = as_float(open_c)
    h = as_float(high_c)
    l = as_float(low_c)
    c = as_float(close_c)
    if vol_c is None:
        v = np.full_like(c, fill_value=float(fill_volume), dtype=np.float32)
    else:
        v = pd.to_numeric(df[vol_c], errors="coerce").fillna(fill_volume).to_numpy(dtype=np.float32)

    dates = df[date_col].to_numpy(dtype="datetime64[ns]")

    mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    dates, o, h, l, c, v = dates[mask], o[mask], h[mask], l[mask], c[mask], v[mask]

    if len(c) < 5:
        raise ValueError(f"Not enough rows after cleaning: {csv_path}")

    return PricesWithDates(dates=dates, open=o, high=h, low=l, close=c, volume=v)


def slice_month(p: PricesWithDates, start: Optional[str], end: Optional[str], fallback_last_n: int = 22) -> PricesWithDates:
    dates = pd.to_datetime(p.dates)

    if start is None and end is None:
        # default: last N trading days
        idx0 = max(0, len(dates) - fallback_last_n)
        sel = np.arange(idx0, len(dates))
    else:
        start_dt = pd.to_datetime(start) if start else dates.min()
        end_dt = pd.to_datetime(end) if end else dates.max()
        sel = np.where((dates >= start_dt) & (dates <= end_dt))[0]
        if sel.size == 0:
            raise ValueError(f"No rows in date range [{start_dt} .. {end_dt}]")

    return PricesWithDates(
        dates=p.dates[sel],
        open=p.open[sel],
        high=p.high[sel],
        low=p.low[sel],
        close=p.close[sel],
        volume=p.volume[sel],
    )


# ---------- Portfolio simulator ----------
@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    pnl: float
    ret: float


@dataclass
class BacktestResult:
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_daily: float
    num_trades: int
    win_rate: float
    equity_curve: np.ndarray
    trades: List[Trade]


def compute_max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())  # negative


def sharpe_daily(returns: np.ndarray, eps: float = 1e-12) -> float:
    # rough daily sharpe (no risk-free), annualization ~ sqrt(252)
    mu = float(np.mean(returns))
    sd = float(np.std(returns))
    if sd < eps:
        return 0.0
    return float((mu / sd) * np.sqrt(252.0))


@torch.no_grad()
def greedy_action(model, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(obs_t)
    return int(torch.argmax(logits, dim=1).item())


def run_backtest_month(
    *,
    ckpt_path: str,
    csv_path: str,
    instrument_name: str,
    bars: int,
    volumes: bool,
    extra_features: bool,
    state_1d: bool,
    initial_cash: float,
    commission_perc: float,
    hold_penalty_per_step: float,
    max_hold_steps: Optional[int],
    start: Optional[str],
    end: Optional[str],
    device: torch.device,
    allow_fractional: bool = True,
) -> BacktestResult:

    raw = load_yf_csv_with_dates(csv_path)
    month = slice_month(raw, start=start, end=end, fallback_last_n=22)

    # Need enough history for bars + execution (t+1)
    if len(month.close) < (bars + 3):
        raise ValueError(
            f"Not enough rows in the selected period for bars={bars}. "
            f"Need at least ~{bars+3}, got {len(month.close)}. "
            f"Expand date range or reduce --bars."
        )

    prices_obj = environ.data.Prices(
        open=month.open,
        high=month.high,
        low=month.low,
        close=month.close,
        volume=month.volume,
    )
    prices_dict = {instrument_name: prices_obj}

    env = environ.StocksEnv(
        prices_dict,
        bars_count=bars,
        commission=commission_perc,
        reset_on_close=False,
        random_ofs_on_reset=False,     # start at bars (chronological)
        reward_on_close=False,
        volumes=volumes,
        extra_features=extra_features,
        reward_mode="close_pnl",
        hold_penalty_per_step=hold_penalty_per_step,
        max_hold_steps=max_hold_steps,
        state_1d=state_1d,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10**9)

    obs, info = env.reset()

    # build model
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    if state_1d:
        # State1D: (channels, time)
        C, T = obs_shape
        model = ActorCriticConv1D(in_channels=C, n_actions=n_actions, bars_count=T)
    else:
        # Flat state: (obs_dim,)
        obs_dim = obs_shape[0]
        model = ActorCriticMLP(obs_dim=obs_dim, n_actions=n_actions)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # Portfolio state
    cash = float(initial_cash)
    shares = 0.0
    in_pos = False

    entry_price = 0.0
    entry_date = ""
    trades: List[Trade] = []

    equity_curve: List[float] = []

    # mark-to-market using CLOSE of current offset
    def mark_to_market() -> float:
        prices = env.unwrapped._state._prices
        ofs = env.unwrapped._state._offset
        close_px = float(prices.close[ofs])
        return cash + shares * close_px

    equity_curve.append(mark_to_market())

    done = False
    while not done:
        prev_have = bool(env.unwrapped._state.have_position)
        prev_ofs = int(env.unwrapped._state._offset)

        action = greedy_action(model, obs, device=device)
        obs, reward, terminated, truncated, info = env.step(action)

        # After step, execution bar is current offset
        prices = env.unwrapped._state._prices
        ofs = int(env.unwrapped._state._offset)
        exec_open = float(prices.open[ofs])
        exec_close = float(prices.close[ofs])

        now_have = bool(env.unwrapped._state.have_position)

        # Transition-based execution (matches env’s single-position semantics)
        if (not prev_have) and now_have:
            # BUY executed at exec_open
            px = exec_open
            fee = commission_perc
            if allow_fractional:
                # all-in
                shares = (cash * (1.0 - fee)) / max(px, 1e-12)
                cash = 0.0
            else:
                # integer shares
                max_sh = int((cash * (1.0 - fee)) // max(px, 1e-12))
                cost = max_sh * px
                cash = cash - cost - (cash * fee)  # approx fee on notional
                shares = float(max_sh)

            in_pos = True
            entry_price = px
            entry_date = str(pd.to_datetime(month.dates[ofs]).date())

        elif prev_have and (not now_have):
            # CLOSED either by action Close, forced close, or terminal liquidation
            # If terminal liquidation, env uses exec_close; otherwise exec_open.
            px = exec_open
            if bool(terminated) and (action != environ.Actions.Close.value):
                # Likely liquidation at end-of-series
                px = exec_close

            fee = commission_perc
            proceeds = shares * px
            cash = proceeds * (1.0 - fee)
            shares = 0.0

            exit_date = str(pd.to_datetime(month.dates[ofs]).date())
            pnl = cash - initial_cash if len(trades) == 0 else cash - (initial_cash if not trades else 0)  # not used directly

            # compute per-trade PnL/return from entry
            trade_pnl = (px - entry_price) * (proceeds / max(px, 1e-12))  # approx
            trade_ret = (px - entry_price) / max(entry_price, 1e-12)

            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=float(entry_price),
                    exit_date=exit_date,
                    exit_price=float(px),
                    pnl=float(trade_pnl),
                    ret=float(trade_ret),
                )
            )

            in_pos = False
            entry_price = 0.0
            entry_date = ""

        equity_curve.append(mark_to_market())
        done = bool(terminated or truncated)

    equity = np.asarray(equity_curve, dtype=np.float64)
    total_ret = float(equity[-1] / initial_cash - 1.0)

    # daily returns from equity curve (one step per bar)
    rets = equity[1:] / np.maximum(equity[:-1], 1e-12) - 1.0
    mdd = compute_max_drawdown(equity)
    sh = sharpe_daily(rets)

    if trades:
        win_rate = float(np.mean([1.0 if t.ret > 0 else 0.0 for t in trades]))
    else:
        win_rate = 0.0

    return BacktestResult(
        final_equity=float(equity[-1]),
        total_return=total_ret,
        max_drawdown=float(mdd),
        sharpe_daily=float(sh),
        num_trades=len(trades),
        win_rate=win_rate,
        equity_curve=equity,
        trades=trades,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Backtest a trained PPO model on stock data. "
        "By default, tests on the last 22 trading days (approx. one month) if no dates specified."
    )
    ap.add_argument("--ckpt", required=True, help="Path to saved .pt checkpoint (state_dict)")
    ap.add_argument("--csv", required=True, help="Path to yfinance CSV (e.g., yf_data/AAPL_1d.csv)")
    ap.add_argument("--name", default="AAPL_1d", help="Instrument name key (any string)")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional, defaults to last 22 trading days if not set)")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional, defaults to last trading day if not set)")
    ap.add_argument("--initial_cash", type=float, default=10_000.0, help="Starting capital in dollars (default: $10,000)")

    # Must match training settings
    ap.add_argument("--bars", type=int, default=10)
    ap.add_argument("--volumes", action="store_true")
    ap.add_argument("--extra_features", action="store_true", help="Set if you trained with extra features ON")
    ap.add_argument("--state_1d", action="store_true", help="Set if you trained with --state_1d")

    # Trading sim params (should match env defaults unless you changed them)
    ap.add_argument("--commission", type=float, default=0.001)
    ap.add_argument("--hold_penalty", type=float, default=0.00002)
    ap.add_argument("--max_hold_steps", type=int, default=250)

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--no_fractional", action="store_true", help="Use integer shares only")

    args = ap.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load data to get date range info for display
    raw_data = load_yf_csv_with_dates(args.csv)
    month_data = slice_month(raw_data, start=args.start, end=args.end, fallback_last_n=22)
    date_start = pd.to_datetime(month_data.dates[0]).date()
    date_end = pd.to_datetime(month_data.dates[-1]).date()

    res = run_backtest_month(
        ckpt_path=args.ckpt,
        csv_path=args.csv,
        instrument_name=args.name,
        bars=args.bars,
        volumes=args.volumes,
        extra_features=args.extra_features,
        state_1d=args.state_1d,
        initial_cash=args.initial_cash,
        commission_perc=args.commission,
        hold_penalty_per_step=args.hold_penalty,
        max_hold_steps=args.max_hold_steps,
        start=args.start,
        end=args.end,
        device=device,
        allow_fractional=(not args.no_fractional),
    )

    print("=" * 60)
    print("PPO Paper-Trading Backtest Results")
    print("=" * 60)
    print(f"Checkpoint:       {args.ckpt}")
    print(f"Data file:        {args.csv}")
    print(f"Date range:       {date_start} to {date_end} ({len(month_data.dates)} trading days)")
    print(f"Initial capital:  ${args.initial_cash:,.2f}")
    print("-" * 60)
    print(f"Final equity:     ${res.final_equity:,.2f}")
    print(f"Total return:     {res.total_return*100:+.2f}%")
    print(f"Profit/Loss:      ${res.final_equity - args.initial_cash:+,.2f}")
    print(f"Max drawdown:     {res.max_drawdown*100:.2f}%")
    print(f"Sharpe (daily):   {res.sharpe_daily:.3f}")
    print(f"Total trades:     {res.num_trades}")
    print(f"Win rate:         {res.win_rate*100:.1f}%")
    if res.num_trades > 0:
        avg_return = np.mean([t.ret for t in res.trades]) * 100
        print(f"Avg trade return: {avg_return:+.2f}%")
    print("-" * 60)
    if res.trades:
        print("Trade Log:")
        for i, t in enumerate(res.trades, 1):
            print(
                f"  {i}. {t.entry_date} BUY @${t.entry_price:.2f} -> "
                f"{t.exit_date} SELL @${t.exit_price:.2f} | "
                f"Return: {t.ret*100:+.2f}% | PnL: ${t.pnl:+,.2f}"
            )
    else:
        print("No trades executed in this period.")
    print("=" * 60)


if __name__ == "__main__":
    main()
