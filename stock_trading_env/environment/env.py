"""
Stock Trading Environment

A Gymnasium-compatible environment for reinforcement learning with stock trading.
"""

import enum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from stock_trading_env.data import Prices

# Default number of past days ("bars") the agent can see
DEFAULT_BARS_COUNT = 10

# Transaction cost applied when buying or closing a position
DEFAULT_COMMISSION_PERC = 0.001  # 0.1% per trade side (10 bps)

# Penalty per step while holding a position (discourages "hold forever")
DEFAULT_HOLD_PENALTY_PERC = 0.00002  # 0.002% per step (tune: 1e-5 to 1e-4)

# Optional cap on holding length (None disables)
DEFAULT_MAX_HOLD_STEPS = 250

# ===== Reason-to-close shaping defaults (OFF by default) =====
# Penalize holding unrealized losses (scaled by magnitude of negative unrealized return)
DEFAULT_UNREALIZED_LOSS_PENALTY_PER_STEP = 0.0
# Bonus for closing profitable trades (scaled by realized return)
DEFAULT_CLOSE_PROFIT_BONUS = 0.0
# Penalize volatility while holding (scaled by recent realized vol)
DEFAULT_UNREALIZED_VOL_PENALTY_PER_STEP = 0.0
# Lookback window for vol penalty
DEFAULT_VOL_LOOKBACK = 20


class Actions(enum.Enum):
    """
    Discrete actions for single-position trading.
    """
    Skip = 0
    Buy = 1
    Close = 2


class State:
    """
    State encoder for the trading environment.

    IMPORTANT (leakage fix):
    - Observation at time t is built from current offset t
    - Action is executed at OPEN(t+1) (next bar), not at CLOSE(t)
    """

    def __init__(
        self,
        bars_count: int,
        commission_perc: float,
        reset_on_close: bool,
        reward_on_close: bool = True,
        volumes: bool = True,
        extra_features: bool = True,
        reward_mode: str = "close_pnl",
        hold_penalty_per_step: float = DEFAULT_HOLD_PENALTY_PERC,
        max_hold_steps: Optional[int] = DEFAULT_MAX_HOLD_STEPS,
        # ===== Reason-to-close shaping (optional) =====
        unrealized_loss_penalty_per_step: float = DEFAULT_UNREALIZED_LOSS_PENALTY_PER_STEP,
        close_profit_bonus: float = DEFAULT_CLOSE_PROFIT_BONUS,
        unrealized_vol_penalty_per_step: float = DEFAULT_UNREALIZED_VOL_PENALTY_PER_STEP,
        vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    ):
        assert isinstance(bars_count, int) and bars_count > 0
        assert isinstance(commission_perc, float) and commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        assert isinstance(volumes, bool)
        assert isinstance(extra_features, bool)
        assert reward_mode in ("close_pnl", "step_logret")

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.hold_penalty_per_step = float(hold_penalty_per_step)
        self.max_hold_steps = max_hold_steps if (max_hold_steps is None) else int(max_hold_steps)
        self.reset_on_close = reset_on_close

        # Backward-compat
        self.reward_on_close = reward_on_close

        # PPO-friendly explicit reward mode
        self.reward_mode = reward_mode

        self.volumes = volumes
        self.extra_features = extra_features

        # Reason-to-close params
        self.unrealized_loss_penalty_per_step = float(unrealized_loss_penalty_per_step)
        self.close_profit_bonus = float(close_profit_bonus)
        self.unrealized_vol_penalty_per_step = float(unrealized_vol_penalty_per_step)
        self.vol_lookback = int(vol_lookback)

        # Will be initialized in reset()
        self._prices: Optional[Prices] = None
        self._offset: int = 0

        self.have_position: bool = False
        self.open_price: float = 0.0
        self.time_in_position: int = 0

    def reset(self, prices: Prices, offset: int):
        assert isinstance(prices, Prices)
        assert offset >= self.bars_count - 1

        self.have_position = False
        self.open_price = 0.0
        self.time_in_position = 0

        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        base = (4 * self.bars_count) if self.volumes else (3 * self.bars_count)
        extras = 3 if self.extra_features else 0  # vol, time_in_pos, atr_like
        return (base + 2 + extras,)

    def encode(self):
        """
        Encode observation as RELATIVE features to improve learning stability.
        """
        res = np.zeros(self.shape, dtype=np.float32)
        shift = 0

        # Window features
        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx

            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c
            if pc == 0.0:
                pc = c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            # price features
            res[shift] = (h / c) - 1.0
            shift += 1
            res[shift] = (l / c) - 1.0
            shift += 1
            res[shift] = (c / pc) - 1.0
            shift += 1

            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v
                if pv <= 0.0:
                    vol_feat = 0.0
                else:
                    vol_feat = (v / pv) - 1.0
                res[shift] = np.clip(vol_feat, -5.0, 5.0)
                shift += 1

        # position flag
        res[shift] = float(self.have_position)
        shift += 1

        # unrealized return (mark-to-market using current close)
        if (not self.have_position) or (self.open_price == 0.0):
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        shift += 1

        # optional extra features
        if self.extra_features:
            # volatility: std of close returns over window
            start = max(1, self._offset - (self.bars_count - 1))
            closes = self._prices.close[start:self._offset + 1].astype(np.float32)
            if closes.shape[0] >= 2:
                rets = closes[1:] / closes[:-1] - 1.0
                vol = float(np.std(rets))
            else:
                vol = 0.0
            res[shift] = np.clip(vol, 0.0, 1.0)
            shift += 1

            # time in position (normalized)
            tnorm = float(self.time_in_position) / float(self.bars_count * 10)
            res[shift] = np.clip(tnorm, 0.0, 1.0)
            shift += 1

            # ATR-like average range
            start2 = max(0, self._offset - (self.bars_count - 1))
            highs = self._prices.high[start2:self._offset + 1].astype(np.float32)
            lows = self._prices.low[start2:self._offset + 1].astype(np.float32)
            closes2 = self._prices.close[start2:self._offset + 1].astype(np.float32)
            denom = np.where(closes2 == 0.0, 1.0, closes2)
            atr_like = float(np.mean((highs - lows) / denom))
            res[shift] = np.clip(atr_like, 0.0, 1.0)
            shift += 1

        return res

    def _cur_close(self) -> float:
        return float(self._prices.close[self._offset])

    def step(self, action) -> tuple[float, bool]:
        """
        Advance one bar and apply action execution at OPEN(t+1).
        Returns: (reward, done)

        Updated logic:
          - Hold penalty applies ONLY on true Hold steps (Skip while in position),
            NOT on Close steps and NOT on forced-close steps.
          - Forced close triggers when you've already held max_hold_steps steps.
          - Optional shaping:
              * penalize holding unrealized losses
              * bonus for closing profitable trades
              * penalize volatility while holding
        """
        assert self._prices is not None, "Call reset() first"

        # Accept either Actions enum or int
        try:
            a = int(action.value)
        except AttributeError:
            a = int(action)

        # Move to t+1 (execution bar)
        self._offset += 1

        # If we stepped beyond available data, terminate cleanly
        if self._offset >= self._prices.close.shape[0] - 1:
            return 0.0, True

        exec_open = float(self._prices.open[self._offset])
        exec_close = float(self._prices.close[self._offset])

        reward = 0.0
        fee = float(self.commission_perc)
        hold_fee = float(self.hold_penalty_per_step)

        # --- Shaping helpers (computed once per step) ---
        unrealized_ret = 0.0
        if self.have_position and self.open_price > 0.0:
            # mark-to-market at exec_open (consistent with execution-at-open)
            unrealized_ret = (exec_open - float(self.open_price)) / float(self.open_price)

        vol = 0.0
        if self.unrealized_vol_penalty_per_step > 0.0:
            end = self._offset
            start = max(1, end - int(self.vol_lookback))
            window = self._prices.close[start:end + 1].astype(np.float32)
            if window.shape[0] >= 3:
                rets = window[1:] / window[:-1] - 1.0
                vol = float(np.std(rets))

        # ---- Optional forced close if holding too long ----
        # Convention: time_in_position counts number of HOLD steps taken since entry.
        # If it already reached max_hold_steps, you must close now.
        forced_close = False
        if (
            self.have_position
            and (self.max_hold_steps is not None)
            and (self.time_in_position >= self.max_hold_steps)
        ):
            a = Actions.Close.value  # override action to close
            forced_close = True

        # 0 = Skip/Hold, 1 = Buy/Open long, 2 = Close
        if a == Actions.Buy.value:
            if not self.have_position:
                self.have_position = True
                self.open_price = exec_open
                self.time_in_position = 0

                # entry commission
                if fee > 0.0:
                    reward -= 100.0 * fee

        elif a == Actions.Close.value:
            if self.have_position:
                # realized return at exec_open (next bar open)
                if self.open_price > 0.0:
                    realized_return = (exec_open - float(self.open_price)) / float(self.open_price)
                else:
                    realized_return = 0.0

                self.have_position = False
                self.open_price = 0.0
                self.time_in_position = 0

                if self.reward_mode == "close_pnl":
                    reward += 100.0 * realized_return

                # bonus for closing profitable trades (encourages intentional exits)
                if self.close_profit_bonus > 0.0 and realized_return > 0.0:
                    reward += 100.0 * self.close_profit_bonus * realized_return

                # exit commission (also applies to forced close)
                if fee > 0.0:
                    reward -= 100.0 * fee

        else:
            # Hold (Skip)
            if self.have_position:
                self.time_in_position += 1

                # apply hold penalty ONLY for holding (not for close/buy)
                if hold_fee > 0.0:
                    reward -= 100.0 * hold_fee

                # penalize holding unrealized losses (scaled by loss magnitude)
                if self.unrealized_loss_penalty_per_step > 0.0 and unrealized_ret < 0.0:
                    reward -= 100.0 * self.unrealized_loss_penalty_per_step * (-unrealized_ret)

                # penalize volatility while holding (optional)
                if self.unrealized_vol_penalty_per_step > 0.0 and vol > 0.0:
                    reward -= 100.0 * self.unrealized_vol_penalty_per_step * vol

        # Terminal liquidation at end of series
        done = (self._offset >= self._prices.close.shape[0] - 2)
        if done and self.have_position:
            if self.open_price > 0.0:
                liq_ret = (exec_close - float(self.open_price)) / float(self.open_price)
            else:
                liq_ret = 0.0

            self.have_position = False
            self.open_price = 0.0
            self.time_in_position = 0

            if self.reward_mode == "close_pnl":
                reward += 100.0 * liq_ret

            # exit commission on liquidation
            if fee > 0.0:
                reward -= 100.0 * fee

        return float(reward), bool(done)


class State1D(State):
    """
    CNN-friendly version: (channels, time).
    Channels include:
      - high_rel, low_rel, ret, (vol_rel), have_position, unrealized_return
    Note: extra_features are NOT included here by default to keep shape simple.
    """

    @property
    def shape(self):
        base_ch = 3 + (1 if self.volumes else 0)
        ch = base_ch + 2
        return (ch, self.bars_count)

    def encode(self):
        res = np.zeros(self.shape, dtype=np.float32)

        t = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx

            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c
            if pc == 0.0:
                pc = c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            ch = 0
            res[ch, t] = (h / c) - 1.0
            ch += 1
            res[ch, t] = (l / c) - 1.0
            ch += 1
            res[ch, t] = (c / pc) - 1.0
            ch += 1

            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v
                vol_feat = (v / pv) - 1.0 if pv > 0 else 0.0
                res[ch, t] = np.clip(vol_feat, -5.0, 5.0)
                ch += 1

            # position features repeat across the time axis
            res[ch, t] = float(self.have_position)
            ch += 1
            if self.have_position and self.open_price != 0.0:
                res[ch, t] = (self._cur_close() - self.open_price) / self.open_price
            else:
                res[ch, t] = 0.0

            t += 1

        return res


class StocksEnv(gym.Env):
    """
    Gymnasium-compatible trading environment wrapper.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices,
        bars_count=DEFAULT_BARS_COUNT,
        commission=DEFAULT_COMMISSION_PERC,
        reset_on_close=True,
        state_1d=False,
        random_ofs_on_reset=True,
        reward_on_close=False,
        volumes=True,
        extra_features=True,
        reward_mode="close_pnl",
        hold_penalty_per_step=DEFAULT_HOLD_PENALTY_PERC,
        max_hold_steps=DEFAULT_MAX_HOLD_STEPS,
        # ===== Reason-to-close shaping kwargs (optional) =====
        unrealized_loss_penalty_per_step=DEFAULT_UNREALIZED_LOSS_PENALTY_PER_STEP,
        close_profit_bonus=DEFAULT_CLOSE_PROFIT_BONUS,
        unrealized_vol_penalty_per_step=DEFAULT_UNREALIZED_VOL_PENALTY_PER_STEP,
        vol_lookback=DEFAULT_VOL_LOOKBACK,
    ):

        self._prices = prices

        if state_1d:
            self._state = State1D(
                bars_count,
                float(commission),
                reset_on_close,
                reward_on_close=reward_on_close,
                volumes=volumes,
                extra_features=False,
                reward_mode=reward_mode,
                hold_penalty_per_step=float(hold_penalty_per_step),
                max_hold_steps=max_hold_steps,
                unrealized_loss_penalty_per_step=float(unrealized_loss_penalty_per_step),
                close_profit_bonus=float(close_profit_bonus),
                unrealized_vol_penalty_per_step=float(unrealized_vol_penalty_per_step),
                vol_lookback=int(vol_lookback),
            )
        else:
            self._state = State(
                bars_count,
                float(commission),
                reset_on_close,
                reward_on_close=reward_on_close,
                volumes=volumes,
                extra_features=extra_features,
                reward_mode=reward_mode,
                hold_penalty_per_step=float(hold_penalty_per_step),
                max_hold_steps=max_hold_steps,
                unrealized_loss_penalty_per_step=float(unrealized_loss_penalty_per_step),
                close_profit_bonus=float(close_profit_bonus),
                unrealized_vol_penalty_per_step=float(unrealized_vol_penalty_per_step),
                vol_lookback=int(vol_lookback),
            )

        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32
        )

        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

        self._instrument = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count

        if self.random_ofs_on_reset:
            max_ofs = max(prices.high.shape[0] - bars * 10, bars + 1)
            # leave room for t+1 execution
            max_ofs = min(max_ofs, prices.high.shape[0] - 2)
            offset = int(self.np_random.integers(bars, max_ofs))
        else:
            offset = bars

        self._state.reset(prices, offset)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": offset}
        return obs, info

    def step(self, action_idx):
        """
        Gymnasium step: accepts an int action, converts to Actions enum, calls State.step()
        """
        action = Actions(int(action_idx))

        # Save observation BEFORE stepping so we can safely return it on terminal
        obs_before = self._state.encode()

        reward, done = self._state.step(action)

        terminated = bool(done)
        truncated = False

        # On terminal, the internal offset might be at boundary; return last valid obs
        obs = obs_before if terminated else self._state.encode()
        info = {"instrument": self._instrument, "offset": int(self._state._offset)}
        return obs, float(reward), terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        """Create environment by loading data from a directory."""
        from stock_trading_env.data import load_many_from_dir
        prices = load_many_from_dir(data_dir)
        return cls(prices, **kwargs)
