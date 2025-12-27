import gymnasium as gym
from gymnasium.utils import seeding

import enum
import numpy as np

from . import data_yf as data

# Default number of past days ("bars") the agent can see
DEFAULT_BARS_COUNT = 10

# Transaction cost applied when buying or closing a position
DEFAULT_COMMISSION_PERC = 0.1


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
        reward_mode: str = "close_pnl",  # "close_pnl" or "step_logret"
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
        self.reset_on_close = reset_on_close

        # Backward-compat
        self.reward_on_close = reward_on_close

        # PPO-friendly explicit reward mode
        self.reward_mode = reward_mode

        self.volumes = volumes
        self.extra_features = extra_features

        # Will be initialized in reset()
        self._prices: data.Prices | None = None
        self._offset: int = 0

        self.have_position: bool = False
        self.open_price: float = 0.0
        self.time_in_position: int = 0

    def reset(self, prices: data.Prices, offset: int):
        assert isinstance(prices, data.Prices)
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

    def step(self, action: Actions):
        """
        Returns (reward, done)

        Execution model:
        - You decide action based on obs at time t
        - Environment advances to t+1
        - Execution occurs at OPEN(t+1)
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False

        # Prices at time t (bar used to build observation)
        close_t = self._cur_close()

        # ---- advance time first: move to t+1 ----
        self._offset += 1

        # Commission in SAME SCALE as reward:
        # reward uses 100 * return (percent-points), so commission_perc is in percent-points too.
        # Example: commission_perc=0.1 means 0.1% cost -> subtract 0.1
        comm = float(self.commission_perc)

        # End-of-data safety (if we ran past the end, liquidate using last close)
        if self._offset >= self._prices.close.shape[0]:
            if self.have_position and self.open_price != 0.0:
                exit_price = float(self._prices.close[-1])
                reward += 100.0 * (exit_price - self.open_price) / self.open_price
                reward -= comm
                self.have_position = False
                self.open_price = 0.0
                self.time_in_position = 0
            return reward, True

        # Prices at time t+1 (used for execution)
        open_t1 = float(self._prices.open[self._offset])
        close_t1 = float(self._prices.close[self._offset])

        # Optional: update holding time if already in position
        if self.have_position:
            self.time_in_position += 1

        # ---- execute action at open(t+1) ----
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = open_t1
            self.time_in_position = 0
            reward -= comm

        elif action == Actions.Close and self.have_position:
            reward -= comm

            # realized PnL at open(t+1)
            if (self.reward_mode == "close_pnl") or self.reward_on_close:
                if self.open_price != 0.0:
                    reward += 100.0 * (open_t1 - self.open_price) / self.open_price

            self.have_position = False
            self.open_price = 0.0
            self.time_in_position = 0
            done |= self.reset_on_close

        # Optional per-step reward while holding (step-based mode)
        if self.have_position and (not self.reward_on_close) and (self.reward_mode != "close_pnl"):
            if close_t != 0.0:
                reward += 100.0 * (close_t1 - close_t) / close_t

        # ---- termination checks ----
        done |= self._offset >= self._prices.close.shape[0] - 1
        if getattr(self, "time_limit", None) is not None:
            done |= self._offset >= int(self.time_limit)

        # ---- BUG 2 FIX: terminal liquidation (realize PnL if episode ends while holding) ----
        if done and self.have_position and self.open_price != 0.0:
            # Close using the same execution convention (OPEN(t+1) for the final step we are at)
            exit_price = open_t1
            reward += 100.0 * (exit_price - self.open_price) / self.open_price
            reward -= comm
            self.have_position = False
            self.open_price = 0.0
            self.time_in_position = 0

        return reward, done


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
        reward_mode="close_pnl",  # "close_pnl" or "step_logret"
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
        prices = data.load_many_from_dir(data_dir)
        return cls(prices, **kwargs)
