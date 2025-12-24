import gymnasium as gym
from gymnasium.utils import seeding

import enum
import numpy as np

import config
from . import data_yf as data

# Default number of past days ("bars") the agent can see
DEFAULT_BARS_COUNT = config.EnvConfig.DEFAULT_BARS_COUNT

# Transaction cost applied when buying or closing a position
DEFAULT_COMMISSION_PERC = config.EnvConfig.DEFAULT_COMMISSION_PERC


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

    Base features (already in your env):
      - For each of last N bars: relative (high/close - 1), (low/close - 1), (close/prev_close - 1),
        and optionally relative volume change.
      - have_position flag
      - unrealized_return (if holding)

    Optional extra features (recommended for PPO):
      - volatility estimate over recent returns (std)
      - time_in_position (normalized)
      - ATR-like range: mean((high-low)/close) over window
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

        # Backward-compat: your older logic used reward_on_close True/False
        self.reward_on_close = reward_on_close

        # PPO-friendly explicit reward mode
        self.reward_mode = reward_mode

        self.volumes = volumes
        self.extra_features = extra_features

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
        # Base:
        #   if volumes: 4 * bars_count
        #   else:       3 * bars_count
        # plus: have_position + unrealized_return
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

        # unrealized return (if holding)
        if not self.have_position or self.open_price == 0.0:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        shift += 1

        # optional extra features
        if self.extra_features:
            # --- volatility: std of close returns over window ---
            start = max(1, self._offset - (self.bars_count - 1))
            closes = self._prices.close[start:self._offset + 1].astype(np.float32)
            if closes.shape[0] >= 2:
                rets = closes[1:] / closes[:-1] - 1.0
                vol = float(np.std(rets))
            else:
                vol = 0.0
            res[shift] = np.clip(vol, 0.0, 1.0)
            shift += 1

            # --- time in position (normalized) ---
            tnorm = float(self.time_in_position) / float(self.bars_count * 10)
            res[shift] = np.clip(tnorm, 0.0, 1.0)
            shift += 1

            # --- ATR-like average range: mean((high-low)/close) ---
            start2 = max(0, self._offset - (self.bars_count - 1))
            highs = self._prices.high[start2:self._offset + 1].astype(np.float32)
            lows = self._prices.low[start2:self._offset + 1].astype(np.float32)
            closes2 = self._prices.close[start2:self._offset + 1].astype(np.float32)
            denom = np.where(closes2 == 0.0, 1.0, closes2)
            atr_like = float(np.mean((highs - lows) / denom))
            res[shift] = np.clip(atr_like, 0.0, 1.0)
            shift += 1

        return res

    def _cur_close(self):
        return float(self._prices.close[self._offset])

    def step(self, action: Actions):
        """
        Returns (reward, done)
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False

        close = self._cur_close()
        prev_close = close  # will be updated after offset++

        # ---- trade logic ----
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            self.time_in_position = 0

            # commission penalty
            reward -= self.commission_perc

        elif action == Actions.Close and self.have_position:
            # commission penalty
            reward -= self.commission_perc

            # realized return on close (classic)
            if self.reward_mode == "close_pnl" or self.reward_on_close:
                if self.open_price != 0.0:
                    reward += 100.0 * (close - self.open_price) / self.open_price

            self.have_position = False
            self.open_price = 0.0
            self.time_in_position = 0

            done |= self.reset_on_close

        # ---- advance time ----
        # Move to the next day
        self._offset += 1
        prev_close = close

        # If we've moved past the last valid index, end episode safely
        if self._offset >= self._prices.close.shape[0]:
            done = True
            return reward, done

        # Now it's safe to read the new close
        close = self._cur_close()

        # If next step would go out of bounds, mark done (so next call won't crash)
        done |= self._offset >= self._prices.close.shape[0] - 1

        # Optional: per-step reward while holding (only if using step-based reward)
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

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
        # channels: price-features + optional volume + 2 position features
        # We return channels x bars_count
        base_ch = 3 + (1 if self.volumes else 0)
        ch = base_ch + 2
        return (ch, self.bars_count)

    def encode(self):
        res = np.zeros(self.shape, dtype=np.float32)

        # fill time dimension left->right (oldest -> newest)
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

            # position features repeat across the time axis (simple trick)
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

    metadata = {'render.modes': ['human']}

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
            # keep State1D simpler by default (extra_features ignored)
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close,
                volumes=volumes,
                extra_features=False,
                reward_mode=reward_mode,
            )
        else:
            self._state = State(
                bars_count, commission, reset_on_close,
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count

        if self.random_ofs_on_reset:
            max_ofs = max(prices.high.shape[0] - bars * 10, bars + 1)
            offset = int(self.np_random.integers(bars, max_ofs))
        else:
            offset = bars

        self._state.reset(prices, offset)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": offset}
        return obs, info

    def step(self, action_idx):
        action = Actions(int(action_idx))
        # Save observation BEFORE stepping so we can safely return it on terminal
        obs_before = self._state.encode()

        reward, done = self._state.step(action)

        terminated = bool(done)
        truncated = False

        # If episode ended because we ran out of data, the new offset may be invalid.
        # In that case, return the last valid observation (obs_before).
        if terminated:
            obs = obs_before
        else:
            obs = self._state.encode()

        info = {"instrument": self._instrument, "offset": int(self._state._offset)}
        return obs, float(reward), terminated, truncated, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = data.load_many_from_dir(data_dir)
        return cls(prices, **kwargs)
