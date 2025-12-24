import gymnasium as gym
from gymnasium.utils import seeding

import enum
import numpy as np

from . import data_yf as data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

    def encode(self):
        """
        Encode as relative features (much easier for NN to learn than raw prices).
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0

        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx
            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            # price features
            res[shift] = (h / c) - 1.0;
            shift += 1
            res[shift] = (l / c) - 1.0;
            shift += 1
            res[shift] = (c / pc) - 1.0;
            shift += 1

            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v
                if pv <= 0:
                    vol_feat = 0.0
                else:
                    vol_feat = (v / pv) - 1.0
                # clip extreme spikes
                res[shift] = np.clip(vol_feat, -5.0, 5.0)
                shift += 1

        res[shift] = float(self.have_position);
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price

        return res

    def _cur_close(self):
        # yfinance data is already absolute close price
        return float(self._prices.close[self._offset])

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        """
        Encode as relative features instead of raw prices/volume.
        Keeps the SAME shape: (4*bars_count + 2,) if volumes=True.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0

        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx

            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c
            if pc == 0.0:
                pc = c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            # Price features near 0:
            res[shift] = (h / c) - 1.0;
            shift += 1  # high relative to close
            res[shift] = (l / c) - 1.0;
            shift += 1  # low relative to close
            res[shift] = (c / pc) - 1.0;
            shift += 1  # close return

            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v
                if pv <= 0.0:
                    vol_feat = 0.0
                else:
                    vol_feat = (v / pv) - 1.0
                res[shift] = np.clip(vol_feat, -5.0, 5.0)  # clip spikes
                shift += 1

        res[shift] = float(self.have_position);
        shift += 1

        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price

        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        #assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count

        if self.random_ofs_on_reset:
            max_ofs = prices.high.shape[0] - bars * 10
            max_ofs = max(max_ofs, bars + 1)
            offset = int(self.np_random.integers(bars, max_ofs))
        else:
            offset = bars

        self._state.reset(prices, offset)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, info

    def step(self, action_idx):
        action = Actions(int(action_idx))
        reward, done = self._state.step(action)

        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}

        terminated = bool(done)  # treat done as terminal for now
        truncated = False  # set True if you add time-limit truncation
        return obs, float(reward), terminated, truncated, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        """
        Load yfinance CSV files from a directory and create the environment.
        """
        prices = data.load_many_from_dir(data_dir)
        return cls(prices, **kwargs)
