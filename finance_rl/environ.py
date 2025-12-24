import gymnasium as gym
from gymnasium.utils import seeding

import enum
import numpy as np

import config
# Import our custom Yahoo Finance data loader
from . import data_yf as data


# Default number of past days ("bars") the agent can see
DEFAULT_BARS_COUNT = config.EnvConfig.DEFAULT_BARS_COUNT

# Transaction cost applied when buying or closing a position
DEFAULT_COMMISSION_PERC = config.EnvConfig.DEFAULT_COMMISSION_PERC


# ---------------------------------------------------------
# Action space definition
# ---------------------------------------------------------
class Actions(enum.Enum):
    """
    All possible actions the trading agent can take.
    This is a discrete action space, suitable for DQN.
    """
    Skip = 0     # Do nothing
    Buy = 1      # Open a position (buy the stock)
    Close = 2    # Close the currently open position


# ---------------------------------------------------------
# State representation (what the agent observes)
# ---------------------------------------------------------
class State:
    """
    Represents the environment state from the agent's perspective.

    The state includes:
    - Recent market history (last N bars)
    - Whether the agent currently holds a position
    - Current profit/loss if holding a position
    """

    def __init__(self, bars_count, commission_perc, reset_on_close,
                 reward_on_close=True, volumes=True):

        # Basic sanity checks
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)

        # Configuration parameters
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        """
        Reset the state at the beginning of an episode.

        prices : Prices object containing OHLCV arrays
        offset : starting index in the price history
        """
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count - 1

        self.have_position = False      # Whether we currently hold the stock
        self.open_price = 0.0           # Price at which position was opened
        self._prices = prices           # Price history
        self._offset = offset           # Current time index

    @property
    def shape(self):
        """
        Shape of the observation vector returned to the agent.

        If volumes=True:
            [high, low, close, volume] * bars + position_flag + profit
        """
        if self.volumes:
            return (4 * self.bars_count + 2,)
        else:
            return (3 * self.bars_count + 2,)

    def encode(self):
        """
        Convert the current state into a numerical observation.

        IMPORTANT:
        We use *relative features* (ratios / returns), not raw prices.
        This makes learning easier and more stable for neural networks.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0

        # Loop over the last `bars_count` days
        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx

            # Current and previous close prices
            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            # Relative price features
            res[shift] = (h / c) - 1.0     # How high price was relative to close
            shift += 1
            res[shift] = (l / c) - 1.0     # How low price was relative to close
            shift += 1
            res[shift] = (c / pc) - 1.0    # Daily return
            shift += 1

            # Volume feature (optional)
            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v

                if pv <= 0:
                    vol_feat = 0.0
                else:
                    vol_feat = (v / pv) - 1.0

                # Clip extreme volume spikes
                res[shift] = np.clip(vol_feat, -5.0, 5.0)
                shift += 1

        # Whether we currently hold a position
        res[shift] = float(self.have_position)
        shift += 1

        # Profit/loss of current position (if any)
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price

        return res

    def _cur_close(self):
        """
        Return the current closing price.
        """
        return float(self._prices.close[self._offset])

    def step(self, action):
        """
        Execute one environment step based on the chosen action.

        Returns:
            reward : numeric reward signal
            done   : whether the episode has ended
        """
        assert isinstance(action, Actions)

        reward = 0.0
        done = False
        close = self._cur_close()

        # Open a new position
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc

        # Close existing position
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close

            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price

            self.have_position = False
            self.open_price = 0.0

        # Move to the next day
        self._offset += 1
        prev_close = close
        close = self._cur_close()

        # End episode if we run out of data
        done |= self._offset >= self._prices.close.shape[0] - 1

        # Continuous reward while holding a position (optional)
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


# ---------------------------------------------------------
# 1D Convolution-friendly state (CNN-based agents)
# ---------------------------------------------------------
class State1D(State):
    """
    Alternative state representation suitable for 1D CNNs.
    """

    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        """
        Same information as State.encode(), but arranged as channels × time.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0

        for bar_idx in range(-self.bars_count + 1, 1):
            i = self._offset + bar_idx

            c = float(self._prices.close[i])
            pc = float(self._prices.close[i - 1]) if i > 0 else c

            h = float(self._prices.high[i])
            l = float(self._prices.low[i])

            res[shift] = (h / c) - 1.0; shift += 1
            res[shift] = (l / c) - 1.0; shift += 1
            res[shift] = (c / pc) - 1.0; shift += 1

            if self.volumes:
                v = float(self._prices.volume[i])
                pv = float(self._prices.volume[i - 1]) if i > 0 else v
                vol_feat = (v / pv) - 1.0 if pv > 0 else 0.0
                res[shift] = np.clip(vol_feat, -5.0, 5.0)
                shift += 1

        res[shift] = float(self.have_position); shift += 1
        res[shift] = (self._cur_close() - self.open_price) / self.open_price if self.have_position else 0.0

        return res


# ---------------------------------------------------------
# Gymnasium Environment Wrapper
# ---------------------------------------------------------
class StocksEnv(gym.Env):
    """
    Gymnasium-compatible trading environment.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True,
                 state_1d=False, random_ofs_on_reset=True,
                 reward_on_close=False, volumes=False):

        self._prices = prices

        # Choose state representation
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close,
                                  reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,
                                reward_on_close=reward_on_close, volumes=volumes)

        # Define Gym spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32
        )

        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self, *, seed=None, options=None):
        """
        Start a new episode.
        """
        super().reset(seed=seed)

        # Randomly choose a stock
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._state.bars_count

        # Random starting point
        if self.random_ofs_on_reset:
            max_ofs = max(prices.high.shape[0] - bars * 10, bars + 1)
            offset = int(self.np_random.integers(bars, max_ofs))
        else:
            offset = bars

        self._state.reset(prices, offset)
        return self._state.encode(), {"instrument": self._instrument, "offset": offset}

    def step(self, action_idx):
        """
        Execute one action in the environment.
        """
        action = Actions(int(action_idx))
        reward, done = self._state.step(action)

        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}

        return obs, float(reward), bool(done), False, info

    def seed(self, seed=None):
        """
        Set a random seed for reproducibility.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        """
        Convenience method to create the environment from a directory of CSV files.
        """
        prices = data.load_many_from_dir(data_dir)
        return cls(prices, **kwargs)
