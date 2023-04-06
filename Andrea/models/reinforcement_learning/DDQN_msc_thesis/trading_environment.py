import gym
from gym import spaces
from gym.utils import seeding

from DDQN_trading import DataSource, TradingSimulator

from typing import *


class TradingEnvironment(gym.Env):
    """
    A simple trading environment for RL.
    - Provides observations for a security price series
    - An episode is defined as a sequence of N steps with random start
    - Each 'step' allows the agent to choose one of three actions:
        - 0: SHORT
        - 1: HOLD
        - 2: LONG
    - Trading has an optional cost (default: 10bps) of the change in position value.
    - Going from short to long implies two trades.
    - Not trading also incurs a default time cost of 1bps per step.
    - An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    - If the NAV drops to 0, the episode ends with a loss.
    - If the NAV hits 2.0, the agent wins.
    - Trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    # metadata = {'render.modes': ['human']}

    def __init__(
            self,
            steps_per_episode: int = 60 * 24,
            trading_cost_bps: float = 1e-3,
            time_cost_bps: float = 1e-4,
            ticker='EURUSD2022_1m',
            start_end: Optional[Tuple[str, str]] = None
    ):
        self.trading_days = steps_per_episode
        self.trading_cost_bps = trading_cost_bps
        self.ticker = ticker
        self.time_cost_bps = time_cost_bps
        self.data_source = DataSource(
            steps_per_episode=self.trading_days,
            ticker=ticker,
            start_end=start_end
        )
        self.simulator = TradingSimulator(
            steps=self.trading_days,
            trading_cost_bps=self.trading_cost_bps,
            time_cost_bps=self.time_cost_bps
        )
        self.action_space = spaces.Discrete(3)  # three actions: {0: HOLD, 1: BUY, 2: SELL}
        self.observation_space = spaces.Box(self.data_source.min_values.values, self.data_source.max_values.values)   # to understand
        self.reset()

    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action) -> tuple:
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action, market_return=observation[0])
        return observation, reward, done, None, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]
