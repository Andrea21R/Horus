import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

from typing import *


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment
    - Loads & preprocesses prices
    - Provides data for each new episode.
    """

    def __init__(
            self,
            steps_per_episode: int = 60 * 24,
            ticker: str = 'EURUSD2022_1m',
            normalize: bool = True,
            start_end: Optional[Tuple[str, str]] = None
    ):
        self.ticker = ticker
        self.steps_per_episode = steps_per_episode
        self.start_end = start_end
        self.normalize = normalize
        self.data = self.load_data()
        self.load_features()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None

    def load_data(self) -> pd.DataFrame:
        log.info('loading data for {}...'.format(self.ticker))
        file_path = os.path.dirname(os.getcwd()) + rf"\data\dataset\{self.ticker}.parquet"
        df = pd.read_parquet(file_path)
        if self.start_end:
            start, end = self.start_end
            df = df.loc[start: end]
        log.info('got data for {}...'.format(self.ticker))
        return df[['close']]

    def load_features(self) -> NoReturn:

        file_path = os.path.dirname(os.getcwd()) + rf"\data\dataset\features_{self.ticker}.parquet"
        fe_df = pd.read_parquet(file_path).loc[self.data.index]

        self.data = pd.concat([self.data, fe_df], axis=1).dropna()
        rets = self.data['close'].pct_change()

        if self.normalize:
            self.data = pd.DataFrame(
                data=scale(self.data),
                columns=self.data.columns,
                index=self.data.index
            )
        self.data.drop(['close'], axis=1, inplace=True)
        self.data['returns'] = rets  # not scale rets
        self.data = self.data.loc[:, ['returns'] + list(self.data.columns.drop('returns'))]
        self.data = self.data.dropna()
        log.info(self.data.info())

    def reset(self) -> NoReturn:
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.steps_per_episode
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self) -> tuple:
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.steps_per_episode
        return obs, done
