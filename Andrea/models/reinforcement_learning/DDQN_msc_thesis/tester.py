import numpy as np
import pandas as pd
from tensorflow import keras
from DDQN_trading import DataSource
from typing import *


class Tester:

    def __init__(self, model_path: str, ticker: str, start_end: Tuple[str, str]):
        self.model_path = model_path
        self.ticker = ticker
        self.start_end = start_end

        self.ann = self.get_model()
        self.test_data = self.get_test_data()

    def get_model(self) -> keras.models.Sequential:
        return keras.models.load_model(self.model_path)


    def get_test_data(self) -> pd.DataFrame:

        data_source = DataSource(
            ticker=self.ticker,
            normalize=True,
            start_end=None
        )
        return data_source.data.loc[self.start_end[0]: self.start_end[1]]


    def predict_actions(self) -> pd.Series:
        predictions = self.ann.predict_on_batch(self.test_data)
        actions = np.argmax(predictions, axis=1)
        return pd.Series(data=actions, index=self.test_data.index).sub(1)  # sub(1) because {0: short, 1: neutral, 2: long}


    def get_test_returns(self, tc_bps: float = 0.0001) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the returns of the strategy, using the weights of the ann for the estimation.
        """
        actions = self.predict_actions()
        asset_rets = self.test_data['returns']

        gross_rets = actions.shift(1).mul(asset_rets)
        tc = actions.diff().abs() * tc_bps
        net_rets = gross_rets.sub(tc)

        return  gross_rets, net_rets


    @staticmethod
    def get_cumulative_rets(rets: pd.Series, comp: bool = True) -> pd.Series:
        if comp:
            return (1 + rets).cumprod() - 1
        else:
            return rets.cumsum()
