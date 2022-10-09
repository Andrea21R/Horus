import os
import pandas as pd
import yfinance as yf

from fe import FeatureEngineering
from signals import Signals
from utils import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# data = yf.Ticker("EURUSD=X").history('5d', interval='1m')['Close']
data = pd.read_parquet(os.path.dirname(os.getcwd()) + "/test_data/EURUSD2022.parquet")#.resample('1min').last().dropna()

rsi = FeatureEngineering.rsi(data['close'], timeperiod=14)
s_risk = FeatureEngineering.vola_sma(data['close'], timeperiod=14, on_rets=False)

rsi, s_risk = Utils.align_series(rsi, s_risk)
data = data.loc[rsi.index]

signal = Signals.from_rsi(
    data=data,
    rsi=rsi,
    open_cv=(70, 30),
    close_cv=(50, 50),
    s_risk=s_risk,
    n_std=2,
    show_graph=True,
    tc_spread=True
)
