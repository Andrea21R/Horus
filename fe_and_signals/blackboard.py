import os
import pandas as pd
import yfinance as yf

from fe import FeatureEngineering
from mkt_fe import get_mkt_features_by_trades
from signals import Signals
from utils import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = yf.Ticker("BTC-USD").history('300d', interval='1h')[['Open', 'High', 'Low', 'Close']]
data.columns = ['open', 'high', 'low', 'close']
data = pd.read_parquet(os.path.dirname(os.getcwd()) + "/test_data/EURUSD2022.parquet")#.resample('1min').last().dropna()

rsi = FeatureEngineering.rsi(data['close'], timeperiod=14)
s_risk = FeatureEngineering.vola_sma(data['close'], timeperiod=14, on_rets=False)

rsi, s_risk = Utils.align_series(rsi, s_risk)
data = data.loc[rsi.index]


signals = Signals.from_rsi(
    data=data,
    rsi=rsi,
    open_cv=(75, 25),
    close_cv=(60, 40),
    s_risk=s_risk,
    n_std=3,
    show_graph=True,
    spread=True,
    tc_perc=None, #0.002
)

# trade_history = RiskSignal.get_trade_history(s_signals=signals, data=data, spread=True, real_start_end=False)  #, tc_perc=0.002)
#
# mkt_fe = get_mkt_features_by_trades(
#     trades_start_dates=trade_history['start'].to_list(),
#     data=data,
#     fe_pars=fe_pars,
# )
# mkt_fe.to_parquet(r"C:\Users\andre\Dropbox\Horus\test_data\mkt_fe_for_logit_EURUSD.parquet")
# trade_history.to_pickle(r"C:\Users\andre\Dropbox\Horus\test_data\rsi_trades_history_EURUSD.pkl")
#
