import os
import pandas as pd
import yfinance as yf

from Andrea.fe_and_signals import Features
from Andrea.fe_and_signals import Utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = yf.Ticker("BTC-USD").history('300d', interval='1h')[['Open', 'High', 'Low', 'Close']]
data.columns = ['open', 'high', 'low', 'close']
data = pd.read_parquet(os.path.dirname(os.getcwd()) + "/test_data/EURUSD2022.parquet")

rsi = Features.Overlap.rsi(data['close'], timeperiod=14)
s_risk = Features.Vola.vola_sma(data['close'], timeperiod=120, on_rets=False)

rsi, s_risk = Utils.align_series(rsi, s_risk)
data = data.loc[rsi.index]

bb = Features.Overlap.bollinger(data.close, 60 * 24 * 5, 2, 2)

# signals = Signals.from_bands(data=data, uband=bb['uband'], mband=bb['mband'], lband=bb['lband'], middle_exit=True,
#                              s_risk=s_risk, n_std=2, fix_talib_bug=True, max_attempts=2, show_graph=True, spread=True)
# s_pnl = RiskSignal.get_pnl_from_signals(signals, data, spread=True)
# cum_pnl = RiskSignal.get_cumulative_pnl(s_pnl['net_pnl'], comp=True)
# trade_hist = RiskSignal.get_trade_history(signals, data)


#
# signals = Signals.from_rsi(
#     data=data,
#     rsi=rsi,
#     open_cv=(75, 25),
#     close_cv=(60, 40),
#     s_risk=s_risk,
#     n_std=3,
#     show_graph=True,
#     spread=True,
#     tc_perc=None, #0.002
# )

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
