import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

from utils import Utils
from graphs import Graphs


class Signals:

    @staticmethod
    def from_rsi(
            data: pd.DataFrame,
            rsi: pd.Series,
            open_cv: Tuple[float, float],
            close_cv: Tuple[float, float],
            s_risk: pd.Series,
            n_std: float,
            show_graph: bool = False,
            spread: bool = True,
            tc_perc: Optional[float] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns signals from RSI indicator
        :param data: pd.DataFrame, with OHLC and spread(%) (or bidask). spread to True to use spread(%)
        :param rsi: pd.Series
        :param open_cv: Tuple[float, float], 1° element: upper_threshold; 2°: lower_threshold
        :param close_cv: Tuple[float, float], 1° element: close_short; 2°: close_long
        :param s_risk: pd.Series
        :param n_std: float
        :param show_graph: bool, True if you want the graph of the trading signal
        :param spread: bool, True if data contain spread (%), otherwise False
        :param tc_perc: if you want to use a general % spread for all periods
        :return:
        """
        Utils.check_len(data['close'], rsi, s_risk)

        price, rsi, risk = Utils.from_series_to_numpy(data['close'], rsi, s_risk)
        signals = np.zeros(len(price))
        s_sl = np.repeat(np.nan, len(price))

        cz = {'long': False, 'short': False}
        open_trade = False
        tmp_sl = None
        signal = 0

        for t in range(len(price)):

            # print(f'--------- t: {t}')

            # --------- CRITIC ZONE
            if not open_trade:
                if rsi[t] > open_cv[0]:
                    cz['short'] = True
                    # print(f'cz Short')
                    continue  # if now we are in CZ, other evaluation is unuseful
                elif rsi[t] < open_cv[1]:
                    cz['long'] = True
                    # print(f'cz Long')
                    continue  # if now we are in CZ, other evaluation is unuseful

            # --------- OPEN TRADE
            if not open_trade:
                if cz['short'] and (rsi[t] < open_cv[0]):
                    signal = -1
                    cz['short'] = False
                    open_trade = True
                    tmp_sl = price[t] + n_std * risk[t]
                    s_sl[t] = tmp_sl
                    # print(f'Open Short')

                elif cz['long'] and (rsi[t] > open_cv[1]):
                    signal = 1
                    cz['long'] = False
                    open_trade = True
                    tmp_sl = price[t] - n_std * risk[t]
                    s_sl[t] = tmp_sl
                    # print(f'Open Long')

            # --------- CLOSE TRADE
            if open_trade:

                if ((signals[t -1] == -1) and (rsi[t] <= close_cv[0])) or ((signals[t -1] == 1) and (rsi[t] >= close_cv[1])):
                    # print(f'Close Short for TP (RSI: {rsi}; P: {price[t]})')
                    signal = 0
                    open_trade = False
                    tmp_sl = None


            # --------- STOP LOSS
            if open_trade:

                if ((signals[t -1] == -1) and (price[t] > tmp_sl)) or ((signals[t -1] == 1) and (price[t] < tmp_sl)):
                    # print(f'Close Long for SL (SL: {tmp_sl}; P: {price[t]})')
                    signal = 0
                    open_trade = False
                    tmp_sl = None

            signals[t] = signal

        s_signals = pd.Series(signals, index=data.index)
        s_sl = pd.Series(s_sl, index=data.index).ffill()

        if show_graph:
            fig, axs = plt.subplots(nrows=3)
            fig.suptitle(
                f"TRADING SYSTEM | RSI | open_cv={open_cv}; close_cv={close_cv}; n_std={n_std}", fontweight='bold'
            )
            Graphs.buy_sell_on_price(
                s_signals=s_signals,
                continuous_signals=False,
                s_close=data['close'],
                s_sl=s_sl,
                ax=axs[0],
                large_data_constraint=True,
                show=False
            )
            Graphs.reversal_indicator(
                indicator=rsi, open_cv=open_cv, close_cv=close_cv, indicator_name="RSI", ax=axs[1], show=False
            )
            Graphs.pnl_graph(
                data=data, s_signals=s_signals, spread=spread, tc_perc=tc_perc, ax=axs[2], show=False
            )
            plt.tight_layout()
            plt.plot()

        return s_signals
