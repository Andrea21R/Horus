import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from Andrea.fe_and_signals.utils import Utils
from Andrea.fe_and_signals.graphs import Graphs


class Signals:

    @staticmethod
    def from_overlapped_filter(
            data: pd.DataFrame,
            s_overlap: pd.Series,
            show_graph: bool = False,
            spread: bool = True,
            tc_perc: Optional[float] = None
    ) -> pd.Series:

        close = data['close']
        s_signals = close.mask(close >= s_overlap, 1).mask(close < s_overlap, -1)

        if show_graph:
            fig, axs = plt.subplots(nrows=2)
            fig.suptitle("TRADING SYSTEM | OVERLAPPED FILTER")
            Graphs.buy_sell_on_price(
                s_signals=s_signals,
                continuous_signals=True,
                s_close=close,
                ax=axs[0],
                large_data_constraint=True
            )
            axs[0].plot(s_overlap)
            Graphs.pnl_graph(
                data=data,
                s_signals=s_signals,
                spread=spread,
                tc_perc=tc_perc,
                ax=axs[1]
            )
            plt.tight_layout()
            plt.show()

        return s_signals


    @staticmethod
    def from_cross_filters(
            s_filter1: pd.Series,
            s_filter2: pd.Series,
            s_control: Optional[pd.Series] = None,
            control_threshold: Optional[float] = None,
            show_graph: bool = False,
            data: Optional[pd.DataFrame] = None,
            spread: bool = True,
            tc_perc: Optional[float] = None
    ):
        if isinstance(s_control, pd.Series) and (not control_threshold):
            raise Exception("If you pass s_control series, you also have to pass a  control threshold")

        s_signals = s_filter1.\
            mask(s_filter1 >= s_filter2, 1).\
            mask(s_filter2 > s_filter1, -1)
        if control_threshold:
            s_signals = s_signals.mask(s_control < control_threshold, 0)

        if show_graph:
            fig, axs = plt.subplots(nrows=3)
            fig.suptitle(f"TRADING SYSTEM | CROSS-FILTER {'with control' if control_threshold else ''}")
            Graphs.buy_sell_on_price(
                s_signals=s_signals,
                continuous_signals=False if control_threshold else True,
                s_close=data.close,
                ax=axs[0],
                large_data_constraint=True
            )
            axs[1].plot(s_filter1, color='green')
            axs[1].plot(s_filter2, color='red')
            legend = ['filter1', 'filter2']
            if control_threshold:
                axs[1].plot(s_control, color='orange')
                axs[1].axhline(control_threshold)
                legend.extend(['control-filter', 'control-threshold'])
            axs[1].legend(legend)
            axs[1].grid(linestyle='--', color='silver')
            axs[1].set_ylabel('FILTERS', fontweight='bold')

            Graphs.pnl_graph(
                data=data,
                s_signals=s_signals,
                spread=spread,
                tc_perc=tc_perc,
                ax=axs[2]
            )

        return s_signals

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
    ) -> pd.Series:
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
        :return: pd.Series
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

    @staticmethod
    def from_bands(
            data: pd.DataFrame,
            uband: pd.Series,
            mband: pd.Series,
            lband: pd.Series,
            middle_exit: bool,
            s_risk: pd.Series,
            n_std: float,
            fix_talib_bug: bool,
            max_attempts: Optional[int] = None,
            show_graph: bool = False,
            spread: bool = True,
            tc_perc: Optional[float] = None
    ) -> pd.Series:
        """
        Return signals {1: long, 0: neutral, -1: short} from bands indicators (bollinger, ...).

        :param data: pd.DataFrame, with at least the column 'close',
        :param uband: pd.Series, upper-band
        :param mband: pd.Series, middle-band
        :param lband: pd.Series, lower-band
        :param middle_exit: bool, True if you want to set a Take-Profit when price crosses the mband
        :param s_risk: pd.Series, volatility series (atr, vola_sma, vola_ema, etc.)
        :param n_std: float, number of std for SL. It will be used with s_risk
        :param fix_talib_bug: bool, if you use Bollinger, talib has a bug. Sometimes uband=mband=lband. If
                              this par is settled to True, when this bug occurs, function will return no signal
        :param max_attempts: Optional[int], number of max SL touchable from one leg.
        :param show_graph: bool, True for showing a trading graph
        :param spread: bool, True if data contain spread (%), otherwise False
        :param tc_perc: if you want to use a general % spread for all periods
        :return: pd.Series
        """
        close, uband, mband, lband = Utils.from_series_to_numpy(data.close, uband, mband, lband)
        Utils.check_len(close, uband, mband, lband)

        signals = np.repeat(0, len(close))
        s_sl = np.repeat(np.nan, len(close))

        signal = 0
        tmp_sl = np.nan
        open_trade = False
        lose_trades = []
        pause = False

        for t in range(len(close)):

            if fix_talib_bug:
                stop_for_bug = True if uband[t] == mband[t] else False
            else:
                stop_for_bug = False

            if not stop_for_bug:
                # ---- PAUSE EVALUATION
                if pause:
                    # given p0=close[t-1], p1=close[t], m=mband[t]
                    # if (p0-m>0 & p1-m<0) or (p0-m<0 & p1-m>0) it means that price crossed the middle-band
                    if (np.sign(close[t - 1] - mband[t]) + np.sign(close[t] - mband[t])) == 0:
                        pause = False
                        lose_trades = []

                # --- SL & TP
                if open_trade:
                    # ---- STOP LOSS
                    if (signal == 1) and (close[t] < tmp_sl):
                        lose_trades.append(signal)
                        signal = 0
                        open_trade = False
                        continue
                    elif (signal == -1) and (close[t] > tmp_sl):
                        lose_trades.append(signal)
                        signal = 0
                        open_trade = False
                        continue
                    # ---- TAKE PROFIT
                    if middle_exit:
                        if (signal == 1) and (close[t] > mband[t]):
                            lose_trades = []
                            signal = 0
                            open_trade = False
                            continue
                        elif (signal == -1) and (close[t] < mband[t]):
                            lose_trades = []
                            signal = 0
                            open_trade = False
                            continue
                    else:
                        if (signal == 1) and (close[t] > uband[t]):
                            lose_trades = []
                            signal = 0
                            open_trade = False
                        elif (signal == -1) and (close[t] < lband[t]):
                            lose_trades = []
                            signal = 0
                            open_trade = False

                # ----- OPEN TRADE
                if not open_trade:

                    if max_attempts and (not pause):
                        if abs(sum(lose_trades[-max_attempts:])) == max_attempts:
                            pause = True

                    if not pause:
                        if close[t] < lband[t]:
                            signal = 1
                            tmp_sl = close[t] - n_std * s_risk[t]
                            open_trade = True
                        elif close[t] > uband[t]:
                            signal = -1
                            tmp_sl = close[t] + n_std * s_risk[t]
                            open_trade = True

                signals[t] = signal
                s_sl[t] = tmp_sl

        signals = pd.Series(signals, index=data.index)
        s_sl = pd.Series(s_sl, index=data.index).ffill()

        if show_graph:

            fig, axs = plt.subplots(nrows=2)
            fig.suptitle(f"TRADING SYSTEM | BANDS | mid_exit={middle_exit}; n_std_risk={n_std}")

            Graphs.buy_sell_on_price(
                s_signals=signals,
                continuous_signals=False,
                s_close=data.close,
                s_sl=s_sl,
                ax=axs[0],
                large_data_constraint=False
            )
            uband, mband, lband = pd.Series(uband, index=data.index, name='uband'), \
                                  pd.Series(mband, index=data.index, name='mband'), \
                                  pd.Series(lband, index=data.index, name='lband')
            axs[0].plot(pd.concat([uband, mband, lband], axis=1))

            Graphs.pnl_graph(data=data, s_signals=signals, spread=spread, tc_perc=tc_perc, ax=axs[1])

            plt.tight_layout()
            plt.plot()

        return signals
