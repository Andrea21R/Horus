import pandas as pd
import numpy as np
import talib as ta

import aot_module as aot
from horus.fe_and_signals.utils import Utils


class FeatureEngineering(object):

    @staticmethod
    def sma(s: pd.Series, timeperiod: int) -> pd.Series:
        """
        Returns Simple Moving Average (SMA)
        :param s: pd.Series
        :param timeperiod: int, window
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod)
        return ta.SMA(s, timeperiod)

    @staticmethod
    def ema(s: pd.Series, timeperiod: int) -> pd.Series:
        """
        Returns Exponential Moving Average (EMA)
        :param s: pd.Series
        :param timeperiod: int, window
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod)
        return ta.EMA(s, timeperiod)

    @staticmethod
    def macd(s: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int) -> tuple:
        """
        Return macd, macdsignal, macdhist
        """
        return ta.MACD(s, fastperiod, slowperiod, signalperiod)

    @staticmethod
    def kama(s: pd.Series, timeperiod: int, fp: int, sp: int) -> pd.Series:
        """
        KAMA(timeperiod (n), fastest_period (fp), slowest_period (sp))

        Formula
        -----------------------------------------------------------------
                KAMA(i) = KAMA(i-1) + SC * [P(i) - KAMA(i-1)]
        -----------------------------------------------------------------

        Appendix
        -----------------------------------------------------------------
        i)   SC = [ER * (Fast - Slow) + Slow] ** 2
        ii)  ER = |P(i) - P(i-n+1)| / sum(|P(t)-P(t-1)|; t=1 to i)
        iii) Fast = 2 / (FP + 1)
        iv)  Slow = 2 / (SP + 1)

        Definition
        -----------------------------------------------------------------
        SC: Smoothing Constants
        ER: Efficiency Ratio
        FP, SP: Fastest Period, Slowest Period
        FC, SC: Fastest Coefficient, Slowest Coefficient
        """
        prices = s.to_numpy()
        kama = np.zeros(prices.shape)
        fastest = 2 / (fp + 1)
        slowest = 2 / (sp + 1)

        kama = aot.loop_kama(prices, kama, timeperiod, fastest, slowest)
        kama = pd.Series(kama, index=s.index).replace(0, np.nan)
        return kama

    @staticmethod
    def roc(s: pd.Series, timeperiod: int) -> pd.Series:
        """
        Calc Return Of Change (ROC)
        :param s: pd.Series, closing price
        :param timeperiod: int, window
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod + 1)
        return ta.ROCP(s, timeperiod)

    @staticmethod
    def rsi(s: pd.Series, timeperiod: int) -> pd.Series:
        """
        Calc Relative Strength Index (RSI)
        :param s: pd.Series, closing price
        :param timeperiod: int, window
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod + 1)
        return ta.RSI(s, timeperiod)

    # ========================================== VOLA ==================================================================

    @staticmethod
    def vola_sma(s: pd.Series, timeperiod: int, on_rets: bool) -> pd.Series:
        """
        Returns Rolling-Volatility using Simple Moving Average (SMA)
        :param s: pd.Series, prices
        :param timeperiod: int, window
        :param on_rets: bool, True if you want to calc vola on returns, otherwise on the prices
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod + 1 if on_rets else timeperiod)
        if on_rets:
            tgt = s.pct_change()
        else:
            tgt = s
        return tgt.rolling(timeperiod).std()

    @staticmethod
    def vola_ema(s: pd.Series, timeperiod: int, on_rets: bool) -> pd.Series:
        """
        Returns Rolling-Volatility using Exponential Moving Average (EMA)
        :param s: pd.Series, prices
        :param timeperiod: int, window
        :param on_rets: bool, True if you want to calc vola on returns, otherwise on the prices
        :return: pd.Series
        """
        Utils.check_min_obs(s, min_len=timeperiod + 1 if on_rets else timeperiod)
        if on_rets:
            s_arr = Utils.from_series_to_numpy(s.pct_change())
        else:
            s_arr = Utils.from_series_to_numpy(s)

        ema = np.repeat(np.nan, repeats=s_arr)

        for t in range(timeperiod, len(s_arr) + 1):

            tgt = s_arr[t - timeperiod: timeperiod]
            tmp_ema = ta.EMA(tgt, timeperiod)

            ema[timeperiod - 1] = tmp_ema[-1]

        return pd.Series(s_arr, index=s.index)
