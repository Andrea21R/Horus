import pandas as pd
import numpy as np
import talib as ta
import hurst
from typing import Tuple

import aot_module as aot
from fe_and_signals.utils import Utils

"""
Migliorie & Commenti:
- velocizzare Hurst (troppo lento ora)
"""


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

    @staticmethod
    def atr(data: pd.DataFrame, timeperiod: int) -> pd.Series:
        """
        Returns the Average-True-Range (ATR)
        :param data: pd.DataFrame, with at least columns: ['high', 'low', 'close']
        :param timeperiod: int, window
        :return: pd.Series
        """
        Utils.check_min_obs(data, min_len=timeperiod + 1)
        return ta.ATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)

    @staticmethod
    def bars_dispersion(data: pd.DataFrame) -> pd.Series:
        """
        It returns the dispersion of the bars. It's computed as follow:
            (HIGH - LOW) / LOW
        Thus, it's bounded between 0 and 1 --> [0, 1]
        :param data: pd.DataFrame, with at least the columns: ['high', 'low']
        :return: pd.Series
        """
        return (data['high'] - data['low']) / data['low']

    @classmethod
    def bars_dispersion_rolling(cls, data: pd.DataFrame, timeperiod: int) -> pd.Series:
        """
        Returns SMA of the dispersion of the bars. See FeaturesEngineering.bars_dispersion func's docs for more details.
        :param data: pd.DataFrame, with at least the columns: ['high', 'low']
        :timeperiod: int, window for SMA
        :return: pd.Series
        """
        Utils.check_min_obs(data, min_len=timeperiod)

        bars_dispersion =  cls.bars_dispersion(data)
        if timeperiod == 1:
            return bars_dispersion
        else:
            return cls.sma(bars_dispersion, timeperiod=timeperiod)

    @staticmethod
    def hurst_exp(s_price: pd.DataFrame, timeperiod: int) -> pd.Series:
        """
        Returns Hurst Exponent Indicator. It detects if the series shows persistent pattern (i.e. trend period) or
        anti-persistent one (i.e. mean-reverting period). For further references see:
            - https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
            - https://en.wikipedia.org/wiki/Hurst_exponent
        For computing Hurst Exponent exist several formulas. This function use the R/S procedure.
        ----------------------------------------------------------------------------------------------------------------
        WARNING.1: Due to a constraint from the formula, minimum obs required are 100
        ----------------------------------------------------------------------------------------------------------------
        :param s_price: pd.Series, closing_price
        :param timeperiod: int, window
        :return: pd.Series
        """
        if timeperiod < 100:
            raise Exception("Due to a Hurst constraint, timeperiod cannot be less then 100")
        Utils.check_min_obs(s_price, min_len=100 if timeperiod < 100 else timeperiod)
        return  s_price.rolling(timeperiod).apply(lambda s: hurst.compute_Hc(s, kind='price', simplified=False)[0])

    # ========================================== VOLA ==================================================================

    @classmethod
    def cross_sma_perc_distance(cls, s_price: pd.Series, lookback: Tuple[int, int]) -> float:
        """
        Returns the percentage distance between a shorter SMA and a longest one, computed on the closing prices.
        ----------------------------------------------------------------------------------------------------------------
        It's computed as follow:
                Shorter_SMA / Longer_SMA - 1
        ----------------------------------------------------------------------------------------------------------------
        It should be useful to feed some ML models. It's a market features that trying to suggest if we're in a bullish
        market (shorter > longer) or into a bear market (longer < shorter), and the magnitude of the trend (this is the
        reason why it returns a distance percentage, instead of a simple dummy {1: bullish; 0: bearish).
        ----------------------------------------------------------------------------------------------------------------
        :param data, pd.DataFrame, with at least closing prices (columns named 'close')
        :param lookback, Tuple[int, int], windows for SMA. First elements will be assigned to the shorter SMA
        """
        Utils.check_min_obs(s_price, min_len=lookback[1])
        short_ma = cls.sma(s_price, timeperiod=lookback[0])
        long_ma  = cls.sma(s_price,timeperiod=lookback[1])
        return short_ma / long_ma - 1
