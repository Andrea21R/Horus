import pandas as pd
import numpy as np
import talib as ta
import hurst
from typing import Tuple, Union, Optional

from fe_and_signals import aot_module as aot
from fe_and_signals.utils import Utils

"""
Migliorie & Commenti:
- velocizzare Hurst (troppo lento ora)
"""


class Features(object):

    class Overlap:

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
        def rsi(s: pd.Series, timeperiod: int) -> pd.Series:
            """
            Calc Relative Strength Index (RSI)
            :param s: pd.Series, closing price
            :param timeperiod: int, window
            :return: pd.Series
            """
            Utils.check_min_obs(s, min_len=timeperiod + 1)
            return ta.RSI(s, timeperiod)

        @staticmethod
        def bollinger(s: pd.Series, timeperiod: int, ndevup: float, ndevdown: float) -> pd.DataFrame:
            """
            Return the Bollinger Bands.
            :param s: pd.Series, prices
            :param timeperiod: int, window
            :param ndevup: number of std for the upper band
            :param ndevdown: number of std for the lower band
            :return: pd.DataFrame with columns ['uband', 'mband', 'lband']
            """
            output = pd.concat(ta.BBANDS(s, timeperiod, ndevup, ndevdown), axis=1)
            output.columns = ['uband', 'mband', 'lband']
            return output

    class Vola:

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

            ema = np.repeat(np.nan, repeats=len(s_arr))

            for t in range(timeperiod, len(s_arr) + 1):

                tgt = s_arr[t - timeperiod: t]
                tmp_ema = ta.EMA(tgt, timeperiod)

                ema[t - 1] = tmp_ema[-1]

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
                return Features.Overlap.sma(bars_dispersion, timeperiod=timeperiod)

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

    class Momentum:

        @staticmethod
        def macd(s: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int) -> tuple:
            """
            Return macd, macdsignal, macdhist
            """
            return ta.MACD(s, fastperiod, slowperiod, signalperiod)

        @staticmethod
        def cross_sma_perc_distance(s_price: pd.Series, lookback: Tuple[int, int]) -> float:
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
            short_ma = Features.Overlap.sma(s_price, timeperiod=lookback[0])
            long_ma  = Features.Overlap.sma(s_price, timeperiod=lookback[1])
            return short_ma / long_ma - 1

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
        def adx(data: pd.DataFrame, timeperiod: int) -> pd.Series:
            """
            Return the Average-Directional-Index (ADX)
            :param data: pd.DataFrame, with at least columns ['high', 'low', 'close']
            :param timeperiod: int
            :return: pd.Series
            """
            return ta.ADX(data.high, data.low, data.close, timeperiod)

        @staticmethod
        def plus_di(data: pd.DataFrame, timeperiod: int) -> pd.Series:
            """
            Return the +DI
            :param data: pd.dataFrame, with at least columns ['high', 'low', 'close']
            :param timeperiod: int
            :return: pd.Series
            """
            return ta.PLUS_DI(data.high, data.low, data.close, timeperiod=timeperiod)

        @staticmethod
        def minus_di(data: pd.DataFrame, timeperiod: int) -> pd.Series:
            """
            Return the -DI
            :param data: pd.dataFrame, with at least columns ['high', 'low', 'close']
            :param timeperiod: int
            :return: pd.Series
            """
            return ta.MINUS_DI(data.high, data.low, data.close, timeperiod=timeperiod)

    class Others:

        @staticmethod
        def returns(s: pd.Series, lags: Optional[Union[int, list]] = None) -> Union[pd.Series, pd.DataFrame]:
            """
            It calc price returns with no lag and with them, depending on the parameter lags
            :param s: pd.Series, prices
            :param lags: Optional[Union[int, list]], type:int if you want all the lags between 0 and lags. type: list
                                                     if you want to specify the lags period.
            """
            if not isinstance(s, pd.Series):
                raise Exception("s must be a pandas.Series object")
            rets = s.pct_change()
            if lags:
                if isinstance(lags, int):
                    lags = list(range(lags + 1))

                if len(rets.dropna()) < max(lags):
                    raise Exception("Observations must be at least equal to max(lags) + 1")

                output = pd.concat([rets.shift(n) for n in lags], axis=1)
                output.columns = [f'lag.{n}' for n in lags]
                return output
            else:
                return rets

    class Dummy:

        @staticmethod
        def extreme_events(s: pd.Series, std_threshold: float, std_window: int, diff: bool = False) -> pd.Series:
            """
            Returns a dummy series {-1, 0, 1} where there was an extreme event, i.e. a price movement greater than N
            standard deviation. It might be use as a reversal signal
            :param s: pd.Series, prices
            :param std_threshold: float, number of standard deviations to identify the extreme event
            :param std_window: int, window for building the rolling standard deviation threshold
            :param diff: bool, True to use differences for price movement, False for percentage returns.
            :return: pd.Series with {-1: extreme-negative event; 0: no extreme event; 1: extreme-positive event}
            """
            s_roll_std = s.rolling(std_window).std() * std_threshold

            if diff:
                delta = s.diff()
            else:
                delta = s.pct_change()

            return delta.\
                mask(delta >  s_roll_std,  1).\
                mask(delta < -s_roll_std, -1).\
                mask((delta <= s_roll_std) & (delta >= -s_roll_std), 0)
