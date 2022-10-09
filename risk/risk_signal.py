import pandas as pd
from typing import Union


class RiskSignal:

    @staticmethod
    def calc_pnl_from_signals(s_signals: pd.Series, data: pd.DataFrame, spread: bool) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns gross/net PNL from signals strategy
        :param s_signals: pd.Series, with {-1: short, 0: neutral, 1: long}
        :param data: pd.DataFrame, with OHLC and spread(%) (or bidask). spread to True to use spread(%)
        :param spread: bool, True to compute net_pnl using spread(%)
        :return: pd.Series
        """
        s_tp = s_signals.diff().abs()  # turning points
        s_gross_pnl = s_signals.shift(1).mul(data['close'].pct_change())

        # spread(%)
        if spread:
            # open_spread because I receive a signal at the end of bar, thus I make a trade at the open of the next one
            s_tc = data['open_spread'] / 2
            s_tc = s_tp.shift(1).mul(s_tc)  # shift(1) because the signal in t is traded at t + 1

            return pd.DataFrame(
                {
                    'net_pnl': s_gross_pnl - s_tc,
                    'gross_pnl': s_gross_pnl
                },
                index=s_gross_pnl.index
            )
        else:
            # no tc
            return s_gross_pnl


    @staticmethod
    def calc_cumulative_pnl(s_pnl: pd.Series, comp: bool) -> pd.Series:
        """
        Returns Cumulative pnl from point-in-time pnl.
        :param s_pnl: pd.Series
        :param comp: bool, True for compounding, False for simple.
        :return: pd.Series
        """
        if comp:
            return (1 + s_pnl).cumprod() - 1
        else:
            return s_pnl.cumsum()
