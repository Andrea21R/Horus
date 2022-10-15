import pandas as pd
import numpy as np
import ray
from itertools import product
from typing import Union, Optional, List, Tuple

from fe_and_signals.utils import Utils
from fe_and_signals.fe import FeatureEngineering as Fe

"""
Da fare & Commenti:
- Risolvere problema start/end in trade_history. Quando metto diff=0 il numero di starts e ends è diverso
- MKT_featuers naming: ora le chiamo vola.0, vola.1. Trovare un modo migliore per fare naming
"""


class RiskSignal:

    @staticmethod
    def get_pnl_from_signals(
            s_signals: pd.Series,
            data: pd.DataFrame,
            spread: bool,
            tc_perc: Optional[float] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Returns gross/net PNL from signals strategy
        ----------------------------------------------------------------------------------------------------------------
        WARNING.1: Calculating the Net-PNL, we subtract the spread% or tc_perc at the first and end pnl (point-in-time).
                   For the first operation (first_pnl - spread%/2 or tc_perc/2) it's ok, because if I gain 3% and there
                   is an half spread of 1%, at the end of the day I gained 2%.
                   In the other hand, for the end calculation (end_pnl - spread%/2 or tc_perc/2) we're underestimating
                   the cost, because the 1% of half spread would be discounted from the final value of the trade.
                   e.g.: if I invested 1$ and at the end I have 1.5$, the 1% of half spread would be discounted from
                         $1.5, not from the end_pnl point-in time.
            Example: start 1$, final value 1.5$ (i.e. 50% gross pnl), half spread 1%,
                        ---> (1-1%) * (1.5) * (1-1%) = 1.47015 --> 47.05% real Net-PNL
                        ---> gross_pnl_series = [25%, 20%] ---> final_comp_pnl = 50% (like before)
                        ---> net_pnl_series = [24%, 19%] ---> 47.56% Net-PNL (+0.56% then Real)
        ----------------------------------------------------------------------------------------------------------------
        :param s_signals: pd.Series, with {-1: short, 0: neutral, 1: long}
        :param data: pd.DataFrame, with OHLC and spread(%) (or bidask). spread to True to use spread(%)
        :param spread: bool, True to compute net_pnl using spread(%)
        :param tc_perc: Optional[float], if you want to use a general % spread for all periods
        :return: pd.Series
        """
        if spread and tc_perc:
            raise Exception("You cannot compute net-pnl using both spread and tc_percent. Choose one of them.")
        s_tp = s_signals.diff().abs()  # turning points
        s_gross_pnl = s_signals.shift(1).mul(data['close'].pct_change())

        # spread(%)
        if spread:
            # open_spread because I receive a signal at the end of bar, thus I make a trade at the open of the next one
            s_tc = data['open_spread'] / 2
            s_tc = s_tp.shift(1).mul(s_tc)  # shift(1) because the signal in t is traded at t + 1
            net_pnl = s_gross_pnl - s_tc
        else:
            s_tc = -s_tp * (tc_perc / 2)
            net_pnl = s_gross_pnl + s_tc

        return pd.DataFrame({'net_pnl': net_pnl, 'gross_pnl': s_gross_pnl}, index=s_gross_pnl.index)

    @staticmethod
    def get_cumulative_pnl(s_pnl: pd.Series, comp: bool) -> pd.Series:
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

    @staticmethod
    def __get_start_end_trades(s_signals: pd.Series) -> Tuple[list, list]:
        """
        Returns a List of (start, end) date for the trades made.
        :param s_signals: pd.Series, with signals {-1, 0, 1} LAGGED-1, because the signal in t will use in t + 1
        :return: List[list]
        """
        signals = Utils.from_series_to_numpy(s_signals)
        idx = s_signals.index

        starts = []
        ends = []
        trade_open = False

        for t in range(len(signals)):

            previous = signals[t - 1] if t > 0 else None
            current = signals[t]

            # new trade opened
            if not trade_open:
                if (current == 1) or (current == -1):
                    trade_open = True
                    starts.append(idx[t])
                    continue  # to avoid entering in the "trade closed" evaluation

            # trade closed
            if trade_open:
                if current != previous:
                    # if current = 0 the trade was closed, otherwise there was a change side, from 1 to -1 and viceversa
                    if current == 0:
                        trade_open = False
                    ends.append(idx[t])

            # last signal
            if t == len(signals) - 1:
                # if there wasn't opened trade
                if current == previous == 0:
                    pass
                # otherwise it considers the trade closed at the end of the period
                else:
                    ends.append(idx[t])

        return starts, ends

    @classmethod
    def get_trade_history(
            cls,
            s_signals: pd.Series,
            data: pd.DataFrame,
            real_start_end: bool = True,
            spread: bool = True,
            tc_perc: Optional[float] = None,
            checks: bool = True
    ) -> pd.DataFrame:
        """
        Return a pd.DataFame with the trades-history. The fields it returns are:
            - start_date
            - end_date
            - leg: {1: long; 0: neutral; -1: short}
            - duration (in terms of bins)
            - entry_price
            - exit_price
            - gross_pnl: pnl excluding transaction costs
            - net_pnl: pnl including transaction costs
            - tc: transaction costs
        :param s_signals: pd.Series, with signals {-1, 0, 1}
        :param data: pd.Series, with OHLC and spread% or bidask prices
        :param real_start_end: bool, True if you want the real start/end (i.e. signal in t is executed in t+1). False
                                     otherwise (eg to avoid lookahead bias in get mkt features for feeding a ML model)
        :param spread: bool, True to compute net_pnl using spread(%)
        :param tc_perc: Optional[float], if you want to use a general % spread for all periods
        :param checks: bool, True if you want to run safe checks within the function
        :return: pd.DataFrame
        """
        signals = s_signals.tz_localize(None)
        data_ = data.tz_localize(None)

        shift = 1 if real_start_end else 0
        starts, ends = cls.__get_start_end_trades(signals.shift(shift))

        if checks:
            if spread and tc_perc:
                raise Exception("You cannot compute net-pnl using both spread and tc_percent. Choose one of them.")
            Utils.check_len(data, s_signals)
            if len(starts) != len(ends):
                raise UserWarning("Start dates and End dates have different length. Strange...")

        if starts:
            trade_hist = pd.DataFrame({'start': starts, 'end': ends}, index=np.arange(len(starts)))
            trade_hist['leg'] = [signals.loc[start] for start in starts]
            trade_hist['duration'] = [len(signals.loc[start: end]) for start, end in zip(starts, ends)]
            trade_hist['entry_p'] = [data_.loc[start, 'open'] for start in starts]
            trade_hist['exit_p'] = [data_.loc[end, 'open'] for end in ends]
            trade_hist['gross_pnl'] = (trade_hist['exit_p'] / trade_hist['entry_p'] - 1) * trade_hist['leg']
            if spread:
                open_tc = data_.loc[trade_hist['start'].values, 'open_spread'].values
                close_tc = data_.loc[trade_hist['end'].values, 'open_spread'].values
                # for a full comprehension of the calculus, read the docs of the func RiskSignal.calc_pnl_from_signals()
                trade_hist['net_pnl'] = (1 - open_tc) * (1 + trade_hist['gross_pnl']) * (1 - close_tc) - 1
            elif tc_perc:
                trade_hist['net_pnl'] = (1 - tc_perc/2) * (1 + trade_hist['gross_pnl']) * (1 - tc_perc/2) - 1

            trade_hist['tc'] = trade_hist['net_pnl'] - trade_hist['gross_pnl']
            # final check
            if checks:
                if not all((trade_hist['net_pnl'] < trade_hist['gross_pnl']).values):
                    raise UserWarning("There occured a strange thing. Net_PNL isn't less then Gross_PNL in all cases.")
        else:
            trade_hist = pd.DataFrame(
                columns=['start', 'end', 'leg', 'duration', 'entry_p', 'exit_p', 'gross_pnl', 'net_pnl']
            )
        return trade_hist

