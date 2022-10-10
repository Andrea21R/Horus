import matplotlib.pyplot as plt
from typing import Optional, Tuple
import pandas as pd

from fe_and_signals.utils import Utils
from risk.risk_signal import RiskSignal


class Graphs:

    @staticmethod
    def buy_sell_on_price(
            s_signals: pd.Series,
            continuous_signals: bool,
            s_close: pd.Series,
            s_sl: Optional[pd.Series] = None,
            ax: Optional[plt.Axes] = None,
            large_data_constraint: bool = False,
            show: bool = False
    ) -> plt.Axes:
        """
        Returns an instance of Axes with a Graph containing the Price Action and the buy/sell actions (highlighted by
        a scatter's markers)
        :param s_signals: pd.Series, with signals {-1, 0, 1}
        :param continuous_signals: bool, True for continuos signals (like Trend-Following), False otherwise (like RSI)
        :param s_close: pd.Series, closing price
        :param s_sl: Optional[pd.Series], series of stop loss
        :param ax: Optional[plt.Axes], specific axes to assign the Graph
        :param large_data_constraint: bool, True if you want to hide the scatter's markers for buy/sell action when
                                            the dataset has more than 50.000 observation
        :param show: bool, True to show the graph, False otherwise
        :return: plt.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        tp_sign = Utils.from_signals_to_tp(s_signals, continuos=continuous_signals)
        open_sign = tp_sign['open_sign']
        close_sign = tp_sign['close_sign']

        ax.plot(s_close)
        ax.grid(linestyle='--', color='silver')
        legend = ['Close']

        scatter_trigger = len(s_close) < 50_000 if large_data_constraint else True

        if scatter_trigger:
            idx = open_sign.index
            ax.scatter(x=idx, y=s_close.where(open_sign > 0), marker='^', color='green', zorder=10)
            ax.scatter(x=idx, y=s_close.where(open_sign < 0), marker='v', color='red', zorder=10)
            ax.scatter(x=idx, y=s_close.where(close_sign == 1), marker='o', color='black', zorder=10)
            if isinstance(s_sl, pd.Series):
                ax.plot(s_sl, linestyle='--', color='red', linewidth=1)
            legend.extend(['Long', 'Short', 'Close_trade', 'SL'])
        ax.legend(legend)
        ax.set_ylabel('Price Action', fontweight='bold')

        if show:
            plt.show()
        return ax

    @staticmethod
    def reversal_indicator(
            indicator: pd.Series,
            open_cv: Tuple[float, float],
            close_cv: Tuple[float, float],
            indicator_name: Optional[str] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = False
    ) -> plt.Axes:
        """
        Returns a Graph showing the indicator path and the thresholds
        :param indicator: pd.Series, like RSI
        :param open_cv: Tuple[float, float], Critical Values (i.e. thresholds) for open trades
        :param close_cv: Tuple[float, float], Critical Values (i.e. thresholds) for close trades
        :param indicator_name: Optional[str]
        :param ax: Optional[plt.Axes], specific axes to assign the Graph
        :param show: bool, True to show the graph, False otherwise
        :return: plt.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        ax.plot(indicator, color='violet')
        ax.axhline(open_cv[0], color='black', linestyle='--')
        ax.axhline(open_cv[1], color='black', linestyle='--')
        ax.axhline(close_cv[0], color='green', linestyle='--')
        ax.axhline(close_cv[1], color='green', linestyle='--')
        ax.grid(linestyle='--', color='silver')
        ax.set_ylabel(f'{"indicator" if not indicator_name else indicator_name} & Thresholds', fontweight='bold')

        if show:
            plt.show()
        else:
            return ax

    @staticmethod
    def pnl_graph(
            data: pd.DataFrame,
            s_signals: pd.Series,
            spread: bool,
            tc_perc: Optional[float] = None,
            ax: Optional[plt.Axes] = None,
            show: bool = False
    ) -> plt.Axes:
        """
        Returns a Graph with the Cumulative PNL (Gross and/or Net) from a signals
        :param data: pd.DataFrame, with OHLC and spread%
        :param s_signals: pd.Series, with signals {-1, 0, 1}
        :param spread: bool, True to compute net_pnl using spread(%)
        :param tc_perc: if you want to use a general % spread for all periods
        :param ax: Optional[plt.Axes], specific axes to assign the Graph
        :param show: bool, True to show the graph, False otherwise
        :return: plt.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        s_pnl = RiskSignal.get_pnl_from_signals(s_signals, data, spread=spread, tc_perc=tc_perc)
        s_net_cum_pnl = RiskSignal.get_cumulative_pnl(s_pnl['net_pnl'], comp=True)
        s_gross_cum_pnl = RiskSignal.get_cumulative_pnl(s_pnl['gross_pnl'], comp=True)
        ax.plot(s_net_cum_pnl, color='red', linewidth=1.5)
        legend = ['Net PNL', 'Gross PNL']
        ax.plot(s_gross_cum_pnl, color='orange', linewidth=1.5)
        ax.legend(legend)
        ax.grid(linestyle='--', color='silver')
        ax.set_ylabel('Cumulative PNL', fontweight='bold')

        if show:
            plt.show()
        return ax
