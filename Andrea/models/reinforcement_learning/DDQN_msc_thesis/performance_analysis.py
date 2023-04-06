import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Analyst():

    def __init__(self, strategy_gross_rets: pd.Series, strategy_net_rets: pd.Series, benchmark_rets: pd.Series):

        self.strategy_gross_rets = strategy_gross_rets
        self.strategy_net_rets = strategy_net_rets
        self.benchmark_rets = benchmark_rets

    @staticmethod
    def _calc_total_return(rets: pd.Series) -> float:
        return rets.add(1).cumprod().sub(1).iloc[-1]

    @staticmethod
    def _calc_avg_ann_rets(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return rets.mean() * 252

    @staticmethod
    def _calc_avg_ann_volatility(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return rets.std() * np.sqrt(252)

    @staticmethod
    def _calc_max_dd(rets: pd.Series) -> float:
        equity_curve = (1 + rets).cumprod()
        return float((equity_curve / equity_curve.cummax() - 1).min())

    @staticmethod
    def _calc_ann_semivolatility(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return float(rets[rets < 0].std() * np.sqrt(252))

    @staticmethod
    def _calc_daily_skew(rets: pd.Series) -> float:
        return rets.skew()

    @staticmethod
    def _calc_daily_kurt(rets: pd.Series) -> float:
        return rets.kurt()

    @staticmethod
    def _calc_ann_sharpe(rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return (rets.mean() / rets.std()) * np.sqrt(252)

    @classmethod
    def _calc_ann_sortino(cls, rets: pd.Series) -> float:
        """
        Assuming daily returns
        """
        return (rets.mean() * 252) / cls._calc_ann_semivolatility(rets)

    @staticmethod
    def _calc_best_day(rets: pd.Series) -> float:
        return float(rets.max())

    @staticmethod
    def _calc_worst_day(rets: pd.Series) -> float:
        return float(rets.min())

    @staticmethod
    def _calc_var(rets: pd.Series, parametric: bool, var_confidence: float = 0.99) -> float:
        if parametric:
            return -rets.std() * 2.33
        else:
            return rets.quantile(q=1-var_confidence, interpolation='linear')

    def get_performance_analysis(self):
        results = []
        for series in (self.strategy_gross_rets, self.strategy_net_rets, self.benchmark_rets):
            tmp_res = pd.Series(
                {
                    'Total Return':             self._calc_total_return(series),
                    'Avg Ann. Return':          self._calc_avg_ann_rets(series),
                    'Avg Ann. Volatility':      self._calc_avg_ann_volatility(series),
                    'Avg Ann. Semi-Volatility': self._calc_ann_semivolatility(series),
                    'Max Drawdown':             self._calc_max_dd(series),
                    'VaR(99%) Parametric':      self._calc_var(series, parametric=True),
                    'VaR(99%) Not Parametric':  self._calc_var(series, parametric=False, var_confidence=0.99),
                    'Daily Skewness':           self._calc_daily_skew(series),
                    'Daily Kurtosis':           self._calc_daily_kurt(series),
                    'Ann. Sharpe':              self._calc_ann_sharpe(series),
                    'Ann. Sortino':             self._calc_ann_sortino(series),
                    'Best Day':                 self._calc_best_day(series),
                    'Worst Day':                self._calc_worst_day(series)
                }
            )
            results.append(tmp_res)

        return pd.concat(results, axis=1, keys=['strategy (gross)', 'strategy (nt)', 'benchmark'])


def plot_final_graph(
        strategy_gross_cumrets: pd.Series,
        strategy_net_cumrets: pd.Series,
        benchmark_cumrets: pd.Series,
        actions: pd.Series
) -> None:

    fig, axs = plt.subplots(nrows=2)
    plt.suptitle('DDQN-strategy VS Benchmark (long-only)', fontsize=16, fontweight='bold')

    axs[0].plot(strategy_gross_cumrets)
    axs[0].plot(strategy_net_cumrets)
    axs[0].plot(benchmark_cumrets, color='green')
    axs[0].grid(linestyle='--', color='silver')
    axs[0].set_ylabel('Cumulative Returns (compounded)', fontweight='bold')
    axs[0].legend(['DDQN-strategy (Gross)', 'DDQN-strategy (Net)', 'Long-Only'])

    actions = actions.copy(deep=True)
    axs[1].step(x=actions.index, y=actions.values, color='black', linewidth=1)
    axs[1].legend(['position'])
    axs[1].grid(linestyle='--', color='silver')
    axs[1].set_ylabel('{-1: short, 0: neutral, 1: long}', fontweight='bold')
    axx = axs[1].twinx()
    axx.plot(benchmark_cumrets, color='green')
    axx.set_ylabel('Price path', fontweight='bold')

    plt.show()
