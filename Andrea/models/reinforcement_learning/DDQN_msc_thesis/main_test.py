import os.path
import matplotlib as mpl

from DDQN_trading import Tester, Analyst, plot_final_graph


if __name__ == "__main__":

    import warnings
    warnings.simplefilter("ignore", FutureWarning)
    mpl.use('TkAgg')

    model_name = "ddqn_target_ann.h5"
    model_path = os.path.join(os.getcwd(), "results", "2023_02_19__10_52", model_name)
    ticker = "AAPL_10y_1d"
    start_end = ("2021-01-04", "2021-06-01")

    tester = Tester(
        model_path=model_path,
        ticker=ticker,
        start_end=start_end
    )

    strategy_gross_rets, strategy_net_rets = tester.get_test_returns(tc_bps=0.0025)
    strategy_gross_cumrets = tester.get_cumulative_rets(strategy_gross_rets, comp=True)
    strategy_net_cumrets = tester.get_cumulative_rets(strategy_net_rets, comp=True)
    plot_final_graph(
        strategy_gross_cumrets,
        strategy_net_cumrets,
        tester.get_cumulative_rets(tester.test_data['returns'], comp=True),
        tester.predict_actions()
    )

    analyst = Analyst(
        strategy_gross_rets=strategy_gross_rets,
        strategy_net_rets=strategy_net_rets,
        benchmark_rets=tester.test_data['returns']
    )
    performance_analysis = analyst.get_performance_analysis()

    print(performance_analysis)
