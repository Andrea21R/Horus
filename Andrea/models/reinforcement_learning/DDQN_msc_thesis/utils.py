import numpy as np
import datetime as dt
from pathlib import Path

import pandas as pd


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


def track_results(
        episode: int,
        nav_ma_100: float,
        nav_ma_10: float,
        market_nav_100: float,
        market_nav_10: float,
        win_ratio: float,
        total: float,
        epsilon: float
):
    # time_ma = np.mean([episode_time[-100:]])
    # T = np.sum(episode_time)

    template = '{:>4d} | {} | {} | Agent: MA(100):{:>6.1%}; MA(10):{:>6.1%} | '
    template += 'Market: MA(100):{:>6.1%}; MA(10):{:>6.1%} | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(
        episode,
        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        format_time(total),
        nav_ma_100 - 1,
        nav_ma_10 - 1,
        market_nav_100 - 1,
        market_nav_10 - 1,
        win_ratio,
        epsilon)
    )


def get_timestamp_for_file() -> str:
    return str(dt.datetime.now())[:16].replace("-", "_").replace(" ", "__").replace(":", "_")


def store_results(config: dict, results: pd.DataFrame, path) -> None:
    dt_str = get_timestamp_for_file()
    writer = pd.ExcelWriter(path / f'{dt_str}_results.xlsx', engine='xlsxwriter')
    pd.Series(config).to_excel(writer, sheet_name="configuration")
    results.to_excel(writer, sheet_name="results")
    results.to_csv(path / f'{dt_str}_results.xlsx')
    writer.close()


def get_results_path():
    return Path('results', get_timestamp_for_file())
