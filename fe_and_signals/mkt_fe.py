import os

import pandas as pd
from itertools import product


def get_mkt_features_by_trades(trades_start_dates: list, data: pd.DataFrame, fe_pars: dict) -> pd.DataFrame:
    """
    Returns market features, according to the start_date from a list of trades.
    Think if it would be better to move it into an autofe.py file
    """
    mkt_fe = {}
    for func_name, func in fe_pars.items():
        pars_combo = product(*list(func['args'].values()))
        func_pars_k = func['args'].keys()
        pars_combo = [
            {
                k: combo[idx]
                for k, idx in zip(func_pars_k, range(len(func_pars_k)))
            }
            for combo in pars_combo
        ]
        print(func_name)

        for idx, combo in enumerate(pars_combo):
            tgt_data = data[func['data_cols']] if len(func['data_cols']) > 1 else data[func['data_cols'][0]]
            mkt_fe[f"{func_name}.{idx}"] = func['func'](tgt_data, **combo)

    mkt_fe = pd.DataFrame(mkt_fe, index=data.index)
    # NOT EFFICIENT. FIND NEW WAY TO DO IT
    mkt_fe = pd.concat(
        [
            mkt_fe.iloc[mkt_fe.index.get_loc(start) - 1]  # -1 because signal[t] is a trade in t+1, but mkt_fe are available in t-1
            for start in trades_start_dates
        ],
        axis=1
    ).transpose()

    return mkt_fe


if __name__ == "__main__":

    import datetime as dt
    from fe_and_signals.mkt_fe_config import fe_pars

    files_path =  os.path.dirname(os.getcwd()) + "/test_data/"
    trade_hist_file_name = "bb_trades_history_EURUSD.pkl"
    trade_hist = pd.read_pickle(files_path + trade_hist_file_name)
    data_file_name = "EURUSD2022.parquet"
    data = pd.read_parquet(files_path + data_file_name)

    mkt_fe = get_mkt_features_by_trades(
        trades_start_dates=trade_hist['start'],
        data=data,
        fe_pars=fe_pars
    )

    date_str = str(dt.datetime.now())[:16].replace(":", "").replace("-", "").replace(" ", "_")
    output_file_name = "mkt_fe_EURUSD_BB.parquet"
    mkt_fe.to_parquet(files_path + output_file_name)
