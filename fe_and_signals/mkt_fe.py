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
    mkt_fe = pd.concat([mkt_fe.loc[start] for start in trades_start_dates], axis=1).transpose()

    return mkt_fe
