import os
import pandas as pd
from itertools import product


class FeGenerator:

    @staticmethod
    def get_fe_by_config(data: pd.DataFrame, fe_pars: dict) -> pd.DataFrame:
        """
        Returns a DataFrame of features (n_sample, n_features), according to the fe_pars (type:dict), i.e. a config.
        ----------------------------------------------------------------------------------------------------------------
        :param data: pd.DataFrame, with at least the columns needed for features computation
        :param fe_pars: dict,
        :return: pd.DataFrame
        """
        fe = {}
        for func_name, func_items in fe_pars.items():
            pars_combo = product(*list(func_items['args'].values()))
            func_pars_k = func_items['args'].keys()
            pars_combo = [
                {
                    k: combo[idx]
                    for k, idx in zip(func_pars_k, range(len(func_pars_k)))
                }
                for combo in pars_combo
            ]
            print(func_name)

            for idx, combo in enumerate(pars_combo):
                tgt_data = data[func_items['data_cols']] if len(func_items['data_cols']) > 1 else data[func_items['data_cols'][0]]
                fe[f"{func_name}.{idx}"] = func_items['func'](tgt_data, **combo)

        return pd.DataFrame(fe, index=data.index)

    @classmethod
    def get_fe_by_config_for_target_dates(cls, data: pd.DataFrame, fe_pars: dict, target_dates: list) -> pd.DataFrame:
        """
        Returns a DataFrame of features, built following the configuration from fe_pars and the output will have the
        target dates on the row. Useful for logit for trade
        ----------------------------------------------------------------------------------------------------------------
        :param data: pd.DataFrame, with at least the columns needed for features computation
        :param fe_pars: dict
        :param target_dates: list
        :return: pd.DataFrame
        """
        fe = cls.get_fe_by_config(data=data, fe_pars=fe_pars)
        fe = pd.concat(
            [
                fe.iloc[fe.index.get_loc(start) - 1]  # -1 because signal[t] is a trade in t+1, but mkt_fe are available in t-1
                for start in target_dates
            ],
            axis=1
        ).transpose()

        return fe


if __name__ == "__main__":

    import datetime as dt
    from fe_and_signals.mkt_fe_config import fe_pars

    files_path =  os.path.dirname(os.getcwd()) + "/test_data/"
    trade_hist_file_name = "bb_trades_history_EURUSD.pkl"
    trade_hist = pd.read_pickle(files_path + trade_hist_file_name)
    data_file_name = "EURUSD2022.parquet"
    data = pd.read_parquet(files_path + data_file_name)

    mkt_fe = FeGenerator.get_fe_by_config_for_target_dates(target_dates=trade_hist['start'], data=data, fe_pars=fe_pars)

    date_str = str(dt.datetime.now())[:16].replace(":", "").replace("-", "").replace(" ", "_")
    output_file_name = "mkt_fe_EURUSD_BB.parquet"
    mkt_fe.to_parquet(files_path + output_file_name)
