import os
import pandas as pd
from itertools import product


class FeGenerator:

    @staticmethod
    def get_fe_by_config(data: pd.DataFrame, fe_pars: dict, verbose: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame of features (n_sample, n_features), according to the fe_pars (type:dict), i.e. a config.
        ----------------------------------------------------------------------------------------------------------------
        :param data: pd.DataFrame, with at least the columns needed for features computation
        :param fe_pars: dict,
        :param verbose: bool, True if you want to print the current calculation in console
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

            for idx, combo in enumerate(pars_combo):
                if verbose:
                    print(f'Func_name: {func_name}.{idx}; args: {combo}')
                tgt_data = data[func_items['data_cols']] if len(func_items['data_cols']) > 1 else data[func_items['data_cols'][0]]
                fe[f"{func_name}.{idx}"] = func_items['func'](tgt_data, **combo)

        fe = pd.concat(fe, axis=1)
        fe.columns = [
            col1
            if isinstance(fe.columns.get_level_values(0).get_loc(col1), int)
            else f"{col1}.{col2}"

            for col1, col2 in fe.columns
        ]
        return fe

    @classmethod
    def get_fe_by_config_for_target_dates(
            cls,
            data: pd.DataFrame,
            fe_pars: dict,
            target_dates: list,
            verbose: bool = False
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of features, built following the configuration from fe_pars and the output will have the
        target dates on the row. Useful for logit for trade
        ----------------------------------------------------------------------------------------------------------------
        :param data: pd.DataFrame, with at least the columns needed for features computation
        :param fe_pars: dict
        :param target_dates: list
        :param verbose: bool, True if you want to print the current calculation in console
        :return: pd.DataFrame
        """
        fe = cls.get_fe_by_config(data=data, fe_pars=fe_pars, verbose=verbose)
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
    from fe_and_signals.fe_config import fe_config_for_trades_logit

    files_path =  os.path.dirname(os.getcwd()) + "/test_data/"
    trade_hist_file_name = "bb_trades_history_EURUSD.pkl"
    trade_hist = pd.read_pickle(files_path + trade_hist_file_name)
    data_file_name = "EURUSD2022.parquet"
    data = pd.read_parquet(files_path + data_file_name)

    mkt_fe = FeGenerator.get_fe_by_config_for_target_dates(
        target_dates=trade_hist['start'],
        data=data,
        fe_pars=fe_config_for_trades_logit,
        verbose=True
    )
    date_str = str(dt.datetime.now())[:16].replace(":", "").replace("-", "").replace(" ", "_")
    output_file_name = "mkt_fe_EURUSD_BB.parquet"
    mkt_fe.to_parquet(files_path + output_file_name)
