import pandas as pd
import numpy as np

from typing import Union, List, NoReturn


class Utils:

    @staticmethod
    def check_len(*args: Union[pd.Series, pd.DataFrame]) -> NoReturn:
        """
        Checks if the length of all the elements is equal

        :param args: Union[pd.Series, pd.DataFrame]
        :return: None or Exception
        """
        eval_str = f'{"==".join([str(len(arg)) for arg in args])}'
        if not eval(eval_str):
            raise Exception("inputs must have the same length")

    @staticmethod
    def check_min_obs(*args: Union[pd.Series, pd.DataFrame], min_len: int) -> NoReturn:
        """
        Checks if all the elements in args have at least N elements

        :param args: Union[pd.Series, pd.DataFrame]
        :return: None or Exception
        """
        for arg in args:
            if len(arg) < min_len:
                raise Exception(f"inputs must have at least {min_len}obs")

    @staticmethod
    def from_series_to_numpy(*args: pd.Series) -> Union[List[np.array], np.array]:
        """
        Transform pd.Series to np.array

        :param args: pd.Series
        :return: np.array if it was passed only one pd.Series, List[np.array] otherwise
        """
        out = []
        for arg in args:
            out.append(arg.to_numpy())
        if len(out) == 1:
            return out[0]
        else:
            return out

    @staticmethod
    def from_signals_to_tp(signals: pd.Series, continuos: bool) -> pd.DataFrame:
        """
        Returns turning points of a signals strategy

        :param signals: pd.Series, with signals {-1, 0, 1}
        :param continuos: bool, True for Trend-Following, False for others (eg RSI reversal that it doesn't trade always)
        :return: pd.DataFrame
        """
        sign = Utils.from_series_to_numpy(signals)
        open_sign = np.zeros(len(signals))
        close_sign = np.zeros(len(signals))

        for t in range(1, len(signals)):

            if not continuos:
                if sign[t - 1] == 0:
                    if sign[t] == 1:
                        open_sign[t] = 1
                    elif sign[t] == -1:
                        open_sign[t] = -1

                elif (sign[t - 1] == 1) or (sign[t - 1] == -1):
                    if sign[t] == 0:
                        close_sign[t] = 1

        return pd.DataFrame({'open_sign': open_sign, 'close_sign': close_sign}, index=signals.index)

    @staticmethod
    def align_series(*args: pd.Series) -> List[pd.Series]:
        df = pd.concat([*args], axis=1).dropna()
        df.columns = np.arange(df.shape[1])
        return [df[col] for col in df.columns]
