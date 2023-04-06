import pandas as pd
from typing import *


def generate_features_dataframe(dataset: Union[pd.Series, pd.DataFrame], parametrization: dict) -> pd.DataFrame:

    fe = {}

    for k,v in parametrization.items():
        func = v['func']
        tgt_cols = v['tgt_cols'] if len(v['tgt_cols']) > 1 else v['tgt_cols'][0]
        tmp_data = dataset[tgt_cols]
        kwargs = v['kwargs']

        fe[k] = func(tmp_data, **kwargs)
    fe_df = pd.concat(fe, axis=1)
    cols = [f"{a}.{b}" for a, b in fe_df.columns]
    fe_df.columns = [
        col.split(".")[0]
        if flag == False else col
        for col, flag
        in zip(
            cols,
            fe_df.columns.get_level_values(0).duplicated(keep=False)
        )
    ]
    return fe_df
