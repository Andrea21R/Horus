import pandas as pd
import numpy as np


def get_label_and_combo_for_one_period(
        data: pd.DataFrame,
        mkt_fe_config: dict,
        indicator_state_spaces: dict
) -> pd.DataFrame:
    pass

# 1. devo creare un config example per creare le combinazioni dell'indicatore (come feci nel GA in qresearch)
# 2. qui dentro creo tutte le combinazioni di pars per generare gli indicatori
# 3. definisco la mia fitness func calcolata per ogni indicatore su un dato periodo (eg 1-day)
# 4. genero le features di mkt per ogni starting period
# 5. preparo il mio dataset di training
#   - Y: fitness func per il periodo target
#   - X: control variables (parametri indicatore) e mkt features
# 6. Faccio train-test del modello e salvo risultati
# 7. Ripeto 4-6 con approccio Walk-Forward Optimization
# 8. Valuto com'è andato
