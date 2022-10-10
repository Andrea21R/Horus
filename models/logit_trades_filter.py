import numpy as np
import pandas as pd
from itertools import product
from statsmodels.api import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from typing import Tuple, NoReturn, Optional, Union, Dict

"""
Da capire:
- fit_intercept quando instanzio la Logit, aggiunge da sola la costante o la devo aggiungere io?
"""


class LogitTrades:

    def __init__(self):
        self.regularization_par = None  # Smaller values specify stronger regularization
        self.fit_intercept = None
        self.Y_train = None
        self.X_train = None
        self.model: Union[LogitTrades, None] = None

    @staticmethod
    def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / df.std()

    def fit(
            self,
            Y_train: pd.Series,
            X_train: pd.DataFrame,
            standardize_X: bool = True,
            penalty: str = "l2",
            regularization_par: float = 1,
            fit_intercept: bool = True,
            n_jobs: int = -1,
            max_iter: int = 100
    ):

        model = LogisticRegression(
            C=regularization_par,  # lambda_par
            penalty=penalty,
            fit_intercept=fit_intercept,
            random_state=42,
            n_jobs=n_jobs,
            max_iter=max_iter
        )
        X = self.standardize_dataset(X_train) if standardize_X else X_train
        X = add_constant(X) if fit_intercept else X

        self.Y_train = Y_train.astype(int)
        self.X_train = X
        self.fit_intercept = fit_intercept
        self.regularization_par = regularization_par

        model.fit(X, Y_train)
        self.model = model

    def __check_X_cols(self, X_test: pd.DataFrame) -> Union[pd.DataFrame, NoReturn]:
        try:
            return X_test[self.X_train.columns]
        except:
            x_train_cols = set(self.X_train.columns)
            x_test_cols = set(X_test)

            missed = x_train_cols.difference(x_test_cols)
            excess = x_test_cols.difference(x_train_cols)

            if missed:
                print(f"Missed the following features: {missed}")
            if excess:
                print(f"The following columns are in excess: {excess}")

    def predict(
            self,
            X_test: pd.DataFrame,
            standardize_X: bool = True,
            show_graph: bool = False
    ) -> pd.Series:

        if not self.model:
            raise Exception("For getting Logit parameters, you have to fit the model")
        X = self.standardize_dataset(X_test) if standardize_X else X_test
        X = add_constant(X) if self.fit_intercept else X
        X = self.__check_X_cols(X)

        # params is a dict, then df.mul(dict) match the columns by names
        probabilities = pd.Series(self.model.predict_proba(X)[:, 1], index=X.index, name='P(Y=1)')

        if show_graph:
            # da rifare
            probabilities.plot.hist()
            plt.axhline(0.5, color='black', linestyle='--', linewidth=3)
            plt.title('Probabilities of Profitability | Logit-Output', fontweight='bold')
            plt.grid(color='silver', linestyle='--')
            plt.tight_layout()
            plt.show()

        return probabilities

    @staticmethod
    def mask_y_pred(y_pred: pd.Series, threshold: float) -> pd.Series:
        return y_pred.mask(y_pred > threshold, 1).mask(y_pred <= threshold, 0).astype(int)

    def get_accuracy_score(
            self,
            y_true: pd.Series,
            y_pred: pd.Series,
            threshold: float,
    ) -> float:
        y_pred = self.mask_y_pred(y_pred, threshold)
        return accuracy_score(y_true, y_pred)

    def get_auroc_score(
            self,
            y_true: pd.Series,
            y_pred: pd.Series,
            threshold: float,
    ) -> float:
        y_pred = self.mask_y_pred(y_pred, threshold)
        return roc_auc_score(y_true, y_pred)

    def get_all_metrics(
            self,
            y_true: pd.Series,
            y_pred: pd.Series,
            threshold: float,
    ) -> Dict[str, float]:
        y_pred = self.mask_y_pred(y_pred, threshold)
        return {
            'accuracy_score': accuracy_score(y_true=y_true, y_pred=y_pred),
            'auroc_score': roc_auc_score(y_true=y_true, y_score=y_pred)
        }

    def get_confusion_matrix(
            self,
            y_true: pd.Series,
            y_pred: pd.Series,
            threshold: float,
    ) -> pd.DataFrame:
        y_pred = self.mask_y_pred(y_pred=y_pred, threshold=threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        matrix = pd.DataFrame(index=['Positive', 'Negative'], columns=['Positive', 'Negative'])
        matrix.columns.name = "Actual-Values"
        matrix.index.name = "Predicted-Values"

        matrix.loc['Positive', 'Positive'] = tp
        matrix.loc['Positive', 'Negative'] = fp
        matrix.loc['Negative', 'Positive'] = fn
        matrix.loc['Negative', 'Negative'] = tn

        return matrix

# ----------------------------------------------------------------------------------------------------------------------
def logit_fine_tuning(
        Y_train: pd.Series,
        Y_test: pd.Series,
        Y_test_pnl: pd.Series,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        num_lambdas: int,
        num_thresholds: int,
        standardize_X: bool = True,
        fit_intercept: bool = True,
        max_iter: int = 100,
        n_jobs: int = -1
) -> pd.DataFrame:

    lambdas = np.linspace(0.001, 1, num_lambdas)
    thresholds = np.linspace(0.1, 0.9, num_thresholds)

    logit = LogitTrades()

    metrics = {}
    metrics['original'] = {
        'accuracy': np.nan,
        'auroc': np.nan,
        'gain_loss': abs(Y_test_pnl[Y_test_pnl > 0].mean() / Y_test_pnl[Y_test_pnl < 0].mean()),
        'sharpe': Y_test_pnl.mean() / Y_test_pnl.std(),
        'trades_excluded': np.nan,
        'tp/pos': np.nan, 'tn/neg': np.nan,
        'tot_pnl': Y_test_pnl.sum()
    }

    for lambda_ in lambdas:
        # I re-train the model for each lambda
        logit.fit(
            Y_train=Y_train,
            X_train=X_train,
            regularization_par=lambda_,
            fit_intercept=fit_intercept,
            n_jobs=n_jobs,
            max_iter=max_iter
        )
        y_pred = logit.predict(X_test=X_test, standardize_X=standardize_X, show_graph=False)

        for threshold in thresholds:
            key = f"l:{lambda_}|thr:{threshold}"
            metrics[key] = {}

            metrics[key]['accuracy'] = logit.get_accuracy_score(y_true=Y_test, y_pred=y_pred, threshold=threshold)
            metrics[key]['auroc'] = logit.get_auroc_score(y_true=Y_test, y_pred=y_pred, threshold=threshold)

            mask = logit.mask_y_pred(y_pred=y_pred, threshold=threshold).replace(0, np.nan)
            pnl_masked = pd.Series(Y_test_pnl.values * mask.values).dropna()  # simulate if we only hold the trade suggested by logit model
            metrics[key]['gain_loss'] = abs(pnl_masked[pnl_masked > 0].mean() / pnl_masked[pnl_masked < 0].mean())
            metrics[key]['sharpe'] = pnl_masked.mean() / pnl_masked.std()
            metrics[key]['trades_excluded'] = (len(Y_test_pnl) - len(pnl_masked)) / len(Y_test_pnl)

            confusion = logit.get_confusion_matrix(y_true=Y_test, y_pred=y_pred, threshold=threshold)
            metrics[key]['tp/pos'] = confusion.loc['Positive', 'Positive'] / confusion.loc[:, 'Positive'].sum()
            metrics[key]['tn/neg'] = confusion.loc['Negative', 'Negative'] / confusion.loc[:, 'Negative'].sum()

            metrics[key]['tot_pnl'] = pnl_masked.sum()

    return pd.DataFrame(metrics).transpose()

def align_trades_pnl_and_mkt_fe(trades_pnl: pd.DataFrame, mkt_fe: pd.DataFrame) -> tuple:

    mkt_fe_ = mkt_fe.copy(deep=True)

    mkt_fe_.dropna(inplace=True)
    idx_mask = [True if idx in mkt_fe_.index.tz_localize(None) else False for idx in trades_pnl.index]

    return (mkt_fe_, trades_pnl.loc[idx_mask])

# ----------------------------------------------------------------------------------------------------------------------
# def print_pnl_stats(pnl: pd.Series) -> NoReturn:
#     print(f"Avg-PNL > 0: {round(pnl.loc[pos_idx].mean() / 1000)}k")
#     print(f"Avg-PNL < 0: {round(pnl.loc[neg_idx].mean() / 1000)}k")
#     print(' ')
#     print(f"GAIN-LOSS ratio: {abs(round(pnl.loc[pos_idx].mean() / pnl.loc[neg_idx].mean(), 2))}")
#     print(' ')
#     print(f'Mean: {round(pnl.mean() / 1000, 2)}k')
#     print(f'Vola: {round(pnl.std() / 1000, 2)}k')
#     print(f"Trade-Sharpe: {round(pnl.mean() / pnl.std(), 2)}")
#     print(' ')
#     print(f'Max: {round(pnl.max() / 1000, 2)}k')
#     print(f'Min: {round(pnl.min() / 1000, 2)}k')
#     print(' ')


if __name__ == "__main__":

    import os
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    path = os.path.dirname(os.getcwd()) + "/test_data/"

    trade_hist = pd.read_pickle(path + "rsi_trades_history_EURUSD.pkl")
    mkt_fe = pd.read_parquet(path + "mkt_fe_for_logit_EURUSD.parquet")
    trades_pnl = trade_hist['net_pnl']
    trades_pnl.index = trade_hist['start'].values

    mkt_fe, trades_pnl = align_trades_pnl_and_mkt_fe(trades_pnl, mkt_fe)

    # It's better to use dataset without index
    mkt_fe.index, trades_pnl.index = np.arange(len(mkt_fe)), np.arange(len(trades_pnl))

    train_size = 0.7
    test_size = 1 - train_size
    shuffle = True

    X_train, X_test = train_test_split(mkt_fe, train_size=train_size, test_size=test_size, shuffle=shuffle)
    Y = LogitTrades().mask_y_pred(trades_pnl, 0)
    Y_train, Y_test = train_test_split(Y, train_size=train_size, test_size=test_size, shuffle=shuffle)
    Y_pnl_train, Y_pnl_test = train_test_split(trades_pnl, train_size=train_size, test_size=test_size, shuffle=shuffle)

    result = logit_fine_tuning(
        Y_train=Y_train,
        Y_test=Y_test,
        Y_test_pnl=Y_pnl_test,
        X_train=X_train,
        X_test=X_test,
        num_lambdas=5,
        num_thresholds=50,
        standardize_X=True,
        fit_intercept=True,
        n_jobs=-1,
        max_iter=300
    )
    result.to_excel(r"C:\Users\andre\Dropbox\Horus\simulation_results\logit_trades_on_rsi_EURUSD_no_shuffle.xlsx")

    # fe_logit = TendersFeLogit(SQL_ALCHEMY_CONNECTION)
    # fe_logit.fit_logit()
    #
    #
    # X_test = fe_logit.X_train.drop('const', axis=1)
    # Y_test = fe_logit.Y_train
    # threshold = 0.4
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # pnl, fe = fe_logit.get_tenders_fe()
    # pos_idx = fe.where(fe['pnl_dummy'] == 1).dropna().index
    # neg_idx = fe.where(fe['pnl_dummy'] == 0).dropna().index
    # # ------------------------------------------------------------------------------------------------------------------
    # print('============================== LOGIT-STATISTICS ===========================================================')
    # print(f'THRESHOLD: {threshold}')
    # print('============================== LOGIT-STATISTICS ===========================================================')
    # print(' ')
    # print(f"Accuracy-Ratio: {fe_logit.get_accuracy_score(Y_test, X_test, threshold)}")
    # print(' ')
    # print(f"AUROC: {fe_logit.get_auroc_score(Y_test, X_test, threshold)}")
    # print(' ')
    # print('--- CONFUSION MATRIX')
    # print(fe_logit.get_confusion_matrix(Y_test, X_test, threshold))
    # # ------------------------------------------------------------------------------------------------------------------
    # print('============================== LOGIT-EFFICIENCY ===========================================================')
    # print(' ')
    # print(' ----- TENDERS-ALGO STATISTICS (pre Logit)')
    # print_pnl_stats(pnl)
    # # ------------------------------------------------------------------------------------------------------------------
    # y_pred = fe_logit.get_prediction(X_test, False)
    # mask = y_pred.mask(y_pred > threshold, 1).mask(y_pred <= threshold, 0)
    # pnl_masked_by_logit = pnl.mul(mask)
    # # ------------------------------------------------------------------------------------------------------------------
    # print('========================= TENDERS-ALGO STATISTICS (post Logit) ============================================')
    # print(' ')
    # print_pnl_stats(pnl_masked_by_logit)
    # print('===========================================================================================================')


