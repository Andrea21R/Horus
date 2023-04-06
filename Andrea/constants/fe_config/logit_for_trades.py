from Andrea.fe_and_signals.features import Features

fe_pars = {
    'vola_sma': {
        'func': Features.Vola.vola_sma,
        'data_cols': ['close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60 * 24], 'on_rets': [False]}
    },
    'vola_ema': {
        'func': Features.Vola.vola_ema,
        'data_cols': ['close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60*24], 'on_rets': [False]}
    },
    'atr': {
        'func': Features.Vola.atr,
        'data_cols': ['high', 'low', 'close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60 * 24]}
    },
    'bars_dispersion_rolling': {
        'func': Features.Vola.bars_dispersion_rolling,
        'data_cols': ['high', 'low'],
        'args': {'timeperiod': [1, 5, 15, 30, 60, 60 * 4]}
    },
    # 'hurst': {
    #     'func': Features.Vola.hurst_exp,
    #     'data_cols': ['close'],
    #     'args': {'timeperiod': [60 * 2]}
    # },
    'rsi': {
        'func': Features.Overlap.rsi,
        'data_cols': ['close'],
        'args': {'timeperiod': [5, 14, 30]}
    },
    'roc': {
        'func': Features.Momentum.roc,
        'data_cols': ['close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60 * 4]}
    },
    'cross_sma_perc_distance': {
        'func': Features.Momentum.cross_sma_perc_distance,
        'data_cols': ['close'],
        'args': {'lookback': [(10, 30), (20, 60)]}
    },
    'adx': {
        'func': Features.Momentum.adx,
        'data_cols': ['high', 'low', 'close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60*24]}
    },
    'returns': {
        'func': Features.Others.returns,
        'data_cols': ['close'],
        'args': {'lags': [15]}
    }
}
