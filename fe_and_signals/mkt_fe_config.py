from fe_and_signals.fe import Fe as Fe

fe_pars = {
    'vola': {
        'func': Fe.Vola.vola_sma,
        'data_cols': ['close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60 * 24], 'on_rets': [False]}
    },
    'roc': {
        'func': Fe.Momentum.roc,
        'data_cols': ['close'],
        'args': {'timeperiod': [1, 5, 15, 30, 60, 60 * 4]}
    },
    'atr': {
        'func': Fe.Vola.atr,
        'data_cols': ['high', 'low', 'close'],
        'args': {'timeperiod': [5, 15, 30, 60, 60 * 24]}
    },
    'bars_dispersion_rolling': {
        'func': Fe.Vola.bars_dispersion_rolling,
        'data_cols': ['high', 'low'],
        'args': {'timeperiod': [1, 5, 15, 30, 60, 60 * 4]}
    },
    # 'hurst': {
    #     'func': Fe.hurst_exp,
    #     'data_cols': ['close'],
    #     'args': {'timeperiod': [60 * 2]}
    # },
    'cross_sma_perc_distance': {
        'func': Fe.Momentum.cross_sma_perc_distance,
        'data_cols': ['close'],
        'args': {'lookback': [(10, 30), (20, 60)]}
    }
}
