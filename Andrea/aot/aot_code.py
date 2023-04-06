import numpy as np
from numba.pycc import CC

cc = CC("aot_code")
# Uncomment the following line to print out the compilation steps
cc.verbose = True

@cc.export('loop_kama', 'f8[:](f8[:], f8[:], f8, f8, f8)')
def loop_kama(prices: np.array, kama: np.array, timeperiod: int, fastest: float, slowest: float) -> np.array:

    for t in range(timeperiod - 1, kama.shape[0]):

        p_change = prices[int(t)] - prices[int(t - timeperiod + 1)]
        period = prices[int(t - timeperiod + 1): int(t + 1)]
        p_sum = np.sum(np.abs(period[1:] - period[0: -1]))
        ER = p_change / p_sum

        SC = (ER * (fastest - slowest) + slowest) ** 2
        kama[t] = kama[t - 1] + SC * (prices[t] - kama[t - 1])

    return kama


if __name__ == "__main__":
    cc.compile()
