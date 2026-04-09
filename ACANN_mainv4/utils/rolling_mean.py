"""
滑动平均（移动平均）工具。
"""
import numpy as np


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')
