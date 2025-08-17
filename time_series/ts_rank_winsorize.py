import numpy as np


def ts_rank(x, d, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)
    current_values = windows[:, :, -1] 
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    result[d - 1:, :] = ranks + constant

    return result


def ts_rank_winsorize(x, d, constant=0, winsorize_std=None):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  

    if winsorize_std is not None:
        mu = np.nanmean(windows, axis=-1, keepdims=True)  
        sigma = np.nanstd(windows, axis=-1, keepdims=True)  

        sigma = np.where(sigma == 0, 1, sigma)

        lower = mu - winsorize_std * sigma
        upper = mu + winsorize_std * sigma

        windows = np.clip(windows, lower, upper)

    current_values = windows[:, :, -1]
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    result[d - 1:, :] = ranks + constant

    return result
