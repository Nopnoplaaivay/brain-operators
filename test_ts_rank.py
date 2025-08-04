import time
import numpy as np
import pandas as pd


price = pd.read_csv("pivoted_adjusted_close_price.csv")
acb_close_adjusted = price["ACB"].values
price = price.drop(columns=["date"])
# close_adjusted = price.values
# close_adjusted = close_adjusted[:10, :3]  # Example with 1000 time periods and 50 stocks
np.random.seed(42)
T, N = 50000, 3  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100


def ts_rank_v1(x, d, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    for t in range(d - 1, T):
        window = x[t - d + 1:t + 1, :]  # shape: (d, N)
        current = x[t, :][None, :]  # shape: (1, N)
        ranks = np.sum(window < current, axis=0) / (d - 1)
        result[t, :] = ranks + constant

    return result

start = time.time()
ranks = ts_rank_v1(close_adjusted, d=20)
print(f"Execution time: {time.time() - start:.4f} seconds")

def ts_rank_v2(x, d, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)
    current_values = windows[:, :, -1]  # shape: (T-d+1, N)
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    result[d - 1:, :] = ranks + constant

    return result

start = time.time()
alpha = ts_rank_v2(close_adjusted, d=20)
print(f"Execution time: {time.time() - start:.4f} seconds")



def ts_rank_winsorize(x, d, constant=0, winsorize_std=None):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    if winsorize_std is not None:
        # Compute mean and std for each window along the time dimension (axis=-1)
        mu = np.nanmean(windows, axis=-1, keepdims=True)  # shape: (T-d+1, N, 1)
        sigma = np.nanstd(windows, axis=-1, keepdims=True)  # shape: (T-d+1, N, 1)

        sigma = np.where(sigma == 0, 1, sigma)

        lower = mu - winsorize_std * sigma
        upper = mu + winsorize_std * sigma

        windows = np.clip(windows, lower, upper)

    current_values = windows[:, :, -1]
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    result[d - 1:, :] = ranks + constant

    return result

start = time.time()
ts_rank_winsorize(close_adjusted, d=20, constant=0, winsorize_std=4)
print(f"Execution time: {time.time() - start:.4f} seconds")
