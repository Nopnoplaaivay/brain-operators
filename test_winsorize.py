import numpy as np
import pandas as pd


price = pd.read_csv("pivoted_adjusted_close_price.csv")
acb = price["ACB"].values

# 1d array
def winsorize(x, std=4):
    mu = np.mean(x)
    sigma = np.std(x)
    lower = mu - std * sigma
    upper = mu + std * sigma
    return np.clip(x, lower, upper)

# Tạo data có outlier
x = np.array([1, 2, 2, 3, 3, 100])
print("Before winsorize:", x)

x_win = winsorize(x, std=1)
print("After winsorize (std=1):", x_win)

x_win2 = winsorize(x, std=4)
print("After winsorize (std=4):", x_win2)


def winsorize_2d(x, std=4):
    """
    Winsorizes a 2D array (T, N) column-wise.
    """
    mu = np.nanmean(x, axis=0, keepdims=True)
    sigma = np.nanstd(x, axis=0, keepdims=True)
    lower = mu - std * sigma
    upper = mu + std * sigma
    return np.clip(x, lower, upper)

x = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 100],  # Outlier in column 2
])
print(winsorize_2d(x, std=1))
