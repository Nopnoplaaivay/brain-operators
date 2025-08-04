import time
import numpy as np
import pandas as pd


from time_series import ts_rank_winsorize, ts_rank
from cross_sectional import winsorize


price = pd.read_csv("pivoted_adjusted_close_price.csv")
acb_close_adjusted = price["ACB"].values
price = price.drop(columns=["date"])
# close_adjusted = price.values
# close_adjusted = close_adjusted[:10, :3]  # Example with 1000 time periods and 50 stocks
np.random.seed(42)
T, N = 50000, 3  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100


x = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 100],  # Outlier in column 2
    [5, 7],  # Outlier in column 2
    [4, 5],  # Outlier in column 2
    [4, 6],  # Outlier in column 2
])

alpha_1 = ts_rank(winsorize(x, std=2), d=3)
alpha_2 = winsorize(ts_rank(x, d=3), std=2)

print("Alpha 1 (winsorized before ts_rank):")
print(alpha_1)

print("Alpha 2 (ts_rank before winsorized):")
print(alpha_2)

