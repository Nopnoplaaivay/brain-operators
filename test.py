import time
import numpy as np

from time_series import (
    ts_rank_winsorize,
    ts_decay_linear,
    ts_ema
)


# price = pd.read_csv("pivoted_adjusted_close_price.csv")
# acb_close_adjusted = price["ACB"].values
# price = price.drop(columns=["date"])
# close_adjusted = price.values


np.random.seed(42)
T, N = 50000, 3  # 1000 time periods, n stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100

group_data = np.random.randn(T, 50).cumsum(axis=0) + 100
group_data[:, :10] *= 10
group_data[:, 10:20] *= 0.1

# Test ts_rank_winsorize
for d in [20, 63, 252]:
    start = time.time()
    alpha = ts_rank_winsorize(close_adjusted, d=d, winsorize_std=4)
    print(f"Execution time for ts_rank_winsorize {d} days: {time.time() - start:.4f}s")
print("=" * 30)

# Test ts_decay_linear
for d in [20, 63, 252]:
    start = time.time()
    alpha_2 = ts_decay_linear(close_adjusted, d=d, dense=False)
    print(f"Execution time for ts_decay_linear {d} days: {time.time() - start:.4f}s")
print("=" * 30)

# Test ts_ema
for d in [20, 63, 252]:
    start = time.time()
    alpha_3 = ts_ema(close_adjusted, d=d, dense=False)
    print(f"Execution time for ts_ema {d} days: {time.time() - start:.4f}s")
print("=" * 30)
