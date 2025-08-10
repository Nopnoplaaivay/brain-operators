import time
import numpy as np

from time_series import (
    ts_rank,
    ts_rank_winsorize,
    ts_rank_group
)

# price = pd.read_csv("pivoted_adjusted_close_price.csv")
# acb_close_adjusted = price["ACB"].values
# price = price.drop(columns=["date"])
# close_adjusted = price.values
np.random.seed(42)
T, N = 50000, 3  # 1000 time periods, n stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100

# Test ts_rank
for d in [20, 63, 252]:
    start = time.time()
    alpha = ts_rank(close_adjusted, d=d)
    print(f"Execution time for {d} days: {time.time() - start:.4f}s")
print("=" * 30)

# Test ts_rank_winsorize
for d in [20, 63, 252]:
    start = time.time()
    alpha_1 = ts_rank_winsorize(close_adjusted, d=d, winsorize_std=4)
    print(f"Execution time for {d} days: {time.time() - start:.4f}s")
print("=" * 30)

# Test ts_rank_group
for d in [20, 63, 252]:
    start = time.time()
    alpha_2 = ts_rank_group(close_adjusted, d=d, n_group=10)
    print(f"Execution time for {d} days: {time.time() - start:.4f}s")
print("=" * 30)
