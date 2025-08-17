import time
import numpy as np
import pandas as pd


from time_series import ts_rank, ts_rank_group_for, ts_rank_group_numpy

np.random.seed(42)
T, N = 50000, 1  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100


ts_rank_df = ts_rank(close_adjusted, d=10)

import time
start = time.time()
ts_rank_group_for_df_10 = ts_rank_group_for(close_adjusted, d=10, n_group=10)
print(f"Execution time for ts_rank_group_for with n_group=10: {time.time() - start:.4f} seconds")
ts_rank_group_for_df_8 = ts_rank_group_for(close_adjusted, d=10, n_group=8)
ts_rank_group_for_df_3 = ts_rank_group_for(close_adjusted, d=10, n_group=3)

start = time.time()
ts_rank_group_numpy_df_10 = ts_rank_group_numpy(close_adjusted, d=10, n_group=10)
print(f"Execution time for ts_rank_group_numpy with n_group=10: {time.time() - start:.4f} seconds")
ts_rank_group_numpy_df_8 = ts_rank_group_numpy(close_adjusted, d=10, n_group=8)
ts_rank_group_numpy_df_3 = ts_rank_group_numpy(close_adjusted, d=10, n_group=3)

# Compare results
results_df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=T, freq='D'),
    'close_adjusted': close_adjusted.flatten(),
    'ts_rank': ts_rank_df.flatten(),
    'ts_rank_group_for_df_10': ts_rank_group_for_df_10.flatten(),
    'ts_rank_group_numpy_df_10': ts_rank_group_numpy_df_10.flatten(),
    'ts_rank_group_for_df_8': ts_rank_group_for_df_8.flatten(),
    'ts_rank_group_numpy_df_8': ts_rank_group_numpy_df_8.flatten(),
    'ts_rank_group_for_df_3': ts_rank_group_for_df_3.flatten(),
    'ts_rank_group_numpy_df_3': ts_rank_group_numpy_df_3.flatten(),
})

# Export the DataFrame to an Excel file
results_df.to_excel('ts_rank_group_results_2.xlsx', index=False)