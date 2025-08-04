import time
import numpy as np
import pandas as pd


from time_series import ts_rank_winsorize, ts_decay_linear, ts_decay_ema

np.random.seed(42)
T, N = 50000, 1  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100


ts_decay_linear_df = ts_decay_linear(close_adjusted, d=100)
ts_decay_ema_df = ts_decay_ema(close_adjusted, d=100)

# Compare results
results_df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=T, freq='D'),
    'close_adjusted': close_adjusted.flatten(),
    'ts_decay_linear': ts_decay_linear_df.flatten(),
    'ts_decay_ema': ts_decay_ema_df.flatten()
})

# Export the DataFrame to an Excel file
results_df.to_excel('ts_decay_results.xlsx', index=False)