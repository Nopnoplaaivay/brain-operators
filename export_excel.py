import time
import numpy as np
import pandas as pd


from time_series import ts_rank_winsorize, ts_decay_linear, ts_decay_ema

np.random.seed(42)
T, N = 50000, 1  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100

# Add outliers - introduce large spikes at random positions
num_outliers = int(T * 0.05)  # 5% of the data will be outliers
outlier_positions = np.random.choice(T, size=num_outliers, replace=False)
outlier_magnitudes = np.random.choice([-1, 1], size=num_outliers) * np.random.uniform(100, 300, size=num_outliers)

for pos, mag in zip(outlier_positions, outlier_magnitudes):
    close_adjusted[pos] += mag

# Add a few extreme outliers
extreme_positions = np.random.choice(T, size=5, replace=False)
extreme_magnitudes = np.random.choice([-1, 1], size=5) * np.random.uniform(40, 80, size=5)
for pos, mag in zip(extreme_positions, extreme_magnitudes):
    close_adjusted[pos] += mag




ts_rank_df = ts_rank_winsorize(close_adjusted, d=100)
ts_rank_winsorize_1sigma = ts_rank_winsorize(close_adjusted, d=100, winsorize_std=1)
ts_rank_winsorize_2sigma = ts_rank_winsorize(close_adjusted, d=100, winsorize_std=2)
ts_rank_winsorize_3sigma = ts_rank_winsorize(close_adjusted, d=100, winsorize_std=3)

# Compare results
results_df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=T, freq='D'),
    'close_adjusted': close_adjusted.flatten(),
    'outlier_positions': np.isin(np.arange(T), outlier_positions).astype(int),
    'ts_rank': ts_rank_df.flatten(),
    'ts_rank_winsorize_1sigma': ts_rank_winsorize_1sigma.flatten(),
    'ts_rank_winsorize_2sigma': ts_rank_winsorize_2sigma.flatten(),
    'ts_rank_winsorize_3sigma': ts_rank_winsorize_3sigma.flatten()
})

# Export the DataFrame to an Excel file
results_df.to_excel('ts_rank_results.xlsx', index=False)