import numpy as np

def ts_decay_linear(x, d, dense=False):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    # Create linear weights (most recent gets highest weight)
    weights = np.arange(1, d + 1, dtype=np.float64)  # [1, 2, 3, ..., d]
    weights_expanded = weights[np.newaxis, np.newaxis, :]  # shape: (1, 1, d)

    if dense:
        # Dense mode: handle NaN by excluding them from calculation
        valid_mask = ~np.isnan(windows)  # shape: (T-d+1, N, d)

        # Replace NaN with 0 for calculation, but weights become 0 for NaN positions
        windows_for_calc = np.where(valid_mask, windows, 0.0)
        weights_for_calc = np.where(valid_mask, weights_expanded, 0.0)

        # Calculate weighted sums
        numerators = np.sum(windows_for_calc * weights_for_calc, axis=2)
        denominators = np.sum(weights_for_calc, axis=2)

        # Calculate results only where we have valid data
        valid_result_mask = denominators > 0
        result_values = np.full((T - d + 1, N), np.nan)
        result_values[valid_result_mask] = numerators[valid_result_mask] / denominators[valid_result_mask]

        result[d - 1:, :] = result_values

    else:
        # Sparse mode: treat NaN as 0 - fully vectorized
        windows_filled = np.nan_to_num(windows, nan=0.0)
        numerators = np.sum(windows_filled * weights_expanded, axis=2)
        weights_sum = np.sum(weights)
        result[d - 1:, :] = numerators / weights_sum

    return result