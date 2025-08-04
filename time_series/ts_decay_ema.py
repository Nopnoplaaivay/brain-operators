import numpy as np

def ts_decay_ema(x, d, dense=False, alpha=None):
    """
    Exponential Moving Average operator with sliding window optimization.

    Calculates EMA using either a specified alpha or derived from lookback period d.
    EMA gives exponentially decreasing weights to older observations.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input time series data
    d : int
        Lookback period for EMA calculation (used to calculate alpha if not provided)
    dense : bool, default=False
        - False (sparse mode): treat NaN as 0
        - True (dense mode): skip NaN values in calculation
    alpha : float or None, default=None
        Smoothing factor. If None, calculated as 2/(d+1)
        Higher alpha = more weight to recent observations

    Returns:
    --------
    ndarray, shape (T, N)
        Exponential moving averages. First d-1 periods are NaN.

    Formula:
    --------
    EMA_t = alpha * X_t + (1-alpha) * EMA_{t-1}
    where alpha = 2/(d+1) if not specified

    Example:
    --------
    For d=5: alpha = 2/(5+1) = 0.333
    EMA gives weights approximately: [0.13, 0.20, 0.27, 0.33] (newest gets most weight)
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Calculate smoothing factor
    if alpha is None:
        alpha = 2.0 / (d + 1)

    # Validate alpha
    if not (0 < alpha <= 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    if dense:
        # Dense mode: handle NaN by calculating EMA only on valid values
        for i in range(T - d + 1):
            for j in range(N):
                window_data = windows[i, j, :]
                valid_mask = ~np.isnan(window_data)

                if np.any(valid_mask):
                    valid_data = window_data[valid_mask]

                    if len(valid_data) > 0:
                        ema = valid_data[0]
                        for k in range(1, len(valid_data)):
                            ema = alpha * valid_data[k] + (1 - alpha) * ema
                        result[d - 1 + i, j] = ema
    else:
        windows_filled = np.nan_to_num(windows, nan=0.0)
        # lay gia tri dau tien trong time range
        ema_values = windows_filled[:, :, 0].copy()  # shape: (T-d+1, N)

        for t in range(1, d):
            ema_values = alpha * windows_filled[:, :, t] + (1 - alpha) * ema_values

        result[d - 1:, :] = ema_values

    return result