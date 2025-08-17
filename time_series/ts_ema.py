import numpy as np


def ts_ema(x, d, dense=False, alpha=None):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    if T < d:
        return result
    if alpha is None:
        alpha = 2.0 / (d + 1)

    if not (0 < alpha <= 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

    if dense:
        for j in range(N):
            col_data = x[:, j]

            for start_idx in range(T - d + 1):
                window = col_data[start_idx:start_idx + d]
                valid_mask = ~np.isnan(window)

                if np.sum(valid_mask) >= d:  
                    sma_init = np.mean(window[valid_mask][:d])
                    result[start_idx + d - 1, j] = sma_init

                    ema_current = sma_init
                    for t in range(start_idx + d, T):
                        if not np.isnan(col_data[t]):
                            ema_current = alpha * col_data[t] + (1 - alpha) * ema_current
                            result[t, j] = ema_current
                        else:
                            pass
                    break
    else:
        x_filled = np.nan_to_num(x, nan=0.0)

        sma_init = np.mean(x_filled[:d, :], axis=0) 
        result[d - 1, :] = sma_init

        ema_current = sma_init.copy() 

        for t in range(d, T):
            ema_current = alpha * x_filled[t, :] + (1 - alpha) * ema_current
            result[t, :] = ema_current

    return result
