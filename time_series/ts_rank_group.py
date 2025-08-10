import numpy as np


def ts_rank_group(x, d, n_group, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)
    current_values = windows[:, :, -1]  # shape: (T-d+1, N)

    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    ranks = ranks + constant

    # shift_amount = (0.5 / n_group) - 1e-9
    shift_amount = (0.5 / n_group)
    grouped_ranks = np.floor((ranks + shift_amount) * n_group)

    # Đảm bảo các giá trị nằm trong khoảng [0, n_group-1]
    grouped_ranks = np.clip(grouped_ranks, 0, n_group - 1)
    grouped_ranks = grouped_ranks + 1

    result[d - 1:, :] = grouped_ranks

    # Chỉ convert phần không phải NaN thành int, giữ NaN như cũ
    result_int = result.copy()
    mask = ~np.isnan(result)
    result_int[mask] = result[mask].astype(int)

    return result_int
