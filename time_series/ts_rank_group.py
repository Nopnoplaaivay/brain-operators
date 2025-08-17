import numpy as np
import math

def ts_rank_group_for(x, d, n_group, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)
    current_values = windows[:, :, -1]  # shape: (T-d+1, N)
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    ranks = ranks + constant

    grouped_ranks = np.zeros_like(ranks)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            rank_val = ranks[i, j]
            if rank_val == 0:
                grouped_ranks[i, j] = 1
            else:
                grouped_ranks[i, j] = math.ceil(rank_val * n_group)

    result[d - 1:, :] = grouped_ranks

    result_int = result.copy()
    mask = ~np.isnan(result)
    result_int[mask] = result[mask].astype(int)

    return result_int


def ts_rank_group_numpy(x, d, n_group, constant=0):
    T, N = x.shape
    result = np.full((T, N), np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)
    current_values = windows[:, :, -1]
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    ranks = ranks + constant

    # Tạo boundaries: [0, 1/n, 2/n, ..., 1]
    bins = np.linspace(0, 1, n_group + 1)
    ranks_flat = ranks.flatten()
    groups_flat = np.digitize(ranks_flat, bins, right=False)
    groups_flat = np.minimum(groups_flat, n_group)

    grouped_ranks = groups_flat.reshape(ranks.shape)

    result[d - 1:, :] = grouped_ranks

    result_int = result.copy()
    mask = ~np.isnan(result)
    result_int[mask] = result[mask].astype(int)

    return result_int

def ts_rank_group_old(x, d, n_group, constant=0):
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
