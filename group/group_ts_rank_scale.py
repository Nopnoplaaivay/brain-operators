import numpy as np

GROUPS = {
    "sector": np.array([0] * 20 + [1] * 20 + [2] * 10),
    "subindustry": np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10),
    "industry": np.array([0] * 25 + [1] * 25)
}

def group_ts_rank_scale(x, group, d=20, constant=0):

    group = GROUPS[group]
    T, N = x.shape

    if len(group) != N:
        raise ValueError(f"Group array length ({len(group)}) must match number of columns ({N})")

    result = np.full_like(x, np.nan, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=d, axis=0)
    current_values = windows[:, :, -1]  # shape: (T-d+1, N)
    comparisons = windows[:, :, :-1] < current_values[:, :, np.newaxis]
    ranks = np.sum(comparisons, axis=2) / (d - 1)
    result[d - 1:, :] = ranks + constant

    unique_groups = np.unique(group)
    for g in unique_groups:
        group_mask = (group == g)
        group_data = x[:, group_mask]  # shape: (T, group_size)

        with np.errstate(invalid='ignore'):
            group_min = np.nanmin(group_data, axis=1, keepdims=True)  # shape: (T, 1)
            group_max = np.nanmax(group_data, axis=1, keepdims=True)  # shape: (T, 1)

        valid_range = (group_max > group_min) & np.isfinite(group_min) & np.isfinite(group_max)
        constant_range = (group_max == group_min) & np.isfinite(group_min)

        scaled_group = np.full_like(group_data, np.nan)

        valid_mask = valid_range[:, 0]
        if np.any(valid_mask):
            range_diff = group_max[valid_mask] - group_min[valid_mask]
            scaled_group[valid_mask, :] = (
                    (group_data[valid_mask, :] - group_min[valid_mask]) / range_diff
            )

        result[:, group_mask] = scaled_group

    return result
