import numpy as np

GROUPS = {
    "sector": np.array([0] * 20 + [1] * 20 + [2] * 10),
    "subindustry": np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10),
    "industry": np.array([0] * 25 + [1] * 25)
}

def group_scale(x, group, handle_constant='midpoint'):

    group = GROUPS[group]
    T, N = x.shape

    # Validate inputs
    if len(group) != N:
        raise ValueError(f"Group array length ({len(group)}) must match number of columns ({N})")

    # Handle constant value options
    constant_values = {
        'midpoint': 0.5,
        'zero': 0.0,
        'one': 1.0,
        'nan': np.nan
    }

    if handle_constant not in constant_values:
        raise ValueError(f"handle_constant must be one of {list(constant_values.keys())}")

    constant_fill = constant_values[handle_constant]

    result = np.full_like(x, np.nan, dtype=np.float64)

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

        constant_mask = constant_range[:, 0]  # Convert to 1D for indexing
        if np.any(constant_mask):
            # Only fill non-NaN values with the constant
            non_nan_mask = ~np.isnan(group_data[constant_mask, :])
            scaled_group[constant_mask, :] = np.where(
                non_nan_mask,
                constant_fill,
                np.nan
            )

        result[:, group_mask] = scaled_group

    return result
