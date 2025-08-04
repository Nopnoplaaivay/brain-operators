import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Generate sample data for testing
np.random.seed(42)
T, N = 1000, 50  # 1000 time periods, 50 stocks
close_adjusted = np.random.randn(T, N).cumsum(axis=0) + 100

# Add some NaN values to test sparse/dense modes
close_with_nan = close_adjusted.copy()
nan_indices = np.random.choice(T * N, size=100, replace=False)
flat_data = close_with_nan.flatten()
flat_data[nan_indices] = np.nan
close_with_nan = flat_data.reshape(T, N)


def ts_ema_v1(x, d, dense=False):
    """
    Original EMA implementation using loops.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input time series data
    d : int
        Lookback period for EMA calculation (used to calculate alpha)
    dense : bool, default=False
        If False (sparse mode): treat NaN as 0
        If True (dense mode): skip NaN values in calculation

    Returns:
    --------
    ndarray, shape (T, N)
        Exponential moving averages with NaN for first d-1 periods
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Calculate smoothing factor: alpha = 2/(d+1)
    alpha = 2.0 / (d + 1)

    for t in range(d - 1, T):
        window = x[t - d + 1:t + 1, :]  # shape: (d, N)

        if dense:
            # Dense mode: handle NaN properly
            for n in range(N):
                col_data = window[:, n]
                valid_mask = ~np.isnan(col_data)

                if np.any(valid_mask):
                    valid_data = col_data[valid_mask]

                    if len(valid_data) > 0:
                        # Calculate EMA on valid data only
                        ema = valid_data[0]  # Start with first valid value
                        for i in range(1, len(valid_data)):
                            ema = alpha * valid_data[i] + (1 - alpha) * ema
                        result[t, n] = ema
                else:
                    result[t, n] = np.nan
        else:
            # Sparse mode: treat NaN as 0
            window_filled = np.nan_to_num(window, nan=0.0)

            # Calculate EMA for each column
            for n in range(N):
                ema = window_filled[0, n]  # Start with first value
                for i in range(1, d):
                    ema = alpha * window_filled[i, n] + (1 - alpha) * ema
                result[t, n] = ema

    return result


def ts_ema_v2(x, d, dense=False):
    """
    Optimized EMA implementation using sliding_window_view with vectorization.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input time series data
    d : int
        Lookback period for EMA calculation
    dense : bool, default=False
        If False (sparse mode): treat NaN as 0
        If True (dense mode): skip NaN values in calculation

    Returns:
    --------
    ndarray, shape (T, N)
        Exponential moving averages with NaN for first d-1 periods
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Create sliding windows
    windows = sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    # Calculate smoothing factor
    alpha = 2.0 / (d + 1)

    if dense:
        # Dense mode: handle NaN by calculating EMA only on valid values
        for i in range(T - d + 1):
            for j in range(N):
                window_data = windows[i, j, :]
                valid_mask = ~np.isnan(window_data)

                if np.any(valid_mask):
                    valid_data = window_data[valid_mask]

                    if len(valid_data) > 0:
                        # Calculate EMA on valid data
                        ema = valid_data[0]
                        for k in range(1, len(valid_data)):
                            ema = alpha * valid_data[k] + (1 - alpha) * ema
                        result[d - 1 + i, j] = ema

    else:
        # Sparse mode: treat NaN as 0 and vectorize where possible
        windows_filled = np.nan_to_num(windows, nan=0.0)

        # Calculate EMA for each window
        for i in range(T - d + 1):
            window = windows_filled[i, :, :]  # shape: (N, d)

            # Initialize EMA with first values
            ema_values = window[:, 0].copy()  # shape: (N,)

            # Apply EMA formula iteratively
            for t in range(1, d):
                ema_values = alpha * window[:, t] + (1 - alpha) * ema_values

            result[d - 1 + i, :] = ema_values

    return result


def ts_ema_v3(x, d, dense=False, method='standard'):
    """
    Highly optimized EMA implementation with multiple calculation methods.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input time series data
    d : int
        Lookback period for EMA calculation
    dense : bool, default=False
        If False (sparse mode): treat NaN as 0
        If True (dense mode): skip NaN values in calculation
    method : str, default='standard'
        - 'standard': Traditional EMA calculation
        - 'weighted': Use exponential weights for all values at once

    Returns:
    --------
    ndarray, shape (T, N)
        Exponential moving averages with NaN for first d-1 periods
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Create sliding windows
    windows = sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    # Calculate smoothing factor
    alpha = 2.0 / (d + 1)

    if method == 'weighted':
        # Alternative method: use exponential weights directly
        # Create exponential weights (oldest to newest)
        exp_weights = (1 - alpha) ** np.arange(d - 1, -1, -1)  # [w_oldest, ..., w_newest]
        exp_weights = exp_weights / np.sum(exp_weights)  # Normalize

        if dense:
            # Dense mode with weighted approach
            valid_mask = ~np.isnan(windows)  # shape: (T-d+1, N, d)

            # Adjust weights for valid values only
            for i in range(T - d + 1):
                for j in range(N):
                    window_data = windows[i, j, :]
                    valid_data_mask = valid_mask[i, j, :]

                    if np.any(valid_data_mask):
                        valid_data = window_data[valid_data_mask]
                        valid_weights = exp_weights[valid_data_mask]
                        valid_weights = valid_weights / np.sum(valid_weights)  # Renormalize

                        result[d - 1 + i, j] = np.sum(valid_data * valid_weights)
        else:
            # Sparse mode with weighted approach - fully vectorized
            windows_filled = np.nan_to_num(windows, nan=0.0)
            weighted_sum = np.sum(windows_filled * exp_weights[np.newaxis, np.newaxis, :], axis=2)
            result[d - 1:, :] = weighted_sum

    else:
        # Standard EMA calculation
        if dense:
            # Dense mode: handle NaN properly
            for i in range(T - d + 1):
                for j in range(N):
                    window_data = windows[i, j, :]
                    valid_mask = ~np.isnan(window_data)

                    if np.any(valid_mask):
                        valid_data = window_data[valid_mask]

                        if len(valid_data) > 0:
                            # Calculate EMA iteratively
                            ema = valid_data[0]
                            for k in range(1, len(valid_data)):
                                ema = alpha * valid_data[k] + (1 - alpha) * ema
                            result[d - 1 + i, j] = ema
        else:
            # Sparse mode: vectorized EMA calculation
            windows_filled = np.nan_to_num(windows, nan=0.0)

            # Initialize with first values
            ema_values = windows_filled[:, :, 0].copy()  # shape: (T-d+1, N)

            # Apply EMA formula iteratively
            for t in range(1, d):
                ema_values = alpha * windows_filled[:, :, t] + (1 - alpha) * ema_values

            result[d - 1:, :] = ema_values

    return result


# Final optimized version - recommended for use
def ts_ema(x, d, dense=False, alpha=None):
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
    windows = sliding_window_view(x, window_shape=d, axis=0)  # shape: (T-d+1, N, d)

    if dense:
        # Dense mode: handle NaN by calculating EMA only on valid values
        for i in range(T - d + 1):
            for j in range(N):
                window_data = windows[i, j, :]
                valid_mask = ~np.isnan(window_data)

                if np.any(valid_mask):
                    valid_data = window_data[valid_mask]

                    if len(valid_data) > 0:
                        # Calculate EMA iteratively on valid data
                        ema = valid_data[0]
                        for k in range(1, len(valid_data)):
                            ema = alpha * valid_data[k] + (1 - alpha) * ema
                        result[d - 1 + i, j] = ema
    else:
        # Sparse mode: treat NaN as 0 and use vectorized calculation
        windows_filled = np.nan_to_num(windows, nan=0.0)

        # Initialize EMA with first values of each window
        ema_values = windows_filled[:, :, 0].copy()  # shape: (T-d+1, N)

        # Apply EMA formula iteratively across time dimension
        for t in range(1, d):
            ema_values = alpha * windows_filled[:, :, t] + (1 - alpha) * ema_values

        result[d - 1:, :] = ema_values

    return result


def compare_ema_vs_linear_decay(x, d=10):
    """Compare EMA vs Linear Decay smoothing"""
    print("EMA vs Linear Decay Comparison:")
    print("=" * 40)

    # Calculate both
    ema_result = ts_ema(x, d, dense=False)

    # Import linear decay from previous implementation
    def ts_decay_linear_simple(x, d):
        """Simplified linear decay for comparison"""
        T, N = x.shape
        result = np.full((T, N), np.nan)
        windows = sliding_window_view(x, window_shape=d, axis=0)
        weights = np.arange(1, d + 1, dtype=np.float64)
        weights_expanded = weights[np.newaxis, np.newaxis, :]
        windows_filled = np.nan_to_num(windows, nan=0.0)
        numerators = np.sum(windows_filled * weights_expanded, axis=2)
        weights_sum = np.sum(weights)
        result[d - 1:, :] = numerators / weights_sum
        return result

    linear_result = ts_decay_linear_simple(x, d)

    # Compare characteristics
    valid_mask = ~np.isnan(ema_result) & ~np.isnan(linear_result)

    if np.any(valid_mask):
        ema_values = ema_result[valid_mask]
        linear_values = linear_result[valid_mask]

        print(f"EMA - Mean: {np.mean(ema_values):.4f}, Std: {np.std(ema_values):.4f}")
        print(f"Linear - Mean: {np.mean(linear_values):.4f}, Std: {np.std(linear_values):.4f}")

        correlation = np.corrcoef(ema_values, linear_values)[0, 1]
        print(f"Correlation between EMA and Linear Decay: {correlation:.4f}")

        # Analyze differences
        diff = np.abs(ema_values - linear_values)
        print(f"Mean absolute difference: {np.mean(diff):.4f}")
        print(f"Max absolute difference: {np.max(diff):.4f}")

    return ema_result, linear_result


def demonstrate_alpha_effects(x, d=10):
    """Demonstrate different alpha values"""
    print("Alpha Parameter Effects:")
    print("=" * 30)

    # Different alpha values
    alphas = [0.1, 0.2, 2 / (d + 1), 0.5, 0.8]  # Include default alpha

    print(f"Default alpha for d={d}: {2 / (d + 1):.3f}")
    print()

    for alpha in alphas:
        result = ts_ema(x, d, alpha=alpha, dense=False)
        valid_mask = ~np.isnan(result)

        if np.any(valid_mask):
            values = result[valid_mask]
            print(f"Alpha {alpha:.3f}: Mean={np.mean(values):.3f}, Std={np.std(values):.3f}")

        # Show effective weights for interpretation
        weights = []
        weight = alpha
        for i in range(d):
            weights.append(weight)
            weight *= (1 - alpha)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize for comparison

        print(f"  Effective weights (newest to oldest): {weights[:5].round(3)}")
        print()


def verify_ema_formula():
    """Verify EMA calculation with manual example"""
    print("EMA Formula Verification:")
    print("=" * 30)

    # Simple example data
    example_data = np.array([[10, 12, 11, 13, 15]]).T  # shape: (5, 1)
    d = 5
    alpha = 2 / (d + 1)  # 0.333

    print(f"Input data: {example_data.flatten()}")
    print(f"Alpha: {alpha:.3f}")

    # Manual calculation
    manual_ema = example_data[0, 0]  # Start with first value: 10
    print(f"EMA[0]: {manual_ema:.3f}")

    for i in range(1, 5):
        manual_ema = alpha * example_data[i, 0] + (1 - alpha) * manual_ema
        print(
            f"EMA[{i}]: {alpha:.3f} * {example_data[i, 0]} + {(1 - alpha):.3f} * {manual_ema / (alpha + (1 - alpha)):.3f} = {manual_ema:.3f}")

    # Calculated result
    result = ts_ema(example_data, d=5, dense=False)
    calculated_ema = result[-1, 0]

    print(f"\nManual final EMA: {manual_ema:.6f}")
    print(f"Calculated EMA: {calculated_ema:.6f}")
    print(f"Difference: {abs(manual_ema - calculated_ema):.10f}")

    return abs(manual_ema - calculated_ema) < 1e-10


def benchmark_ema_versions(x, d=10, num_runs=3):
    """Benchmark EMA implementations"""
    print("EMA Performance Benchmark:")
    print("=" * 30)

    functions = [
        ("V1 Original (sparse)", lambda x, d: ts_ema_v1(x, d, dense=False)),
        ("V1 Original (dense)", lambda x, d: ts_ema_v1(x, d, dense=True)),
        ("V2 Optimized (sparse)", lambda x, d: ts_ema_v2(x, d, dense=False)),
        ("V2 Optimized (dense)", lambda x, d: ts_ema_v2(x, d, dense=True)),
        ("V3 Standard (sparse)", lambda x, d: ts_ema_v3(x, d, dense=False, method='standard')),
        ("V3 Weighted (sparse)", lambda x, d: ts_ema_v3(x, d, dense=False, method='weighted')),
        ("Final ts_ema (sparse)", lambda x, d: ts_ema(x, d, dense=False)),
        ("Final ts_ema (dense)", lambda x, d: ts_ema(x, d, dense=True)),
    ]

    results = {}

    for name, func in functions:
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = func(x, d)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[name] = {'time': avg_time, 'result': result}

        print(f"{name:25s}: {avg_time:.6f}s ± {std_time:.6f}s")

    # Calculate speedups
    print("\nSpeedup Analysis:")
    print("=" * 20)
    baseline = results["V1 Original (sparse)"]['time']

    for name, data in results.items():
        if name != "V1 Original (sparse)":
            speedup = baseline / data['time']
            print(f"{name:25s}: {speedup:.2f}x")

    return results


def verify_ema_correctness(x, d=5):
    """Verify all EMA implementations produce same results"""
    print("EMA Correctness Verification:")
    print("=" * 35)

    # Test sparse mode
    result_v1 = ts_ema_v1(x, d, dense=False)
    result_v2 = ts_ema_v2(x, d, dense=False)
    result_final = ts_ema(x, d, dense=False)

    valid_mask = ~np.isnan(result_v1)

    if np.any(valid_mask):
        diff_v1_v2 = np.abs(result_v1[valid_mask] - result_v2[valid_mask])
        diff_v1_final = np.abs(result_v1[valid_mask] - result_final[valid_mask])

        max_diff_v2 = np.max(diff_v1_v2)
        max_diff_final = np.max(diff_v1_final)

        print(f"Sparse mode:")
        print(f"  V1 vs V2 max difference: {max_diff_v2:.10f}")
        print(f"  V1 vs Final max difference: {max_diff_final:.10f}")

        sparse_correct = max_diff_v2 < 1e-10 and max_diff_final < 1e-10
    else:
        sparse_correct = True
        print("Sparse mode: No valid results to compare")

    # Test dense mode if data has NaN
    if np.any(np.isnan(x)):
        result_v1_dense = ts_ema_v1(x, d, dense=True)
        result_final_dense = ts_ema(x, d, dense=True)

        valid_mask_dense = ~np.isnan(result_v1_dense)

        if np.any(valid_mask_dense):
            diff_dense = np.abs(result_v1_dense[valid_mask_dense] - result_final_dense[valid_mask_dense])
            max_diff_dense = np.max(diff_dense)

            print(f"Dense mode:")
            print(f"  V1 vs Final max difference: {max_diff_dense:.10f}")

            dense_correct = max_diff_dense < 1e-10
        else:
            dense_correct = True
            print("Dense mode: No valid results to compare")
    else:
        dense_correct = True
        print("Dense mode: No NaN values in data")

    overall_correct = sparse_correct and dense_correct
    print(f"\n✅ Overall correctness: {'PASSED' if overall_correct else 'FAILED'}")

    return overall_correct


# Run comprehensive tests
print("Comprehensive Testing of ts_ema Operator")
print("=" * 45)

# Verify EMA formula
print("1. Formula Verification:")
formula_correct = verify_ema_formula()

print("\n" + "=" * 45)

# Test correctness
print("2. Implementation Correctness:")
correctness_passed = verify_ema_correctness(close_with_nan, d=5)

print("\n" + "=" * 45)

# Demonstrate alpha effects
print("3. Alpha Parameter Effects:")
demonstrate_alpha_effects(close_with_nan, d=10)

print("\n" + "=" * 45)

# Compare with linear decay
print("4. EMA vs Linear Decay:")
ema_result, linear_result = compare_ema_vs_linear_decay(close_with_nan, d=10)

print("\n" + "=" * 45)

# Benchmark performance
print("5. Performance Benchmark:")
benchmark_results = benchmark_ema_versions(close_with_nan, d=10)

print("\n" + "=" * 45)
print("Usage Examples:")
print("=" * 15)

print("\n# Basic EMA (default alpha = 2/(d+1))")
print("ema = ts_ema(data, d=10)")

print("\n# EMA with custom alpha (more reactive to recent changes)")
print("fast_ema = ts_ema(data, d=10, alpha=0.5)")

print("\n# EMA in dense mode (handles NaN properly)")
print("ema_dense = ts_ema(data, d=10, dense=True)")

print("\n# Different time horizons")
print("short_ema = ts_ema(data, d=5)   # Fast EMA")
print("long_ema = ts_ema(data, d=50)   # Slow EMA")

# Demonstrate actual usage with different parameters
print(f"\nExample Results:")
basic_ema = ts_ema(close_with_nan, d=10, dense=False)
fast_ema = ts_ema(close_with_nan, d=10, alpha=0.5, dense=False)
slow_ema = ts_ema(close_with_nan, d=30, dense=False)

print(f"Input shape: {close_with_nan.shape}")
print(f"Basic EMA (d=10): {np.sum(~np.isnan(basic_ema))} valid values")
print(f"Fast EMA (α=0.5): {np.sum(~np.isnan(fast_ema))} valid values")
print(f"Slow EMA (d=30): {np.sum(~np.isnan(slow_ema))} valid values")

print("\nKey Advantages of EMA:")
print("- More responsive to recent price changes than linear decay")
print("- Exponential weighting naturally emphasizes recent data")
print("- Computational efficiency with sliding window")
print("- Flexible alpha parameter for different smoothing needs")
print("- Widely used in technical analysis and algorithmic trading")