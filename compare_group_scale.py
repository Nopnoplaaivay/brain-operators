import time
import numpy as np
import pandas as pd

# Generate sample data for testing
np.random.seed(42)
T, N = 1000, 50  # 1000 time periods, 50 stocks

# Create sample data with different scales
base_data = np.random.randn(T, N).cumsum(axis=0) + 100
# Make some stocks have much larger values to test scaling
base_data[:, :10] *= 10  # First 10 stocks have 10x larger values
base_data[:, 10:20] *= 0.1  # Next 10 stocks have much smaller values

# Create group assignments (e.g., sectors)
# Group 0: Large cap (first 20 stocks)
# Group 1: Mid cap (next 20 stocks)
# Group 2: Small cap (last 10 stocks)
group_assignment = np.array([0] * 20 + [1] * 20 + [2] * 10)

# Add some NaN values to test handling
data_with_nan = base_data.copy()
nan_indices = np.random.choice(T * N, size=100, replace=False)
flat_data = data_with_nan.flatten()
flat_data[nan_indices] = np.nan
data_with_nan = flat_data.reshape(T, N)


def group_scale_v1(x, group):
    """
    Basic implementation using loops.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input data to be scaled
    group : ndarray, shape (N,)
        Group assignment for each column/asset

    Returns:
    --------
    ndarray, shape (T, N)
        Scaled data where each group is normalized to [0, 1]
        Formula: (x - group_min) / (group_max - group_min)
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Get unique groups
    unique_groups = np.unique(group)

    for t in range(T):
        row = x[t, :]

        for g in unique_groups:
            # Find assets in this group
            group_mask = (group == g)
            group_values = row[group_mask]

            # Skip if all NaN
            if np.all(np.isnan(group_values)):
                result[t, group_mask] = np.nan
                continue

            # Calculate group min and max (ignoring NaN)
            group_min = np.nanmin(group_values)
            group_max = np.nanmax(group_values)

            # Handle case where min == max
            if group_max == group_min:
                # All values are the same, set to 0.5 or 0
                result[t, group_mask] = np.where(np.isnan(group_values), np.nan, 0.5)
            else:
                # Apply scaling formula
                scaled = (group_values - group_min) / (group_max - group_min)
                result[t, group_mask] = scaled

    return result


def group_scale_v2(x, group):
    """
    Optimized implementation using vectorization.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input data to be scaled
    group : ndarray, shape (N,)
        Group assignment for each column/asset

    Returns:
    --------
    ndarray, shape (T, N)
        Scaled data where each group is normalized to [0, 1]
    """
    T, N = x.shape
    result = np.full((T, N), np.nan)

    # Get unique groups
    unique_groups = np.unique(group)

    for g in unique_groups:
        # Find assets in this group
        group_mask = (group == g)
        group_data = x[:, group_mask]  # shape: (T, num_assets_in_group)

        # Calculate min and max for each time period within the group
        group_min = np.nanmin(group_data, axis=1, keepdims=True)  # shape: (T, 1)
        group_max = np.nanmax(group_data, axis=1, keepdims=True)  # shape: (T, 1)

        # Handle case where min == max
        same_values_mask = (group_max == group_min)

        # Calculate scaling
        denominator = group_max - group_min
        denominator = np.where(same_values_mask, 1.0, denominator)  # Avoid division by zero

        scaled_group = (group_data - group_min) / denominator

        # Set to 0.5 where all values in group are the same (and not NaN)
        scaled_group = np.where(
            same_values_mask & ~np.isnan(group_data),
            0.5,
            scaled_group
        )

        # Place back into result
        result[:, group_mask] = scaled_group

    return result


def group_scale_v3(x, group):
    """
    Highly optimized implementation with better NaN handling.

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input data to be scaled
    group : ndarray, shape (N,)
        Group assignment for each column/asset

    Returns:
    --------
    ndarray, shape (T, N)
        Scaled data where each group is normalized to [0, 1]
    """
    T, N = x.shape
    result = np.full_like(x, np.nan)

    # Get unique groups and create efficient indexing
    unique_groups, group_indices = np.unique(group, return_inverse=True)

    # Process each group
    for i, g in enumerate(unique_groups):
        # Boolean mask for this group
        group_mask = (group_indices == i)

        if not np.any(group_mask):
            continue

        # Extract group data
        group_data = x[:, group_mask]

        # Vectorized min/max calculation across assets in group for each time period
        with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
            group_min = np.nanmin(group_data, axis=1, keepdims=True)
            group_max = np.nanmax(group_data, axis=1, keepdims=True)

        # Create masks for different conditions
        valid_range_mask = (group_max > group_min) & np.isfinite(group_min) & np.isfinite(group_max)
        same_values_mask = (group_max == group_min) & np.isfinite(group_min)

        # Initialize scaled data
        scaled_group = np.full_like(group_data, np.nan)

        # Apply scaling where we have valid range
        if np.any(valid_range_mask):
            valid_indices = np.where(valid_range_mask)[0]
            for t in valid_indices:
                scaled_group[t, :] = (group_data[t, :] - group_min[t, 0]) / (group_max[t, 0] - group_min[t, 0])

        # Handle same values case (set to 0.5)
        if np.any(same_values_mask):
            same_indices = np.where(same_values_mask)[0]
            for t in same_indices:
                mask = ~np.isnan(group_data[t, :])
                scaled_group[t, mask] = 0.5

        # Place results back
        result[:, group_mask] = scaled_group

    return result


# Final optimized version - recommended for use
def group_scale(x, group, handle_constant='midpoint'):
    """
    Group-wise min-max scaling operator.

    Normalizes values within each group to be between 0 and 1 using:
    scaled_value = (value - group_min) / (group_max - group_min)

    Parameters:
    -----------
    x : ndarray, shape (T, N)
        Input data to be scaled
    group : ndarray, shape (N,)
        Group assignment for each column. Each unique value represents a different group.
    handle_constant : str, default='midpoint'
        How to handle cases where group_max == group_min:
        - 'midpoint': Set all values to 0.5
        - 'zero': Set all values to 0.0
        - 'one': Set all values to 1.0
        - 'nan': Set all values to NaN

    Returns:
    --------
    ndarray, shape (T, N)
        Scaled data where each group is independently normalized to [0, 1].
        NaN values are preserved in their original positions.

    Example:
    --------
    x = [[10, 20, 100, 200],    # Group 0: [10,20], Group 1: [100,200]
         [15, 25, 150, 250]]    # Group 0: [15,25], Group 1: [150,250]
    group = [0, 0, 1, 1]

    Result:
    Group 0 at time 0: [10,20] -> [(10-10)/(20-10), (20-10)/(20-10)] = [0.0, 1.0]
    Group 1 at time 0: [100,200] -> [(100-100)/(200-100), (200-100)/(200-100)] = [0.0, 1.0]
    """
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

    # Initialize result
    result = np.full_like(x, np.nan, dtype=np.float64)

    # Get unique groups for efficient processing
    unique_groups = np.unique(group)

    # Process each group
    for g in unique_groups:
        # Create mask for current group
        group_mask = (group == g)

        # Extract data for this group
        group_data = x[:, group_mask]  # shape: (T, group_size)

        # Calculate min and max for each time period within the group
        with np.errstate(invalid='ignore'):  # Suppress all-NaN warnings
            group_min = np.nanmin(group_data, axis=1, keepdims=True)  # shape: (T, 1)
            group_max = np.nanmax(group_data, axis=1, keepdims=True)  # shape: (T, 1)

        # Identify different cases
        valid_range = (group_max > group_min) & np.isfinite(group_min) & np.isfinite(group_max)
        constant_range = (group_max == group_min) & np.isfinite(group_min)

        # Initialize scaled group data
        scaled_group = np.full_like(group_data, np.nan)

        # Apply scaling for valid ranges
        valid_mask = valid_range[:, 0]  # Convert to 1D for indexing
        if np.any(valid_mask):
            range_diff = group_max[valid_mask] - group_min[valid_mask]
            scaled_group[valid_mask, :] = (
                    (group_data[valid_mask, :] - group_min[valid_mask]) / range_diff
            )

        # Handle constant values
        constant_mask = constant_range[:, 0]  # Convert to 1D for indexing
        if np.any(constant_mask):
            # Only fill non-NaN values with the constant
            non_nan_mask = ~np.isnan(group_data[constant_mask, :])
            scaled_group[constant_mask, :] = np.where(
                non_nan_mask,
                constant_fill,
                np.nan
            )

        # Place results back into main result array
        result[:, group_mask] = scaled_group

    return result


def demonstrate_group_scaling():
    """Demonstrate group scaling with clear examples"""
    print("Group Scaling Demonstration:")
    print("=" * 35)

    # Create simple example data
    example_data = np.array([
        [10, 20, 100, 200, 5],  # Time 0
        [15, 25, 150, 250, 10],  # Time 1
        [12, 22, 120, 220, 8]  # Time 2
    ])

    # Group assignments: [0, 0, 1, 1, 2]
    # Group 0: columns 0,1 (values 10-25)
    # Group 1: columns 2,3 (values 100-250)
    # Group 2: column 4 (single asset)
    example_groups = np.array([0, 0, 1, 1, 2])

    print("Input data:")
    print(example_data)
    print(f"Groups: {example_groups}")
    print()

    # Apply scaling
    scaled = group_scale(example_data, example_groups)

    print("Scaled data:")
    print(scaled.round(3))
    print()

    # Show calculations for first time period
    print("Manual verification for Time 0:")
    print("Group 0 (cols 0,1): [10, 20]")
    print(f"  Min: 10, Max: 20, Range: 10")
    print(f"  Scaled: [(10-10)/10, (20-10)/10] = [0.0, 1.0]")

    print("Group 1 (cols 2,3): [100, 200]")
    print(f"  Min: 100, Max: 200, Range: 100")
    print(f"  Scaled: [(100-100)/100, (200-100)/100] = [0.0, 1.0]")

    print("Group 2 (col 4): [5]")
    print(f"  Single value -> set to 0.5 (midpoint)")
    print()

    return scaled


def test_nan_handling():
    """Test how the function handles NaN values"""
    print("NaN Handling Test:")
    print("=" * 20)

    # Data with NaN values
    nan_data = np.array([
        [10, np.nan, 100, 200],
        [15, 25, np.nan, 250],
        [np.nan, np.nan, 150, np.nan]
    ])

    groups = np.array([0, 0, 1, 1])

    print("Input data with NaN:")
    print(nan_data)
    print(f"Groups: {groups}")
    print()

    scaled = group_scale(nan_data, groups)

    print("Scaled result:")
    print(scaled)
    print()

    # Explain the logic
    print("Explanation:")
    print("Time 0, Group 0: [10, NaN] -> Min:10, Max:10 -> [0.5, NaN]")
    print("Time 0, Group 1: [100, 200] -> [0.0, 1.0]")
    print("Time 1, Group 0: [15, 25] -> [0.0, 1.0]")
    print("Time 1, Group 1: [NaN, 250] -> [NaN, 0.5]")
    print("Time 2: All NaN or single values")

    return scaled


def test_constant_handling():
    """Test different ways to handle constant values"""
    print("Constant Value Handling:")
    print("=" * 30)

    # Data where groups have same values
    const_data = np.array([
        [10, 10, 100, 150],  # Group 0: all same, Group 1: different
        [20, 20, 200, 200]  # Group 0: all same, Group 1: all same
    ])

    groups = np.array([0, 0, 1, 1])

    print("Input data:")
    print(const_data)
    print(f"Groups: {groups}")
    print()

    # Test different handling methods
    methods = ['midpoint', 'zero', 'one', 'nan']

    for method in methods:
        result = group_scale(const_data, groups, handle_constant=method)
        print(f"Method '{method}':")
        print(result)
        print()

    return methods


def benchmark_group_scale(x, group, num_runs=3):
    """Benchmark different implementations"""
    print("Performance Benchmark:")
    print("=" * 25)

    functions = [
        ("V1 Basic", group_scale_v1),
        ("V2 Vectorized", group_scale_v2),
        ("V3 Optimized", group_scale_v3),
        ("Final", group_scale)
    ]

    results = {}

    for name, func in functions:
        times = []
        for _ in range(num_runs):
            start = time.time()
            if name == "Final":
                result = func(x, group)
            else:
                result = func(x, group)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[name] = {'time': avg_time, 'result': result}

        print(f"{name:15s}: {avg_time:.6f}s ± {std_time:.6f}s")

    # Calculate speedups
    print("\nSpeedup Analysis:")
    print("=" * 20)
    baseline = results["V1 Basic"]['time']

    for name, data in results.items():
        if name != "V1 Basic":
            speedup = baseline / data['time']
            print(f"{name:15s}: {speedup:.2f}x faster")

    return results


def verify_correctness(x, group):
    """Verify all implementations produce same results"""
    print("Correctness Verification:")
    print("=" * 30)

    result_v1 = group_scale_v1(x, group)
    result_v2 = group_scale_v2(x, group)
    result_final = group_scale(x, group)

    # Compare results (handling NaN properly)
    def compare_arrays(a, b, name_a, name_b):
        # Find positions where both are finite
        both_finite = np.isfinite(a) & np.isfinite(b)

        if np.any(both_finite):
            diff = np.abs(a[both_finite] - b[both_finite])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"{name_a} vs {name_b}:")
            print(f"  Max difference: {max_diff:.10f}")
            print(f"  Mean difference: {mean_diff:.10f}")

            return max_diff < 1e-10
        else:
            print(f"{name_a} vs {name_b}: No finite values to compare")
            return True

    v1_v2_correct = compare_arrays(result_v1, result_v2, "V1", "V2")
    v1_final_correct = compare_arrays(result_v1, result_final, "V1", "Final")

    # Check NaN positions match
    nan_v1 = np.isnan(result_v1)
    nan_final = np.isnan(result_final)
    nan_match = np.array_equal(nan_v1, nan_final)

    print(f"NaN positions match: {nan_match}")

    overall_correct = v1_v2_correct and v1_final_correct and nan_match
    print(f"\n✅ Overall correctness: {'PASSED' if overall_correct else 'FAILED'}")

    return overall_correct


def analyze_scaling_properties(x, group):
    """Analyze properties of the scaling"""
    print("Scaling Properties Analysis:")
    print("=" * 35)

    original = x.copy()
    scaled = group_scale(x, group)

    unique_groups = np.unique(group)

    for g in unique_groups:
        group_mask = (group == g)

        # Get data for this group
        orig_group = original[:, group_mask]
        scaled_group = scaled[:, group_mask]

        # Calculate statistics (ignoring NaN)
        valid_orig = orig_group[np.isfinite(orig_group)]
        valid_scaled = scaled_group[np.isfinite(scaled_group)]

        if len(valid_orig) > 0 and len(valid_scaled) > 0:
            print(f"Group {g} ({np.sum(group_mask)} assets):")
            print(f"  Original - Min: {np.min(valid_orig):.2f}, Max: {np.max(valid_orig):.2f}")
            print(f"  Scaled   - Min: {np.min(valid_scaled):.3f}, Max: {np.max(valid_scaled):.3f}")
            print(f"  Scaling preserved order: {np.allclose(np.argsort(valid_orig), np.argsort(valid_scaled))}")
            print()


# Run comprehensive tests
print("Comprehensive Testing of group_scale Operator")
print("=" * 50)

# Demonstrate basic functionality
print("1. Basic Demonstration:")
demo_result = demonstrate_group_scaling()

print("\n" + "=" * 50)

# Test NaN handling
print("2. NaN Handling:")
nan_result = test_nan_handling()

print("\n" + "=" * 50)

# Test constant value handling
print("3. Constant Value Handling:")
const_methods = test_constant_handling()

print("\n" + "=" * 50)

# Verify correctness
print("4. Implementation Correctness:")
correctness_passed = verify_correctness(data_with_nan, group_assignment)

print("\n" + "=" * 50)

# Analyze scaling properties
print("5. Scaling Properties:")
analyze_scaling_properties(base_data, group_assignment)

print("\n" + "=" * 50)

# Benchmark performance
print("6. Performance Benchmark:")
benchmark_results = benchmark_group_scale(data_with_nan, group_assignment)

print("\n" + "=" * 50)
print("Usage Examples:")
print("=" * 15)

print("\n# Basic group scaling")
print("scaled = group_scale(data, group_ids)")

print("\n# Handle constant values differently")
print("scaled = group_scale(data, group_ids, handle_constant='zero')")

print("\n# Common use cases:")
print("# Sector-neutral scaling")
print("sector_neutral = group_scale(factor_scores, sector_groups)")

print("\n# Market cap group scaling")
print("size_neutral = group_scale(returns, size_buckets)")

print("\n# Industry relative scaling")
print("industry_relative = group_scale(fundamentals, industry_codes)")

# Real example with the generated data
print(f"\nExample Results:")
example_scaled = group_scale(base_data[:5, :10], group_assignment[:10])

print(f"Input shape: {base_data.shape}")
print(
    f"Groups: {np.unique(group_assignment)} (sizes: {[np.sum(group_assignment == g) for g in np.unique(group_assignment)]})")
print(f"Scaled data range: [{np.nanmin(example_scaled):.3f}, {np.nanmax(example_scaled):.3f}]")

print("\nKey Benefits:")
print("- Group-neutral factor exposure")
print("- Eliminates group-level biases")
print("- Preserves within-group ranking")
print("- Handles missing data gracefully")
print("- Essential for long-short equity strategies")