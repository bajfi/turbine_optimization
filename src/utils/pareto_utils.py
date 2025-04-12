from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def identify_pareto_front(F: np.ndarray) -> np.ndarray:
    """
    Identify the Pareto front from a set of solutions.

    Args:
        F: Objective values, shape (n_solutions, n_objectives)

    Returns:
        Boolean array indicating which solutions are on the Pareto front
    """
    n_points = F.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    # Use vectorized operations instead of nested loops
    for i in range(n_points):
        # Only compare against points that are still in the Pareto set
        if is_pareto[i]:
            # Create masks for the dominance check
            # A solution i dominates j if all objectives are <= and at least one is <
            dominates = np.all(F[i] <= F[np.where(is_pareto)[0]], axis=1) & np.any(
                F[i] < F[np.where(is_pareto)[0]], axis=1
            )

            # Don't mark the current point as dominated
            dominates_idx = np.where(is_pareto)[0][dominates]
            dominates_idx = dominates_idx[dominates_idx != i]

            # Update the is_pareto array
            is_pareto[dominates_idx] = False

    return is_pareto


def clean_pareto_front(
    F: np.ndarray, decimal_places: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean the Pareto front by removing duplicates and sorting by the first objective.

    Args:
        F: Objective values, shape (n_solutions, n_objectives)
        decimal_places: Number of decimal places to round to for comparison

    Returns:
        Tuple of (unique_indices, cleaned_F)
    """
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(F, columns=[f"obj_{i}" for i in range(F.shape[1])])

    # Round values for comparison
    rounded_df = df.round(decimal_places)

    # Remove duplicates
    unique_df = rounded_df.drop_duplicates()

    # Get the indices of the unique values
    unique_indices = unique_df.index.values

    # Sort by first objective
    sorted_indices = unique_indices[np.argsort(df.iloc[unique_indices, 0].values)]

    # Get the cleaned Pareto front
    cleaned_F = F[sorted_indices]

    return sorted_indices, cleaned_F


def interpolate_pareto_front(
    x: np.ndarray, y: np.ndarray, num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the Pareto front to create a smooth curve.

    Args:
        x: x-coordinates (first objective)
        y: y-coordinates (second objective)
        num_points: Number of points to interpolate

    Returns:
        Tuple of (x_interp, y_interp)
    """
    # Sort points by x value
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Create unique x values
    unique_x, indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[indices]

    # Create interpolated points
    x_interp = np.linspace(min(unique_x), max(unique_x), num_points)

    # Use linear interpolation for simplicity
    y_interp = np.interp(x_interp, unique_x, unique_y)

    return x_interp, y_interp


def smooth_pareto_front(x: np.ndarray, y: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply Gaussian smoothing to the Pareto front.

    Args:
        x: x-coordinates (first objective)
        y: y-coordinates (second objective)
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Smoothed y values
    """
    if len(x) <= 3:
        return y

    # Sort by x values
    sorted_indices = np.argsort(x)
    y_sorted = y[sorted_indices]

    # Apply Gaussian smoothing
    y_smooth = gaussian_filter1d(y_sorted, sigma=sigma)

    return y_smooth


def find_closest_solution(
    X: np.ndarray, F: np.ndarray, target_f: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Find the solution closest to a target objective value.

    Args:
        X: Decision variables, shape (n_solutions, n_variables)
        F: Objective values, shape (n_solutions, n_objectives)
        target_f: Target objective values, shape (n_objectives,)

    Returns:
        Tuple of (closest_x, closest_index)
    """
    # Calculate distances to target
    distances = np.sqrt(np.sum((F - target_f) ** 2, axis=1))

    # Find the index of the closest solution
    closest_idx = np.argmin(distances)

    return X[closest_idx], closest_idx
