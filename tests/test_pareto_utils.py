import numpy as np

from src.utils.pareto_utils import (
    clean_pareto_front,
    find_closest_solution,
    identify_pareto_front,
    interpolate_pareto_front,
    smooth_pareto_front,
)


class TestIdentifyParetoFront:
    def test_simple_2d_case(self):
        # Minimize both objectives
        F = np.array(
            [
                [2, 3],  # Pareto optimal
                [1, 5],  # Pareto optimal
                [3, 2],  # Pareto optimal
                [5, 1],  # Pareto optimal
                [4, 4],  # Dominated
                [2, 4],  # Dominated
                [3, 3],  # Dominated
            ]
        )

        expected = np.array([True, True, True, True, False, False, False])
        result = identify_pareto_front(F)

        np.testing.assert_array_equal(result, expected)

    def test_3d_objectives(self):
        # Minimize all three objectives
        F = np.array(
            [
                [1, 2, 3],  # Pareto optimal
                [2, 1, 3],  # Pareto optimal
                [3, 2, 1],  # Pareto optimal
                [2, 2, 2],  # Non-dominated (in strict Pareto sense)
                [3, 3, 3],  # Dominated
            ]
        )

        expected = np.array([True, True, True, True, False])
        result = identify_pareto_front(F)

        np.testing.assert_array_equal(result, expected)

    def test_identical_solutions(self):
        # Solutions with identical objective values
        F = np.array(
            [
                [1, 2],
                [1, 2],  # Duplicate
                [2, 1],
                [3, 3],
            ]
        )

        expected = np.array([True, True, True, False])
        result = identify_pareto_front(F)

        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        F = np.array([]).reshape(0, 2)
        result = identify_pareto_front(F)
        assert len(result) == 0


class TestCleanParetoFront:
    def test_remove_duplicates(self):
        F = np.array(
            [
                [1.00001, 2.00001],
                [1.00002, 2.00002],  # Duplicate when rounded
                [2, 1],
                [3, 3],
            ]
        )

        sorted_indices, cleaned_F = clean_pareto_front(F, decimal_places=3)

        # Should have 3 unique solutions when rounded to 3 decimal places
        assert len(sorted_indices) == 3
        assert cleaned_F.shape[0] == 3

        # First objective should be sorted
        assert np.all(np.diff(cleaned_F[:, 0]) >= 0)

    def test_no_duplicates(self):
        F = np.array(
            [
                [1, 5],
                [2, 4],
                [3, 3],
                [4, 2],
                [5, 1],
            ]
        )

        sorted_indices, cleaned_F = clean_pareto_front(F)

        # All solutions should be kept
        assert len(sorted_indices) == 5
        assert cleaned_F.shape[0] == 5

        # First objective should be sorted
        assert np.all(np.diff(cleaned_F[:, 0]) >= 0)


class TestInterpolateParetoFront:
    def test_interpolation(self):
        x = np.array([1, 3, 5, 7])
        y = np.array([7, 5, 3, 1])

        x_interp, y_interp = interpolate_pareto_front(x, y, num_points=7)

        # Check dimensions
        assert len(x_interp) == 7
        assert len(y_interp) == 7

        # Check boundary values
        assert x_interp[0] == 1
        assert x_interp[-1] == 7
        assert y_interp[0] == 7
        assert y_interp[-1] == 1

        # Check if points are evenly distributed
        assert np.allclose(np.diff(x_interp), (7 - 1) / 6)

    def test_with_duplicate_x_values(self):
        x = np.array([1, 2, 2, 3])
        y = np.array([4, 3, 2, 1])

        x_interp, y_interp = interpolate_pareto_front(x, y, num_points=5)

        # Only the first occurrence of each x-value should be used
        assert len(x_interp) == 5
        assert x_interp[0] == 1
        assert x_interp[-1] == 3

        # y value for x=2 should be 3 (first occurrence), not 2
        mid_idx = len(x_interp) // 2
        assert x_interp[mid_idx] == 2
        assert y_interp[mid_idx] == 3


class TestSmoothParetoFront:
    def test_smoothing(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])

        y_smooth = smooth_pareto_front(x, y, sigma=1.0)

        # Smoothed array should have the same length
        assert len(y_smooth) == len(y)

        # For this simple linear case, the smoothed values should be close to original
        # but not identical due to Gaussian smoothing
        assert not np.array_equal(y_smooth, y)

        # Values should be sorted according to x
        x_sorted_indices = np.argsort(x)
        assert np.array_equal(y_smooth, y_smooth[x_sorted_indices])

    def test_small_array(self):
        # Test with array of length <= 3
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])

        y_smooth = smooth_pareto_front(x, y)

        # For small arrays, original values should be returned
        np.testing.assert_array_equal(y_smooth, y)


class TestFindClosestSolution:
    def test_exact_match(self):
        X = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
            ]
        )

        F = np.array(
            [
                [10, 20],
                [20, 10],
                [15, 15],
            ]
        )

        target_f = np.array([15, 15])

        closest_x, closest_idx = find_closest_solution(X, F, target_f)

        # Should find the exact match
        assert closest_idx == 2
        np.testing.assert_array_equal(closest_x, X[2])

    def test_approximate_match(self):
        X = np.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
            ]
        )

        F = np.array(
            [
                [10, 20],
                [20, 10],
                [16, 16],
            ]
        )

        target_f = np.array([15, 15])

        closest_x, closest_idx = find_closest_solution(X, F, target_f)

        # Should find the closest match (Euclidean distance)
        assert closest_idx == 2
        np.testing.assert_array_equal(closest_x, X[2])
