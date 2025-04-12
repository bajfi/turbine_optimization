from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class OptimizationResultHandler:
    """
    Handles processing and storing of optimization results.
    This class is responsible for converting raw optimization results to useful formats.
    """

    def __init__(self, result=None):
        """
        Initialize the result handler with optional result data.

        Args:
            result: Optional raw optimization result
        """
        self.result = result

    def set_result(self, result) -> None:
        """
        Set the optimization result to process.

        Args:
            result: Raw optimization result
        """
        self.result = result

    def create_result_summary(self, algorithm_name: str, n_gen: int) -> Dict[str, Any]:
        """
        Create a summary dictionary from the optimization result.

        Args:
            algorithm_name: Name of the algorithm used
            n_gen: Number of generations

        Returns:
            Dictionary containing the result summary

        Raises:
            ValueError: If no result has been set
        """
        if self.result is None:
            raise ValueError("No optimization result has been set")

        # Create a more structured result
        result_dict = {
            "X": self.result.X,  # Decision variables
            "F": self.result.F,  # Objective values
            "algorithm": algorithm_name,
            "n_gen": n_gen,
            "n_evals": self.result.algorithm.evaluator.n_eval,
            "exec_time": self.result.exec_time,
        }

        return result_dict

    def get_pareto_solutions(
        self,
        problem=None,
        parameter_names: Optional[List[str]] = None,
        objective_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the Pareto-optimal solutions as a DataFrame.

        Args:
            problem: The optimization problem (needed for integer variables)
            parameter_names: Names of the decision variables
            objective_names: Names of the objectives

        Returns:
            DataFrame containing the Pareto-optimal solutions

        Raises:
            ValueError: If no result has been set
        """
        if self.result is None:
            raise ValueError("No optimization result has been set")

        X = self.result.X
        F = self.result.F

        # Use default names if not provided
        if parameter_names is None:
            parameter_names = [f"var_{i}" for i in range(X.shape[1])]

        if objective_names is None:
            objective_names = [f"obj_{i}" for i in range(F.shape[1])]

        # Create DataFrame with parameters and objectives
        data = {}

        # Add parameters
        for i, name in enumerate(parameter_names):
            # Handle integer variables if problem is provided
            if problem is not None and i in getattr(problem, "integer_vars", []):
                data[name] = [int(round(x[i])) for x in X]
            else:
                data[name] = [x[i] for x in X]

        # Add objectives
        for i, name in enumerate(objective_names):
            data[name] = [f[i] for f in F]

        return pd.DataFrame(data)

    def get_best_solution(self, objective_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Get the best solution for a single objective.

        Args:
            objective_index: Index of the objective to optimize

        Returns:
            Dictionary with the best X and F values

        Raises:
            ValueError: If no result has been set
        """
        if self.result is None:
            raise ValueError("No optimization result has been set")

        # Get the index of the best solution for the specified objective
        best_idx = np.argmin(self.result.F[:, objective_index])

        return {
            "X": self.result.X[best_idx],
            "F": self.result.F[best_idx],
        }
