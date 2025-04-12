from typing import Any, Dict, List, Optional

import pandas as pd
from pymoo.optimize import minimize

from .algorithms import create_algorithm, get_available_algorithms
from .problem_definition import TurbineOptimizationProblem
from .result_handler import OptimizationResultHandler


class MultiObjectiveOptimizer:
    """
    Class for performing multi-objective optimization using various algorithms.
    This class delegates algorithm creation to the AlgorithmFactory,
    following both the Strategy and Factory patterns.
    """

    def __init__(self, problem: TurbineOptimizationProblem):
        """
        Initialize the optimizer with a problem definition.

        Args:
            problem: The optimization problem to solve
        """
        self.problem = problem
        self.result_handler = OptimizationResultHandler()
        self._last_algorithm_name = None
        self._last_algorithm_params = None

    def get_available_algorithms(self) -> Dict[str, str]:
        """
        Get a list of available optimization algorithms.

        Returns:
            Dictionary of algorithm identifiers and their descriptive names
        """
        return get_available_algorithms()

    def run_optimization(
        self,
        algorithm_name: str = "nsga2",
        algorithm_params: Optional[Dict[str, Any]] = None,
        n_gen: int = 200,
        seed: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the optimization with the specified algorithm.

        Args:
            algorithm_name: Name of the algorithm to use
            algorithm_params: Parameters for the algorithm setup
            n_gen: Number of generations
            seed: Random seed for reproducibility
            verbose: Whether to print progress information

        Returns:
            Dictionary containing optimization results
        """
        # Store algorithm configuration for reference
        self._last_algorithm_name = algorithm_name
        self._last_algorithm_params = algorithm_params or {}

        # Create the algorithm instance
        algorithm_instance = create_algorithm(algorithm_name)
        algorithm = algorithm_instance.setup(**(algorithm_params or {}))

        # Run the optimization
        result = minimize(
            self.problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose
        )

        # Store and process the result
        self.result_handler.set_result(result)
        return self.result_handler.create_result_summary(
            algorithm_instance.get_name(), n_gen
        )

    def get_pareto_solutions(
        self,
        parameter_names: Optional[List[str]] = None,
        objective_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the Pareto-optimal solutions as a DataFrame.

        Args:
            parameter_names: Names of the decision variables
            objective_names: Names of the objectives

        Returns:
            DataFrame containing the Pareto-optimal solutions

        Raises:
            ValueError: If no optimization has been run yet
        """
        return self.result_handler.get_pareto_solutions(
            problem=self.problem,
            parameter_names=parameter_names,
            objective_names=objective_names,
        )

    def get_best_solution(self, objective_index: int = 0) -> Dict[str, Any]:
        """
        Get the best solution for a single objective.

        Args:
            objective_index: Index of the objective to optimize

        Returns:
            Dictionary with the best X and F values
        """
        return self.result_handler.get_best_solution(objective_index)

    def get_last_configuration(self) -> Dict[str, Any]:
        """
        Get the configuration used in the last optimization run.

        Returns:
            Dictionary with algorithm configuration
        """
        if self._last_algorithm_name is None:
            return {}

        return {
            "algorithm": self._last_algorithm_name,
            "parameters": self._last_algorithm_params,
        }
