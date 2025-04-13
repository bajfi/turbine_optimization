import os
import sys

import numpy as np
from matplotlib import pyplot as plt

# Add the project root directory to Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), r"../")))

from src.models.base_model import BaseModel
from src.optimization.optimizer import MultiObjectiveOptimizer
from src.optimization.problem_definition import TurbineOptimizationProblem


class DummyModel(BaseModel):
    """
    Dummy model for demonstration purposes.
    """

    def __init__(self, function):
        self.function = function

    def predict(self, X):
        """
        Predict function result from input.

        Args:
            X: Input data

        Returns:
            Predicted values
        """
        return self.function(X)

    def fit(self, X, y):
        """
        Dummy fit method implementation.

        Args:
            X: Training data features
            y: Training data targets
        """
        # Dummy implementation - does nothing
        pass

    def get_params(self):
        """
        Get model parameters.

        Returns:
            Empty parameter dictionary
        """
        # Return empty params dictionary
        return {}

    def set_params(self, **params):
        """
        Set model parameters.

        Args:
            params: Parameters to set
        """
        # Dummy implementation - does nothing
        pass

    def save_model(self, file_path):
        """
        Save model to file.

        Args:
            file_path: Path to save model
        """
        # Dummy implementation - does nothing
        pass

    def load_model(self, file_path):
        """
        Load model from file.

        Args:
            file_path: Path to load model from
        """
        # Dummy implementation - does nothing
        pass


def run_single_algorithm(
    optimizer, algorithm_name, algorithm_params, parameter_names, objective_names
):
    """
    Run optimization with a specific algorithm and get Pareto solutions.

    Args:
        optimizer: The MultiObjectiveOptimizer instance
        algorithm_name: Name of the algorithm to run
        algorithm_params: Parameters for the algorithm
        parameter_names: Names of the parameters
        objective_names: Names of the objectives

    Returns:
        Pandas DataFrame with Pareto solutions
    """
    print(f"\nRunning optimization with {algorithm_name.upper()} algorithm...")
    optimizer.run_optimization(
        algorithm_name=algorithm_name,
        algorithm_params=algorithm_params,
        n_gen=50,
        verbose=True,
    )

    # Get and print Pareto solutions
    pareto_solutions = optimizer.get_pareto_solutions(
        parameter_names=parameter_names, objective_names=objective_names
    )
    print(f"\nPareto solutions with {algorithm_name.upper()}:")
    print(pareto_solutions.head())

    return pareto_solutions


def plot_comparison(pareto_solutions_dict):
    """
    Create a comparative plot of Pareto fronts from different algorithms.

    Args:
        pareto_solutions_dict: Dictionary mapping algorithm names to their Pareto solutions
    """
    plt.figure(figsize=(10, 6))

    for algo_name, solutions in pareto_solutions_dict.items():
        plt.scatter(
            solutions["f1"],
            solutions["f2"],
            label=algo_name.upper(),
            alpha=0.7,
        )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Comparison of Pareto Fronts")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_front_comparison.png")
    plt.show()


def main():
    """
    Main function that demonstrates multi-algorithm optimization.

    Creates dummy objective functions, optimizes them with different algorithms,
    and compares results.
    """

    # Create dummy objective functions
    def objective1(x):
        """
        First objective function: sum of squared values.

        Args:
            x: Input array

        Returns:
            Sum of squared values
        """
        return np.sum(x**2, axis=1)

    def objective2(x):
        """
        Second objective function: sum of squared differences from 2.

        Args:
            x: Input array

        Returns:
            Sum of squared differences from 2
        """
        return np.sum((x - 2) ** 2, axis=1)

    # Create dummy models
    models = [DummyModel(objective1), DummyModel(objective2)]

    # Setup problem definition
    n_var = 3
    xl = np.zeros(n_var)
    xu = np.ones(n_var) * 5.0
    integer_vars = [0]  # First variable is integer

    problem = TurbineOptimizationProblem(
        models=models, xl=xl, xu=xu, integer_vars=integer_vars
    )

    # Create optimizer
    optimizer = MultiObjectiveOptimizer(problem)

    # Show available algorithms
    print("Available algorithms:")
    algorithms = optimizer.get_available_algorithms()
    for name, description in algorithms.items():
        print(f"- {name}: {description}")

    # Parameter and objective names for all algorithms
    parameter_names = ["x1", "x2", "x3"]
    objective_names = ["f1", "f2"]

    # Dictionary to store Pareto solutions from each algorithm
    pareto_solutions = {}

    # Run optimization with NSGA-II
    pareto_solutions["nsga2"] = run_single_algorithm(
        optimizer, "nsga2", {"pop_size": 100}, parameter_names, objective_names
    )

    # Run optimization with MOEA/D
    pareto_solutions["moead"] = run_single_algorithm(
        optimizer, "moead", {"n_neighbors": 10}, parameter_names, objective_names
    )

    # Run optimization with SMS-EMOA
    pareto_solutions["sms-emoa"] = run_single_algorithm(
        optimizer, "sms-emoa", {"pop_size": 80}, parameter_names, objective_names
    )

    # Compare results with visualization
    plot_comparison(pareto_solutions)

    # Get best solutions
    best_solution = optimizer.get_best_solution(objective_index=0)
    print(f"\nBest solution for f1 with most recent algorithm: {best_solution}")

    # Show the last configuration used
    print("\nLast algorithm configuration:")
    print(optimizer.get_last_configuration())


if __name__ == "__main__":
    main()
