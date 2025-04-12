from typing import Any, Dict, List

import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from .problem_definition import TurbineOptimizationProblem


class MultiObjectiveOptimizer:
    """
    Class for performing multi-objective optimization using various algorithms.
    This class follows the Strategy pattern by allowing different optimization
    algorithms to be selected at runtime.
    """

    def __init__(self, problem: TurbineOptimizationProblem):
        """
        Initialize the optimizer with a problem definition.

        Args:
            problem: The optimization problem to solve
        """
        self.problem = problem
        self.result = None

    def setup_nsga2(
        self,
        pop_size: int = 200,
        n_offsprings: int = 50,
        crossover_prob: float = 0.9,
        crossover_eta: float = 15,
        mutation_eta: float = 20,
    ) -> NSGA2:
        """
        Setup the NSGA-II algorithm with the specified parameters.

        Args:
            pop_size: Population size
            n_offsprings: Number of offspring per generation
            crossover_prob: Crossover probability
            crossover_eta: Crossover distribution index
            mutation_eta: Mutation distribution index

        Returns:
            Configured NSGA-II algorithm
        """
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(eta=mutation_eta),
            eliminate_duplicates=True,
        )

        return algorithm

    def run_optimization(
        self, algorithm=None, n_gen: int = 200, seed: int = 1, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run the optimization algorithm.

        Args:
            algorithm: The optimization algorithm to use (if None, NSGA-II will be used)
            n_gen: Number of generations
            seed: Random seed for reproducibility
            verbose: Whether to print progress information

        Returns:
            Dictionary containing optimization results
        """
        if algorithm is None:
            algorithm = self.setup_nsga2()

        # Run the optimization
        self.result = minimize(
            self.problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose
        )

        # Create a more structured result
        result_dict = {
            "X": self.result.X,  # Decision variables
            "F": self.result.F,  # Objective values
            "algorithm": algorithm.__class__.__name__,
            "n_gen": n_gen,
            "n_evals": self.result.algorithm.evaluator.n_eval,
            "exec_time": self.result.exec_time,
        }

        return result_dict

    def get_pareto_solutions(
        self, parameter_names: List[str] = None, objective_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Get the Pareto-optimal solutions as a DataFrame.

        Args:
            parameter_names: Names of the decision variables
            objective_names: Names of the objectives

        Returns:
            DataFrame containing the Pareto-optimal solutions
        """
        if self.result is None:
            raise ValueError(
                "No optimization has been run yet. Call run_optimization first."
            )

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
            # Handle integer variables
            if i in getattr(self.problem, "integer_vars", []):
                data[name] = [int(round(x[i])) for x in X]
            else:
                data[name] = [x[i] for x in X]

        # Add objectives
        for i, name in enumerate(objective_names):
            data[name] = [f[i] for f in F]

        return pd.DataFrame(data)
