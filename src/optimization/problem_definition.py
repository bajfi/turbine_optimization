from typing import List

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.models.base_model import BaseModel


class TurbineOptimizationProblem(ElementwiseProblem):
    """
    Definition of the turbine optimization problem for Pymoo.
    This class encapsulates the optimization problem formulation.
    """

    def __init__(
        self, models: List[BaseModel], xl: np.ndarray, xu: np.ndarray, **kwargs
    ):
        """
        Initialize the optimization problem.

        Args:
            models: List of surrogate models for each objective
            xl: Lower bounds for variables
            xu: Upper bounds for variables
            **kwargs: Optional parameters:
                integer_vars: Indices of variables that should be treated as integers
                constraints: List of constraint functions
        """
        self.models = models
        self.integer_vars = kwargs.get("integer_vars", [])
        self.constraints = kwargs.get("constraints", [])

        # Call parent constructor with problem definition
        super().__init__(
            n_var=len(xl),
            n_obj=len(models),
            n_constr=len(self.constraints),
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a solution for the multi-objective optimization problem.

        Args:
            x: Decision variables
            out: Output dictionary

        Returns:
            Dictionary with objective values and constraint values
        """
        # Make a copy of x to avoid modifying the original
        x_copy = x.copy()

        # Enforce integer variables if any
        for idx in self.integer_vars:
            x_copy[idx] = int(round(x_copy[idx]))

        # Reshape for model prediction (models expect 2D input)
        x_reshaped = x_copy.reshape(1, -1)

        # Predict objectives using surrogate models
        objectives = np.array([model.predict(x_reshaped)[0] for model in self.models])

        # Evaluate constraints if any
        if self.constraints:
            constraint_values = np.array(
                [constraint(x_copy) for constraint in self.constraints]
            )
            out["G"] = constraint_values

        # Set objective values
        out["F"] = objectives
