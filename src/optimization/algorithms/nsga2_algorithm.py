from typing import Any, Dict, override

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from src.optimization.algorithms.base_algorithm import OptimizationAlgorithm


class NSGA2Algorithm(OptimizationAlgorithm):
    """
    Implementation of the NSGA-II algorithm.
    """

    @override
    def get_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        return "NSGA-II"

    @override
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the NSGA-II algorithm.

        Returns:
            Dictionary of default parameters
        """
        return {
            "pop_size": 200,
            "n_offsprings": 50,
            "crossover_prob": 0.9,
            "crossover_eta": 15,
            "mutation_eta": 20,
            "sampling": IntegerRandomSampling(),
            "eliminate_duplicates": True,
        }

    @override
    def setup(self, **kwargs) -> NSGA2:
        """
        Setup the NSGA-II algorithm with the specified parameters.

        Args:
            **kwargs: Algorithm-specific parameters including:
                pop_size: Population size
                n_offsprings: Number of offspring per generation
                crossover_prob: Crossover probability
                crossover_eta: Crossover distribution index
                mutation_eta: Mutation distribution index

        Returns:
            Configured NSGA-II algorithm
        """
        # Use provided parameters or fall back to defaults
        params = self.get_default_params()
        params.update(kwargs)

        algorithm = NSGA2(
            pop_size=params["pop_size"],
            n_offsprings=params["n_offsprings"],
            sampling=params["sampling"],
            crossover=SBX(prob=params["crossover_prob"], eta=params["crossover_eta"]),
            mutation=PM(eta=params["mutation_eta"]),
            eliminate_duplicates=params["eliminate_duplicates"],
        )

        return algorithm
