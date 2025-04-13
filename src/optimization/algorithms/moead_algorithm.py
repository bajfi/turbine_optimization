from typing import Any, Dict, override

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from .base_algorithm import OptimizationAlgorithm


class MOEADAlgorithm(OptimizationAlgorithm):
    """
    Implementation of the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) algorithm.
    """

    @override
    def get_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        return "MOEA/D"

    @override
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the MOEA/D algorithm.

        Returns:
            Dictionary of default parameters
        """
        return {
            "n_neighbors": 20,
            "pop_size": 200,
            "crossover_prob": 1,
            "crossover_eta": 20,
            "mutation_eta": 20,
            "sampling": IntegerRandomSampling(),
        }

    @override
    def setup(self, **kwargs) -> MOEAD:
        """
        Setup the MOEA/D algorithm with the specified parameters.

        Args:
            **kwargs: Algorithm-specific parameters including:
                n_neighbors: Number of neighbors
                pop_size: Population size
                crossover_prob: Crossover probability
                crossover_eta: Crossover distribution index
                mutation_eta: Mutation distribution index
                n_obj: Number of objectives (default: 2)

        Returns:
            Configured MOEA/D algorithm
        """
        # Use provided parameters or fall back to defaults
        params = self.get_default_params()
        params.update(kwargs)

        # Create reference directions using Das and Dennis's systematic approach
        n_obj = kwargs.get("n_obj", 2)  # Default to 2 objectives if not specified
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)

        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=params["n_neighbors"],
            sampling=params["sampling"],
            crossover=SBX(prob=params["crossover_prob"], eta=params["crossover_eta"]),
            mutation=PM(eta=params["mutation_eta"]),
        )

        return algorithm
