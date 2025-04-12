from typing import Any, Dict, override

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from .base_algorithm import OptimizationAlgorithm


class SMSEMOAAlgorithm(OptimizationAlgorithm):
    """
    Implementation of the SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm).
    This algorithm focuses on maximizing the hypervolume indicator.
    """

    @override
    def get_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        return "SMS-EMOA"

    @override
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the SMS-EMOA algorithm.

        Returns:
            Dictionary of default parameters
        """
        return {
            "pop_size": 100,
            "n_offsprings": 50,
            "crossover_prob": 0.9,
            "crossover_eta": 15,
            "mutation_eta": 20,
            "sampling": IntegerRandomSampling(),
            "eliminate_duplicates": True,
        }

    @override
    def setup(self, **kwargs) -> SMSEMOA:
        """
        Setup the SMS-EMOA algorithm with the specified parameters.

        Args:
            **kwargs: Algorithm-specific parameters including:
                pop_size: Population size
                crossover_prob: Crossover probability
                crossover_eta: Crossover distribution index
                mutation_eta: Mutation distribution index

        Returns:
            Configured SMS-EMOA algorithm
        """
        # Use provided parameters or fall back to defaults
        params = self.get_default_params()
        params.update(kwargs)

        algorithm = SMSEMOA(
            pop_size=params["pop_size"],
            sampling=params["sampling"],
            crossover=SBX(prob=params["crossover_prob"], eta=params["crossover_eta"]),
            mutation=PM(eta=params["mutation_eta"]),
            eliminate_duplicates=params["eliminate_duplicates"],
            n_offsprings=params["n_offsprings"],
        )

        return algorithm
