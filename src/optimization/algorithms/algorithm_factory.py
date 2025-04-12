from typing import Any, Dict, Optional, Type

from .base_algorithm import OptimizationAlgorithm
from .moead_algorithm import MOEADAlgorithm
from .nsga2_algorithm import NSGA2Algorithm
from .sms_emoa_algorithm import SMSEMOAAlgorithm


class AlgorithmFactory:
    """
    Factory class for creating optimization algorithm instances.
    This follows the Factory pattern to centralize algorithm creation.
    """

    # Registry of available algorithms
    _algorithms: Dict[str, Type[OptimizationAlgorithm]] = {
        "nsga2": NSGA2Algorithm,
        "moead": MOEADAlgorithm,
        "sms-emoa": SMSEMOAAlgorithm,
    }

    @classmethod
    def register_algorithm(
        cls, name: str, algorithm_class: Type[OptimizationAlgorithm]
    ) -> None:
        """
        Register a new algorithm type.

        Args:
            name: Name identifier for the algorithm
            algorithm_class: The algorithm class to register
        """
        cls._algorithms[name.lower()] = algorithm_class

    @classmethod
    def create_algorithm(cls, name: str, **kwargs) -> Optional[OptimizationAlgorithm]:
        """
        Create an instance of the requested algorithm.

        Args:
            name: Name of the algorithm to create
            **kwargs: Parameters to pass to the algorithm setup

        Returns:
            Configured algorithm instance or None if algorithm not found

        Raises:
            ValueError: If the algorithm name is not recognized
        """
        name = name.lower()
        if name not in cls._algorithms:
            raise ValueError(
                f"Unknown algorithm: {name}. Available algorithms: {list(cls._algorithms.keys())}"
            )

        algorithm_class = cls._algorithms[name]
        return algorithm_class()

    @classmethod
    def list_algorithms(cls) -> Dict[str, str]:
        """
        List all available algorithms.

        Returns:
            Dictionary mapping algorithm names to their descriptive names
        """
        return {
            name: algo_class().get_name()
            for name, algo_class in cls._algorithms.items()
        }

    @classmethod
    def get_default_params(cls, name: str) -> Dict[str, Any]:
        """
        Get the default parameters for a specific algorithm.

        Args:
            name: Name of the algorithm

        Returns:
            Dictionary of default parameters

        Raises:
            ValueError: If the algorithm name is not recognized
        """
        name = name.lower()
        if name not in cls._algorithms:
            raise ValueError(
                f"Unknown algorithm: {name}. Available algorithms: {list(cls._algorithms.keys())}"
            )

        algorithm = cls._algorithms[name]()
        return algorithm.get_default_params()
