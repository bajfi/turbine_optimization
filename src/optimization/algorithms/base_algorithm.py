from abc import ABC, abstractmethod
from typing import Any, Dict


class OptimizationAlgorithm(ABC):
    """
    Abstract base class defining the interface for optimization algorithms.
    Following the Strategy pattern, this allows different optimization algorithms
    to be used interchangeably.
    """

    @abstractmethod
    def setup(self, **kwargs) -> Any:
        """
        Setup the algorithm with the given parameters.

        Args:
            **kwargs: Algorithm-specific parameters

        Returns:
            The configured algorithm instance
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the algorithm.

        Returns:
            Name of the algorithm
        """
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the algorithm.

        Returns:
            Dictionary of default parameters
        """
        pass
