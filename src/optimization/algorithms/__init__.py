from .algorithm_factory import AlgorithmFactory
from .base_algorithm import OptimizationAlgorithm
from .moead_algorithm import MOEADAlgorithm
from .nsga2_algorithm import NSGA2Algorithm
from .sms_emoa_algorithm import SMSEMOAAlgorithm

# Export the key classes
__all__ = [
    "AlgorithmFactory",
    "OptimizationAlgorithm",
    "NSGA2Algorithm",
    "MOEADAlgorithm",
    "SMSEMOAAlgorithm",
]

# Simple API for getting available algorithms
get_available_algorithms = AlgorithmFactory.list_algorithms
create_algorithm = AlgorithmFactory.create_algorithm
get_default_params = AlgorithmFactory.get_default_params
register_algorithm = AlgorithmFactory.register_algorithm
