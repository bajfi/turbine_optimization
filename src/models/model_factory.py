from typing import Dict, Type

from src.models.base_model import BaseModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.neural_network import NeuralNetworkModel
from src.models.random_forest import RandomForestModel
from src.models.svr import SVRModel
from src.models.xgboost_model import XGBoostModel


class ModelFactory:
    """
    Factory class for creating surrogate models.
    This follows the Factory pattern to create different model instances.
    """

    _models: Dict[str, Type[BaseModel]] = {
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
        "svr": SVRModel,
        "neural_network": NeuralNetworkModel,
        "xgboost": XGBoostModel,
        # Add more model types here as they are implemented
    }

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type with the factory.

        Args:
            model_type: String identifier for the model type
            model_class: The model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"Model class must be a subclass of BaseModel, got {model_class}"
            )

        cls._models[model_type] = model_class

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance of the specified type.

        Args:
            model_type: String identifier for the model type
            **kwargs: Parameters to pass to the model constructor

        Returns:
            Instance of the requested model

        Raises:
            ValueError: If the requested model type is not registered
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}"
            )

        model_class = cls._models[model_type]
        return model_class(**kwargs)

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get a list of all available model types.

        Returns:
            List of available model type names
        """
        return list(cls._models.keys())
