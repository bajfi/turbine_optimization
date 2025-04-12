from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all surrogate models.
    Following the Strategy pattern, this allows different model implementations
    to be used interchangeably.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        pass

    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        """
        return {}

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the model.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set the parameters of the model.

        Args:
            **params: Model parameters
        """
        pass

    @abstractmethod
    def save_model(self, file_path: str | Path) -> None:
        """
        Save the model to a file.

        Args:
            file_path: Path to save the model to
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str | Path) -> None:
        """
        Load the model from a file.

        Args:
            file_path: Path to load the model from
        """
        pass
