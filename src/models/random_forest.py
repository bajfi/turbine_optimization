import os
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest regression model implementation using scikit-learn.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to RandomForestRegressor
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the model.

        Returns:
            Dictionary of model parameters
        """
        return self.model.get_params()

    def set_params(self, **params) -> None:
        """
        Set the parameters of the model.

        Args:
            **params: Model parameters
        """
        self.model.set_params(**params)

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.

        Returns:
            Array of feature importances
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        else:
            raise ValueError("Model has not been trained yet")

    def save_model(self, file_path: str) -> None:
        """
        Save the model to a file using pickle.

        Args:
            file_path: Path to save the model to
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path: str) -> None:
        """
        Load the model from a file using pickle.

        Args:
            file_path: Path to load the model from
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
