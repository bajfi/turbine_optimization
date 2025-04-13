import os
import pickle
from typing import Any, Dict, override

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.models.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting regression model implementation using scikit-learn.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to GradientBoostingRegressor
        """
        params = self.default_params
        params.update(kwargs)
        self.model = GradientBoostingRegressor(**params)

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gradient Boosting model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        self.model.fit(X, y)

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        return self.model.predict(X)

    @override
    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        """
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "random_state": 42,
            "verbose": 0,
        }

    @override
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the model.

        Returns:
            Dictionary of model parameters
        """
        return self.model.get_params()

    @override
    def set_params(self, **params) -> None:
        """
        Set the parameters of the model.

        Args:
            **params: Model parameters
        """
        self.model.set_params(**params)

    @override
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.

        Returns:
            Array of feature importances
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        raise ValueError("Model has not been trained yet")

    @override
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

    @override
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
