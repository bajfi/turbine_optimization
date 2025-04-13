import pickle
from pathlib import Path
from typing import Any, Dict, override

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest regression model implementation using scikit-learn.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to RandomForestRegressor
        """
        params = self.default_params
        params.update(kwargs)
        self.model = RandomForestRegressor(**params)

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.

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
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
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
    def save_model(self, file_path: str | Path) -> None:
        """
        Save the model to a file using pickle.

        Args:
            file_path: Path to save the model to
        """
        directory = Path(file_path).parent
        if directory and not directory.exists():
            directory.mkdir(parents=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    @override
    def load_model(self, file_path: str | Path) -> None:
        """
        Load the model from a file using pickle.

        Args:
            file_path: Path to load the model from
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
