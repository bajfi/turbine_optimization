import pickle
from pathlib import Path
from typing import Any, Dict, override

import numpy as np
import xgboost as xgb

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost regression model implementation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model.

        Args:
            n_estimators: Number of gradient boosted trees
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth for base learners
            objective: Specify the learning task and the corresponding learning objective
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to XGBRegressor
        """
        params = self.default_params
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the XGBoost model.

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
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "auto",
            "n_jobs": -1,
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
