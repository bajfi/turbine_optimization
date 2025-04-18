import pickle
from pathlib import Path
from typing import Any, Dict, override

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural Network regression model implementation using scikit-learn's MLPRegressor.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Neural Network model.

        Args:
            hidden_layer_sizes: The size of hidden layers
            activation: Activation function
            solver: The solver for weight optimization
            alpha: L2 penalty parameter
            learning_rate: Learning rate schedule
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to MLPRegressor
        """
        params = self.default_params
        params.update(kwargs)
        self.model = MLPRegressor(**params)
        self.scaler = StandardScaler()
        self._is_fitted = False

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Neural Network model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        # Scale features for better convergence
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Scale features using the same scaler used during training
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @override
    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        """
        return {
            "hidden_layer_sizes": (
                100,
                50,
            ),  # Two hidden layers with 100 and 50 neurons
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "tol": 1e-4,
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

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.
        For neural networks, we approximate importance using the magnitudes of weights
        in the first layer.

        Returns:
            Array of feature importances
        """
        if not hasattr(self.model, "coefs_") or len(self.model.coefs_) == 0:
            raise ValueError("Model has not been trained yet")

        # Use first layer weights as feature importance approximation
        return np.abs(self.model.coefs_[0]).sum(axis=1)

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

        # Save both the model and the scaler
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "is_fitted": self._is_fitted,
                },
                f,
            )

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
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self._is_fitted = saved_data["is_fitted"]
