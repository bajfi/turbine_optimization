import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, override

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural Network regression model implementation using scikit-learn MLPRegressor.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Neural Network model.

        Args:
            hidden_layer_sizes: The sizes of hidden layers
            activation: Activation function for the hidden layers
            solver: The solver for weight optimization
            alpha: L2 penalty parameter
            learning_rate: Learning rate schedule
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to MLPRegressor
        """
        params = self.default_params

        # Convert hidden_layer_sizes if provided in kwargs
        if "hidden_layer_sizes" in kwargs:
            kwargs["hidden_layer_sizes"] = self._convert_hidden_layer_sizes(
                kwargs["hidden_layer_sizes"]
            )

        # Update with provided parameters
        params.update(kwargs)

        self.model = MLPRegressor(**params)

    @staticmethod
    def _convert_hidden_layer_sizes(hidden_sizes: Any) -> Tuple[int, ...]:
        """
        Convert various formats of hidden_layer_sizes to the tuple format required by MLPRegressor.

        Args:
            hidden_sizes: Hidden layer sizes in various formats (string, list, tuple)

        Returns:
            Tuple of integers representing hidden layer sizes
        """
        if isinstance(hidden_sizes, str):
            # Handle string format like "(100, 50)"
            if hidden_sizes.startswith("(") and hidden_sizes.endswith(")"):
                return tuple(int(x.strip()) for x in hidden_sizes[1:-1].split(","))
            # Handle other string formats if needed

        # Convert list or numpy array to tuple
        if isinstance(hidden_sizes, (list, np.ndarray)):
            return tuple(hidden_sizes)

        # Already a tuple or other format
        return hidden_sizes

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Neural Network model.

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
            "shuffle": True,
            "random_state": 42,
            "tol": 1e-4,
            "verbose": False,
            "early_stopping": False,
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
        # Handle hidden_layer_sizes conversion
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = self._convert_hidden_layer_sizes(
                params["hidden_layer_sizes"]
            )

        self.model.set_params(**params)

    @override
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.
        Note: MLPRegressor does not provide feature importances directly.
        This implementation returns the absolute values of the weights
        from the first layer as a rough approximation of feature importance.

        Returns:
            Array of feature importances
        """
        if not hasattr(self.model, "coefs_"):
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
