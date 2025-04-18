import pickle
from pathlib import Path
from typing import Any, Dict, override

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from src.models.base_model import BaseModel


class GaussianProcessModel(BaseModel):
    """
    Gaussian Process regression model implementation using scikit-learn.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Gaussian Process model.

        Args:
            kernel: The kernel specifying the covariance function
            alpha: Value added to the diagonal of the kernel matrix during fitting
            optimizer: The optimizer to use for kernel parameter optimization
            n_restarts_optimizer: Number of restarts of the optimizer for finding kernel params
            normalize_y: Whether to normalize the target values
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to GaussianProcessRegressor
        """
        params = self.default_params.copy()

        # Handle special case for kernel parameter
        if "kernel" in kwargs and isinstance(kwargs["kernel"], str):
            kwargs["kernel"] = self._create_kernel_from_string(kwargs["kernel"])

        params.update(kwargs)
        self.model = GaussianProcessRegressor(**params)

    def _create_kernel_from_string(self, kernel_name: str):
        """
        Create a kernel object from a string identifier.

        Args:
            kernel_name: String identifier for the kernel

        Returns:
            A kernel object

        Raises:
            ValueError: If the kernel name is not recognized
        """
        kernel_name = kernel_name.lower()

        if kernel_name == "rbf":
            return RBF(length_scale=1.0)
        if kernel_name == "matern":
            return Matern(length_scale=1.0, nu=2.5)
        if kernel_name == "constant":
            return ConstantKernel(1.0)
        if kernel_name == "white":
            return WhiteKernel(noise_level=0.1)
        if kernel_name == "rq":
            # Combination of kernels to approximate Rational Quadratic
            return ConstantKernel(1.0) * RBF(length_scale=1.0)
        if kernel_name == "default":
            # Default kernel from default_params
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=0.1
            )
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gaussian Process model.

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

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with standard deviation estimates.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Tuple containing:
                - Predicted values of shape (n_samples,)
                - Standard deviation estimates of shape (n_samples,)
        """
        return self.model.predict(X, return_std=True)

    @override
    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        """
        # Default kernel: Matern kernel with a constant term and noise term
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
            noise_level=0.1
        )

        return {
            "kernel": kernel,
            "alpha": 1e-10,  # Small regularization to ensure numerical stability
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 5,
            "normalize_y": True,
            "random_state": 42,
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
        # Handle special case for kernel parameter
        if "kernel" in params and isinstance(params["kernel"], str):
            params["kernel"] = self._create_kernel_from_string(params["kernel"])

        self.model.set_params(**params)

    def get_kernel_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the kernel.

        Returns:
            Dictionary of kernel parameters
        """
        if hasattr(self.model, "kernel_"):
            return self.model.kernel_.get_params()
        return self.model.kernel.get_params()

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
            self.model = pickle.load(f)
