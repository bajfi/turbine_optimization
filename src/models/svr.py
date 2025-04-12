import pickle
from pathlib import Path
from typing import Any, Dict, Union, override

import numpy as np
from sklearn.svm import SVR

from .base_model import BaseModel


class SVRModel(BaseModel):
    """
    Support Vector Regression model implementation using scikit-learn.
    """

    # Define numeric parameters as class attributes for better maintainability
    FLOAT_PARAMS = {"C", "epsilon", "tol", "gamma"}
    INT_PARAMS = {"max_iter"}
    STRING_PARAMS = {"kernel", "verbose"}

    def __init__(self, **kwargs):
        """
        Initialize the SVR model.

        Args:
            kernel: Specifies the kernel type to be used
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            gamma: Kernel coefficient ('scale', 'auto' or float)
            tol: Tolerance for stopping criterion
            max_iter: Hard limit on iterations within solver
            **kwargs: Additional parameters to pass to SVR
        """
        # Start with default params and update with processed kwargs
        params = self.default_params
        processed_kwargs = self._convert_param_types(kwargs)
        params.update(processed_kwargs)

        self.model = SVR(**params)

    @classmethod
    def _convert_param_types(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameters to their appropriate types.

        Args:
            params: Parameters to convert

        Returns:
            Dictionary with converted parameter values
        """
        result = {}

        for name, value in params.items():
            if name in cls.FLOAT_PARAMS:
                result[name] = cls._try_convert_to_float(value)
            elif name in cls.INT_PARAMS:
                result[name] = cls._try_convert_to_int(value)
            else:
                # Keep original value for string parameters or unknown parameters
                result[name] = value

        return result

    @staticmethod
    def _try_convert_to_float(value: Any) -> Union[float, Any]:
        """
        Try to convert a value to float.

        Args:
            value: Value to convert

        Returns:
            Converted float value or original value if conversion fails or isn't needed
        """
        # Skip conversion if already a float or not a string
        if isinstance(value, float) or not isinstance(value, str):
            return value

        # Special case for gamma which can be 'scale' or 'auto'
        if value in {"scale", "auto"}:
            return value

        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _try_convert_to_int(value: Any) -> Union[int, Any]:
        """
        Try to convert a value to int.

        Args:
            value: Value to convert

        Returns:
            Converted int value or original value if conversion fails or isn't needed
        """
        # Skip conversion if already an int or not a string
        if isinstance(value, int) or not isinstance(value, str):
            return value

        try:
            return int(float(value))
        except (ValueError, TypeError):
            return value

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVR model.

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
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "tol": 0.001,
            "verbose": False,
            "max_iter": -1,  # No limit
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
        processed_params = self._convert_param_types(params)
        self.model.set_params(**processed_params)

    @override
    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the trained model.
        Note: SVR does not provide feature importances directly.
        This implementation returns the absolute values of the coefficients
        for linear kernels or raises an error for non-linear kernels.

        Returns:
            Array of feature importances (or coefficients for linear kernel)
        Raises:
            ValueError: If model is not trained or kernel is not linear
        """
        if not hasattr(self.model, "dual_coef_"):
            raise ValueError("Model has not been trained yet")

        if self.model.kernel == "linear" and hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])

        raise ValueError("Feature importances are only available for linear kernel")

    @override
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save the model to a file using pickle.

        Args:
            file_path: Path to save the model to
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @override
    def load_model(self, file_path: Union[str, Path]) -> None:
        """
        Load the model from a file using pickle.

        Args:
            file_path: Path to load the model from
        Raises:
            FileNotFoundError: If the model file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(path, "rb") as f:
            self.model = pickle.load(f)
