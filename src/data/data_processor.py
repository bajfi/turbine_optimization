from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader


class DataProcessor:
    """
    Class responsible for processing turbine data for optimization.
    """

    def __init__(self):
        """
        Initialize the data processor.
        """
        self.x_min = None
        self.x_max = None
        self.parameter_indices = {}
        self.integer_indices = []

    def compute_feature_bounds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the min and max bounds for each feature.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (min_bounds, max_bounds)
        """
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        return self.x_min, self.x_max

    def get_optimization_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the optimization bounds for the features.

        Returns:
            Tuple of (min_bounds, max_bounds)
        """
        if self.x_min is None or self.x_max is None:
            raise ValueError(
                "Bounds have not been computed. Call compute_feature_bounds first."
            )

        return self.x_min, self.x_max

    def prepare_training_data(
        self,
        data: pd.DataFrame,
        parameter_mappings: Dict[str, str],
        target_mappings: Dict[str, Dict[str, str]],
        integer_parameters: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare training data for model building with flexible column mappings.

        Args:
            data: DataFrame containing the data
            parameter_mappings: Dictionary mapping parameter names to column names
            target_mappings: Dictionary mapping target names to settings (column and type)
            integer_parameters: List of parameter names that should be treated as integers

        Returns:
            Tuple of (X_train, targets_dict)
        """
        # Create a DataLoader to extract features and targets
        loader = DataLoader()

        # Get features
        features_df, parameter_names = loader.get_parameter_features(
            data, parameter_mappings
        )
        X_train = features_df.values.astype(np.float64)

        # Store parameter indices for optimization
        self.parameter_indices = {param: i for i, param in enumerate(parameter_names)}

        # Get integer parameter indices
        if integer_parameters:
            self.integer_indices = [
                self.parameter_indices[param]
                for param in integer_parameters
                if param in self.parameter_indices
            ]

        # Get targets
        targets_dict = loader.get_target_variables(data, target_mappings)

        # Compute bounds for later use in optimization
        self.compute_feature_bounds(X_train)

        return X_train, targets_dict

    def get_integer_parameter_indices(self) -> List[int]:
        """
        Get the indices of integer parameters for optimization.

        Returns:
            List of integer parameter indices
        """
        return self.integer_indices
