import os
from typing import Dict, List, Tuple

import pandas as pd


class DataLoader:
    """
    Class responsible for loading and providing access to turbine optimization data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader with the directory containing data files.

        Args:
            data_dir: Path to the directory containing the data files
        """
        self.data_dir = data_dir

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filename: Name of the CSV file to load

        Returns:
            DataFrame containing the loaded data
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        return pd.read_csv(file_path)

    def get_parameter_features(
        self, data: pd.DataFrame, column_mappings: Dict[str, str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract parameter features from the data using column mappings.

        Args:
            data: DataFrame containing the data
            column_mappings: Dictionary mapping parameter names to column names in the data

        Returns:
            Tuple of (features_df, parameter_names)
        """
        parameter_names = list(column_mappings.keys())
        column_names = list(column_mappings.values())

        # Validate that all columns exist in the DataFrame
        missing_columns = [col for col in column_names if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns are missing from the data: {missing_columns}"
            )

        # Extract the features using the column mappings
        features_df = data[column_names].copy()

        # Rename columns to parameter names for consistency
        rename_map = {column_mappings[param]: param for param in parameter_names}
        features_df.rename(columns=rename_map, inplace=True)

        return features_df, parameter_names

    def get_target_variables(
        self, data: pd.DataFrame, target_mappings: Dict[str, Dict[str, str]]
    ) -> Dict[str, pd.Series]:
        """
        Extract target variables from the data using column mappings with objective type.

        Args:
            data: DataFrame containing the data
            target_mappings: Dictionary mapping target names to settings (column and type)

        Returns:
            Dictionary of {target_name: target_series}
        """
        # Get column names from target_mappings
        column_names = [
            target_info["column"] for target_info in target_mappings.values()
        ]

        # Validate that all columns exist in the DataFrame
        missing_columns = [col for col in column_names if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"The following target columns are missing from the data: {missing_columns}"
            )

        # Extract the targets, create a dictionary of name to Series
        targets = {}
        for target_name, target_info in target_mappings.items():
            column_name = target_info["column"]
            obj_type = target_info.get(
                "type", "minimize"
            ).lower()  # Default to minimize

            # Get the raw data
            target_values = data[column_name].values.astype(float)

            # Negate values for maximization (since optimization always minimizes)
            if obj_type == "maximize":
                target_values = -target_values

            targets[target_name] = target_values

        return targets
