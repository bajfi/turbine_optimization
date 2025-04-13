import itertools
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class ParameterVisualizer:
    """
    Class for visualizing parameter relationships and dependencies.
    """

    @staticmethod
    def plot_parameter_pairwise(
        X: np.ndarray,
        F: np.ndarray,
        parameter_names: List[str],
        objective_idx: int = 0,
        colormap: str = "viridis",
        integer_params: List[int] = None,
        title: str = "Parameter Relationships",
        save_path: Optional[str | Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (18, 12),
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Create pairwise plots showing relationships between parameters.

        Args:
            X: Decision variables, shape (n_solutions, n_parameters)
            F: Objective values, shape (n_solutions, n_objectives)
            parameter_names: Names of the parameters
            objective_idx: Index of the objective to use for coloring
            colormap: Colormap to use
            integer_params: Indices of parameters that should be treated as integers
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to show the plot
            figsize: Figure size
            dpi: DPI for saved figure

        Returns:
            The matplotlib figure
        """
        if integer_params is None:
            integer_params = []

        # Create subplots
        n_params = X.shape[1]
        n_plots = n_params * (n_params - 1) // 2  # Number of pairwise combinations

        # Calculate grid dimensions
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        axs = axs.flatten()

        # Get pairwise parameter combinations
        param_indices = list(itertools.combinations(range(n_params), 2))

        # Initialize i to avoid undefined loop variable issue
        i = 0

        # Create plots for each parameter combination
        for i, (idx1, idx2) in enumerate(param_indices):
            if i >= len(axs):  # In case we calculated wrong
                break

            # Plot parameters with objective as color
            scatter = axs[i].scatter(
                X[:, idx1],
                X[:, idx2],
                c=-F[:, objective_idx],  # Use negative for consistent color scale
                cmap=colormap,
                s=50,
                alpha=0.8,
            )

            axs[i].set_xlabel(parameter_names[idx1], fontsize=12)
            axs[i].set_ylabel(parameter_names[idx2], fontsize=12)
            axs[i].set_title(
                f"{parameter_names[idx1]} vs {parameter_names[idx2]}", fontsize=14
            )
            axs[i].grid(True, linestyle="--", alpha=0.3)

            # Set integer ticks for integer parameters
            if idx1 in integer_params:
                param_values = np.unique(np.round(X[:, idx1])).astype(int)
                axs[i].set_xticks(param_values)

            if idx2 in integer_params:
                param_values = np.unique(np.round(X[:, idx2])).astype(int)
                axs[i].set_yticks(param_values)

        # Turn off any unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        # Add colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Objective Value", fontsize=14)

        # Set overall title
        fig.suptitle(title, fontsize=16)

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    @staticmethod
    def plot_3d_parameter_space(
        X: np.ndarray,
        F: np.ndarray,
        param_indices: List[int],
        parameter_names: List[str],
        objective_idx: int = 0,
        colormap: str = "viridis",
        integer_params: List[int] = None,
        title: str = "Parameter Space",
        save_path: Optional[str | Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Create a 3D visualization of the parameter space.

        Args:
            X: Decision variables, shape (n_solutions, n_parameters)
            F: Objective values, shape (n_solutions, n_objectives)
            param_indices: Indices of three parameters to visualize
            parameter_names: Names of the parameters
            objective_idx: Index of the objective to use for coloring
            colormap: Colormap to use
            integer_params: Indices of parameters that should be treated as integers
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to show the plot
            figsize: Figure size
            dpi: DPI for saved figure

        Returns:
            The matplotlib figure
        """
        if len(param_indices) != 3:
            raise ValueError("param_indices must contain exactly 3 indices")

        if integer_params is None:
            integer_params = []

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Get parameter values
        x_vals = X[:, param_indices[0]]
        y_vals = X[:, param_indices[1]]
        z_vals = X[:, param_indices[2]]

        # Round integer parameters
        if param_indices[0] in integer_params:
            x_vals = np.round(x_vals).astype(int)
        if param_indices[1] in integer_params:
            y_vals = np.round(y_vals).astype(int)
        if param_indices[2] in integer_params:
            z_vals = np.round(z_vals).astype(int)

        # Create 3D scatter plot
        sc = ax.scatter(
            x_vals,
            y_vals,
            z_vals,
            c=-F[:, objective_idx],  # Use negative for consistent color scale
            cmap=colormap,
            s=50,
            alpha=0.8,
        )

        # Set labels
        ax.set_xlabel(parameter_names[param_indices[0]], fontsize=12)
        ax.set_ylabel(parameter_names[param_indices[1]], fontsize=12)
        ax.set_zlabel(parameter_names[param_indices[2]], fontsize=12)
        ax.set_title(title, fontsize=14)

        # Set integer ticks for integer parameters
        if param_indices[0] in integer_params:
            ax.set_xticks(np.unique(x_vals))
        if param_indices[1] in integer_params:
            ax.set_yticks(np.unique(y_vals))
        if param_indices[2] in integer_params:
            ax.set_zticks(np.unique(z_vals))

        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label("Objective Value", fontsize=12)

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig
