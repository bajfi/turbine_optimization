from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils.pareto_utils import (
    clean_pareto_front,
    interpolate_pareto_front,
    smooth_pareto_front,
)


class ParetoVisualizer:
    """
    Class for visualizing Pareto fronts.
    """

    @staticmethod
    def plot_pareto_front(
        F: np.ndarray,
        title: str = "Pareto Front",
        x_label: str = "f₁",
        y_label: str = "f₂",
        save_path: Optional[str | Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 9),
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Plot the Pareto front for a bi-objective optimization problem.

        Args:
            F: Objective values, shape (n_solutions, 2)
            title: Plot title
            x_label: Label for x-axis
            y_label: Label for y-axis
            save_path: Path to save the figure
            show_plot: Whether to show the plot
            figsize: Figure size
            dpi: DPI for saved figure

        Returns:
            The matplotlib figure
        """
        if F.shape[1] != 2:
            raise ValueError("This function can only plot 2D Pareto fronts")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot of all solutions
        ax.scatter(
            F[:, 0], F[:, 1], c="red", s=30, alpha=0.8, label="Pareto-optimal solutions"
        )

        # Get cleaned Pareto front
        _, cleaned_F = clean_pareto_front(F)
        x_pareto = cleaned_F[:, 0]
        y_pareto = cleaned_F[:, 1]

        # Plot Pareto front as a connected line
        ax.plot(
            x_pareto,
            y_pareto,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Pareto front",
        )

        # Apply smoothing if enough points
        if len(x_pareto) > 3:
            y_smooth = smooth_pareto_front(x_pareto, y_pareto, sigma=1.5)
            ax.plot(
                x_pareto,
                y_smooth,
                color="green",
                linewidth=2.5,
                label="Smoothed Pareto front",
            )

        # Improve styling
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=12, loc="best")

        # Add annotation showing the number of Pareto-optimal solutions
        ax.annotate(
            f"Number of unique Pareto solutions: {len(x_pareto)}",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "gray",
                "alpha": 0.8,
            },
        )

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig

    @staticmethod
    def plot_interpolated_pareto_front(
        F: np.ndarray,
        num_points: int = 200,
        title: str = "Interpolated Pareto Front",
        x_label: str = "f₁",
        y_label: str = "f₂",
        save_path: Optional[str | Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 9),
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Plot an interpolated Pareto front.

        Args:
            F: Objective values, shape (n_solutions, 2)
            num_points: Number of points to use for interpolation
            title: Plot title
            x_label: Label for x-axis
            y_label: Label for y-axis
            save_path: Path to save the figure
            show_plot: Whether to show the plot
            figsize: Figure size
            dpi: DPI for saved figure

        Returns:
            The matplotlib figure
        """
        if F.shape[1] != 2:
            raise ValueError("This function can only plot 2D Pareto fronts")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot of all solutions
        ax.scatter(
            F[:, 0],
            F[:, 1],
            c="red",
            s=20,
            alpha=0.6,
            label="Original optimal solutions",
        )

        # Get cleaned Pareto front
        _, cleaned_F = clean_pareto_front(F)
        x_pareto = cleaned_F[:, 0]
        y_pareto = cleaned_F[:, 1]

        # Plot Pareto front as a connected line
        ax.plot(
            x_pareto,
            y_pareto,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Discrete Pareto front",
        )

        # Create interpolated Pareto front
        x_interp, y_interp = interpolate_pareto_front(
            x_pareto, y_pareto, num_points=num_points
        )

        # Plot interpolated Pareto front
        ax.plot(
            x_interp,
            y_interp,
            color="green",
            linewidth=2.5,
            label=f"Interpolated Pareto front ({num_points} points)",
        )

        # Improve styling
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=12, loc="best")

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show plot if requested
        if show_plot:
            plt.show()

        return fig
