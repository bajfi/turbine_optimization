#!/usr/bin/env python3
import argparse
import os
import sys

# Add the project root directory to Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import DEFAULT_CONFIG, ConfigManager, init_config

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.optimization.optimizer import MultiObjectiveOptimizer
from src.optimization.problem_definition import TurbineOptimizationProblem
from src.visualization.parameter_plots import ParameterVisualizer
from src.visualization.pareto_plots import ParetoVisualizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Turbine Optimization Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Initialize a new configuration file with default settings",
    )

    return parser.parse_args()


def setup_config(args):
    """Set up the configuration from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ConfigManager instance
    """
    # Initialize configuration
    if args.init_config:
        print(f"Initializing configuration file: {args.config}")
        config_manager = init_config(args.config)
    else:
        # Load configuration
        config_manager = ConfigManager(args.config)
        if not os.path.exists(args.config):
            print(f"Configuration file not found: {args.config}")
            print("Using default configuration...")
            config_manager.config = DEFAULT_CONFIG.copy()

    return config_manager


def load_and_process_data(config_manager):
    """Load and process data according to configuration.

    Args:
        config_manager: ConfigManager instance

    Returns:
        Tuple of (X_train, targets_dict, parameter_names, integer_indices, x_min, x_max)
    """
    # Load data
    data_dir = config_manager.get("data.data_dir", "data")
    data_file = config_manager.get("data.data_file")

    print(f"Loading data from: {os.path.join(data_dir, data_file)}")
    loader = DataLoader(data_dir)
    data = loader.load_data(data_file)

    # Get column mappings from configuration
    column_config = config_manager.get("data.columns", {})
    parameter_mappings = column_config.get("parameters", {})
    target_mappings = column_config.get("targets", {})
    integer_parameters = column_config.get("integer_parameters", [])

    # Validate that we have parameter and target mappings
    if not parameter_mappings:
        raise ValueError(
            "No parameter mappings defined in the configuration. "
            "Please check the 'data.columns.parameters' section."
        )
    if not target_mappings:
        raise ValueError(
            "No target mappings defined in the configuration. "
            "Please check the 'data.columns.targets' section."
        )

    # Process data with flexible column mappings
    processor = DataProcessor()
    X_train, targets_dict = processor.prepare_training_data(
        data, parameter_mappings, target_mappings, integer_parameters
    )
    x_min, x_max = processor.get_optimization_bounds()

    # Get parameter names and indices for integer variables
    parameter_names = list(parameter_mappings.keys())
    integer_indices = processor.get_integer_parameter_indices()

    # Update integer_vars in the config for visualization
    config_manager.set("optimization.integer_vars", integer_indices)

    return (
        X_train,
        targets_dict,
        parameter_names,
        integer_indices,
        x_min,
        x_max,
        target_mappings,
    )


def create_and_train_models(config_manager, X_train, targets_dict, target_names):
    """Create and train surrogate models for each target.

    Args:
        config_manager: ConfigManager instance
        X_train: Training features
        targets_dict: Dictionary of target values
        target_names: List of target names

    Returns:
        List of trained models
    """
    print("Creating surrogate models...")
    models = []

    for target_name in target_names:
        model_config = config_manager.get(f"models.{target_name}_model", {})
        model = ModelFactory.create_model(
            model_config.get("type", "random_forest"), **model_config.get("params", {})
        )
        models.append(model)

    # Train models
    print("Training surrogate models...")
    for i, target_name in enumerate(target_names):
        models[i].fit(X_train, targets_dict[target_name])

    return models


def run_optimization(config_manager, models, x_min, x_max, integer_indices):
    """Run the optimization process.

    Args:
        config_manager: ConfigManager instance
        models: List of trained models
        x_min: Lower bounds for variables
        x_max: Upper bounds for variables
        integer_indices: Indices of integer variables

    Returns:
        Dictionary containing optimization results
    """
    # Setup optimization problem with integer variables from config
    print("Setting up optimization problem...")
    problem = TurbineOptimizationProblem(
        models=models, xl=x_min, xu=x_max, integer_vars=integer_indices
    )

    # Run optimization
    print("Running optimization...")
    optimizer = MultiObjectiveOptimizer(problem)

    # Setup algorithm based on configuration
    algorithm_name = config_manager.get("optimization.algorithm", "nsga2")
    opt_params = config_manager.get("optimization.algorithm_params", {})
    run_params = config_manager.get("optimization.run_params", {})

    return (
        optimizer.run_optimization(
            algorithm_name=algorithm_name,
            algorithm_params=opt_params,
            **run_params,
        ),
        optimizer,
        problem,
    )


def create_visualizations(
    results,
    parameter_names,
    objective_names,
    optimizer,
    vis_config,
    output_dir,
    target_mappings,
):
    """Create visualizations of the optimization results.

    Args:
        results: Optimization results dictionary
        parameter_names: List of parameter names
        objective_names: List of objective names
        optimizer: MultiObjectiveOptimizer instance
        vis_config: Visualization configuration
        output_dir: Output directory for saving plots
        target_mappings: Dictionary mapping targets to their settings
    """
    print("Processing results...")
    pareto_df = optimizer.get_pareto_solutions(
        parameter_names=parameter_names, objective_names=objective_names
    )

    # Get visualization configuration
    dpi = vis_config.get("dpi", 300)
    show_plots = vis_config.get("show_plots", True)
    save_plots = vis_config.get("save_plots", True)

    if not (save_plots or show_plots):
        return pareto_df

    print("Creating visualizations...")

    # Get proper labels for objectives, accounting for objective type
    x_label = objective_names[0]
    y_label = objective_names[1] if len(objective_names) > 1 else "f₂"

    # Get objective types
    objective_types = {}
    for obj_name, obj_info in target_mappings.items():
        if isinstance(obj_info, dict):
            objective_types[obj_name] = obj_info.get("type", "minimize").lower()
        else:
            objective_types[obj_name] = "minimize"

    # Adjust labels based on objective type
    if x_label in objective_types and objective_types[x_label] == "maximize":
        x_label = f"Negative {x_label}"
    if y_label in objective_types and objective_types[y_label] == "maximize":
        y_label = f"Negative {y_label}"

    # Common visualization config
    viz_config = {
        "x_label": x_label,
        "y_label": y_label,
        "show_plot": show_plots,
        "dpi": dpi,
    }

    # Create Pareto front plot
    ParetoVisualizer.plot_pareto_front(
        results["F"],
        title="Pareto Front for Turbine Optimization",
        save_path=os.path.join(output_dir, "pareto_front.png") if save_plots else None,
        **viz_config,
    )

    # Create interpolated Pareto front plot
    ParetoVisualizer.plot_interpolated_pareto_front(
        results["F"],
        num_points=200,
        title="Interpolated Pareto Front",
        save_path=os.path.join(output_dir, "interpolated_pareto_front.png")
        if save_plots
        else None,
        **viz_config,
    )

    # Create parameter relationship plots
    ParameterVisualizer.plot_parameter_pairwise(
        results["X"],
        results["F"],
        parameter_names=parameter_names,
        objective_idx=1
        if len(objective_names) > 1
        else 0,  # Use second objective (usually efficiency) for coloring
        integer_params=optimizer.problem.integer_vars,
        title="Parameter Relationships for Pareto-Optimal Solutions",
        save_path=os.path.join(output_dir, "parameter_relationships.png")
        if save_plots
        else None,
        **viz_config,
    )

    # Create 3D parameter plot if we have at least 3 parameters
    if len(parameter_names) >= 3:
        ParameterVisualizer.plot_3d_parameter_space(
            results["X"],
            results["F"],
            param_indices=[0, 1, 2],  # First three parameters
            parameter_names=parameter_names,
            objective_idx=1
            if len(objective_names) > 1
            else 0,  # Use second objective for coloring
            integer_params=optimizer.problem.integer_vars,
            title="Parameter Space of Pareto-Optimal Solutions",
            save_path=os.path.join(output_dir, "parameter_space_3d.png")
            if save_plots
            else None,
            **viz_config,
        )

    return pareto_df


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    # Setup configuration
    config_manager = setup_config(args)

    # Create output directory if it doesn't exist
    output_dir = config_manager.get("visualization.output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    (
        X_train,
        targets_dict,
        parameter_names,
        integer_indices,
        x_min,
        x_max,
        target_mappings,
    ) = load_and_process_data(config_manager)

    # Create and train models
    objective_names = list(target_mappings.keys())
    models = create_and_train_models(
        config_manager, X_train, targets_dict, objective_names
    )

    # Run optimization
    results, optimizer, _ = run_optimization(
        config_manager, models, x_min, x_max, integer_indices
    )

    # Visualize results
    vis_config = config_manager.get("visualization", {})
    pareto_df = create_visualizations(
        results,
        parameter_names,
        objective_names,
        optimizer,
        vis_config,
        output_dir,
        target_mappings,
    )

    # Save results
    results_file = os.path.join(output_dir, "pareto_solutions.csv")
    print(f"Saving Pareto-optimal solutions to: {results_file}")
    pareto_df.to_csv(results_file, index=False)

    print("Optimization completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
