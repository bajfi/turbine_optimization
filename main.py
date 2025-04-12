#!/usr/bin/env python3
import argparse
import os

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


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

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

    # Create output directory if it doesn't exist
    output_dir = config_manager.get("visualization.output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

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

    # Create models for each target
    print("Creating surrogate models...")
    models = []
    objective_names = list(target_mappings.keys())
    for target_name in objective_names:
        model_config = config_manager.get(f"models.{target_name}_model", {})
        model = ModelFactory.create_model(
            model_config.get("type", "random_forest"), **model_config.get("params", {})
        )
        models.append(model)

    # Train models
    print("Training surrogate models...")
    for i, target_name in enumerate(objective_names):
        models[i].fit(X_train, targets_dict[target_name])

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

    # Run optimization
    opt_params = config_manager.get("optimization.algorithm_params", {})
    run_params = config_manager.get("optimization.run_params", {})
    results = optimizer.run_optimization(
        algorithm_name=algorithm_name,
        algorithm_params=opt_params,
        **run_params,
    )

    # Get Pareto solutions with the parameter names from config
    print("Processing results...")
    pareto_df = optimizer.get_pareto_solutions(
        parameter_names=parameter_names, objective_names=objective_names
    )

    # Visualize results
    vis_config = config_manager.get("visualization", {})
    dpi = vis_config.get("dpi", 300)
    show_plots = vis_config.get("show_plots", True)
    save_plots = vis_config.get("save_plots", True)

    if save_plots or show_plots:
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

        # Create Pareto front plot
        ParetoVisualizer.plot_pareto_front(
            results["F"],
            title="Pareto Front for Turbine Optimization",
            x_label=x_label,
            y_label=y_label,
            save_path=os.path.join(output_dir, "pareto_front.png")
            if save_plots
            else None,
            show_plot=show_plots,
            dpi=dpi,
        )

        # Create interpolated Pareto front plot
        ParetoVisualizer.plot_interpolated_pareto_front(
            results["F"],
            num_points=200,
            title="Interpolated Pareto Front",
            x_label=x_label,
            y_label=y_label,
            save_path=os.path.join(output_dir, "interpolated_pareto_front.png")
            if save_plots
            else None,
            show_plot=show_plots,
            dpi=dpi,
        )

        # Create parameter relationship plots
        ParameterVisualizer.plot_parameter_pairwise(
            results["X"],
            results["F"],
            parameter_names=parameter_names,
            objective_idx=1
            if len(objective_names) > 1
            else 0,  # Use second objective (usually efficiency) for coloring
            integer_params=integer_indices,
            title="Parameter Relationships for Pareto-Optimal Solutions",
            save_path=os.path.join(output_dir, "parameter_relationships.png")
            if save_plots
            else None,
            show_plot=show_plots,
            dpi=dpi,
        )

        # Create 3D parameter plot
        if len(parameter_names) >= 3:
            ParameterVisualizer.plot_3d_parameter_space(
                results["X"],
                results["F"],
                param_indices=[0, 1, 2],  # First three parameters
                parameter_names=parameter_names,
                objective_idx=1
                if len(objective_names) > 1
                else 0,  # Use second objective for coloring
                integer_params=integer_indices,
                title="Parameter Space of Pareto-Optimal Solutions",
                save_path=os.path.join(output_dir, "parameter_space_3d.png")
                if save_plots
                else None,
                show_plot=show_plots,
                dpi=dpi,
            )

    # Save results
    results_file = os.path.join(output_dir, "pareto_solutions.csv")
    print(f"Saving Pareto-optimal solutions to: {results_file}")
    pareto_df.to_csv(results_file, index=False)

    print("Optimization completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
