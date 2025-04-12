import numpy as np
from matplotlib import pyplot as plt

from ..models.base_model import BaseModel
from ..optimization.optimizer import MultiObjectiveOptimizer
from ..optimization.problem_definition import TurbineOptimizationProblem


class DummyModel(BaseModel):
    """
    Dummy model for demonstration purposes.
    """

    def __init__(self, function):
        self.function = function

    def predict(self, x):
        return self.function(x)

    def fit(self, X, y):
        # Dummy implementation - does nothing
        pass

    def get_params(self):
        # Return empty params dictionary
        return {}

    def set_params(self, **params):
        # Dummy implementation - does nothing
        pass

    def save_model(self, file_path):
        # Dummy implementation - does nothing
        pass

    def load_model(self, file_path):
        # Dummy implementation - does nothing
        pass


def main():
    # Create dummy objective functions
    def objective1(x):
        return np.sum(x**2, axis=1)

    def objective2(x):
        return np.sum((x - 2) ** 2, axis=1)

    # Create dummy models
    models = [DummyModel(objective1), DummyModel(objective2)]

    # Setup problem definition
    n_var = 3
    xl = np.zeros(n_var)
    xu = np.ones(n_var) * 5.0
    integer_vars = [0]  # First variable is integer

    problem = TurbineOptimizationProblem(
        models=models, xl=xl, xu=xu, integer_vars=integer_vars
    )

    # Create optimizer
    optimizer = MultiObjectiveOptimizer(problem)

    # Show available algorithms
    print("Available algorithms:")
    algorithms = optimizer.get_available_algorithms()
    for name, description in algorithms.items():
        print(f"- {name}: {description}")

    # Run optimization with NSGA-II
    print("\nRunning optimization with NSGA-II algorithm...")
    optimizer.run_optimization(
        algorithm_name="nsga2",
        algorithm_params={"pop_size": 100},
        n_gen=50,
        verbose=True,
    )

    # Get and print Pareto solutions
    pareto_solutions_nsga2 = optimizer.get_pareto_solutions(
        parameter_names=["x1", "x2", "x3"], objective_names=["f1", "f2"]
    )
    print("\nPareto solutions with NSGA-II:")
    print(pareto_solutions_nsga2.head())

    # Run optimization with MOEA/D
    print("\nRunning optimization with MOEA/D algorithm...")
    optimizer.run_optimization(
        algorithm_name="moead",
        algorithm_params={"n_neighbors": 10},
        n_gen=50,
        verbose=True,
    )

    # Get and print Pareto solutions
    pareto_solutions_moead = optimizer.get_pareto_solutions(
        parameter_names=["x1", "x2", "x3"], objective_names=["f1", "f2"]
    )
    print("\nPareto solutions with MOEA/D:")
    print(pareto_solutions_moead.head())

    # Run optimization with SMS-EMOA
    print("\nRunning optimization with SMS-EMOA algorithm...")
    optimizer.run_optimization(
        algorithm_name="sms-emoa",
        algorithm_params={"pop_size": 80},
        n_gen=50,
        verbose=True,
    )

    # Get and print Pareto solutions
    pareto_solutions_smsemoa = optimizer.get_pareto_solutions(
        parameter_names=["x1", "x2", "x3"], objective_names=["f1", "f2"]
    )
    print("\nPareto solutions with SMS-EMOA:")
    print(pareto_solutions_smsemoa.head())

    # Compare results with visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(
        pareto_solutions_nsga2["f1"],
        pareto_solutions_nsga2["f2"],
        label="NSGA-II",
        alpha=0.7,
    )
    plt.scatter(
        pareto_solutions_moead["f1"],
        pareto_solutions_moead["f2"],
        label="MOEA/D",
        alpha=0.7,
    )
    plt.scatter(
        pareto_solutions_smsemoa["f1"],
        pareto_solutions_smsemoa["f2"],
        label="SMS-EMOA",
        alpha=0.7,
    )
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Comparison of Pareto Fronts")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_front_comparison.png")
    plt.show()

    # Get best solutions
    best_nsga2 = optimizer.get_best_solution(objective_index=0)
    print(f"\nBest solution for f1 with most recent algorithm: {best_nsga2}")

    # Show the last configuration used
    print("\nLast algorithm configuration:")
    print(optimizer.get_last_configuration())


if __name__ == "__main__":
    main()
