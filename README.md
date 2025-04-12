# Turbine Optimization Framework

This framework provides tools for optimizing turbine designs using surrogate modeling and multi-objective optimization techniques.

## Project Structure

```bash
turbine_optimization/
├── config/               # Configuration files
├── data/                 # Data storage
├── src/                  # Source code
│   ├── data/             # Data access and processing
│   ├── models/           # Model building and evaluation
│   ├── optimization/     # Optimization algorithms
│   ├── visualization/    # Plotting and visualization
│   └── utils/            # Utility functions
├── tests/                # Test suite
└── main.py               # Entry point
```

## Features

- **Modular Design**: Each component is designed to be replaceable and extensible
- **Multiple Surrogate Models**: Support for different surrogate models (currently RandomForest)
- **Multi-Objective Optimization**: Using NSGA-II and other algorithms
- **Visualization Tools**: Comprehensive plotting of Pareto fronts and parameter relationships
- **Configuration System**: Easy configuration via YAML or JSON files
- **Flexible Data Handling**: Configurable column mappings to work with any data format

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- Pymoo
- Matplotlib
- PyYAML

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/turbine_optimization.git
   cd turbine_optimization
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Initialize configuration (optional):

   ```bash
   python main.py --init-config
   ```

## Usage

### Basic Usage

```bash
python main.py
```

This will run the optimization with default settings, looking for data in the `data/` directory.

### Configuration

You can specify a custom configuration file:

```bash
python main.py --config path/to/config.yaml
```

### Column Mapping Configuration

The framework allows flexible column mapping to work with any CSV data format. You can configure which columns to use as input parameters and target variables in the configuration file.

The configuration uses generic parameter and target naming:

- Parameters (inputs): x1, x2, x3, ...
- Targets (outputs/objectives): y1, y2, ...

Example configuration:

```yaml
data:
  columns:
    # Map generic parameter names to column names in your CSV
    parameters:
      x1: blads            # First parameter maps to 'blads' column
      x2: angle-wrap       # Second parameter maps to 'baojiao' column
      x3: angle-inlet      # Third parameter maps to 'angle_in' column
      x4: angle-outlet     # Fourth parameter maps to 'angle_out' column
    
    # Map generic target names to column names
    targets:
      y1:
        column: power      # First objective maps to 'power' column
        type: maximize     # We want to maximize this objective
      y2:
        column: efficiency # Second objective maps to 'efficiency' column
        type: maximize     # We want to maximize this objective
    
    # Specify which parameters should be treated as integers
    integer_parameters:
      - x1
```

#### Target Configuration

For target variables (objectives), you can specify:

- `column`: The column name in the CSV file
- `type`: Whether to `maximize` or `minimize` the objective

This is important for multi-objective optimization, as it determines how the objective is treated. For `maximize` objectives, the values are automatically negated internally (since the optimizer always minimizes).

This allows you to use any CSV data file without needing to modify the code.

### Adding New Models

To add a new surrogate model:

1. Create a new model class that extends `BaseModel` in `src/models/`
2. Register the model in `ModelFactory`
3. Update your configuration to use the new model

Example for adding a Neural Network model:

```python
# src/models/neural_network.py
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    # Implement required methods
    ...

# Register in model_factory.py
ModelFactory.register_model('neural_network', NeuralNetworkModel)
```

Then in your configuration:

```yaml
models:
  y1_model:
    type: neural_network
    params:
      hidden_layers: [64, 32]
      activation: relu
```

## Examples

The following examples demonstrate how to use the framework:

### Basic Optimization

```python
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.optimization.problem_definition import TurbineOptimizationProblem
from src.optimization.optimizer import MultiObjectiveOptimizer

# Load data
loader = DataLoader("data")
data = loader.load_data("turbine_data.csv")

# Define column mappings using generic names
parameter_mappings = {
    "x1": "blads",     # First parameter (integer)
    "x2": "baojiao",   # Second parameter
    "x3": "angle_in",  # Third parameter
    "x4": "angle_out"  # Fourth parameter
}
target_mappings = {
    "y1": {
        "column": "power",
        "type": "maximize"  # We want to maximize power
    },
    "y2": {
        "column": "efficiency",
        "type": "maximize"  # We want to maximize efficiency
    }
}
integer_parameters = ["x1"]

# Process data
processor = DataProcessor()
X_train, targets_dict = processor.prepare_training_data(
    data, parameter_mappings, target_mappings, integer_parameters
)

# Create and train models
models = []
for target_name in target_mappings.keys():
    model = ModelFactory.create_model('random_forest')
    model.fit(X_train, targets_dict[target_name])
    models.append(model)

# Setup optimization
problem = TurbineOptimizationProblem(
    models=models,
    xl=processor.x_min,
    xu=processor.x_max,
    integer_vars=processor.get_integer_parameter_indices()
)

# Run optimization
optimizer = MultiObjectiveOptimizer(problem)
results = optimizer.run_optimization()
```

## Documentation

For more detailed documentation, see the docstrings in the code or the [project wiki](link-to-wiki).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
