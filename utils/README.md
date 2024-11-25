# Utilities Directory

The `utils/` directory contains utility scripts and modules that provide supporting functionalities for the **Dynamic Neural Network Refinement** project. These utilities are designed to enhance code reusability, simplify complex tasks, and streamline development workflows.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Available Utilities](#available-utilities)
  - [data_loader.py](#data_loaderpy)
  - [model_utils.py](#model_utilspy)
  - [logger.py](#loggerpy)
  - [config_loader.py](#config_loaderpy)
  - [visualization_utils.py](#visualization_utilspy)
- [Usage](#usage)
  - [Loading Configurations](#loading-configurations)
  - [Model Management](#model-management)
  - [Logging Setup](#logging-setup)
  - [Data Loading](#data-loading)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Utilities are essential for maintaining clean and efficient codebases. The `utils/` directory centralizes common functions and classes that are used across different modules and scripts in the project. This promotes DRY (Don't Repeat Yourself) principles and facilitates easier maintenance and scalability.

## Directory Structure

````

utils/ ├── data_loader.py ├── model_utils.py ├── logger.py ├── config_loader.py ├── visualization_utils.py └── README.md

````

- **data_loader.py**: Functions for loading and preprocessing data.
- **model_utils.py**: Utilities for model management, such as saving and loading models.
- **logger.py**: Configures and initializes structured logging.
- **config_loader.py**: Functions to load and validate configuration files.
- **visualization_utils.py**: Helper functions for generating and managing visualizations.

## Available Utilities

### `data_loader.py`

**Purpose:**  
Provides functions to load and preprocess datasets, ensuring consistency across training and evaluation processes.

**Key Functions:**

- `load_dataset(path, batch_size, shuffle, num_workers)`: Loads a dataset from the specified path with given parameters.
- `preprocess_data(data)`: Applies preprocessing steps such as normalization, encoding, or augmentation.

**Example Usage:**

```python
from utils.data_loader import load_dataset

train_loader = load_dataset(
    path='data/processed/train_processed.csv',
    batch_size=64,
    shuffle=True,
    num_workers=4
)
````

### `model_utils.py`

**Purpose:**  
Contains utilities for managing models, including saving, loading, and evaluating model states.

**Key Functions:**

- `save_model(model, path)`: Saves the model's state dictionary to the specified path.
- `load_model(model_class, path, device)`: Loads the model's state dictionary from the specified path.
- `get_model_summary(model, input_size)`: Generates a summary of the model architecture.

**Example Usage:**

```python
from utils.model_utils import save_model, load_model
from src.model import DynamicNeuralNetwork

model = DynamicNeuralNetwork(...)
save_model(model, 'models/final/model_v1.0_final.pth')

# Later...
model = load_model(DynamicNeuralNetwork, 'models/final/model_v1.0_final.pth', device='cuda')
```

### `logger.py`

**Purpose:**  
Configures and initializes structured logging for consistent and comprehensive log management across the project.

**Key Components:**

- `setup_logging(log_path)`: Sets up logging configurations and returns a logger instance.

**Example Usage:**

```python
from utils.logger import setup_logging

logger = setup_logging('logs/app.log')
logger.info('Application started successfully.')
```

### `config_loader.py`

**Purpose:**  
Provides functions to load and validate configuration files, ensuring that all necessary parameters are correctly set before execution.

**Key Functions:**

- `load_config(config_path)`: Loads a YAML configuration file from the specified path.
- `validate_config(config, schema_path)`: Validates the loaded configuration against a predefined schema.

**Example Usage:**

```python
from utils.config_loader import load_config, validate_config

config = load_config('config/train_config.yaml')
validate_config(config, 'config/schema.yaml')
```

### `visualization_utils.py`

**Purpose:**  
Contains helper functions to streamline the creation and management of visualizations, enhancing the functionality of visualization scripts.

**Key Functions:**

- `save_plot(fig, path)`: Saves a Matplotlib figure to the specified path.
- `create_directory(path)`: Creates a directory if it does not exist.
- `generate_color_palette(n_colors)`: Generates a color palette for plotting.

**Example Usage:**

```python
from utils.visualization_utils import create_directory, save_plot
import matplotlib.pyplot as plt

create_directory('visualizations/')
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
save_plot(fig, 'visualizations/sample_plot.png')
```

## Usage

### Loading Configurations

Utilize the `config_loader.py` to load and validate configuration files, ensuring that scripts receive the correct parameters.

```python
from utils.config_loader import load_config, validate_config

config = load_config('config/train_config.yaml')
validate_config(config, 'config/schema.yaml')
```

### Model Management

Manage model states seamlessly using `model_utils.py`, enabling easy saving and loading of models.

```python
from utils.model_utils import save_model, load_model
from src.model import DynamicNeuralNetwork

# Initialize and train model
model = DynamicNeuralNetwork(...)
# After training
save_model(model, 'models/final/model_v1.0_final.pth')

# To load the model later
model = load_model(DynamicNeuralNetwork, 'models/final/model_v1.0_final.pth', device='cuda')
```

### Logging Setup

Set up structured logging to track application behavior and performance.

```python
from utils.logger import setup_logging

logger = setup_logging('logs/app.log')
logger.info('Training started.')
```

### Data Loading

Load and preprocess data efficiently using the provided data loader functions.

```python
from utils.data_loader import load_dataset

train_loader = load_dataset(
    path='data/processed/train_processed.csv',
    batch_size=64,
    shuffle=True,
    num_workers=4
)
```

## Best Practices

- **Modular Design:**  
    Keep utility functions modular and focused on single responsibilities to enhance reusability and maintainability.
    
- **Documentation:**  
    Document each utility function with clear descriptions, parameters, and usage examples to facilitate ease of use.
    
- **Error Handling:**  
    Implement robust error handling within utility functions to manage unexpected scenarios gracefully.
    
- **Testing:**  
    Write unit tests for utility functions to ensure their reliability and correctness.
    
- **Consistent Naming:**  
    Use descriptive and consistent naming conventions for utility functions and modules to improve code readability.
    

## Troubleshooting

- **Function Not Found Errors:**
    
    - **Solution:**  
        Ensure that you have correctly imported the utility functions. Verify the function names and module paths.
- **Configuration Loading Issues:**
    
    - **Solution:**  
        Check that the configuration files are correctly formatted in YAML. Use a YAML validator to identify syntax errors.
- **Logging Issues:**
    
    - **Solution:**  
        Ensure that the log directory exists and that the application has write permissions. Verify the log path provided to the `setup_logging` function.
- **Data Loading Failures:**
    
    - **Solution:**  
        Confirm that the data files exist at the specified paths and are accessible. Check for correct file permissions and paths.

## Contributing

Contributions to the utilities are highly encouraged! To contribute:

1. **Fork the Repository**
    
2. **Create a Feature Branch**
    
    ```bash
    git checkout -b feature/add-new-utility
    ```
    
3. **Implement the Utility Function**
    
4. **Commit Your Changes**
    
    ```bash
    git commit -m "feat: add utility function for advanced data preprocessing"
    ```
    
5. **Push to Your Fork**
    
    ```bash
    git push origin feature/add-new-utility
    ```
    
6. **Open a Pull Request**
    
    Provide a clear description of the utility added, its purpose, and usage examples.
    

For detailed guidelines, refer to the [Best Practices](https://chatgpt.com/docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](https://chatgpt.com/LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
