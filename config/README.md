# Configurations Directory

The `config/` directory houses all configuration files essential for the **Dynamic Neural Network Refinement** project. These configuration files define various parameters and settings used across different components of the project, ensuring consistency and flexibility in operations such as training, evaluation, pruning, quantization, and deployment.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
  - [train_config.yaml](#train_configyaml)
  - [eval_config.yaml](#eval_configyaml)
  - [nas_config.yaml](#nas_configyaml)
  - [prune_config.yaml](#prune_configyaml)
  - [quantize_config.yaml](#quantize_configyaml)
  - [docker-compose.yml](#docker-composeyml)
- [Usage](#usage)
  - [Modifying Configurations](#modifying-configurations)
  - [Loading Configurations in Scripts](#loading-configurations-in-scripts)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Directory Structure

```
config/
├── train_config.yaml
├── eval_config.yaml
├── nas_config.yaml
├── prune_config.yaml
├── quantize_config.yaml
└── docker-compose.yml
```

## Configuration Files

### `train_config.yaml`

**Purpose:**  
Defines the parameters and settings for training the neural network model.

**Key Sections:**

- **Data Configuration:**  
  Specifies paths to training and validation datasets, batch sizes, and data augmentation techniques.

- **Model Configuration:**  
  Defines model architecture parameters, such as input size, hidden layers, and activation functions.

- **Training Hyperparameters:**  
  Includes learning rate, optimizer type, number of epochs, and loss functions.

- **Logging and Checkpointing:**  
  Sets up logging intervals, checkpoint saving frequency, and paths for storing logs and models.

**Example Configuration:**

```yaml
# config/train_config.yaml

data:
  train_dataset: "data/train_dataset.csv"
  val_dataset: "data/val_dataset.csv"
  batch_size: 64
  shuffle: true
  num_workers: 4

model:
  input_size: 100
  hidden_layers:
    - 256
    - 256
    - 128
  output_size: 10
  activation: "ReLU"

training:
  optimizer: "Adam"
  learning_rate: 0.001
  epochs: 20
  loss_function: "CrossEntropyLoss"
  scheduler:
    type: "StepLR"
    step_size: 10
    gamma: 0.1

logging:
  log_interval: 10
  checkpoint_interval: 5
  log_dir: "logs/"
  model_save_path: "models/checkpoints/"

device:
  use_cuda: true
  gpu_ids: [0, 1]
```

### `eval_config.yaml`

**Purpose:**  
Defines the parameters and settings for evaluating the trained neural network model.

**Key Sections:**

- **Data Configuration:**  
  Specifies paths to validation datasets and batch sizes.

- **Model Configuration:**  
  Defines model architecture parameters to ensure consistency with the trained model.

- **Evaluation Metrics:**  
  Includes metrics like accuracy, precision, recall, and F1-score.

**Example Configuration:**

```yaml
# config/eval_config.yaml

data:
  val_dataset: "data/val_dataset.csv"
  batch_size: 64
  shuffle: false
  num_workers: 4

model:
  input_size: 100
  hidden_layers:
    - 256
    - 256
    - 128
  output_size: 10
  activation: "ReLU"

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score

device:
  use_cuda: true
  gpu_ids: [0, 1]
```

### `nas_config.yaml`

**Purpose:**  
Defines the configuration parameters for the Neural Architecture Search (NAS) process.

**Key Sections:**

- **Search Space:**  
  Specifies the mutation strategies and their respective parameters.

- **NAS Parameters:**  
  Includes population size, number of generations, mutation rates, and selection criteria.

- **Evaluation Metrics:**  
  Defines the metrics used to evaluate and compare different model architectures.

**Example Configuration:**

```yaml
# config/nas_config.yaml

search_space:
  add_layer:
    enabled: true
    layer_type: "Linear"
    input_size: 256
    output_size: 128
  remove_layer:
    enabled: true
    layer_name: "layer3"
  increase_units:
    enabled: true
    layer_name: "layer1"
    new_output_size: 512
  decrease_units:
    enabled: true
    layer_name: "layer1"
    new_output_size: 128

nas_parameters:
  population_size: 5
  generations: 5
  mutation_rate: 0.3
  selection_method: "tournament"
  tournament_size: 2

evaluation:
  metric: "accuracy"
  higher_is_better: true
```

### `prune_config.yaml`

**Purpose:**  
Defines the parameters and settings for pruning the neural network model to reduce its size and complexity.

**Key Sections:**

- **Pruning Parameters:**  
  Includes the pruning percentage, target layers, and pruning method.

- **Thresholds:**  
  Specifies the thresholds used to determine which weights to prune based on importance.

- **Logging and Checkpointing:**  
  Sets up logging intervals and paths for storing pruned models and logs.

**Example Configuration:**

```yaml
# config/prune_config.yaml

pruning:
  method: "magnitude"
  amount: 0.2  # Prune 20% of the weights
  target_layers:
    - "layer1"
    - "layer2"

thresholds:
  magnitude_threshold: 0.01

logging:
  log_interval: 10
  pruned_model_save_path: "models/pruned/pruned_model.pth"
  log_dir: "logs/pruning/"
```

### `quantize_config.yaml`

**Purpose:**  
Defines the parameters and settings for quantizing the neural network model to lower precision, enhancing inference speed and reducing memory usage.

**Key Sections:**

- **Quantization Parameters:**  
  Includes quantization type, target precision, and layers to quantize.

- **Calibration Settings:**  
  Specifies calibration dataset paths and batch sizes for calibration.

- **Logging and Checkpointing:**  
  Sets up logging intervals and paths for storing quantized models and logs.

**Example Configuration:**

```yaml
# config/quantize_config.yaml

quantization:
  dtype: "int8"
  approach: "dynamic"
  target_layers:
    - "layer1"
    - "layer2"
    - "layer3"

calibration:
  calibration_dataset: "data/calibration_dataset.csv"
  batch_size: 64
  num_workers: 4

logging:
  log_interval: 10
  quantized_model_save_path: "models/quantized/quantized_model.pth"
  log_dir: "logs/quantization/"
```

### `docker-compose.yml`

**Purpose:**  
Defines the Docker Compose configuration to orchestrate multiple Docker containers, including the FastAPI API server, Prometheus for monitoring, and Grafana for visualization.

**Key Sections:**

- **Services:**  
  Defines each service (API server, Prometheus, Grafana) with their respective configurations, images, ports, volumes, and dependencies.

- **Networks:**  
  Configures networks for inter-service communication.

- **Volumes:**  
  Sets up persistent storage volumes for data persistence across container restarts.

**Example Configuration:**

```yaml
# config/docker-compose.yml

version: '3.8'

services:
  api:
    build:
      context: ../docker/
      dockerfile: Dockerfile
    image: dynamic_nn_refinement_api:latest
    container_name: dynamic_nn_refinement_api
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
      - ../logs:/app/logs
      - ../config:/app/config
    environment:
      - ENV_FILE=/app/.env
    depends_on:
      - prometheus
      - grafana

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ../prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ../prometheus/alert.rules.yml:/etc/prometheus/alert.rules.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

networks:
  default:
    driver: bridge

volumes:
  grafana-storage:
```

## Usage

### Modifying Configurations

To customize the project according to your requirements:

1. **Edit Configuration Files:**
   - Navigate to the `config/` directory.
   - Open the relevant YAML file (e.g., `train_config.yaml`) and modify parameters as needed.

2. **Reload Services:**
   - After making changes to configuration files related to Docker services (e.g., `docker-compose.yml`), rebuild and restart the services:
   
     ```bash
     docker-compose up -d --build
     ```

### Loading Configurations in Scripts

Scripts within the `scripts/` and `src/` directories load configuration files to set up parameters dynamically.

**Example: Loading a Configuration File**

```python
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Usage
config = load_config('config/train_config.yaml')
batch_size = config['data']['batch_size']
learning_rate = config['training']['learning_rate']
```

## Best Practices

- **Version Control:**  
  Track changes to configuration files using Git to maintain a history of modifications and facilitate collaboration.

- **Environment-Specific Configurations:**  
  Maintain separate configuration files for different environments (development, testing, production) to ensure appropriate settings are applied.

- **Sensitive Information Management:**  
  Avoid storing sensitive information directly in configuration files. Use environment variables or secret management tools to handle credentials securely.

- **Validation:**  
  Implement validation checks in scripts to ensure that configurations are correctly loaded and contain valid values, preventing runtime errors.

## Contributing

Contributions to the configuration files are welcome! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/update-configurations
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "chore: update training configuration with new hyperparameters"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/update-configurations
   ```

5. **Open a Pull Request**

   Provide a clear description of the changes made and their impact on the project.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
