### `scripts/README.md`

# Scripts Directory

The `scripts/` directory contains a collection of utility scripts that facilitate various tasks within the **Dynamic Neural Network Refinement** project. These scripts handle operations such as training, evaluation, pruning, quantization, profiling, benchmarking, visualization, and model exporting. They are designed to streamline workflows, automate processes, and enhance the overall efficiency of the project.

## Table of Contents

- [Overview](#overview)
- [Available Scripts](#available-scripts)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Pruning the Model](#pruning-the-model)
  - [Quantizing the Model](#quantizing-the-model)
  - [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
  - [Profiling](#profiling)
  - [Benchmarking](#benchmarking)
  - [Visualization](#visualization)
  - [Exporting the Model to ONNX](#exporting-the-model-to-onnx)
  - [Saving the Model](#saving-the-model)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)

## Overview

Each script is designed to perform a specific function, ensuring modularity and ease of maintenance. Below is a detailed description of each available script and instructions on how to use them effectively.

## Available Scripts

- **train.py**: Trains the neural network model based on the provided configuration.
- **evaluate.py**: Evaluates the trained model's performance on a validation dataset.
- **prune.py**: Prunes the model to remove less significant weights, reducing its size and complexity.
- **quantify.py**: Quantizes the model to lower precision, enhancing inference speed and reducing memory footprint.
- **nas.py**: Performs Neural Architecture Search to identify optimal model architectures.
- **profile_training.py**: Profiles the training process to identify performance bottlenecks.
- **profile_inference.py**: Profiles the inference process to assess efficiency and resource utilization.
- **benchmark.py**: Benchmarks the model's training and inference speed under different conditions.
- **visualize.py**: Generates visualizations of training metrics and model performance.
- **export_to_onnx.py**: Exports the trained PyTorch model to the ONNX format for broader compatibility.
- **save_model.py**: Saves the trained model to a specified path.
- **utils.py**: Contains utility functions shared across multiple scripts, such as model loading and logging setup.

## Usage

### Training the Model

Train the neural network using the training script with a specified configuration.

```bash
python scripts/train.py --config config/train_config.yaml
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.

### Evaluating the Model

Evaluate the trained model's performance on the validation dataset.

```bash
python scripts/evaluate.py --config config/eval_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the evaluation configuration YAML file.
- `--model_path`: Path to the trained model file.

### Pruning the Model

Prune the model to remove less significant weights, optimizing its size and performance.

```bash
python scripts/prune.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the trained model file.

### Quantizing the Model

Quantize the model to lower precision for faster inference and reduced memory usage.

```bash
python scripts/quantify.py --config config/train_config.yaml --model_path models/pruned/pruned_model.pth --quantized_model_path models/quantized/quantized_model.pth
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the pruned model file.
- `--quantized_model_path`: Path to save the quantized model.

### Neural Architecture Search (NAS)

Perform Neural Architecture Search to discover optimal model architectures.

```bash
python scripts/nas.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the trained model file.

### Profiling

#### Profile Training

Profile the training process to identify performance bottlenecks.

```bash
python scripts/profile_training.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the trained model file.

#### Profile Inference

Profile the inference process to assess efficiency.

```bash
python scripts/profile_inference.py --config config/eval_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the evaluation configuration YAML file.
- `--model_path`: Path to the trained model file.

### Benchmarking

Benchmark the model's training and inference speed.

```bash
python scripts/benchmark.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth
```

**Parameters:**

- `--config`: Path to the configuration YAML file.
- `--model_path`: Path to the trained model file.

### Visualization

Generate visualizations of training metrics and model performance.

```bash
python scripts/visualize.py --config config/train_config.yaml
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.

### Exporting the Model to ONNX

Export the trained PyTorch model to the ONNX format.

```bash
python scripts/export_to_onnx.py --config config/train_config.yaml --model_path models/quantized/quantized_model.pth --onnx_path models/final/model_v1.0_final.onnx
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the quantized model file.
- `--onnx_path`: Path to save the exported ONNX model.

### Saving the Model

Save the trained model to a specified path.

```bash
python scripts/save_model.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth --save_path models/final/custom_saved_model.pth
```

**Parameters:**

- `--config`: Path to the training configuration YAML file.
- `--model_path`: Path to the trained model file.
- `--save_path`: Path to save the model.

## Utilities

The `utils.py` script contains shared utility functions used across multiple scripts, such as:

- **Model Loading:**  
  Functions to load models from specified paths and devices.
  
- **Logging Setup:**  
  Functions to configure and initialize structured logging for consistent log management.

**Example Usage:**

```python
from scripts.utils import load_model, setup_logging

# Setup logging
logger = setup_logging('logs/script_log.json.log')

# Load model
model = load_model(model, 'models/final/model_v1.0_final.pth', device='cuda')
```

## Contributing

We welcome contributions from the community! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "feat: add your feature"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

   Submit a pull request detailing your changes and their benefits.

For detailed guidelines, refer to the [Best Practices](../docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](../LICENSE).

---
```
