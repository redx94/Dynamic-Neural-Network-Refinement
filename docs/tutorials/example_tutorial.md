# Example Tutorial: Training and Deploying the Dynamic Neural Network

This tutorial walks you through the process of training the Dynamic Neural Network Refinement model, deploying the FastAPI-based API for inference, and setting up monitoring with Prometheus and Grafana.

## Prerequisites

- Completed the [Getting Started](getting_started.md) guide.
- Docker and Docker Compose installed.
- Necessary environment variables set in the `.env` file.
- Data prepared and versioned using DVC.

## Step 1: Activate the Virtual Environment

If you're not using Docker, activate the virtual environment created by Poetry:

```bash
poetry shell
````

## Step 2: Train the Model

Training the model involves executing the training script with the appropriate configuration.

```bash
make train
```

**Description:**

- **Command Breakdown:**
    - `make train`: Invokes the `train` target in the `Makefile`, which runs the training script.
- **Process:**
    - Initializes distributed training if configured.
    - Loads training configurations from `config/train_config.yaml`.
    - Analyzes input data complexities and adjusts model architecture dynamically.
    - Logs training metrics to Weights & Biases (W&B).
    - Saves model checkpoints at specified intervals.

**Expected Outcome:**

- Model checkpoints saved in `models/checkpoints/`.
- Final trained model saved at `models/final/model_v1.0_final.pth`.
- Training logs available in `logs/train.json.log`.

## Step 3: Evaluate the Model

After training, evaluate the model's performance on the validation dataset.

```bash
make evaluate
```

**Description:**

- **Command Breakdown:**
    - `make evaluate`: Invokes the `evaluate` target in the `Makefile`, which runs the evaluation script.
- **Process:**
    - Loads evaluation configurations from `config/eval_config.yaml`.
    - Computes loss and accuracy on the validation dataset.
    - Saves evaluation results to `logs/evaluation_results.csv`.

**Expected Outcome:**

- Evaluation metrics indicating the model's performance.
- Logs available in `logs/evaluation.json.log`.

## Step 4: Prune the Model

Pruning reduces the model's size by removing less significant weights, leading to a more efficient model.

```bash
make prune
```

**Description:**

- **Command Breakdown:**
    - `make prune`: Invokes the `prune` target in the `Makefile`, which runs the pruning script.
- **Process:**
    - Loads the trained model from `models/final/model_v1.0_final.pth`.
    - Applies pruning based on predefined thresholds.
    - Saves the pruned model to `models/pruned/pruned_model.pth`.

**Expected Outcome:**

- Pruned model saved at `models/pruned/pruned_model.pth`.
- Logs available in `logs/prune.json.log`.

## Step 5: Quantize the Model

Quantization further optimizes the model by reducing the precision of its weights, enhancing inference speed.

```bash
make quantize
```

**Description:**

- **Command Breakdown:**
    - `make quantize`: Invokes the `quantize` target in the `Makefile`, which runs the quantization script.
- **Process:**
    - Loads the pruned model from `models/pruned/pruned_model.pth`.
    - Applies dynamic quantization to the model.
    - Saves the quantized model to `models/quantized/quantized_model.pth`.

**Expected Outcome:**

- Quantized model saved at `models/quantized/quantized_model.pth`.
- Logs available in `logs/quantify.json.log`.

## Step 6: Deploy the Model

Deploy the final model to the designated deployment directory or cloud storage.

```bash
make deploy-model
```

**Description:**

- **Command Breakdown:**
    - `make deploy-model`: Invokes the `deploy-model` target in the `Makefile`, which runs the deployment script.
- **Process:**
    - Copies the quantized model to the `deploy/` directory.
    - (Optional) Uploads the model to cloud storage or integrates with deployment pipelines.

**Expected Outcome:**

- Model deployed to the `deploy/` directory.
- Logs available in `logs/deploy_model.json.log`.

## Step 7: Export the Model to ONNX

Exporting the model to the ONNX format allows for broader compatibility across different platforms and frameworks.

```bash
make export-onnx
```

**Description:**

- **Command Breakdown:**
    - `make export-onnx`: Invokes the `export-onnx` target in the `Makefile`, which runs the export script.
- **Process:**
    - Loads the quantized model from `models/quantized/quantized_model.pth`.
    - Exports the model to `models/final/model_v1.0_final.onnx`.

**Expected Outcome:**

- ONNX model saved at `models/final/model_v1.0_final.onnx`.
- Logs available in `logs/export_to_onnx.json.log`.

## Step 8: Serve the Model via API

Deploy the FastAPI-based API to serve the model for real-time inference.

### a. Using Docker Compose

Start all services, including the API server, Prometheus, and Grafana.

```bash
make docker-compose-up
```

**Access Points:**

- **API Server:** `http://localhost:8000/docs`
- **Prometheus:** `http://localhost:9090/`
- **Grafana:** `http://localhost:3000/`

### b. Running Locally Without Docker

Activate the virtual environment and run the API server:

```bash
make run-api-local
```

Access the API documentation at `http://127.0.0.1:8000/docs`.

## Step 9: Monitor Application Metrics

Use Prometheus and Grafana to monitor the application's performance and health.

### a. Access Prometheus

Navigate to `http://localhost:9090/` to access the Prometheus dashboard. Here, you can query and visualize various metrics collected from the API server.

### b. Access Grafana

Navigate to `http://localhost:3000/` to access the Grafana dashboard. Use pre-configured dashboards or create custom ones to visualize metrics from Prometheus.

**Default Credentials:**

- **Username:** `admin`
- **Password:** `admin`

_It's recommended to change the default password upon first login._

## Step 10: Profiling and Benchmarking

Profile and benchmark the model to identify performance bottlenecks and optimize efficiency.

### a. Profile Training

```bash
make profile-training
```

**Description:**

- Profiles the training process, capturing metrics like CPU and GPU usage, memory consumption, and execution time.
- Generates a Chrome trace file at `logs/training_trace.json`.

### b. Profile Inference

```bash
make profile-inference
```

**Description:**

- Profiles the inference process, capturing similar metrics as training.
- Generates a Chrome trace file at `logs/inference_trace.json`.

### c. Benchmark Performance

```bash
make benchmark
```

**Description:**

- Runs benchmarking scripts to evaluate the model's training and inference speed.
- Outputs the time taken for a set number of iterations.

## Step 11: Pre-commit Hooks and Code Quality

Ensure code quality and consistency by running pre-commit hooks.

```bash
make pre-commit
```

**Description:**

- Runs pre-commit hooks that check for trailing whitespaces, fix formatting with Black, lint code with Flake8, and scan for security issues with Bandit.

**Note:**  
Pre-commit hooks are also automatically triggered on every commit, ensuring that all code adheres to the project's standards before being pushed.

## Conclusion

By following this tutorial, you've successfully trained, evaluated, pruned, quantized, and deployed the Dynamic Neural Network Refinement model. Additionally, you've set up monitoring and profiling tools to ensure optimal performance and reliability.

For more advanced tutorials and best practices, explore the [Best Practices](https://chatgpt.com/c/best_practices.md) and [API Documentation](https://chatgpt.com/c/API_documentation.md) sections.

---

_If you encounter any issues or have suggestions for improving this tutorial, please open an issue or submit a pull request._
