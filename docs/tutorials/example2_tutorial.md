# Tutorial: End-to-End Workflow for Dynamic Neural Network Refinement

In this tutorial, we'll guide you through the complete workflow of the **Dynamic Neural Network Refinement** project. You'll learn how to train the model, evaluate its performance, optimize it through pruning and quantization, deploy it via the FastAPI-based API, and monitor its performance using Prometheus and Grafana.

## Prerequisites

Before you begin, ensure you've completed the following:

- Completed the [Getting Started](getting_started.md) guide.
- Installed all necessary dependencies.
- Configured environment variables in the `.env` file.
- Prepared and versioned your datasets using DVC.

## Step 1: Start the Services with Docker Compose

Docker Compose orchestrates multiple services required by the project, including the API server, Prometheus, and Grafana.

```bash
make docker-compose-up
````

**Description:**

- **Builds** Docker images for the API server.
- **Starts** containers for the API server, Prometheus, and Grafana.
- **Exposes Ports:**
    - **API Server:** `http://localhost:8000/`
    - **Prometheus:** `http://localhost:9090/`
    - **Grafana:** `http://localhost:3000/`

**Expected Outcome:**

- All services are up and running.
- Access the API documentation at `http://localhost:8000/docs`.

## Step 2: Train the Model

Training adjusts the neural network architecture dynamically based on data complexity.

```bash
make train
```

**Description:**

- **Initiates** the training process using the configuration specified in `config/train_config.yaml`.
- **Analyzes** data complexities (variance, entropy, sparsity) to refine the model architecture.
- **Logs** training metrics to Weights & Biases (W&B).
- **Saves** model checkpoints and the final trained model.

**Expected Outcome:**

- Model checkpoints saved in `models/checkpoints/`.
- Final model saved at `models/final/model_v1.0_final.pth`.
- Training logs available in `logs/train.json.log`.

## Step 3: Evaluate the Model

Assess the model's performance on the validation dataset.

```bash
make evaluate
```

**Description:**

- **Loads** the trained model.
- **Computes** loss and accuracy on the validation dataset.
- **Saves** evaluation results to `logs/evaluation_results.csv`.

**Expected Outcome:**

- Evaluation metrics indicating model performance.
- Logs available in `logs/evaluate.json.log`.

## Step 4: Prune the Model

Reduce the model's size by removing less significant weights.

```bash
make prune
```

**Description:**

- **Loads** the trained model.
- **Applies** pruning based on predefined thresholds.
- **Saves** the pruned model to `models/pruned/pruned_model.pth`.

**Expected Outcome:**

- Pruned model saved at `models/pruned/pruned_model.pth`.
- Logs available in `logs/prune.json.log`.

## Step 5: Quantize the Model

Optimize the model by reducing the precision of its weights.

```bash
make quantize
```

**Description:**

- **Loads** the pruned model.
- **Applies** dynamic quantization to the model.
- **Saves** the quantized model to `models/quantized/quantized_model.pth`.

**Expected Outcome:**

- Quantized model saved at `models/quantized/quantized_model.pth`.
- Logs available in `logs/quantify.json.log`.

## Step 6: Export the Model to ONNX

Convert the PyTorch model to the ONNX format for broader compatibility.

```bash
make export-onnx
```

**Description:**

- **Loads** the quantized model.
- **Exports** the model to `models/final/model_v1.0_final.onnx`.

**Expected Outcome:**

- ONNX model saved at `models/final/model_v1.0_final.onnx`.
- Logs available in `logs/export_to_onnx.json.log`.

## Step 7: Deploy the Model via API

Make the model accessible for real-time inference through the FastAPI-based API.

### a. Access the API Documentation

Navigate to `http://localhost:8000/docs` in your web browser to access the interactive Swagger UI.

### b. Perform a Prediction

Use the `/predict` endpoint to submit data and receive predictions.

**Example Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "input_data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
           "current_epoch": 10
         }'
```

**Example Response:**

```json
{
  "predictions": [2, 0]
}
```

## Step 8: Monitor Performance with Prometheus and Grafana

Ensure the application is running smoothly by monitoring key metrics.

### a. Access Prometheus

Navigate to `http://localhost:9090/` to access the Prometheus dashboard. Here, you can query metrics and set up alerts.

### b. Access Grafana

Navigate to `http://localhost:3000/` to access Grafana. Use the pre-configured dashboards to visualize metrics collected by Prometheus.

**Default Credentials:**

- **Username:** `admin`
- **Password:** `admin`

_It's recommended to change the default password upon first login._

### c. Explore Metrics

Prometheus collects various metrics such as:

- **Total Requests:** Number of API requests received.
- **Request Latency:** Time taken to process each request.
- **CPU and Memory Usage:** System resource utilization.
- **Model Performance Metrics:** Custom metrics like prediction accuracy and loss.

Use Grafana to create insightful dashboards by connecting it to your Prometheus data source.

## Step 9: Profiling and Benchmarking

Identify performance bottlenecks and optimize model efficiency.

### a. Profile Training

```bash
make profile-training
```

**Description:**

- **Profiles** the training process, capturing CPU and GPU usage, memory consumption, and execution time.
- **Generates** a Chrome trace file at `logs/training_trace.json`.

### b. Profile Inference

```bash
make profile-inference
```

**Description:**

- **Profiles** the inference process, capturing similar metrics as training.
- **Generates** a Chrome trace file at `logs/inference_trace.json`.

### c. Benchmark Performance

```bash
make benchmark
```

**Description:**

- **Runs** benchmarking scripts to evaluate the model's training and inference speed.
- **Outputs** the time taken for a set number of iterations.

**Example Output:**

```
Training Benchmark: 100 iterations took 30.50 seconds.
Inference Benchmark: 100 iterations took 15.25 seconds.
```

## Step 10: Clean Up

After completing your tasks, you can stop the Docker Compose services.

```bash
make docker-compose-down
```

**Description:**

- **Stops** and **removes** all containers, networks, and volumes created by Docker Compose.

---

_Congratulations! You've successfully completed the end-to-end workflow for training, deploying, and monitoring the Dynamic Neural Network Refinement model. Explore more advanced features and best practices in the subsequent documentation sections._

---

_If you encounter any issues or have suggestions for improving this tutorial, please open an issue or submit a pull request._
