# Dynamic Neural Network Refinement

![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)
![Build Status](https://github.com/redx94/Dynamic-Neural-Network-Refinement/actions/workflows/main.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Docker](https://img.shields.io/badge/docker-supported-brightgreen.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

**Dynamic Neural Network Refinement** is an advanced machine learning project designed to enhance neural network architectures dynamically based on real-time complexity metrics. By analyzing factors such as variance, entropy, and sparsity, the system intelligently adjusts the network's structure during training and inference to optimize performance, efficiency, and scalability.

This project integrates modern software engineering practices, including containerization, continuous integration/deployment (CI/CD), comprehensive testing, monitoring, and logging, ensuring a robust and maintainable codebase.

## Features

- **Dynamic Architecture Adjustment:** Automatically refines neural network layers based on data complexity metrics.
- **Distributed Training:** Supports scalable training across multiple GPUs or machines using PyTorch Distributed Data Parallel (DDP).
- **Neural Architecture Search (NAS):** Implements NAS to explore and identify optimal model architectures.
- **Model Optimization:** Includes pruning and quantization techniques to enhance model efficiency.
- **Containerization:** Docker and Docker Compose configurations facilitate easy deployment and environment setup.
- **Continuous Integration/Continuous Deployment (CI/CD):** GitHub Actions workflows automate testing, deployment, and monitoring processes.
- **Monitoring and Logging:** Integrates Prometheus and Grafana for real-time monitoring, along with structured logging for efficient log management.
- **Model Serving:** FastAPI-based API serves models for inference, ensuring quick and reliable predictions.
- **Data Versioning and Validation:** Utilizes DVC for data management and Great Expectations for robust data validation.
- **Performance Profiling and Benchmarking:** Tools and scripts to profile and benchmark both training and inference processes.
- **Automated Model Export:** Exports models to formats like ONNX for broader compatibility.
- **Code Quality Enforcement:** Pre-commit hooks and CI workflows maintain high code standards.

## Installation

Follow these steps to set up the project locally.

### Prerequisites

- **Operating System:** Linux or macOS (Windows support may require additional configurations)
- **Python:** Version 3.8 or higher
- **Docker:** Latest version
- **Docker Compose:** Latest version
- **Git:** For version control
- **Poetry:** For dependency management
- **GitHub Account:** To access repository secrets and manage contributions

### 1. Clone the Repository

```bash
git clone https://github.com/redx94/Dynamic-Neural-Network-Refinement.git
cd Dynamic-Neural-Network-Refinement
```

### 2. Install Dependencies

The project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure Poetry is installed on your system. If not, install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your system's PATH by following the on-screen instructions after installation.

### 3. Set Up the Python Environment

```bash
poetry install
```

This command will create a virtual environment and install all required dependencies as specified in `pyproject.toml`.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory of the project to store environment variables. This file is excluded from version control via `.gitignore` for security reasons.

```bash
touch .env
```

Add the following variables to the `.env` file, replacing placeholder values with your actual credentials:

```env
WANDB_API_KEY=your_wandb_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=your_aws_region
OTHER_SECRET=your_other_secrets
```

**Note:**  
Ensure that you have the necessary permissions and credentials for services like Weights & Biases (W&B) and AWS if you intend to use them for logging and model deployment.

### 5. Configure DVC Remote Storage

Data Version Control (DVC) is used for data management. Configure your remote storage where datasets will be stored.

```bash
dvc remote add -d myremote s3://your-bucket/path
```

Replace `s3://your-bucket/path` with your actual remote storage path. Ensure that your AWS credentials are correctly set in the `.env` file.

### 6. Initialize DVC

```bash
dvc init
dvc pull
```

These commands initialize DVC and pull the necessary data from the remote storage.

## Usage

### Running the Application

#### Using Docker Compose

Docker Compose simplifies the process of running multiple services required by the project, including the API server, Prometheus, and Grafana.

1. **Build and Start Services**

   ```bash
   make docker-build
   make docker-compose-up
   ```

   This command builds the Docker images and starts the containers. The FastAPI server will be accessible at `http://localhost:8000/`, Prometheus at `http://localhost:9090/`, and Grafana at `http://localhost:3000/`.

2. **Accessing the API**

   Once the services are running, you can access the API documentation at `http://localhost:8000/docs`.

#### Running Locally Without Docker

If you prefer to run the application without Docker, follow these steps:

1. **Activate the Virtual Environment**

   ```bash
   poetry shell
   ```

2. **Run the FastAPI Server**

   ```bash
   uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
   ```

   The API will be accessible at `http://127.0.0.1:8000/docs`.

### Training the Model

To train the model using the provided training script:

```bash
make train
```

This command executes the training process as defined in the `Makefile`, utilizing configurations from `config/train_config.yaml`.

### Evaluating the Model

After training, evaluate the model's performance:

```bash
make evaluate
```

This command runs the evaluation script, assessing metrics like loss and accuracy on the validation dataset.

### Pruning and Quantization

Optimize the model by pruning and quantizing:

```bash
make prune
make quantize
```

These commands execute scripts to prune unnecessary weights and quantize the model for enhanced efficiency.

### Deploying the Model

Deploy the trained and optimized model:

```bash
make deploy-model
```

This command copies the model to the designated deployment directory or uploads it to cloud storage, depending on your configuration.

### Exporting the Model to ONNX

Convert the PyTorch model to the ONNX format:

```bash
make export-onnx
```

This facilitates broader compatibility across different platforms and frameworks.

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory, covering the following sections:

- [Getting Started](docs/getting_started.md)
- [API Documentation](docs/API_documentation.md)
- [Best Practices](docs/best_practices.md)
- [License Information](docs/license_information.md)
- [Tutorials](docs/tutorials/example_tutorial.md)

## Testing

The project includes a comprehensive test suite to ensure reliability and correctness.

### Running Tests

Execute all tests using the Makefile:

```bash
make test
```

This command runs unit, integration, and end-to-end tests using `pytest` and `unittest`.

### Test Coverage

To check test coverage:

```bash
make coverage
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

For detailed guidelines, refer to the [Best Practices](docs/best_practices.md) section.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---

*Empower your neural networks with dynamic refinement for optimal performance and efficiency.*
```
---

