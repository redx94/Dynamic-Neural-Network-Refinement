# Getting Started

Welcome to the **Dynamic Neural Network Refinement** project! This guide will walk you through the steps to set up the project locally, ensuring you have all the necessary tools and dependencies to get started.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Operating System:** Linux or macOS (Windows support may require additional configurations)
- **Python:** Version 3.8 or higher
- **Docker:** Latest version
- **Docker Compose:** Latest version
- **Git:** For version control
- **Poetry:** For dependency management
- **GitHub Account:** To access repository secrets and manage contributions

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Dynamic-Neural-Network-Refinement.git
cd Dynamic-Neural-Network-Refinement
````

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

## Running the Application

### Using Docker Compose

Docker Compose simplifies the process of running multiple services required by the project, including the API server, Prometheus, and Grafana.

1. **Build and Start Services**
    
    ```bash
    docker-compose up --build
    ```
    
    This command builds the Docker images and starts the containers. The FastAPI server will be accessible at `http://localhost:8000/`, Prometheus at `http://localhost:9090/`, and Grafana at `http://localhost:3000/`.
    
2. **Accessing the API**
    
    Once the services are running, you can access the API documentation at `http://localhost:8000/docs`.
    

### Running Locally Without Docker

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
    

## Running Tests

The project includes a comprehensive test suite to ensure reliability and correctness.

```bash
make test
```

This command runs all unit, integration, and end-to-end tests using `pytest` and `unittest`.

## Continuous Integration

GitHub Actions workflows are set up to automate testing, linting, security scanning, and deployment processes. These workflows are triggered on code pushes and pull requests to the main branch.

## Monitoring and Logging

- **Prometheus:** Monitors application metrics.
- **Grafana:** Visualizes metrics collected by Prometheus.
- **Structured Logging:** Logs are formatted in JSON for easy parsing and analysis.

## Contributing

Interested in contributing to the project? Please refer to our [Best Practices](https://chatgpt.com/c/best_practices.md) guide for guidelines on how to contribute effectively.

---

_For more detailed information, explore other sections of the documentation._
