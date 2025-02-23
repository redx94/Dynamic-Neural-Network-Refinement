# Makefile

# Variables
PYTHON=python
POETRY=poetry

# Targets

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(POETRY) install
	@echo "Dependencies installed."

# Run unit tests
unit-test:
	@echo "Running unit tests..."
	pytest tests/unit/
	@echo "Unit tests completed."

# Run integration tests
integration-test:
	@echo "Running integration tests..."
	python -m unittest tests/test_integration.py
	@echo "Integration tests completed."

# Run end-to-end tests
e2e-test:
	@echo "Running end-to-end tests..."
	python -m unittest tests/test_e2e.py
	@echo "End-to-end tests completed."

# Run all tests
test: unit-test integration-test e2e-test
	@echo "All tests completed."

# Lint code
lint:
	@echo "Linting code with flake8..."
	flake8 src/ scripts/ tests/
	@echo "Linting completed."

# Format code
format:
	@echo "Formatting code with black..."
	black src/ scripts/ tests/
	@echo "Formatting completed."

# Run pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files
	@echo "Pre-commit hooks completed."

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t dynamic_nn_image .
	@echo "Docker image built."

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -it --rm -p 8000:8000 dynamic_nn_image
	@echo "Docker container stopped."

# Start services with Docker Compose
docker-compose-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up --build
	@echo "Services started."

# Stop services with Docker Compose
docker-compose-down:
	@echo "Stopping services with Docker Compose..."
	docker-compose down
	@echo "Services stopped."

# Run API server locally
run-api-local:
	@echo "Running API server locally..."
	uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
	@echo "API server is running at http://127.0.0.1:8000/"

# Deploy model manually
deploy-model:
	@echo "Deploying the final model..."
	python scripts/deploy_model.py --model_path models/final/model_v1.0_final.pth --deploy_dir deploy/
	@echo "Model deployed to deploy/ directory."

# Export model to ONNX
export-onnx:
	@echo "Exporting model to ONNX format..."
	python scripts/export_to_onnx.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth --output_path models/final/model_v1.0_final.onnx
	@echo "Model exported to ONNX."

# Profile training
profile-training:
	@echo "Profiling training process..."
	python scripts/profile_training.py --config config/train_config.yaml
	@echo "Training profiling completed."

# Profile inference
profile-inference:
	@echo "Profiling inference process..."
	python scripts/profile_inference.py --config config/eval_config.yaml --model_path models/final/model_v1.0_final.pth
	@echo "Inference profiling completed."

# Benchmark performance
benchmark:
	@echo "Benchmarking model performance..."
	python scripts/benchmark.py --config config/train_config.yaml --model_path models/final/model_v1.0_final.pth
	@echo "Benchmarking completed."

# Deploy documentation
deploy-docs:
	@echo "Building documentation..."
	mkdocs build
	@echo "Deploying documentation to GitHub Pages..."
	mkdocs gh-deploy --force
	@echo "Documentation deployed."

# Start monitoring services
start-monitoring:
	@echo "Starting Prometheus and Grafana..."
	docker-compose up -d prometheus grafana
	@echo "Prometheus and Grafana are running."

# Stop monitoring services
stop-monitoring:
	@echo "Stopping Prometheus and Grafana..."
	docker-compose down
	@echo "Prometheus and Grafana have been stopped."

# Default target
all: install test lint format
