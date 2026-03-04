# src/metrics.py

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
import time

# API metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Latency of requests in seconds')

# Model metrics
MODEL_COMPLEXITY = Gauge('model_complexity_score', 'Current model complexity score')
TRAINING_LOSS = Gauge('training_loss', 'Current training loss')
TRAINING_ACCURACY = Gauge('training_accuracy', 'Current training accuracy')
INFERENCE_TIME = Histogram('model_inference_seconds', 'Time taken for model inference')

# Architecture metrics
LAYER_COUNT = Gauge('model_layer_count', 'Number of active layers in the model')
PARAMETER_COUNT = Gauge('model_parameter_count', 'Total number of model parameters')

# Complexity metrics
VARIANCE_THRESHOLD = Gauge('complexity_variance_threshold', 'Current variance threshold')
ENTROPY_THRESHOLD = Gauge('complexity_entropy_threshold', 'Current entropy threshold')
SPARSITY_THRESHOLD = Gauge('complexity_sparsity_threshold', 'Current sparsity threshold')

def setup_metrics(app: FastAPI):
    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        REQUEST_COUNT.inc()
        start_time = time.time()
        response = await call_next(request)
        REQUEST_LATENCY.observe(time.time() - start_time)
        return response

    @app.get("/metrics")
    async def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

def update_training_metrics(loss: float, accuracy: float):
    """Update training-related metrics."""
    TRAINING_LOSS.set(loss)
    TRAINING_ACCURACY.set(accuracy)

def update_complexity_metrics(complexity_scores: dict):
    """Update model complexity-related metrics."""
    MODEL_COMPLEXITY.set(sum(complexity_scores.values()) / len(complexity_scores))
    for metric_name, value in complexity_scores.items():
        if hasattr(globals(), f"complexity_{metric_name}_score"):
            globals()[f"complexity_{metric_name}_score"].set(value)

def update_architecture_metrics(layer_count: int, parameter_count: int):
    """Update model architecture metrics."""
    LAYER_COUNT.set(layer_count)
    PARAMETER_COUNT.set(parameter_count)

def update_threshold_metrics(thresholds: dict):
    """Update complexity threshold metrics."""
    VARIANCE_THRESHOLD.set(thresholds.get('variance', 0))
    ENTROPY_THRESHOLD.set(thresholds.get('entropy', 0))
    SPARSITY_THRESHOLD.set(thresholds.get('sparsity', 0))
