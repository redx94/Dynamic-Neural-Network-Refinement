
import time
from prometheus_client import Counter, Gauge, Histogram

# Metrics definitions
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total number of inference requests')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Time spent on inference')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Current memory usage')

class MetricsCollector:
    @staticmethod
    def record_inference(func):
        def wrapper(*args, **kwargs):
            INFERENCE_REQUESTS.inc()
            start_time = time.time()
            result = func(*args, **kwargs)
            INFERENCE_LATENCY.observe(time.time() - start_time)
            return result
        return wrapper

    @staticmethod
    def update_accuracy(accuracy: float):
        MODEL_ACCURACY.set(accuracy)

    @staticmethod
    def update_memory_usage(usage: int):
        MEMORY_USAGE.set(usage)
