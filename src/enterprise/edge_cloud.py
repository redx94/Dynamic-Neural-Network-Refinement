"""
Edge-Cloud Orchestration module for enterprise deployments.
Implements intelligent workload distribution, model synchronization, and failover.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Callable
import time
import threading
import queue
import json
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib


class DeviceType(Enum):
    """Device types for edge-cloud orchestration."""
    EDGE_MOBILE = "edge_mobile"
    EDGE_IOT = "edge_iot"
    EDGE_SERVER = "edge_server"
    CLOUD_CPU = "cloud_cpu"
    CLOUD_GPU = "cloud_gpu"
    CLOUD_TPU = "cloud_tpu"


class WorkloadPriority(Enum):
    """Priority levels for workload scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DeviceCapabilities:
    """Capabilities and status of a compute device."""
    device_id: str
    device_type: DeviceType
    memory_mb: int
    compute_capability: float  # Relative FLOPS score
    latency_ms: float  # Network latency to this device
    is_available: bool = True
    current_load: float = 0.0
    supported_ops: List[str] = field(default_factory=lambda: ['matmul', 'conv', 'attention'])

    def get_effective_capacity(self) -> float:
        """Calculate effective capacity considering load."""
        return self.compute_capability * (1 - self.current_load)


@dataclass
class WorkloadRequest:
    """Request for compute workload processing."""
    request_id: str
    model_name: str
    input_data: torch.Tensor
    complexity_metrics: Dict[str, float]
    priority: WorkloadPriority = WorkloadPriority.NORMAL
    max_latency_ms: float = 100.0
    requires_encryption: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkloadResult:
    """Result of workload processing."""
    request_id: str
    output_data: torch.Tensor
    processing_device: str
    processing_time_ms: float
    total_latency_ms: float
    success: bool = True
    error_message: Optional[str] = None


class DeviceRegistry:
    """Registry for managing available compute devices."""

    def __init__(self):
        self.devices: Dict[str, DeviceCapabilities] = {}
        self._lock = threading.Lock()

    def register_device(self, capabilities: DeviceCapabilities):
        """Register a new compute device."""
        with self._lock:
            self.devices[capabilities.device_id] = capabilities

    def unregister_device(self, device_id: str):
        """Unregister a device."""
        with self._lock:
            if device_id in self.devices:
                del self.devices[device_id]

    def get_available_devices(self) -> List[DeviceCapabilities]:
        """Get all currently available devices."""
        with self._lock:
            return [d for d in self.devices.values() if d.is_available]

    def update_device_load(self, device_id: str, load_delta: float):
        """Update the load on a device."""
        with self._lock:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.current_load = max(0.0, min(1.0, device.current_load + load_delta))


class WorkloadScheduler(ABC):
    """Abstract base class for workload scheduling strategies."""

    @abstractmethod
    def select_device(
        self,
        request: WorkloadRequest,
        available_devices: List[DeviceCapabilities]
    ) -> Optional[DeviceCapabilities]:
        """Select the best device for a workload."""
        pass


class LatencyOptimizedScheduler(WorkloadScheduler):
    """Scheduler optimized for minimum latency."""

    def select_device(
        self,
        request: WorkloadRequest,
        available_devices: List[DeviceCapabilities]
    ) -> Optional[DeviceCapabilities]:
        """Select device that minimizes total latency."""
        if not available_devices:
            return None

        best_device = None
        best_score = float('inf')

        for device in available_devices:
            # Estimate processing time based on complexity and device capability
            complexity_factor = request.complexity_metrics.get('compute_intensity', 1.0)
            estimated_processing = (complexity_factor / device.get_effective_capacity()) * 10  # ms

            # Total latency = network + processing
            total_latency = device.latency_ms + estimated_processing

            # Check if meets latency constraint
            if total_latency <= request.max_latency_ms and total_latency < best_score:
                best_score = total_latency
                best_device = device

        return best_device


class CostOptimizedScheduler(WorkloadScheduler):
    """Scheduler optimized for minimum cost."""

    # Relative cost per FLOP for each device type
    COST_FACTORS = {
        DeviceType.EDGE_MOBILE: 0.01,
        DeviceType.EDGE_IOT: 0.005,
        DeviceType.EDGE_SERVER: 0.02,
        DeviceType.CLOUD_CPU: 0.05,
        DeviceType.CLOUD_GPU: 0.10,
        DeviceType.CLOUD_TPU: 0.15
    }

    def select_device(
        self,
        request: WorkloadRequest,
        available_devices: List[DeviceCapabilities]
    ) -> Optional[DeviceCapabilities]:
        """Select device that minimizes cost while meeting constraints."""
        if not available_devices:
            return None

        best_device = None
        best_cost = float('inf')

        for device in available_devices:
            complexity_factor = request.complexity_metrics.get('compute_intensity', 1.0)
            estimated_processing = (complexity_factor / device.get_effective_capacity()) * 10

            total_latency = device.latency_ms + estimated_processing

            # Check latency constraint
            if total_latency > request.max_latency_ms:
                continue

            # Calculate cost
            cost_factor = self.COST_FACTORS.get(device.device_type, 0.05)
            cost = cost_factor * estimated_processing

            if cost < best_cost:
                best_cost = cost
                best_device = device

        return best_device


class BalancedScheduler(WorkloadScheduler):
    """Scheduler that balances latency, cost, and device utilization."""

    def __init__(
        self,
        latency_weight: float = 0.4,
        cost_weight: float = 0.3,
        load_weight: float = 0.3
    ):
        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.load_weight = load_weight

    def select_device(
        self,
        request: WorkloadRequest,
        available_devices: List[DeviceCapabilities]
    ) -> Optional[DeviceCapabilities]:
        """Select device based on balanced scoring."""
        if not available_devices:
            return None

        best_device = None
        best_score = -float('inf')

        max_latency = max(d.latency_ms for d in available_devices)
        max_load = max(d.current_load for d in available_devices)

        for device in available_devices:
            complexity_factor = request.complexity_metrics.get('compute_intensity', 1.0)
            estimated_processing = (complexity_factor / device.get_effective_capacity()) * 10
            total_latency = device.latency_ms + estimated_processing

            # Check constraints
            if total_latency > request.max_latency_ms:
                continue

            # Normalize scores (lower is better for latency and load, higher is better for capacity)
            latency_score = 1 - (total_latency / (max_latency + estimated_processing + 1))
            load_score = 1 - (device.current_load / (max_load + 0.01))

            # Simple cost estimation
            cost_score = 1 - (0.5 if 'CLOUD' in device.device_type.value else 0.1)

            # Combined score
            combined_score = (
                self.latency_weight * latency_score +
                self.cost_weight * cost_score +
                self.load_weight * load_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_device = device

        return best_device


class ModelSynchronizer:
    """Synchronizes model weights across edge and cloud devices."""

    def __init__(self, compression_threshold: float = 0.1):
        self.compression_threshold = compression_threshold
        self.model_versions: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

    def compute_delta(
        self,
        old_state: Dict[str, torch.Tensor],
        new_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute compressed delta between model states."""
        delta = {}
        for key in new_state:
            if key in old_state:
                diff = new_state[key] - old_state[key]

                # Sparsify: only keep significant changes
                mask = diff.abs() > self.compression_threshold
                if mask.any():
                    delta[key] = diff * mask.float()
            else:
                delta[key] = new_state[key]

        return delta

    def apply_delta(
        self,
        base_state: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply delta to base state."""
        result = {k: v.clone() for k, v in base_state.items()}

        for key, delta_tensor in delta.items():
            if key in result:
                result[key] = result[key] + delta_tensor
            else:
                result[key] = delta_tensor

        return result

    def broadcast_update(
        self,
        model_name: str,
        new_version: int,
        target_devices: List[str]
    ) -> Dict[str, bool]:
        """Broadcast model update to target devices."""
        results = {}

        with self._lock:
            for device_id in target_devices:
                try:
                    # In real implementation, this would send to actual devices
                    self.model_versions[model_name] = {
                        'version': new_version,
                        'last_sync': time.time()
                    }
                    results[device_id] = True
                except Exception:
                    results[device_id] = False

        return results


class EdgeCloudOrchestrator:
    """
    Main orchestrator for edge-cloud compute distribution.
    """

    def __init__(
        self,
        scheduler: Optional[WorkloadScheduler] = None,
        default_model: Optional[nn.Module] = None
    ):
        self.registry = DeviceRegistry()
        self.scheduler = scheduler or BalancedScheduler()
        self.model_sync = ModelSynchronizer()

        self.models: Dict[str, nn.Module] = {}
        if default_model:
            self.models['default'] = default_model

        self.request_queue = queue.PriorityQueue()
        self.result_cache: Dict[str, WorkloadResult] = {}

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def register_model(self, name: str, model: nn.Module):
        """Register a model for inference."""
        self.models[name] = model

    def add_device(self, device: DeviceCapabilities):
        """Add a compute device to the registry."""
        self.registry.register_device(device)

    def remove_device(self, device_id: str):
        """Remove a device from the registry."""
        self.registry.unregister_device(device_id)

    def submit_request(
        self,
        request: WorkloadRequest,
        callback: Optional[Callable[[WorkloadResult], None]] = None
    ) -> str:
        """
        Submit a workload request for processing.

        Returns:
            Request ID for tracking
        """
        # Priority queue uses min-heap, so negate priority for higher priority first
        priority_value = -request.priority.value
        self.request_queue.put((priority_value, request, callback))

        return request.request_id

    def start(self, num_workers: int = 4):
        """Start the orchestrator worker threads."""
        self._running = True

        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()

    def stop(self):
        """Stop the orchestrator."""
        self._running = False

    def _worker_loop(self):
        """Worker thread for processing requests."""
        while self._running:
            try:
                priority, request, callback = self.request_queue.get(timeout=1.0)
                result = self._process_request(request)
                self.result_cache[request.request_id] = result

                if callback:
                    callback(result)

            except queue.Empty:
                continue
            except Exception as e:
                pass

    def _process_request(self, request: WorkloadRequest) -> WorkloadResult:
        """Process a single workload request."""
        start_time = time.time()

        # Select device
        available_devices = self.registry.get_available_devices()
        device = self.scheduler.select_device(request, available_devices)

        if device is None:
            return WorkloadResult(
                request_id=request.request_id,
                output_data=torch.tensor([]),
                processing_device='none',
                processing_time_ms=0,
                total_latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message='No available device meets requirements'
            )

        # Update device load
        self.registry.update_device_load(device.device_id, 0.1)

        try:
            # Get model
            model = self.models.get(request.model_name, self.models.get('default'))

            if model is None:
                raise ValueError(f"Model not found: {request.model_name}")

            # Run inference
            processing_start = time.time()
            model.eval()

            with torch.no_grad():
                output = model(request.input_data)

            processing_time = (time.time() - processing_start) * 1000

            # Handle dict output
            if isinstance(output, dict):
                output = output.get('logits', output.get('output', output[list(output.keys())[0]]))

            return WorkloadResult(
                request_id=request.request_id,
                output_data=output,
                processing_device=device.device_id,
                processing_time_ms=processing_time,
                total_latency_ms=(time.time() - start_time) * 1000,
                success=True
            )

        except Exception as e:
            return WorkloadResult(
                request_id=request.request_id,
                output_data=torch.tensor([]),
                processing_device=device.device_id,
                processing_time_ms=0,
                total_latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )

        finally:
            # Release device load
            self.registry.update_device_load(device.device_id, -0.1)

    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[WorkloadResult]:
        """Get result for a submitted request."""
        start = time.time()

        while time.time() - start < timeout:
            if request_id in self.result_cache:
                return self.result_cache.pop(request_id)
            time.sleep(0.1)

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        devices = self.registry.get_available_devices()

        return {
            'running': self._running,
            'registered_devices': len(devices),
            'available_devices': sum(1 for d in devices if d.is_available),
            'pending_requests': self.request_queue.qsize(),
            'cached_results': len(self.result_cache),
            'registered_models': list(self.models.keys()),
            'device_details': [
                {
                    'id': d.device_id,
                    'type': d.device_type.value,
                    'load': d.current_load,
                    'available': d.is_available
                }
                for d in devices
            ]
        }


class FailoverManager:
    """Manages failover and redundancy for critical workloads."""

    def __init__(
        self,
        orchestrator: EdgeCloudOrchestrator,
        max_retries: int = 3,
        retry_delay_ms: float = 100
    ):
        self.orchestrator = orchestrator
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

    def execute_with_failover(
        self,
        request: WorkloadRequest,
        fallback_devices: Optional[List[str]] = None
    ) -> WorkloadResult:
        """
        Execute request with automatic failover.

        Args:
            request: The workload request
            fallback_devices: List of device IDs to use as fallback

        Returns:
            Workload result
        """
        for attempt in range(self.max_retries):
            request_id = self.orchestrator.submit_request(request)
            result = self.orchestrator.get_result(request_id)

            if result and result.success:
                return result

            # Modify request for retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay_ms / 1000)
                request.max_latency_ms *= 1.5  # Relax latency constraint

        return WorkloadResult(
            request_id=request.request_id,
            output_data=torch.tensor([]),
            processing_device='failed',
            processing_time_ms=0,
            total_latency_ms=self.max_retries * self.retry_delay_ms,
            success=False,
            error_message='All retries failed'
        )


class MetricsCollector:
    """Collects and aggregates orchestration metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, request: WorkloadRequest, result: WorkloadResult):
        """Record metrics for a completed request."""
        with self._lock:
            self.metrics.append({
                'timestamp': time.time(),
                'request_id': request.request_id,
                'model_name': request.model_name,
                'device_id': result.processing_device,
                'processing_time_ms': result.processing_time_ms,
                'total_latency_ms': result.total_latency_ms,
                'success': result.success,
                'priority': request.priority.value
            })

            # Trim old metrics
            if len(self.metrics) > self.window_size:
                self.metrics = self.metrics[-self.window_size:]

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self._lock:
            if not self.metrics:
                return {}

            successful = [m for m in self.metrics if m['success']]

            return {
                'total_requests': len(self.metrics),
                'success_rate': len(successful) / len(self.metrics),
                'avg_latency_ms': sum(m['total_latency_ms'] for m in successful) / len(successful) if successful else 0,
                'avg_processing_time_ms': sum(m['processing_time_ms'] for m in successful) / len(successful) if successful else 0,
                'p99_latency_ms': sorted(m['total_latency_ms'] for m in successful)[int(len(successful) * 0.99)] if len(successful) > 10 else 0
            }

    def get_device_metrics(self, device_id: str) -> Dict[str, Any]:
        """Get metrics for a specific device."""
        with self._lock:
            device_metrics = [m for m in self.metrics if m['device_id'] == device_id]

            if not device_metrics:
                return {}

            successful = [m for m in device_metrics if m['success']]

            return {
                'device_id': device_id,
                'total_requests': len(device_metrics),
                'success_rate': len(successful) / len(device_metrics),
                'avg_processing_time_ms': sum(m['processing_time_ms'] for m in successful) / len(successful) if successful else 0
            }
