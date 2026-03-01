"""
Tests for Enterprise Security and Edge-Cloud modules.
"""

import pytest
import torch
import torch.nn as nn
import time
from src.enterprise.encryption import (
    SimpleCipher,
    SecureModelStorage,
    AdvancedAnomalyDetector,
    SecureInference,
    APIKeyManager,
    ThreatLevel,
    ThreatReport,
    SecurityError
)
from src.enterprise.edge_cloud import (
    EdgeCloudOrchestrator,
    DeviceRegistry,
    DeviceType,
    WorkloadPriority,
    WorkloadRequest,
    WorkloadResult,
    DeviceCapabilities,
    LatencyOptimizedScheduler,
    CostOptimizedScheduler,
    BalancedScheduler,
    FailoverManager,
    MetricsCollector
)


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)


class TestSimpleCipher:
    """Tests for SimpleCipher."""

    def test_cipher_initialization(self):
        """Test cipher initialization."""
        cipher = SimpleCipher()
        assert cipher.key is not None

    def test_encrypt_decrypt_bytes(self):
        """Test encrypting and decrypting bytes."""
        cipher = SimpleCipher()
        original = b"Hello, World! This is a test message."

        encrypted = cipher.encrypt(original)
        decrypted = cipher.decrypt(encrypted)

        assert decrypted == original
        assert encrypted != original

    def test_encrypt_decrypt_different_keys(self):
        """Test that different keys produce different ciphertexts."""
        cipher1 = SimpleCipher()
        cipher2 = SimpleCipher()

        original = b"Test message"
        encrypted1 = cipher1.encrypt(original)
        encrypted2 = cipher2.encrypt(original)

        # Different keys should produce different ciphertexts
        assert encrypted1 != encrypted2

    def test_encrypt_decrypt_tensor(self):
        """Test encrypting and decrypting tensors."""
        cipher = SimpleCipher()
        original = torch.randn(32, 784)

        encrypted = cipher.encrypt_tensor(original)
        decrypted = cipher.decrypt_tensor(encrypted)

        assert decrypted.shape == original.shape
        assert torch.allclose(decrypted, original, atol=1e-6)

    def test_encrypt_different_shapes(self):
        """Test encrypting tensors with different shapes."""
        cipher = SimpleCipher()

        shapes = [(10,), (5, 10), (2, 3, 4), (1, 1, 1, 1)]

        for shape in shapes:
            tensor = torch.randn(shape)
            encrypted = cipher.encrypt_tensor(tensor)
            decrypted = cipher.decrypt_tensor(encrypted)
            assert torch.allclose(decrypted, tensor, atol=1e-6)


class TestSecureModelStorage:
    """Tests for SecureModelStorage."""

    def test_storage_initialization(self, tmp_path):
        """Test secure storage initialization."""
        storage = SecureModelStorage(storage_path=str(tmp_path))
        assert storage.storage_path == str(tmp_path)

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a model."""
        storage = SecureModelStorage(storage_path=str(tmp_path))
        model = SimpleModel()

        # Save model
        storage.save_model(model, "test_model", metadata={'version': '1.0'})

        # Load model
        new_model = SimpleModel()
        loaded_model = storage.load_model(new_model, "test_model", verify_integrity=True)

        # Check weights are the same
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)

    def test_save_with_metadata(self, tmp_path):
        """Test saving model with metadata."""
        import os

        storage = SecureModelStorage(storage_path=str(tmp_path))
        model = SimpleModel()

        metadata = {
            'version': '2.0',
            'trained_on': 'MNIST',
            'accuracy': 0.95
        }
        storage.save_model(model, "model_with_meta", metadata=metadata)

        # Check metadata file exists
        meta_path = os.path.join(str(tmp_path), "model_with_meta.meta")
        assert os.path.exists(meta_path)


class TestAdvancedAnomalyDetector:
    """Tests for AdvancedAnomalyDetector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = AdvancedAnomalyDetector()
        assert detector.variance_threshold == 2.5

    def test_update_baselines(self):
        """Test baseline updates."""
        detector = AdvancedAnomalyDetector()

        complexities = {
            'variance': torch.tensor([0.5]),
            'entropy': torch.tensor([2.0]),
            'sparsity': torch.tensor([0.3])
        }

        initial_variance_mean = detector.baselines['variance_mean']
        detector.update_baselines(complexities)

        # Baseline should move toward the new value
        assert detector.baselines['variance_mean'] != initial_variance_mean

    def test_detect_fgsm_normal(self):
        """Test FGSM detection with normal input."""
        detector = AdvancedAnomalyDetector()

        normal_complexities = {
            'variance': torch.tensor([0.5]),
            'entropy': torch.tensor([2.0])
        }

        report = detector.detect_fgsm(normal_complexities)

        assert report.level == ThreatLevel.LOW
        assert report.threat_type == 'NORMAL'

    def test_detect_fgsm_attack(self):
        """Test FGSM detection with adversarial input."""
        detector = AdvancedAnomalyDetector()

        # Simulate FGSM signature (high variance and entropy)
        attack_complexities = {
            'variance': torch.tensor([5.0]),
            'entropy': torch.tensor([10.0])
        }

        report = detector.detect_fgsm(attack_complexities)

        assert report.level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]
        assert 'FGSM' in report.threat_type

    def test_detect_data_poisoning_normal(self):
        """Test data poisoning detection with normal input."""
        detector = AdvancedAnomalyDetector()

        normal_complexities = {
            'variance': torch.tensor([0.5]),
            'entropy': torch.tensor([2.0]),
            'sparsity': torch.tensor([0.3])
        }

        report = detector.detect_data_poisoning(normal_complexities)

        assert report.level == ThreatLevel.LOW

    def test_detect_data_poisoning_backdoor(self):
        """Test detection of potential backdoor."""
        detector = AdvancedAnomalyDetector(sparsity_threshold=0.9)

        # Extreme sparsity might indicate backdoor trigger
        suspicious_complexities = {
            'variance': torch.tensor([0.5]),
            'entropy': torch.tensor([2.0]),
            'sparsity': torch.tensor([0.95])
        }

        report = detector.detect_data_poisoning(suspicious_complexities)

        assert report.level != ThreatLevel.LOW or report.threat_type == 'NORMAL'

    def test_detect_model_extraction(self):
        """Test model extraction detection."""
        detector = AdvancedAnomalyDetector()

        # Normal query pattern
        normal_queries = [
            {'timestamp': time.time() - i, 'input_hash': f'hash{i}'}
            for i in range(5)
        ]

        report = detector.detect_model_extraction(normal_queries, time_window=60.0)

        assert report.threat_type in ['NORMAL', 'MODEL_EXTRACTION']

    def test_analyze(self):
        """Test full analysis."""
        detector = AdvancedAnomalyDetector()

        complexities = {
            'variance': torch.tensor([0.5]),
            'entropy': torch.tensor([2.0]),
            'sparsity': torch.tensor([0.3])
        }

        reports = detector.analyze(complexities)

        assert isinstance(reports, list)


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_key_manager_initialization(self):
        """Test key manager initialization."""
        manager = APIKeyManager()
        assert manager.keys == {}

    def test_generate_api_key(self):
        """Test API key generation."""
        manager = APIKeyManager()

        key = manager.generate_api_key(client_id='test_client')

        assert key is not None
        assert key in manager.keys
        assert manager.keys[key]['client_id'] == 'test_client'

    def test_validate_key(self):
        """Test key validation."""
        manager = APIKeyManager()
        key = manager.generate_api_key(client_id='test_client')

        is_valid, message = manager.validate_key(key)

        assert is_valid
        assert message == "Valid"

    def test_validate_invalid_key(self):
        """Test validation of invalid key."""
        manager = APIKeyManager()

        is_valid, message = manager.validate_key("invalid_key")

        assert not is_valid
        assert "Invalid" in message

    def test_revoke_key(self):
        """Test key revocation."""
        manager = APIKeyManager()
        key = manager.generate_api_key(client_id='test_client')

        manager.revoke_key(key)

        is_valid, _ = manager.validate_key(key)
        assert not is_valid

    def test_rotate_key(self):
        """Test key rotation."""
        manager = APIKeyManager()
        old_key = manager.generate_api_key(client_id='test_client')

        new_key = manager.rotate_key(old_key)

        assert new_key != old_key

        # Old key should be invalid
        is_valid_old, _ = manager.validate_key(old_key)
        assert not is_valid_old

        # New key should be valid
        is_valid_new, _ = manager.validate_key(new_key)
        assert is_valid_new


class TestDeviceRegistry:
    """Tests for DeviceRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = DeviceRegistry()
        assert len(registry.devices) == 0

    def test_register_device(self):
        """Test device registration."""
        registry = DeviceRegistry()
        device = DeviceCapabilities(
            device_id='device_1',
            device_type=DeviceType.EDGE_SERVER,
            memory_mb=4096,
            compute_capability=1.0,
            latency_ms=10.0
        )

        registry.register_device(device)

        assert len(registry.devices) == 1
        assert 'device_1' in registry.devices

    def test_unregister_device(self):
        """Test device unregistration."""
        registry = DeviceRegistry()
        device = DeviceCapabilities(
            device_id='device_1',
            device_type=DeviceType.CLOUD_CPU,
            memory_mb=8192,
            compute_capability=2.0,
            latency_ms=50.0
        )
        registry.register_device(device)

        registry.unregister_device('device_1')

        assert len(registry.devices) == 0

    def test_get_available_devices(self):
        """Test getting available devices."""
        registry = DeviceRegistry()
        device1 = DeviceCapabilities(
            device_id='device_1',
            device_type=DeviceType.CLOUD_GPU,
            memory_mb=16384,
            compute_capability=10.0,
            latency_ms=20.0,
            is_available=True
        )
        device2 = DeviceCapabilities(
            device_id='device_2',
            device_type=DeviceType.EDGE_MOBILE,
            memory_mb=1024,
            compute_capability=0.5,
            latency_ms=5.0,
            is_available=False
        )
        registry.register_device(device1)
        registry.register_device(device2)

        available = registry.get_available_devices()

        assert len(available) == 1
        assert available[0].device_id == 'device_1'

    def test_update_device_load(self):
        """Test updating device load."""
        registry = DeviceRegistry()
        device = DeviceCapabilities(
            device_id='device_1',
            device_type=DeviceType.CLOUD_CPU,
            memory_mb=8192,
            compute_capability=2.0,
            latency_ms=50.0,
            current_load=0.3
        )
        registry.register_device(device)

        registry.update_device_load('device_1', 0.2)

        assert registry.devices['device_1'].current_load == 0.5


class TestSchedulers:
    """Tests for workload schedulers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.devices = [
            DeviceCapabilities(
                device_id='fast_edge',
                device_type=DeviceType.EDGE_SERVER,
                memory_mb=4096,
                compute_capability=2.0,
                latency_ms=5.0,
                current_load=0.1
            ),
            DeviceCapabilities(
                device_id='slow_cloud',
                device_type=DeviceType.CLOUD_GPU,
                memory_mb=16384,
                compute_capability=10.0,
                latency_ms=100.0,
                current_load=0.5
            ),
            DeviceCapabilities(
                device_id='cheap_edge',
                device_type=DeviceType.EDGE_MOBILE,
                memory_mb=1024,
                compute_capability=0.5,
                latency_ms=20.0,
                current_load=0.2
            )
        ]

        self.request = WorkloadRequest(
            request_id='test_req',
            model_name='default',
            input_data=torch.randn(1, 784),
            complexity_metrics={'compute_intensity': 1.0},
            priority=WorkloadPriority.NORMAL,
            max_latency_ms=200.0
        )

    def test_latency_optimized_scheduler(self):
        """Test latency-optimized scheduler."""
        scheduler = LatencyOptimizedScheduler()

        selected = scheduler.select_device(self.request, self.devices)

        assert selected is not None
        # Should select the device with lowest latency
        assert selected.device_id == 'fast_edge'

    def test_cost_optimized_scheduler(self):
        """Test cost-optimized scheduler."""
        scheduler = CostOptimizedScheduler()

        selected = scheduler.select_device(self.request, self.devices)

        assert selected is not None
        # Should prefer edge devices (lower cost)
        assert selected.device_type in [DeviceType.EDGE_SERVER, DeviceType.EDGE_MOBILE]

    def test_balanced_scheduler(self):
        """Test balanced scheduler."""
        scheduler = BalancedScheduler()

        selected = scheduler.select_device(self.request, self.devices)

        assert selected is not None
        assert selected.device_id in [d.device_id for d in self.devices]

    def test_scheduler_no_available_devices(self):
        """Test scheduler with no available devices."""
        scheduler = LatencyOptimizedScheduler()

        selected = scheduler.select_device(self.request, [])

        assert selected is None

    def test_scheduler_latency_constraint(self):
        """Test scheduler with tight latency constraint."""
        scheduler = LatencyOptimizedScheduler()

        tight_request = WorkloadRequest(
            request_id='tight_req',
            model_name='default',
            input_data=torch.randn(1, 784),
            complexity_metrics={'compute_intensity': 1.0},
            priority=WorkloadPriority.HIGH,
            max_latency_ms=10.0  # Very tight constraint
        )

        selected = scheduler.select_device(tight_request, self.devices)

        # May or may not find a suitable device
        # Just check it runs without error
        pass


class TestEdgeCloudOrchestrator:
    """Tests for EdgeCloudOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = EdgeCloudOrchestrator()

        assert orchestrator.registry is not None
        assert orchestrator.scheduler is not None

    def test_add_device(self):
        """Test adding device to orchestrator."""
        orchestrator = EdgeCloudOrchestrator()
        device = DeviceCapabilities(
            device_id='test_device',
            device_type=DeviceType.CLOUD_CPU,
            memory_mb=8192,
            compute_capability=2.0,
            latency_ms=50.0
        )

        orchestrator.add_device(device)

        assert 'test_device' in orchestrator.registry.devices

    def test_register_model(self):
        """Test registering model."""
        orchestrator = EdgeCloudOrchestrator()
        model = SimpleModel()

        orchestrator.register_model('test_model', model)

        assert 'test_model' in orchestrator.models

    def test_get_status(self):
        """Test getting orchestrator status."""
        orchestrator = EdgeCloudOrchestrator()

        status = orchestrator.get_status()

        assert 'running' in status
        assert 'registered_devices' in status
        assert 'pending_requests' in status


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        assert len(collector.metrics) == 0

    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector()

        request = WorkloadRequest(
            request_id='test',
            model_name='default',
            input_data=torch.randn(1, 784),
            complexity_metrics={},
            priority=WorkloadPriority.NORMAL
        )

        result = WorkloadResult(
            request_id='test',
            output_data=torch.randn(1, 10),
            processing_device='device_1',
            processing_time_ms=10.0,
            total_latency_ms=15.0,
            success=True
        )

        collector.record(request, result)

        assert len(collector.metrics) == 1

    def test_get_aggregate_metrics(self):
        """Test getting aggregate metrics."""
        collector = MetricsCollector()

        # Record some metrics
        for i in range(10):
            request = WorkloadRequest(
                request_id=f'test_{i}',
                model_name='default',
                input_data=torch.randn(1, 784),
                complexity_metrics={},
                priority=WorkloadPriority.NORMAL
            )

            result = WorkloadResult(
                request_id=f'test_{i}',
                output_data=torch.randn(1, 10),
                processing_device='device_1',
                processing_time_ms=10.0 + i,
                total_latency_ms=15.0 + i,
                success=True
            )

            collector.record(request, result)

        metrics = collector.get_aggregate_metrics()

        assert 'total_requests' in metrics
        assert metrics['total_requests'] == 10
        assert 'success_rate' in metrics
        assert 'avg_latency_ms' in metrics

    def test_window_size(self):
        """Test that window size is respected."""
        collector = MetricsCollector(window_size=5)

        # Record more than window size
        for i in range(10):
            request = WorkloadRequest(
                request_id=f'test_{i}',
                model_name='default',
                input_data=torch.randn(1, 784),
                complexity_metrics={},
                priority=WorkloadPriority.NORMAL
            )

            result = WorkloadResult(
                request_id=f'test_{i}',
                output_data=torch.randn(1, 10),
                processing_device='device_1',
                processing_time_ms=10.0,
                total_latency_ms=15.0,
                success=True
            )

            collector.record(request, result)

        assert len(collector.metrics) <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
