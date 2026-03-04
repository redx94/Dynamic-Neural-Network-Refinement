# DNNR Advanced Features Guide

This guide covers the advanced features added in version 0.3.0 to transform DNNR into a world-class AI platform.

## Table of Contents

1. [Mixture of Experts](#mixture-of-experts)
2. [Reinforcement Learning Routing](#reinforcement-learning-routing)
3. [Neural Architecture Search](#neural-architecture-search)
4. [Pre-trained Models & Benchmarks](#pre-trained-models--benchmarks)
5. [Enterprise Security](#enterprise-security)
6. [Edge-Cloud Orchestration](#edge-cloud-orchestration)

---

## Mixture of Experts

The MoE module implements sparse gating with efficient expert routing for scalable inference.

### Basic Usage

```python
from src.mixture_of_experts import MixtureOfExperts, MoEDynamicNetwork

# Standalone MoE layer
moe = MixtureOfExperts(
    input_dim=256,
    output_dim=128,
    num_experts=8,
    expert_hidden_dim=256,
    top_k=2  # Select top-2 experts per sample
)

# Forward pass
result = moe(input_tensor, training=True)
output = result['output']
load_balance_loss = result['load_balance_loss']
```

### End-to-End MoE Network

```python
from src.mixture_of_experts import MoEDynamicNetwork

model = MoEDynamicNetwork(
    input_dim=784,
    num_classes=10,
    num_experts=8,
    expert_hidden_dim=256,
    top_k=2
)

# Training
result = model(x, training=True)
logits = result['logits']
routing_stats = result['routing_stats']
```

### Hierarchical MoE

```python
moe = MixtureOfExperts(
    input_dim=256,
    output_dim=128,
    num_experts=16,
    hierarchical=True,
    num_groups=4
)
```

---

## Reinforcement Learning Routing

RL-based routing learns optimal path selection from experience.

### Basic Usage

```python
from src.rl_routing import RLRoutingAgent, HybridRouter

# Initialize agent
agent = RLRoutingAgent(
    learning_rate=3e-4,
    gamma=0.99
)

# Select route
complexities = {
    'variance': torch.tensor(0.6),
    'entropy': torch.tensor(0.7),
    'sparsity': torch.tensor(0.3)
}
route, log_prob = agent.select_route(complexities)

# Training loop
for batch in dataloader:
    # ... compute complexities and reward ...
    agent.store_transition(state, action, reward, value, log_prob, done)
    
metrics = agent.update()
```

### Hybrid Router

```python
from src.rl_routing import HybridRouter

router = HybridRouter(
    complexity_thresholds={'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5},
    rl_weight=0.5  # Balance between rules and RL
)

route, info = router.route(complexities, deterministic=True)
```

---

## Neural Architecture Search

Automated architecture discovery with multiple search strategies.

### Evolutionary Search

```python
from src.neural_architecture_search import NASController, ArchitectureConfig

controller = NASController(
    search_method='evolutionary',
    use_predictor=True
)

def evaluate_architecture(config: ArchitectureConfig) -> float:
    # Build and train model
    # Return accuracy or fitness score
    return fitness_score

best_architecture = controller.search(
    evaluate_fn=evaluate_architecture,
    num_iterations=100
)
```

### Differentiable NAS (DARTS-style)

```python
from src.neural_architecture_search import DifferentiableNASCell

cell = DifferentiableNASCell(
    input_dim=128,
    output_dim=64,
    num_operations=4,
    num_nodes=4
)

# Training with temperature annealing
for epoch in range(num_epochs):
    temperature = max(0.1, 1.0 - epoch / num_epochs)
    output = cell(x, temperature=temperature)
```

### Performance Prediction

```python
from src.neural_architecture_search import PerformancePredictor

predictor = PerformancePredictor()
arch_vector = config.to_vector()

predictions = predictor(arch_vector)
accuracy = predictions['predicted_accuracy']
latency = predictions['predicted_latency_ms']
memory = predictions['predicted_memory_mb']
```

---

## Pre-trained Models & Benchmarks

### Loading Pre-trained Models

```python
from src.pretrained import PretrainedLoader, PretrainedModelRegistry

# List available models
print(PretrainedModelRegistry.list_models())
# ['dnnr-mnist-base', 'dnnr-mnist-moe', 'dnnr-fashion-mnist', 'dnnr-cifar10-base']

# Load pre-trained model
loader = PretrainedLoader()
model = loader.load('dnnr-mnist-base', model_class=DynamicNeuralNetwork)
```

### Fine-tuning

```python
from src.pretrained import FineTuner

# Initialize fine-tuner
tuner = FineTuner(
    model=model,
    learning_rate=1e-4,
    freeze_backbone=True,  # Freeze all but output layers
    classifier_lr=1e-3
)

# Gradual unfreezing
tuner.gradual_unfreeze(num_layers=1)  # Unfreeze last layer
```

### Running Benchmarks

```python
from src.pretrained import BenchmarkDataset, BenchmarkRunner

# Load dataset
dataset = BenchmarkDataset('mnist')
train_loader, test_loader = dataset.get_loaders(batch_size=64)

# Run benchmarks
runner = BenchmarkRunner(model, device='cuda')
results = runner.run_full_benchmark(dataset_name='mnist')

print(f"Accuracy: {results['evaluation']['accuracy']:.4f}")
print(f"Latency p95: {results['latency']['p95_ms']:.2f}ms")
print(f"Memory: {results['memory']['peak_memory_mb']:.1f}MB")
```

---

## Enterprise Security

### Model Encryption

```python
from src.enterprise.encryption import SimpleCipher, SecureModelStorage

# Encrypt model weights
cipher = SimpleCipher()
encrypted_tensor = cipher.encrypt_tensor(model_weights)

# Secure storage
storage = SecureModelStorage('secure_models')
storage.save_model(model, 'production_model', metadata={'version': '1.0'})
```

### Anomaly Detection

```python
from src.enterprise.encryption import AdvancedAnomalyDetector

detector = AdvancedAnomalyDetector(
    variance_threshold=2.5,
    entropy_threshold=6.0
)

# Detect threats
threats = detector.analyze(complexities)
for threat in threats:
    print(f"Threat: {threat.threat_type} - {threat.description}")
```

### API Key Management

```python
from src.enterprise.encryption import APIKeyManager

manager = APIKeyManager()

# Generate key
api_key = manager.generate_api_key(
    client_id='client_123',
    permissions=['read', 'inference'],
    expires_days=365
)

# Validate
is_valid, message = manager.validate_key(api_key, required_permission='inference')

# Rotate key
new_key = manager.rotate_key(old_key)
```

---

## Edge-Cloud Orchestration

### Basic Setup

```python
from src.enterprise.edge_cloud import (
    EdgeCloudOrchestrator,
    DeviceCapabilities,
    DeviceType,
    WorkloadRequest,
    WorkloadPriority
)

# Initialize orchestrator
orchestrator = EdgeCloudOrchestrator()

# Register devices
edge_device = DeviceCapabilities(
    device_id='edge_1',
    device_type=DeviceType.EDGE_SERVER,
    memory_mb=4096,
    compute_capability=2.0,
    latency_ms=5.0
)
orchestrator.add_device(edge_device)

# Register model
orchestrator.register_model('classifier', model)

# Submit workload
request = WorkloadRequest(
    request_id='req_001',
    model_name='classifier',
    input_data=input_tensor,
    complexity_metrics={'compute_intensity': 0.5},
    priority=WorkloadPriority.HIGH,
    max_latency_ms=50.0
)

orchestrator.start()
request_id = orchestrator.submit_request(request)
result = orchestrator.get_result(request_id)
```

### Scheduling Strategies

```python
from src.enterprise.edge_cloud import (
    LatencyOptimizedScheduler,
    CostOptimizedScheduler,
    BalancedScheduler
)

# Latency-optimized
orchestrator = EdgeCloudOrchestrator(
    scheduler=LatencyOptimizedScheduler()
)

# Cost-optimized
orchestrator = EdgeCloudOrchestrator(
    scheduler=CostOptimizedScheduler()
)

# Balanced (default)
orchestrator = EdgeCloudOrchestrator(
    scheduler=BalancedScheduler(
        latency_weight=0.4,
        cost_weight=0.3,
        load_weight=0.3
    )
)
```

### Failover

```python
from src.enterprise.edge_cloud import FailoverManager

failover = FailoverManager(
    orchestrator=orchestrator,
    max_retries=3,
    retry_delay_ms=100
)

result = failover.execute_with_failover(request)
```

### Metrics Collection

```python
from src.enterprise.edge_cloud import MetricsCollector

collector = MetricsCollector(window_size=1000)

# Record after each request
collector.record(request, result)

# Get aggregate metrics
metrics = collector.get_aggregate_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"P99 latency: {metrics['p99_latency_ms']:.1f}ms")
```

---

## Integration Example

Complete example combining all features:

```python
import torch
from src import (
    MoEDynamicNetwork,
    HybridRouter,
    PretrainedLoader,
    BenchmarkRunner,
    BenchmarkDataset
)
from src.enterprise import (
    EdgeCloudOrchestrator,
    DeviceCapabilities,
    DeviceType,
    LatencyOptimizedScheduler,
    MetricsCollector,
    APIKeyManager
)

# 1. Load pre-trained MoE model
loader = PretrainedLoader()
model = loader.load('dnnr-mnist-moe', MoEDynamicNetwork)

# 2. Initialize hybrid router
router = HybridRouter(rl_weight=0.7)

# 3. Set up edge-cloud orchestration
orchestrator = EdgeCloudOrchestrator(
    scheduler=LatencyOptimizedScheduler()
)

# Register edge and cloud devices
orchestrator.add_device(DeviceCapabilities(
    device_id='edge_gpu',
    device_type=DeviceType.EDGE_SERVER,
    memory_mb=8192,
    compute_capability=5.0,
    latency_ms=10.0
))

orchestrator.add_device(DeviceCapabilities(
    device_id='cloud_tpu',
    device_type=DeviceType.CLOUD_TPU,
    memory_mb=32768,
    compute_capability=50.0,
    latency_ms=100.0
))

orchestrator.register_model('mnist_classifier', model)
orchestrator.start()

# 4. API security
api_manager = APIKeyManager(max_requests_per_hour=1000)
api_key = api_manager.generate_api_key('client_1', ['inference'])

# 5. Metrics
collector = MetricsCollector()

# 6. Run inference with orchestration
def secure_inference(input_tensor, api_key):
    # Validate API key
    is_valid, msg = api_manager.validate_key(api_key, 'inference')
    if not is_valid:
        raise ValueError(f"Authentication failed: {msg}")
    
    # Determine routing
    from src.analyzer import Analyzer
    analyzer = Analyzer()
    complexities = analyzer.analyze(input_tensor)
    
    route, routing_info = router.route(complexities)
    
    # Submit to orchestrator
    from src.enterprise.edge_cloud import WorkloadRequest, WorkloadPriority
    request = WorkloadRequest(
        request_id=f'req_{time.time()}',
        model_name='mnist_classifier',
        input_data=input_tensor,
        complexity_metrics=routing_info['complexity_scores'],
        priority=WorkloadPriority.NORMAL,
        max_latency_ms=100.0
    )
    
    request_id = orchestrator.submit_request(request)
    result = orchestrator.get_result(request_id, timeout=5.0)
    
    # Record metrics
    collector.record(request, result)
    
    return result.output_data, routing_info
```

---

## Performance Tips

1. **MoE**: Use `top_k=2` for best balance of performance and accuracy
2. **RL Routing**: Train with diverse data to prevent overfitting to specific patterns
3. **NAS**: Use performance predictor to reduce search time
4. **Security**: Rotate API keys monthly and monitor threat logs
5. **Orchestration**: Use balanced scheduler for mixed workloads

## API Reference

Full API documentation is available in the `docs/` directory and through Python's help system:

```python
from src import MoEDynamicNetwork
help(MoEDynamicNetwork)
```
