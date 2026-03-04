"""
Pre-trained models and benchmark datasets support.
Provides pre-trained weights loading, fine-tuning utilities, and standard benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
import urllib.request
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelCheckpoint:
    """Metadata for a model checkpoint."""
    name: str
    version: str
    architecture: str
    dataset: str
    accuracy: float
    num_parameters: int
    url: Optional[str] = None
    local_path: Optional[str] = None


class PretrainedModelRegistry:
    """Registry for pre-trained model checkpoints."""

    _MODELS: Dict[str, ModelCheckpoint] = {
        'dnnr-mnist-base': ModelCheckpoint(
            name='dnnr-mnist-base',
            version='1.0.0',
            architecture='DynamicNeuralNetwork',
            dataset='MNIST',
            accuracy=0.97,
            num_parameters=267_000,
            url='https://models.dnnr.ai/mnist-base-v1.pth',
            local_path='models/mnist_base.pth'
        ),
        'dnnr-mnist-moe': ModelCheckpoint(
            name='dnnr-mnist-moe',
            version='1.0.0',
            architecture='MoEDynamicNetwork',
            dataset='MNIST',
            accuracy=0.98,
            num_parameters=450_000,
            url='https://models.dnnr.ai/mnist-moe-v1.pth',
            local_path='models/mnist_moe.pth'
        ),
        'dnnr-fashion-mnist': ModelCheckpoint(
            name='dnnr-fashion-mnist',
            version='1.0.0',
            architecture='DynamicNeuralNetwork',
            dataset='FashionMNIST',
            accuracy=0.89,
            num_parameters=267_000,
            url='https://models.dnnr.ai/fashion-mnist-v1.pth',
            local_path='models/fashion_mnist.pth'
        ),
        'dnnr-cifar10-base': ModelCheckpoint(
            name='dnnr-cifar10-base',
            version='1.0.0',
            architecture='DynamicNeuralNetwork',
            dataset='CIFAR10',
            accuracy=0.82,
            num_parameters=520_000,
            url='https://models.dnnr.ai/cifar10-base-v1.pth',
            local_path='models/cifar10_base.pth'
        ),
    }

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available pre-trained models."""
        return list(cls._MODELS.keys())

    @classmethod
    def get_model_info(cls, name: str) -> ModelCheckpoint:
        """Get information about a specific model."""
        if name not in cls._MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {cls.list_models()}")
        return cls._MODELS[name]

    @classmethod
    def register_model(cls, checkpoint: ModelCheckpoint):
        """Register a new model checkpoint."""
        cls._MODELS[checkpoint.name] = checkpoint


class ModelDownloader:
    """Downloads and caches pre-trained models."""

    def __init__(self, cache_dir: str = 'models'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_name: str, force: bool = False) -> str:
        """
        Download a pre-trained model.

        Args:
            model_name: Name of the model to download
            force: Force re-download even if cached

        Returns:
            Path to the downloaded model
        """
        checkpoint = PretrainedModelRegistry.get_model_info(model_name)
        local_path = self.cache_dir / f"{model_name}.pth"

        if local_path.exists() and not force:
            return str(local_path)

        if checkpoint.url is None:
            raise ValueError(f"No download URL for model: {model_name}")

        print(f"Downloading {model_name} from {checkpoint.url}...")
        urllib.request.urlretrieve(checkpoint.url, local_path)
        print(f"Downloaded to {local_path}")

        return str(local_path)


class PretrainedLoader:
    """Load and initialize pre-trained models."""

    def __init__(self, cache_dir: str = 'models'):
        self.downloader = ModelDownloader(cache_dir)
        self.cache_dir = cache_dir

    def load(
        self,
        model_name: str,
        model_class: nn.Module,
        strict: bool = True,
        map_location: str = 'cpu'
    ) -> nn.Module:
        """
        Load a pre-trained model.

        Args:
            model_name: Name of the pre-trained model
            model_class: Model class to instantiate
            strict: Whether to strictly enforce state_dict matching
            map_location: Device to map the model to

        Returns:
            Initialized model with loaded weights
        """
        checkpoint_path = self._get_checkpoint_path(model_name)

        if not os.path.exists(checkpoint_path):
            checkpoint_path = self.downloader.download(model_name)

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model = model_class()
        model.load_state_dict(state_dict, strict=strict)

        return model

    def _get_checkpoint_path(self, model_name: str) -> str:
        """Get local path for a model checkpoint."""
        info = PretrainedModelRegistry.get_model_info(model_name)
        if info.local_path:
            return info.local_path
        return os.path.join(self.cache_dir, f"{model_name}.pth")

    def load_partial(
        self,
        model_name: str,
        model: nn.Module,
        prefix: Optional[str] = None,
        exclude_layers: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Load partial weights from a pre-trained model (transfer learning).

        Args:
            model_name: Name of the pre-trained model
            model: Target model to load weights into
            prefix: Optional prefix to match in state_dict keys
            exclude_layers: List of layer names to exclude

        Returns:
            Model with loaded weights
        """
        checkpoint_path = self._get_checkpoint_path(model_name)

        if not os.path.exists(checkpoint_path):
            checkpoint_path = self.downloader.download(model_name)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        # Filter state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            # Apply prefix filter
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix):]

            # Apply exclusion filter
            if exclude_layers and any(ex in key for ex in exclude_layers):
                continue

            new_state_dict[key] = value

        # Load with strict=False to allow partial loading
        model.load_state_dict(new_state_dict, strict=False)

        return model


class FineTuner:
    """Utilities for fine-tuning pre-trained models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        classifier_lr: float = 1e-3
    ):
        self.model = model
        self.freeze_backbone = freeze_backbone

        # Separate parameters for different learning rates
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'output' in name or 'classifier' in name or 'head' in name:
                classifier_params.append(param)
                if freeze_backbone:
                    param.requires_grad = True
            else:
                backbone_params.append(param)
                if freeze_backbone:
                    param.requires_grad = False

        param_groups = [
            {'params': backbone_params, 'lr': learning_rate},
            {'params': classifier_params, 'lr': classifier_lr}
        ]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers for fine-tuning."""
        for name, param in self.model.named_parameters():
            if any(ln in name for ln in layer_names):
                param.requires_grad = True

    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers."""
        for name, param in self.model.named_parameters():
            if any(ln in name for ln in layer_names):
                param.requires_grad = False

    def gradual_unfreeze(self, num_layers: int = 1):
        """Gradually unfreeze layers from the output backwards."""
        layer_names = []
        for name, _ in self.model.named_parameters():
            if 'layer' in name.lower():
                layer_names.append(name)

        # Unfreeze the last N layers
        layers_to_unfreeze = layer_names[-num_layers:]
        self.unfreeze_layers(layers_to_unfreeze)


class BenchmarkDataset:
    """Standard benchmark datasets for model evaluation."""

    DATASETS = {
        'mnist': {
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'input_dim': 784,
            'train_size': 60_000,
            'test_size': 10_000
        },
        'fashion_mnist': {
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'input_dim': 784,
            'train_size': 60_000,
            'test_size': 10_000
        },
        'cifar10': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'input_dim': 3072,
            'train_size': 50_000,
            'test_size': 10_000
        },
        'cifar100': {
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'input_dim': 3072,
            'train_size': 50_000,
            'test_size': 10_000
        }
    }

    def __init__(self, name: str, data_dir: str = 'data'):
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASETS.keys())}")

        self.name = name
        self.info = self.DATASETS[name]
        self.data_dir = data_dir

    def get_loaders(
        self,
        batch_size: int = 64,
        train_transforms: Optional[List] = None,
        test_transforms: Optional[List] = None,
        num_workers: int = 4
    ) -> Tuple[Any, Any]:
        """
        Get data loaders for the dataset.

        Returns:
            Tuple of (train_loader, test_loader)
        """
        from torchvision import datasets, transforms

        if train_transforms is None:
            train_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) if self.info['input_shape'][0] == 1
                else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        if test_transforms is None:
            test_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) if self.info['input_shape'][0] == 1
                else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        train_transform = transforms.Compose(train_transforms)
        test_transform = transforms.Compose(test_transforms)

        if self.name == 'mnist':
            train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.name == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.name == 'cifar10':
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=test_transform)
        elif self.name == 'cifar100':
            train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=test_transform)
        else:
            raise ValueError(f"Dataset {self.name} not implemented")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader


class BenchmarkRunner:
    """Run standard benchmarks on models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose

    def evaluate(
        self,
        test_loader: Any,
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a test set.

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        criterion = criterion or nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        per_class_correct = torch.zeros(10)
        per_class_total = torch.zeros(10)

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                # Flatten if needed
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)

                output = self.model(data)

                # Handle dict output from some models
                if isinstance(output, dict):
                    output = output.get('logits', output.get('output', output[list(output.keys())[0]]))

                loss = criterion(output, target)
                total_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

                # Per-class accuracy
                for i in range(target.size(0)):
                    per_class_total[target[i]] += 1
                    if pred[i] == target[i]:
                        per_class_correct[target[i]] += 1

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        per_class_accuracy = (per_class_correct / (per_class_total + 1e-6)).tolist()

        if self.verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Per-class Accuracy: {[f'{a:.4f}' for a in per_class_accuracy]}")

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'per_class_accuracy': per_class_accuracy,
            'correct': correct,
            'total': total
        }

    def benchmark_latency(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference latency.

        Returns:
            Dictionary with latency statistics (in milliseconds)
        """
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        if dummy_input.dim() > 2:
            dummy_input = dummy_input.view(1, -1)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        # Synchronize if using CUDA
        if self.device != 'cpu':
            torch.cuda.synchronize()

        # Benchmark
        import time
        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()

                if self.device != 'cpu':
                    torch.cuda.synchronize()

                _ = self.model(dummy_input)

                if self.device != 'cpu':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = sorted(latencies)

        return {
            'mean_ms': sum(latencies) / len(latencies),
            'std_ms': (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies)) ** 0.5,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'p50_ms': latencies[len(latencies) // 2],
            'p95_ms': latencies[int(len(latencies) * 0.95)],
            'p99_ms': latencies[int(len(latencies) * 0.99)]
        }

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

    def profile_memory(self, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Profile model memory usage."""
        self.model.eval()

        dummy_input = torch.randn(1, *input_shape).to(self.device)
        if dummy_input.dim() > 2:
            dummy_input = dummy_input.view(1, -1)

        if self.device != 'cpu':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            with torch.no_grad():
                _ = self.model(dummy_input)

            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB

            return {
                'peak_memory_mb': memory_allocated,
                'reserved_memory_mb': memory_reserved
            }
        else:
            import tracemalloc
            tracemalloc.start()

            with torch.no_grad():
                _ = self.model(dummy_input)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                'peak_memory_mb': peak / 1024**2
            }

    def run_full_benchmark(
        self,
        dataset_name: str = 'mnist',
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark suite.

        Returns:
            Dictionary with all benchmark results
        """
        dataset = BenchmarkDataset(dataset_name)
        _, test_loader = dataset.get_loaders(batch_size=batch_size)

        input_dim = dataset.info['input_dim']

        results = {
            'dataset': dataset_name,
            'evaluation': self.evaluate(test_loader),
            'latency': self.benchmark_latency((input_dim,)),
            'parameters': self.count_parameters(),
            'memory': self.profile_memory((input_dim,))
        }

        return results
