import torch
import pytest
from src.model import DynamicNeuralNetwork

def test_shallow_routing():
    """Verify that simple data takes the shallow routing path."""
    model = DynamicNeuralNetwork(hybrid_thresholds=None)
    # mock simple data
    x = torch.randn(1, 784)
    # Low variance, low entropy, high sparsity = simple data
    complexities = {
        'variance': torch.tensor([0.2]),
        'entropy': torch.tensor([0.2]),
        'sparsity': torch.tensor([0.8])
    }
    
    output = model(x, complexities)
    
    assert output is not None
    assert output.shape == (1, 10)

def test_deep_routing():
    """Verify that complex data takes the deep routing path."""
    model = DynamicNeuralNetwork(hybrid_thresholds=None)
    # mock complex data
    x = torch.randn(1, 784)
    # High variance, high entropy, low sparsity = complex data
    complexities = {
        'variance': torch.tensor([0.8]),
        'entropy': torch.tensor([0.8]),
        'sparsity': torch.tensor([0.2])
    }
    
    output = model(x, complexities)
    
    assert output is not None
    assert output.shape == (1, 10)
    
def test_replace_layer():
    """Verify that replacing a layer works correctly."""
    import torch.nn as nn
    model = DynamicNeuralNetwork(hybrid_thresholds=None)
    new_layer = nn.Linear(784, 128)
    model.replace_layer(0, new_layer)
    assert model.layers[0] is new_layer
