import torch
import torch.nn as nn
from typing import Dict, List, Optional

class BaseLayer(nn.Module):
    """Base layer with dynamic routing capabilities."""
    def __init__(self, in_features: int, out_features: int):
        super(BaseLayer, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer(x))

class DynamicNeuralNetwork(nn.Module):
    """
    Dynamic Neural Network with complexity-based routing.
    """
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128], 
                 output_dim: int = 10):
        super(DynamicNeuralNetwork, self).__init__()
        
        # Create dynamic layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(BaseLayer(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(BaseLayer(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor, complexities: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass with dynamic routing based on input complexities.
        Args:
            x: Input tensor
            complexities: Dictionary containing complexity metrics
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input if needed
        
        if complexities is None:
            # Default behavior: use all layers
            for layer in self.layers:
                x = layer(x)
        else:
            # Dynamic routing based on complexities
            for i, layer in enumerate(self.layers):
                # Skip certain layers based on complexity thresholds
                if complexities.get('skip_layer_{}'.format(i), torch.tensor(False)):
                    continue
                x = layer(x)
        
        return self.output_layer(x)
