import torch
import torch.nn as nn
from src.layers import BaseLayer

class DynamicNeuralNetwork(nn.Module):
    """Dynamic Neural Network with complexity-based routing."""
    
    def __init__(self, hybrid_thresholds, network_config=None):
        super(DynamicNeuralNetwork, self).__init__()
        self.hybrid_thresholds = hybrid_thresholds
        self.network_config = network_config or [
            {"type": "BaseLayer", "input_dim": 784, "output_dim": 256},
            {"type": "BaseLayer", "input_dim": 256, "output_dim": 256},
            {"type": "BaseLayer", "input_dim": 256, "output_dim": 128}
        ]
        self._init_layers()

    def _init_layers(self):
        """Initialize network layers based on the configuration."""
        self.layers = nn.ModuleList()
        last_dim = 784  # Assuming input dimension is always 784
        for layer_config in self.network_config:
            layer_type = layer_config.get("type")
            input_dim = layer_config.get("input_dim", last_dim)
            output_dim = layer_config.get("output_dim")

            if layer_type == "BaseLayer":
                layer = BaseLayer(input_dim, output_dim)
            elif layer_type == "Linear":
                layer = nn.Linear(input_dim, output_dim)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            self.layers.append(layer)
            last_dim = output_dim

        self.output_layer = nn.Linear(last_dim, 10)
        self.shortcut_layer = nn.Linear(256, last_dim) # Add a shortcut layer

    def forward(self, x: torch.Tensor, complexities: dict) -> torch.Tensor:
        """
        Routes data through different layers based on complexity metrics.
        
        Args:
            x: Input tensor
            complexities: Dict containing variance, entropy, and sparsity metrics
            
        Returns:
            Output tensor after forward pass
        """
        x = self.layers[0](x)
        x = self.layers[1](x)
        
        if self._should_use_deep_path(complexities):
            x = self.layers[2](x)
        else:
            x = self.shortcut_layer(x) # Use the shortcut layer
        
        return self.output_layer(x)
    
    def _should_use_deep_path(self, complexities: dict) -> bool:
        """Determine if deep path should be used based on complexities."""
        return (complexities['variance'].mean().item() > 0.5 and
                complexities['entropy'].mean().item() > 0.5 and
                complexities['sparsity'].mean().item() < 0.5)
