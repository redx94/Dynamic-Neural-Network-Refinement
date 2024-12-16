
import torch
import torch.nn as nn
from src.layers import BaseLayer

class DynamicNeuralNetwork(nn.Module):
    """Dynamic Neural Network with complexity-based routing."""
    
    def __init__(self, hybrid_thresholds):
        super(DynamicNeuralNetwork, self).__init__()
        self.hybrid_thresholds = hybrid_thresholds
        self._init_layers()

    def _init_layers(self):
        """Initialize network layers."""
        self.layer1 = BaseLayer(100, 256)
        self.layer2 = BaseLayer(256, 256)
        self.layer3 = BaseLayer(256, 128)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor, complexities: dict) -> torch.Tensor:
        """
        Routes data through different layers based on complexity metrics.
        
        Args:
            x: Input tensor
            complexities: Dict containing variance, entropy, and sparsity metrics
            
        Returns:
            Output tensor after forward pass
        """
        x = self.layer1(x)
        x = self.layer2(x)
        
        if self._should_use_deep_path(complexities):
            x = self.layer3(x)
            
        return self.output_layer(x)
    
    def _should_use_deep_path(self, complexities: dict) -> bool:
        """Determine if deep path should be used based on complexities."""
        return (complexities['variance'] and 
                complexities['entropy'] and 
                not complexities['sparsity'])
