# src/model.py

import torch
import torch.nn as nn
from src.layers import BaseLayer

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, hybrid_thresholds):
        super(DynamicNeuralNetwork, self).__init__()
        self.hybrid_thresholds = hybrid_thresholds

        # Define layers using modular BaseLayer
        self.layer1 = BaseLayer(100, 256)
        self.layer2 = BaseLayer(256, 256)
        self.layer3 = BaseLayer(256, 128)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor, complexities: dict) -> torch.Tensor:
        """
        Routes data through different layers based on complexity metrics.
        """
        if complexities['variance'] and complexities['entropy'] and not complexities['sparsity']:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            # Skip layer3 for less complex data
        x = self.output_layer(x)
        return x
