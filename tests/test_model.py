# tests/test_model.py

import unittest
import torch
from src.model import DynamicNeuralNetwork
from src.layers import BaseLayer
from src.hybrid_thresholds import HybridThresholds

class TestDynamicNeuralNetwork(unittest.TestCase):
    def setUp(self):
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        self.hybrid_thresholds = HybridThresholds(initial_thresholds, annealing_start_epoch=5, total_epochs=20)
        self.model = DynamicNeuralNetwork(hybrid_thresholds=self.hybrid_thresholds)

    def test_forward_pass(self):
        input_tensor = torch.randn(10, 784)
        complexities = {
            'variance': torch.tensor([0.6]*10),
            'entropy': torch.tensor([0.6]*10),
            'sparsity': torch.tensor([0.4]*10)
        }
        output = self.model(input_tensor, complexities)
        self.assertEqual(output.shape, (10, 10))

    def test_forward_pass_skip_layer3(self):
        input_tensor = torch.randn(10, 784)
        complexities = {
            'variance': torch.tensor([0.4]*10),  # Below threshold
            'entropy': torch.tensor([0.4]*10),  # Below threshold
            'sparsity': torch.tensor([0.6]*10),  # Above threshold
        }
        # Simulate the forward pass through layer1 and layer2
        x = self.model.layers[0](input_tensor)
        x = self.model.layers[1](x)
        if not self.model._should_use_deep_path(complexities):
            output = self.model.output_layer(x)
        else:
            x = self.model.layers[2](x)
        output = self.model.output_layer(x)
        self.assertEqual(output.shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
