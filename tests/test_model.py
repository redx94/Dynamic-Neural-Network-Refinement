# tests/test_model.py

import unittest
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds

class TestDynamicNeuralNetwork(unittest.TestCase):
    def setUp(self):
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        self.hybrid_thresholds = HybridThresholds(initial_thresholds, annealing_start_epoch=5, total_epochs=20)
        self.model = DynamicNeuralNetwork(hybrid_thresholds=self.hybrid_thresholds)

    def test_forward_pass(self):
        input_tensor = torch.randn(10, 100)
        complexities = {
            'variance': torch.tensor([0.6]*10),
            'entropy': torch.tensor([0.6]*10),
            'sparsity': torch.tensor([0.4]*10)
        }
        output = self.model(input_tensor, complexities)
        self.assertEqual(output.shape, (10, 10))

    def test_forward_pass_skip_layer3(self):
        input_tensor = torch.randn(10, 100)
        complexities = {
            'variance': torch.tensor([0.4]*10),  # Below threshold
            'entropy': torch.tensor([0.4]*10),  # Below threshold
            'sparsity': torch.tensor([0.6]*10)  # Above threshold
        }
        output = self.model(input_tensor, complexities)
        self.assertEqual(output.shape, (10, 10))
        # Further checks can be implemented to ensure layer3 was skipped

if __name__ == '__main__':
    unittest.main()
