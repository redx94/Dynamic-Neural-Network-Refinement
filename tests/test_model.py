import unittest
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds

class TestDynamicNeuralNetwork(unittest.TestCase):
    def setUp(self):
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        hybrid_thresholds = HybridThresholds(initial_thresholds, annealing_start_epoch=5, total_epochs=20)
        self.model = DynamicNeuralNetwork(hybrid_thresholds)
        self.input = torch.randn(10, 100)
        # Mock complexities
        self.complexities = {
            'variance': 0.5,
            'entropy': 0.5,
            'sparsity': 0.5
        }
    
    def test_forward(self):
        output = self.model(self.input, self.complexities)
        self.assertEqual(output.shape, (10, 10))  # Assuming 10 classes

if __name__ == '__main__':
    unittest.main()
