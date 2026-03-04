import torch
import unittest
from src.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = Analyzer()

    def test_compute_variance(self):
        data = torch.tensor([[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]])
        variance = self.analyzer.compute_variance(data)
        
        # PyTorch defaults to unbiased variance (correction=1)
        # For [0,0,0,0]: var is 0
        self.assertAlmostEqual(variance[0].item(), 0.0, places=4)
        
        # For [100,0,0,0]: mean=25. sum_sq = 75^2 + 3*(-25)^2 = 5625 + 1875 = 7500.
        # unbiased var = 7500 / 3 = 2500
        self.assertAlmostEqual(variance[1].item(), 2500.0, places=4)

    def test_compute_entropy(self):
        data = torch.tensor([[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]])
        entropy = self.analyzer.compute_entropy(data)
        
        # For [0,0,0,0]: softmax is [0.25, 0.25, 0.25, 0.25]
        # entropy is -4 * (0.25 * ln(0.25 + 1e-10)) = 1.38629...
        import math
        expected_g1 = -math.log(0.25)
        self.assertAlmostEqual(entropy[0].item(), expected_g1, places=4)
        
        # For [100,0,0,0]: softmax is approx [1.0, 0.0, 0.0, 0.0]
        # entropy is approx 0 (-1 * ln(1) - 3 * 0 * ln(1e-10)) = 0
        self.assertAlmostEqual(entropy[1].item(), 0.0, places=4)

    def test_compute_sparsity(self):
        data = torch.tensor([[0.0, 0.05, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], [0.2, 0.3, 0.4, 0.5]])
        sparsity = self.analyzer.compute_sparsity(data)
        
        # threshold is 0.1
        # [0.0, 0.05, 0.0, 0.0] -> 4/4 < 0.1 -> 1.0
        self.assertAlmostEqual(sparsity[0].item(), 1.0, places=4)
        # [1.0, 0.5, 0.0, 0.0] -> 2/4 < 0.1 -> 0.5
        self.assertAlmostEqual(sparsity[1].item(), 0.5, places=4)
        # [0.2, 0.3, 0.4, 0.5] -> 0/4 < 0.1 -> 0.0
        self.assertAlmostEqual(sparsity[2].item(), 0.0, places=4)

    def test_analyze_return_types(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        complexities = self.analyzer.analyze(data)
        
        self.assertIsInstance(complexities, dict)
        self.assertIn('variance', complexities)
        self.assertIn('entropy', complexities)
        self.assertIn('sparsity', complexities)
        
        self.assertIsInstance(complexities['variance'], torch.Tensor)
        self.assertIsInstance(complexities['entropy'], torch.Tensor)
        self.assertIsInstance(complexities['sparsity'], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
