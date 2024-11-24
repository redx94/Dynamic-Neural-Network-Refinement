import unittest
import torch
from src.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = Analyzer()
        self.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    def test_compute_variance(self):
        var = self.analyzer.compute_variance(self.data)
        expected = torch.var(self.data, dim=1)
        self.assertTrue(torch.allclose(var, expected))
    
    def test_compute_entropy(self):
        ent = self.analyzer.compute_entropy(self.data)
        # Add specific entropy test cases
        self.assertEqual(ent.shape, (2,))
    
    def test_compute_sparsity(self):
        spar = self.analyzer.compute_sparsity(self.data)
        # Add specific sparsity test cases
        self.assertEqual(spar.shape, (2,))

if __name__ == '__main__':
    unittest.main()
