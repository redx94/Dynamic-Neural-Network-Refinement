# tests/test_analyzer.py

import unittest
import torch
from src.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = Analyzer()

    def test_compute_variance(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variance = self.analyzer.compute_variance(data)
        expected = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(variance, expected))

    def test_compute_entropy(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        entropy = self.analyzer.compute_entropy(data)
        # Manually calculated entropy for the first tensor
        expected = torch.tensor([1.0986, 0.0])
        self.assertTrue(torch.allclose(entropy, expected, atol=1e-4))

    def test_compute_sparsity(self):
        data = torch.tensor([[0.05, 0.2, 0.03], [0.0, 0.0, 0.0]])
        sparsity = self.analyzer.compute_sparsity(data)
        expected = torch.tensor([0.6667, 1.0])
        self.assertTrue(torch.allclose(sparsity, expected, atol=1e-4))

    def test_analyze(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        complexities = self.analyzer.analyze(data)
        expected = {
            'variance': torch.tensor([1.0, 1.0]),
            'entropy': torch.tensor([1.0986, 1.0986]),
            'sparsity': torch.tensor([0.3333, 0.3333])
        }
        self.assertTrue(torch.allclose(complexities['variance'], expected['variance']))
        self.assertTrue(torch.allclose(complexities['entropy'], expected['entropy'], atol=1e-4))
        self.assertTrue(torch.allclose(complexities['sparsity'], expected['sparsity'], atol=1e-4))

if __name__ == '__main__':
    unittest.main()
