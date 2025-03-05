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
        self.assertTrue(torch.allclose(variance, expected, atol=1e-4))

    def test_compute_sparsity(self):
        data = torch.tensor([[0.05, 0.2, 0.03], [0.0, 0.0, 0.0]])
        sparsity = self.analyzer.compute_sparsity(data)
        expected = torch.tensor([0.6667, 1.0])
        self.assertTrue(torch.allclose(sparsity, expected, atol=1e-4))

    def test_analyze(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        complexities = self.analyzer.analyze(data)
        expected_variance = torch.tensor([0.6666667, 0.0], dtype=torch.float32)
        expected_entropy = torch.tensor([0.8324, 1.0986], dtype=torch.float32)
        expected_sparsity = torch.tensor([0.6667, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(complexities["variance"], expected_variance, atol=1e-4))
        print(f"Computed variance: {complexities['variance']}")
        print(f"Expected variance: {expected_variance}")
        self.assertTrue(torch.allclose(complexities["entropy"], expected_entropy, atol=1e-4))
        print(f"Computed entropy: {complexities['entropy']}")
        self.assertTrue(torch.allclose(complexities["sparsity"], expected_sparsity, atol=1e-4))

    def test_compute_entropy(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        entropy = self.analyzer.compute_entropy(data)
        expected = torch.tensor([0.8324, 1.0986], dtype=torch.float32)
        print(f"Computed entropy: {entropy}")
        print(f"Expected entropy: {expected}")
        self.assertTrue(torch.allclose(entropy, expected, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
