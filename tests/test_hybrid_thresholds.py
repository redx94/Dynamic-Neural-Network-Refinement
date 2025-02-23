# tests/test_hybrid_thresholds.py

import unittest
import torch
from src.hybrid_thresholds import HybridThresholds

class TestHybridThresholds(unittest.TestCase):
    def setUp(self):
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        self.hybrid = HybridThresholds(initial_thresholds, annealing_start_epoch=5, total_epochs=20)

    def test_before_annealing(self):
        current_epoch = 3
        variance = torch.tensor([0.6])
        entropy = torch.tensor([0.6])
        sparsity = torch.tensor([0.4])
        thresholds = self.hybrid(variance, entropy, sparsity, current_epoch)
        expected = {
            'variance': torch.tensor([True]),
            'entropy': torch.tensor([True]),
            'sparsity': torch.tensor([True])
        }
        self.assertTrue(torch.equal(thresholds['variance'], expected['variance']))
        self.assertTrue(torch.equal(thresholds['entropy'], expected['entropy']))
        self.assertTrue(torch.equal(thresholds['sparsity'], expected['sparsity']))

    def test_during_annealing(self):
        current_epoch = 10
        thresholds = self.hybrid.anneal_thresholds(current_epoch)
        expected = {
            'variance': 0.5 * (1 - (10 - 5) / (20 - 5)),
            'entropy': 0.5 * (1 - (10 - 5) / (20 - 5)),
            'sparsity': 0.5 * (1 - (10 - 5) / (20 - 5))
        }
        for key in expected:
            self.assertAlmostEqual(thresholds[key], expected[key], places=4)

    def test_after_annealing(self):
        current_epoch = 20
        thresholds = self.hybrid.anneal_thresholds(current_epoch)
        expected = {key: 0.0 for key in thresholds}
        for key in thresholds:
            self.assertAlmostEqual(thresholds[key], expected[key], places=4)

if __name__ == '__main__':
    unittest.main()
