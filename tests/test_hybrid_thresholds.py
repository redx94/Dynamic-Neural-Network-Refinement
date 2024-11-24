import unittest
import torch
from src.hybrid_thresholds import HybridThresholds

class TestHybridThresholds(unittest.TestCase):
    def setUp(self):
        initial_thresholds = {'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}
        self.hybrid_thresholds = HybridThresholds(initial_thresholds, annealing_start_epoch=5, total_epochs=20)
        self.var = torch.tensor([0.1, 0.9])
        self.ent = torch.tensor([0.2, 0.8])
        self.spar = torch.tensor([0.3, 0.7])
    
    def test_forward(self):
        complexities = self.hybrid_thresholds(self.var, self.ent, self.spar, current_epoch=10)
        self.assertIn('variance', complexities)
        self.assertIn('entropy', complexities)
        self.assertIn('sparsity', complexities)
    
    def test_calculate_statistical_thresholds(self):
        thresholds = self.hybrid_thresholds.calculate_statistical_thresholds(self.var, self.ent, self.spar)
        self.assertIn('variance', thresholds)
        self.assertIn('entropy', thresholds)
        self.assertIn('sparsity', thresholds)

if __name__ == '__main__':
    unittest.main()
