
import unittest
import torch
from src.adaptive_thresholds import AdaptiveThresholds

class TestAdaptiveThresholds(unittest.TestCase):
    def setUp(self):
        self.thresholds = AdaptiveThresholds()
        
    def test_threshold_update(self):
        initial_threshold = self.thresholds.get_current_threshold()
        self.thresholds.update(0.8)
        new_threshold = self.thresholds.get_current_threshold()
        self.assertNotEqual(initial_threshold, new_threshold)
        
    def test_threshold_bounds(self):
        self.thresholds.update(1.5)
        self.assertLessEqual(self.thresholds.get_current_threshold(), 1.0)
        self.thresholds.update(-0.5)
        self.assertGreaterEqual(self.thresholds.get_current_threshold(), 0.0)

if __name__ == '__main__':
    unittest.main()
