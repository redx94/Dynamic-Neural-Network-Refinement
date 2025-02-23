import unittest
from src.adaptive_thresholds import AdaptiveThresholds


class TestAdaptiveThresholds(unittest.TestCase):
    """
    Unit tests for the AdaptiveThresholds class.
    """

    def setUp(self):
        """
        Sets up the test instance for AdaptiveThresholds.
        """
        self.thresholds = AdaptiveThresholds()

    def test_threshold_update(self):
        """
        Tests updating the adaptive threshold.
        """
        initial_threshold = self.thresholds.get_current_threshold()
        self.thresholds.update(0.8)
        new_threshold = self.thresholds.get_current_threshold()
        self.assertNotEqual(initial_threshold, new_threshold)

    def test_threshold_bounds(self):
        """
        Tests that the threshold does not exceed predefined limits.
        """
        self.thresholds.update(1.5)
        self.assertLessEqual(self.thresholds.get_current_threshold(), 1.0)

        self.thresholds.update(-0.5)
        self.assertGreaterEqual(self.thresholds.get_current_threshold(), 0.0)


if __name__ == "__main__":
    unittest.main()
