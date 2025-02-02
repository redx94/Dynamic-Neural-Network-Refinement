import unittest
import torch
import torch.nn as nn
from src.per_sample_complexity import process_batch_dynamic


class MockModel(nn.Module):
    """
    Mock model for testing per-sample complexity processing.
    """

    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(10, 3)

    def forward(self, data, complexity):
        """
        Mock forward method.

        Args:
            data (torch.Tensor): Input tensor.
            complexity (torch.Tensor): Complexity tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.randn(data.size(0), 3)


class TestPerSampleComplexity(unittest.TestCase):
    """
    Unit tests for per-sample complexity module.
    """

    def setUp(self):
        """
        Sets up a mock model for testing.
        """
        self.model = MockModel()

    def test_process_batch(self):
        """
        Tests batch processing for complexity calculations.
        """
        data = torch.randn(20, 10)
        complexities = torch.randint(0, 3, (20,))

        result = process_batch_dynamic(self.model, data, complexities, 'cpu')
        self.assertEqual(result.size(0), data.size(0))


if __name__ == "__main__":
    unittest.main()
